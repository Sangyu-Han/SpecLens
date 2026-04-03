# src/core/indexing/topn_aggregator.py
"""
TopNAggregator — feature별 단일 min-heap으로 global top-N 수집.

DecileTopKParquet 대비:
  - D개 단일 heap만 사용 (decile × D 대신) → 메모리/속도 향상
  - update() 내부에서 bucket 계산 없음 → hot path 단순화
  - Feature frequency 추적: _freq_counts[d] = unique sample count per feature

Parquet 호환: 기존 DECILES_SCHEMA 그대로 사용, decile=0 고정, rank_in_decile=global rank.
"""
from __future__ import annotations

import heapq
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .decile_aggregator import RunFingerprint
from .decile_parquet_ledger import DecileParquetLedger

_PROV_DEFAULTS = {
    "sample_id": 0,
    "frame_idx": 0,
    "y": -1,
    "x": -1,
    "prompt_id": 0,
    "uid": -1,
}


class TopNAggregator:
    """
    Per-feature global top-N aggregator with optional frequency tracking.

    Interface contract (same as DecileTopKParquet):
      - update(acts_cpu, prov_cpu, *, stride_step, batch_max)
      - finalize_and_write(*, progress_cb)  → parquet rows written
      - state_dict() / load_state_dict(sd)
      - get_feature_frequencies()           → Dict[int, int]
    """

    def __init__(
        self,
        *,
        dict_size: int,
        top_n: int,
        layer_name: str,
        fp: RunFingerprint,
        ledger: DecileParquetLedger,
        prov_cols: Sequence[str],
        track_frequency: bool = True,
        dedupe_key_fn: Optional[Callable[[tuple], Any]] = None,
        slack: int = 4,
        rank: int = 0,
    ):
        self.D = int(dict_size)
        self.top_n = int(top_n)
        self.n_internal = int(top_n + slack)
        self.layer = str(layer_name)
        self.fp = fp
        self.ledger = ledger
        self.rank = int(rank)
        self.track_frequency = bool(track_frequency)

        # Per-feature single min-heap: item = (value, *prov_vals, stride_step)
        self.heaps: List[List[tuple]] = [[] for _ in range(self.D)]

        # Maxima tracking (for compatibility with checkpoint/DDP reduce)
        self.maxima = torch.zeros(self.D, dtype=torch.float32)

        # Provenance column mapping
        self.prov_cols = tuple(prov_cols)
        self._prov_len = len(self.prov_cols)
        self._prov_index = {name: idx for idx, name in enumerate(self.prov_cols)}
        self._idx_sample_id = self._prov_index.get("sample_id")
        self._idx_frame_idx = self._prov_index.get("frame_idx")
        self._idx_y = self._prov_index.get("y")
        self._idx_x = self._prov_index.get("x")
        self._idx_prompt_id = self._prov_index.get("prompt_id")
        self._idx_uid = self._prov_index.get("uid")

        # Feature frequency: count unique sample_ids per feature
        if self.track_frequency:
            self._freq_sets: List[Set[int]] = [set() for _ in range(self.D)]
        else:
            self._freq_sets = []

        self._total_samples_seen: Set[int] = set()

        # Dedupe support (same pattern as DecileTopKParquet)
        self._dedupe_key_fn = dedupe_key_fn
        if dedupe_key_fn is not None:
            self._dedupe_store: Optional[List[Dict[Any, tuple]]] = [
                dict() for _ in range(self.D)
            ]
        else:
            self._dedupe_store = None

        # Checkpoint counters
        self.run_epoch = 0
        self.run_b_in_epoch = 0
        self.global_steps = 0

    # ------------------------------------------------------------------ #
    # Dedupe helpers (simplified vs DecileTopKParquet — heap only)
    # ------------------------------------------------------------------ #
    def _dedupe_lookup(self, d: int, key: Any) -> Optional[tuple]:
        if self._dedupe_store is None:
            return None
        return self._dedupe_store[d].get(key)

    def _dedupe_register(self, d: int, item: tuple, key: Any) -> None:
        if self._dedupe_store is None:
            return
        self._dedupe_store[d][key] = item

    def _dedupe_forget_item(self, d: int, item: tuple) -> None:
        if self._dedupe_store is None:
            return
        key = self._dedupe_key_fn(item)
        store = self._dedupe_store[d]
        entry = store.get(key)
        if entry is not None and entry == item:
            store.pop(key, None)

    def _dedupe_remove_from_heap(self, d: int, item: tuple) -> None:
        heap = self.heaps[d]
        for idx, candidate in enumerate(heap):
            if candidate == item:
                heap.pop(idx)
                heapq.heapify(heap)
                return

    # ------------------------------------------------------------------ #
    # Update (hot path) — vectorized via torch.topk
    # ------------------------------------------------------------------ #
    def update(
        self,
        acts_cpu: torch.Tensor,
        prov_cpu: torch.Tensor,
        *,
        stride_step: int,
        batch_max: Optional[torch.Tensor] = None,
    ):
        """
        acts_cpu: [N, D] float32
        prov_cpu: [N, ?] long

        Vectorized hot path: uses torch.topk(acts_cpu, n_internal, dim=0) to find
        top-n_internal token candidates per feature in one C++ call, replacing the
        original O(N) Python loop + N torch.nonzero calls.

        Correctness guarantee: taking top-n_internal tokens per feature per batch
        is sufficient because any batch entry ranked > n_internal for feature d has
        value <= top_vals[n_internal-1, d]. After inserting the top-n_internal batch
        entries, the heap minimum is >= top_vals[n_internal-1, d], so the skipped
        entries cannot improve the heap.

        Early break: topk returns values in descending order, so when v <= heap_min
        for feature d, all remaining candidates for d are also <= heap_min → break.
        """
        import numpy as np

        N, D = acts_cpu.shape

        # Update maxima
        if batch_max is not None:
            max_vals = batch_max.detach()
            if max_vals.dim() != 1 or max_vals.numel() != D:
                raise ValueError(
                    f"batch_max must be a 1D tensor of length {D}, "
                    f"got shape {tuple(max_vals.shape)}"
                )
            if max_vals.device != self.maxima.device:
                max_vals = max_vals.to(self.maxima.device)
            if max_vals.dtype != self.maxima.dtype:
                max_vals = max_vals.to(self.maxima.dtype)
        else:
            max_vals = acts_cpu.max(dim=0).values
        self.maxima = torch.maximum(self.maxima, max_vals)

        prov_cols_len = self._prov_len
        track_freq = self.track_frequency
        freq_sets = self._freq_sets
        total_seen = self._total_samples_seen
        idx_sample_id = self._idx_sample_id
        stride_step_int = int(stride_step)
        prov_cols = self.prov_cols
        _pd = _PROV_DEFAULTS

        # ── provenance ──
        if prov_cpu.ndim == 2:
            prov_np: np.ndarray = prov_cpu.numpy()   # [N, prov_len]
        else:
            prov_np = prov_cpu.view(N, -1).numpy()
        prov_ncols = prov_np.shape[1]

        # ── track total_seen: O(N) ──
        if track_freq and idx_sample_id is not None and idx_sample_id < prov_ncols:
            total_seen.update(int(x) for x in prov_np[:, idx_sample_id])

        # ── per-feature default prov padding ──
        _prov_defaults_list: List[int] = [
            int(_pd.get(prov_cols[ci] if ci < len(prov_cols) else None, 0))
            for ci in range(prov_cols_len)
        ]

        # ── per-feature batch max for early-skip ──
        if batch_max is not None:
            bmax_np: np.ndarray = max_vals.numpy()  # [D] already on CPU
        else:
            bmax_np = acts_cpu.max(dim=0).values.numpy()

        # Heap-mins array: maintained for O(D) vectorized early-skip comparison
        if not hasattr(self, "_heap_mins_np"):
            self._heap_mins_np: np.ndarray = np.full(D, -1e38, dtype=np.float64)
        heap_mins: np.ndarray = self._heap_mins_np

        # ── vectorized nonzero: one C++ call → M (tok_idx, feat_idx, val) triples ──
        # This replaces N individual torch.nonzero(row) calls (original bottleneck).
        # We filter BEFORE extracting: only consider features where batch_max > heap_min
        # so that inactive features are entirely skipped.
        active_mask_d = bmax_np > heap_mins  # [D] bool
        active_d_arr: np.ndarray = np.where(active_mask_d)[0]  # indices of active features
        if active_d_arr.size == 0:
            return

        # One nonzero call on full acts to get all (tok, feat) pairs
        nz = torch.nonzero(acts_cpu, as_tuple=False)  # [M, 2]: M ≈ N×K active entries
        if nz.numel() == 0:
            return
        feat_ids_np: np.ndarray = nz[:, 1].numpy()     # [M] global feature indices
        tok_ids_np: np.ndarray  = nz[:, 0].numpy()     # [M]
        vals_np: np.ndarray     = acts_cpu[nz[:, 0], nz[:, 1]].numpy()  # [M]

        # Filter to entries belonging to active features
        in_active: np.ndarray = active_mask_d[feat_ids_np]  # [M] bool, vectorized
        if not in_active.any():
            return
        feat_ids_f = feat_ids_np[in_active]
        tok_ids_f  = tok_ids_np[in_active]
        vals_f     = vals_np[in_active]

        # Sort by feature id so entries for each feature are contiguous
        sort_order = np.argsort(feat_ids_f, kind='stable')
        feat_ids_s = feat_ids_f[sort_order]
        tok_ids_s  = tok_ids_f[sort_order]
        vals_s     = vals_f[sort_order]

        # Pre-convert to Python lists: list element access is 3-5x faster than
        # numpy scalar access in a Python for loop.
        vals_list: list  = vals_s.tolist()
        tok_list: list   = tok_ids_s.tolist()

        # Process per feature using counts from np.unique (avoids np.append)
        unique_feats, boundaries, counts = np.unique(feat_ids_s, return_index=True, return_counts=True)

        pv_len     = min(prov_ncols, prov_cols_len)
        n_internal = self.n_internal
        heaps      = self.heaps

        for uf_idx in range(len(unique_feats)):
            d     = int(unique_feats[uf_idx])
            start = int(boundaries[uf_idx])
            end   = start + int(counts[uf_idx])
            heap  = heaps[d]
            heap_min = float(heap_mins[d])

            for m in range(start, end):
                v = vals_list[m]  # Python float from pre-converted list
                if len(heap) >= n_internal and v <= heap_min:
                    continue  # below current heap threshold

                tok_idx   = tok_list[m]   # Python int from pre-converted list
                prov_row  = prov_np[tok_idx]
                prov_vals: List[int] = prov_row[:pv_len].tolist()
                if pv_len < prov_cols_len:
                    prov_vals = prov_vals + _prov_defaults_list[pv_len:]

                item = (v, *prov_vals, stride_step_int)

                # Frequency tracking
                if track_freq and idx_sample_id is not None and idx_sample_id < len(prov_vals):
                    freq_sets[d].add(prov_vals[idx_sample_id])

                # Dedupe check
                dedupe_key = None
                if self._dedupe_key_fn is not None:
                    dedupe_key = self._dedupe_key_fn(item)
                    existing = self._dedupe_lookup(d, dedupe_key)
                    if existing is not None:
                        if v <= existing[0]:
                            continue
                        self._dedupe_remove_from_heap(d, existing)
                        self._dedupe_store[d].pop(dedupe_key, None)

                # Min-heap insert
                removed = None
                if len(heap) < n_internal:
                    heapq.heappush(heap, item)
                elif v > heap_min:
                    removed = heapq.heapreplace(heap, item)
                else:
                    continue

                # Update heap_min after any heap change
                heap_min = heap[0][0]
                heap_mins[d] = heap_min

                # Dedupe bookkeeping
                if self._dedupe_key_fn is not None:
                    if removed is not None:
                        self._dedupe_forget_item(d, removed)
                    self._dedupe_register(d, item, dedupe_key)

    # ------------------------------------------------------------------ #
    # Finalize & write to Parquet
    # ------------------------------------------------------------------ #
    def finalize_and_write(
        self, *, progress_cb: Optional[Callable[[int], None]] = None
    ) -> int:
        rows: List[Dict[str, Any]] = []
        prov_cols_len = self._prov_len

        def _prov_value(values: tuple, idx: Optional[int], default: int) -> int:
            if idx is None or idx >= len(values):
                return default
            return int(values[idx])

        total_wrote = 0
        for d in range(self.D):
            heap = self.heaps[d]
            if not heap:
                continue
            # Sort descending by score, take top_n
            best = sorted(heap, key=lambda x: -x[0])[: self.top_n]
            for rnk, item in enumerate(best):
                v = item[0]
                prov_vals = item[1 : 1 + prov_cols_len]
                s_step = item[-1]
                rows.append(
                    {
                        "run_id": self.fp.run_id,
                        "layer": self.layer,
                        "unit": int(d),
                        "score": float(v),
                        "decile": 0,  # topn mode: always 0
                        "rank_in_decile": int(rnk),
                        "sample_id": _prov_value(
                            prov_vals, self._idx_sample_id, _PROV_DEFAULTS["sample_id"]
                        ),
                        "frame_idx": _prov_value(
                            prov_vals,
                            self._idx_frame_idx,
                            _PROV_DEFAULTS["frame_idx"],
                        ),
                        "y": _prov_value(prov_vals, self._idx_y, _PROV_DEFAULTS["y"]),
                        "x": _prov_value(prov_vals, self._idx_x, _PROV_DEFAULTS["x"]),
                        "prompt_id": _prov_value(
                            prov_vals,
                            self._idx_prompt_id,
                            _PROV_DEFAULTS["prompt_id"],
                        ),
                        "uid": _prov_value(
                            prov_vals, self._idx_uid, _PROV_DEFAULTS["uid"]
                        ),
                        "stride_step": int(s_step),
                        "meta_json": "",
                    }
                )
                total_wrote += 1
                if progress_cb:
                    progress_cb(1)

        if rows:
            self.ledger.write_rows(rows, rank=self.rank)
        return total_wrote

    # ------------------------------------------------------------------ #
    # Feature frequency
    # ------------------------------------------------------------------ #
    def get_feature_frequencies(self) -> Dict[int, int]:
        """Return {feature_id: num_unique_samples} for features with count > 0."""
        if not self.track_frequency:
            return {}
        return {d: len(s) for d, s in enumerate(self._freq_sets) if len(s) > 0}

    def get_total_samples_seen(self) -> int:
        return len(self._total_samples_seen)

    # ------------------------------------------------------------------ #
    # Checkpoint state
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        freq_counts = None
        if self.track_frequency:
            # Save as counts only (sets can be huge); for resume we lose
            # exact dedup but counts are approximate-correct
            freq_counts = [len(s) for s in self._freq_sets]
        return {
            "version": 1,
            "aggregator_type": "topn",
            "D": self.D,
            "top_n": self.top_n,
            "n_internal": self.n_internal,
            "heaps": self.heaps,
            "maxima": self.maxima,
            "layer": self.layer,
            "fp": self.fp.__dict__,
            "run_epoch": self.run_epoch,
            "run_b_in_epoch": self.run_b_in_epoch,
            "global_steps": self.global_steps,
            "prov_cols": list(self.prov_cols),
            "track_frequency": self.track_frequency,
            "freq_counts": freq_counts,
            "total_samples_seen": len(self._total_samples_seen),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.D = int(sd["D"])
        self.top_n = int(sd["top_n"])
        self.n_internal = int(sd.get("n_internal", self.top_n + 4))
        self.heaps = sd["heaps"]
        # Rebuild _heap_mins_np from loaded heaps
        D_loaded = int(sd["D"])
        self._heap_mins_np = np.full(D_loaded, -1e38, dtype=np.float64)
        for _d, _h in enumerate(self.heaps):
            if _h:
                self._heap_mins_np[_d] = _h[0][0]
        self.maxima = (
            sd["maxima"].clone().cpu()
            if isinstance(sd["maxima"], torch.Tensor)
            else torch.as_tensor(sd["maxima"], dtype=torch.float32)
        ).clamp_min(0)
        self.run_epoch = int(sd.get("run_epoch", 0))
        self.run_b_in_epoch = int(sd.get("run_b_in_epoch", 0))
        self.global_steps = int(sd.get("global_steps", 0))

        prov_cols = sd.get("prov_cols")
        if prov_cols is not None:
            self.prov_cols = tuple(prov_cols)
            self._prov_len = len(self.prov_cols)
            self._prov_index = {name: idx for idx, name in enumerate(self.prov_cols)}
            self._idx_sample_id = self._prov_index.get("sample_id")
            self._idx_frame_idx = self._prov_index.get("frame_idx")
            self._idx_y = self._prov_index.get("y")
            self._idx_x = self._prov_index.get("x")
            self._idx_prompt_id = self._prov_index.get("prompt_id")
            self._idx_uid = self._prov_index.get("uid")

        # Frequency: exact sets are not serialized (too large).
        # On resume, frequency tracking restarts from zero.
        # The final frequency will only reflect post-resume data.
        self.track_frequency = sd.get("track_frequency", self.track_frequency)
        if self.track_frequency:
            self._freq_sets = [set() for _ in range(self.D)]
            prev_counts = sd.get("freq_counts")
            if prev_counts is not None:
                import logging
                logging.getLogger("sae_index").warning(
                    "[TopNAggregator] Resumed from checkpoint — frequency "
                    "tracking restarted from zero (previous counts not restored)."
                )
        self._total_samples_seen = set()

        # Rebuild dedupe store from heaps
        if self._dedupe_key_fn is not None:
            self._dedupe_store = [dict() for _ in range(self.D)]
            for d in range(self.D):
                store_d = self._dedupe_store[d]
                for item in self.heaps[d]:
                    store_d[self._dedupe_key_fn(item)] = item

    # ------------------------------------------------------------------ #
    # Estimate rows (for progress reporting compatibility)
    # ------------------------------------------------------------------ #
    def estimate_final_rows(self) -> int:
        total = 0
        for d in range(self.D):
            total += min(self.top_n, len(self.heaps[d]))
        return total
