from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml


def _to_python_list(t: Optional[Any]) -> List[int]:
    if t is None:
        return []
    if torch.is_tensor(t):
        return [int(v) for v in t.detach().cpu().tolist()]
    if isinstance(t, Iterable) and not isinstance(t, (str, bytes)):
        return [int(v) for v in t]
    return [int(t)]


class ClipJSONLedger:
    """Light-weight offline ledger for CLIP batches (JSONL per rank)."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        part_modulus: int = 128,
        filename_prefix: str = "clip_samples",
    ) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        pid = os.getpid()
        self.path = self.root / f"{filename_prefix}_r{rank}_p{pid}.jsonl"
        self.part_modulus = max(1, int(part_modulus))
        self._cache: Dict[int, str] = {}

    def write_from_batch(self, batch: Dict[str, Any]) -> None:
        """Persist the minimal metadata for the current batch."""
        sample_ids = _to_python_list(batch.get("sample_id"))
        if not sample_ids:
            return
        labels = _to_python_list(batch.get("label"))
        paths: List[str] = list(batch.get("path") or [])
        # Align list lengths without crashing when optional fields are missing
        while len(labels) < len(sample_ids):
            labels.append(-1)
        if len(paths) < len(sample_ids):
            paths.extend([""] * (len(sample_ids) - len(paths)))

        records = []
        for idx, sid in enumerate(sample_ids):
            rec = {
                "sample_id": int(sid),
                "path": paths[idx],
                "label": int(labels[idx]) if labels else None,
                "part": int(sid) % self.part_modulus,
            }
            records.append(rec)

        if self._cache is not None:
            for rec in records:
                if rec["path"]:
                    self._cache[int(rec["sample_id"])] = rec["path"]

        with self.path.open("a", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_from_bvd(self, batch: Dict[str, Any]) -> None:
        """Legacy alias; prefer write_from_batch."""
        self.write_from_batch(batch)

    def lookup(self, sample_ids: Iterable[int]) -> Dict[int, str]:
        wanted = {int(s) for s in sample_ids if s is not None}
        if not wanted or not self.root.exists():
            return {}
        if not self._cache:
            cache: Dict[int, str] = {}
            for jsonl in self.root.glob("*.jsonl"):
                try:
                    for line in jsonl.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            rec = yaml.safe_load(line)
                        sid = rec.get("sample_id")
                        if sid is None:
                            continue
                        p = rec.get("path") or ""
                        if p and int(sid) not in cache:
                            cache[int(sid)] = p
                except Exception:
                    continue
            self._cache = cache
        return {sid: self._cache.get(sid, "") for sid in wanted if sid in self._cache}


__all__ = ["ClipJSONLedger"]
