from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pyarrow as pa
import pyarrow.dataset as ds

from src.core.indexing.registry_utils import ensure_dir, unique_basename


def _to_python_list(t: Optional[Any]) -> List[int]:
    if t is None:
        return []
    if hasattr(t, "detach") and hasattr(t, "cpu"):
        try:
            return [int(v) for v in t.detach().cpu().tolist()]
        except Exception:
            pass
    if isinstance(t, Iterable) and not isinstance(t, (str, bytes)):
        return [int(v) for v in t]
    return [int(t)]


def _to_str_list(val: Optional[Any]) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (str, bytes, Path)):
        return [str(val)]
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return [str(v) for v in val]
    return [str(val)]


def _normalize_meta_json(val: Optional[Any]) -> List[str]:
    def _one(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    if val is None:
        return []
    if isinstance(val, dict):
        return [_one(val)]
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return [_one(v) for v in val]
    return [_one(val)]


def _part(sample_id: int, modulus: int) -> int:
    return int(sample_id) % int(modulus)


class OfflineMetaParquetLedger:
    """
    Offline meta ledger for CLIP-style runs (sample_id -> path lookup via Parquet).
    Hive partitions: part=int(sample_id % M) and optionally run_id for pruning.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        part_modulus: int = 128,
        compression: str = "zstd",
        run_id: str | None = None,
        partition_by_run_id: bool = False,
        basename_prefix: str = "offline_meta",
    ) -> None:
        self.root = Path(root_dir)
        self.dir = self.root / "parquet"
        ensure_dir(self.dir)
        self.part_modulus = max(1, int(part_modulus))
        self.run_id = str(run_id) if run_id is not None else None
        self.partition_by_run_id = bool(partition_by_run_id)
        self.basename_prefix = basename_prefix
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))

        self._pq_kwargs = {
            "compression": compression,
            "use_dictionary": ["path", "run_id", "meta_json"],
            "write_statistics": True,
        }

        self._schema = pa.schema(
            [
                ("sample_id", pa.int64()),
                ("path", pa.string()),
                ("label", pa.int32()),
                ("run_id", pa.string()),
                ("meta_json", pa.string()),
                ("part", pa.int32()),
            ]
        )

        part_fields = [pa.field("part", pa.int32())]
        if self.partition_by_run_id:
            part_fields.insert(0, pa.field("run_id", pa.string()))
        self._partition_schema = pa.schema(part_fields)
        self._partitioning = ds.partitioning(self._partition_schema, flavor="hive")

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _table_from_rows(self, rows: List[Dict[str, Any]]) -> pa.Table:
        cols = {}
        for f in self._schema:
            name = f.name
            vals = [r.get(name, None) for r in rows]
            cols[name] = pa.array(vals, type=f.type)
        return pa.table(cols, schema=self._schema)

    def write_from_batch(self, batch: Dict[str, Any]) -> int:
        sample_ids = _to_python_list(batch.get("sample_id"))
        if not sample_ids:
            return 0

        paths = _to_str_list(batch.get("path"))
        labels = _to_python_list(batch.get("label"))
        run_ids = _to_str_list(batch.get("run_id"))
        meta_json = _normalize_meta_json(batch.get("meta_json"))

        while len(paths) < len(sample_ids):
            paths.append("")
        while len(labels) < len(sample_ids):
            labels.append(None)
        if run_ids or self.run_id is not None or self.partition_by_run_id:
            fill = self.run_id if self.run_id is not None else ""
            while len(run_ids) < len(sample_ids):
                run_ids.append(fill)
        while len(meta_json) < len(sample_ids):
            meta_json.append("")

        rows: List[Dict[str, Any]] = []
        for idx, sid in enumerate(sample_ids):
            if sid is None:
                continue
            rows.append(
                {
                    "sample_id": int(sid),
                    "path": paths[idx] if paths else "",
                    "label": int(labels[idx]) if labels and labels[idx] is not None else None,
                    "run_id": run_ids[idx] if run_ids else self.run_id,
                    "meta_json": meta_json[idx] if meta_json else "",
                    "part": _part(int(sid), self.part_modulus),
                }
            )

        if not rows:
            return 0

        tbl = self._table_from_rows(rows)
        base = unique_basename(self.basename_prefix, rank=self.rank)
        fmt = ds.ParquetFileFormat()
        file_options = fmt.make_write_options(**self._pq_kwargs)
        max_parts = max(8192, self.part_modulus * 16)

        ds.write_dataset(
            data=tbl,
            base_dir=str(self.dir),
            format="parquet",
            partitioning=self._partitioning,
            basename_template=f"{base}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            file_options=file_options,
            max_partitions=max_parts,
            use_threads=True,
        )
        return len(rows)

    def write_from_bvd(self, batch: Dict[str, Any]) -> int:
        """Legacy alias; prefer write_from_batch."""
        return self.write_from_batch(batch)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def as_dataset(self):
        return ds.dataset(str(self.dir), format="parquet", partitioning=self._partitioning)

    def lookup(self, sample_ids: Sequence[int]) -> Dict[int, str]:
        wanted = {int(s) for s in sample_ids if s is not None}
        if not wanted:
            return {}
        try:
            dset = self.as_dataset()
        except Exception:
            return {}

        parts = sorted({_part(sid, self.part_modulus) for sid in wanted})
        filt = ds.field("part").isin(parts) & ds.field("sample_id").isin(sorted(wanted))
        if self.run_id is not None:
            filt = (ds.field("run_id") == self.run_id) & filt

        tbl = dset.to_table(filter=filt, columns=["sample_id", "path"])
        out: Dict[int, str] = {}
        sids = tbl.column("sample_id").to_pylist() if tbl.num_rows else []
        paths = tbl.column("path").to_pylist() if tbl.num_rows else []
        for sid, path in zip(sids, paths):
            if path and sid not in out:
                out[int(sid)] = str(path)
        return out

    def __repr__(self) -> str:
        compression = self._pq_kwargs.get("compression")
        return (
            f"OfflineMetaParquetLedger(root='{self.root}', dir='{self.dir}', "
            f"part_modulus={self.part_modulus}, run_id={self.run_id!r}, "
            f"compression={compression!r})"
        )


__all__ = ["OfflineMetaParquetLedger"]
