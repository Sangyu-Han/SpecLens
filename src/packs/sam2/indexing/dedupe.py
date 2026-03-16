from __future__ import annotations
from typing import Sequence, Callable, Dict, Optional, Any


def by_sample_and_uid(*, prov_cols: Sequence[str], config: Optional[Dict[str, Any]] = None) -> Callable[[tuple], tuple]:
    if not prov_cols:
        raise ValueError("SAM2 provenance columns are required to build a dedupe key")

    cfg = config or {}
    fields = cfg.get("fields") or ["sample_id", "uid"]
    prov_cols = tuple(prov_cols)
    indices = []
    for name in fields:
        if name not in prov_cols:
            raise ValueError(
                f"[SAM2 dedupe] field '{name}' is not present in provenance columns {prov_cols}"
            )
        indices.append(prov_cols.index(name))

    prov_len = len(prov_cols)
    offset = 1  # DecileTopKParquet item layout: (score, *prov, stride_step)

    def _key(item: tuple) -> tuple:
        prov_vals = item[offset:offset + prov_len]
        return tuple(prov_vals[idx] for idx in indices)

    return _key