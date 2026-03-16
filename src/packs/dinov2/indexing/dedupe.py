from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple


def by_sample_id(*, prov_cols: Sequence[str], config: Dict[str, Any] | None = None) -> Callable[[Tuple], Tuple]:
    if not prov_cols:
        raise ValueError("Provenance columns required to build a dedupe key")
    try:
        idx = prov_cols.index("sample_id")
    except ValueError as exc:
        raise ValueError("'sample_id' must be present in provenance columns for DINOv2 dedupe") from exc

    offset = 1  # score column in DecileTopKParquet rows

    def _key(item: Tuple) -> Tuple:
        prov = item[offset : offset + len(prov_cols)]
        return (prov[idx],)

    return _key


__all__ = ["by_sample_id"]
