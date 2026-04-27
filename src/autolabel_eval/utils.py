from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row)))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def feature_key(block_idx: int, feature_id: int) -> str:
    return f"block_{int(block_idx)}/feature_{int(feature_id)}"


def token_uid(block_idx: int, sample_id: int, token_idx: int) -> str:
    return f"block_{int(block_idx)}/sample_{int(sample_id)}/tok_{int(token_idx)}"

