from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autolabel_eval.config import EvalConfig
from autolabel_eval.legacy import LegacyRuntime


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        workspace_root=Path(args.workspace_root),
        repo_root=ROOT,
        model_name=str(args.vision_model_name),
        deciles_root_override=Path(args.deciles_root),
        offline_meta_root_override=Path(args.offline_meta_root) if str(args.offline_meta_root).strip() else None,
        checkpoints_root_override=Path(args.checkpoints_root),
        checkpoint_relpath_template=str(args.checkpoint_pattern),
        dataset_root_override=Path(args.dataset_root),
    )


def _parquet_glob(config: EvalConfig, block_idx: int) -> str:
    return str((config.deciles_root / f"layer_part=model.blocks.{int(block_idx)}" / "**/*.parquet").as_posix())


def _load_top5_rows(con: duckdb.DuckDBPyConnection, config: EvalConfig, block_idx: int) -> pd.DataFrame:
    path = _parquet_glob(config, block_idx)
    return con.execute(
        f"""
        with ranked as (
          select unit, score, sample_id, frame_idx, y, x,
                 row_number() over (partition by unit order by score desc, sample_id asc, x asc) as rn
          from read_parquet('{path}')
        )
        select unit, score, sample_id, frame_idx, y, x, rn
        from ranked
        where rn <= 5
        order by unit asc, rn asc
        """
    ).fetchdf()


def _cls_feature_matrix_for_paths(
    runtime: LegacyRuntime,
    *,
    block_idx: int,
    sample_paths: list[tuple[int, str]],
    batch_size: int,
) -> dict[int, np.ndarray]:
    sae = runtime.load_sae(block_idx)
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(sample_paths), int(batch_size)):
        chunk = sample_paths[start : start + int(batch_size)]
        tensors = [runtime.load_image_tensor(path) for _sid, path in chunk]
        batch = torch.cat(tensors, dim=0)
        buf: list[torch.Tensor] = []

        handle = runtime.model.blocks[int(block_idx)].register_forward_hook(
            lambda _m, _i, output: buf.append(output if torch.is_tensor(output) else output[0])
        )
        try:
            with torch.no_grad():
                runtime.model(batch)
        finally:
            handle.remove()
        if not buf:
            raise RuntimeError(f"Failed to capture block output for block {block_idx}")
        block_out = buf[0]
        with torch.no_grad():
            feature_acts = sae(block_out).get("feature_acts")
        cls_acts = feature_acts[:, 0, :].detach().cpu().numpy().astype(np.float32)
        for (sample_id, _path), vec in zip(chunk, cls_acts, strict=True):
            out[int(sample_id)] = vec
    return out


def _discover_block(
    con: duckdb.DuckDBPyConnection,
    runtime: LegacyRuntime,
    config: EvalConfig,
    *,
    block_idx: int,
    batch_size: int,
) -> dict[str, Any]:
    top5 = _load_top5_rows(con, config, int(block_idx))
    groups = {int(unit): df.sort_values("rn") for unit, df in top5.groupby("unit", sort=True)}

    seed_units: list[int] = []
    unit_rows: dict[int, list[dict[str, Any]]] = {}
    unit_sample_ids: dict[int, list[int]] = {}
    all_sample_ids: set[int] = set()

    for unit, df in groups.items():
        # Seed criterion: at least one CLS row appears in the feature's top-5 ledger rows.
        if not ((df["y"] == -1) & (df["x"] == 0)).any():
            continue
        rows = [
            {
                "rank": int(row.rn),
                "sample_id": int(row.sample_id),
                "frame_idx": int(row.frame_idx),
                "score": float(row.score),
                "y": int(row.y),
                "x": int(row.x),
            }
            for row in df.itertuples(index=False)
        ]
        sample_ids = [int(v) for v in pd.unique(df["sample_id"])]
        seed_units.append(int(unit))
        unit_rows[int(unit)] = rows
        unit_sample_ids[int(unit)] = sample_ids
        all_sample_ids.update(sample_ids)

    path_map = runtime.lookup_paths(sorted(all_sample_ids))
    sample_paths = [(int(sid), str(path_map[int(sid)])) for sid in sorted(all_sample_ids) if int(sid) in path_map]
    cls_cache = _cls_feature_matrix_for_paths(
        runtime,
        block_idx=int(block_idx),
        sample_paths=sample_paths,
        batch_size=int(batch_size),
    )

    accepted_features: list[dict[str, Any]] = []
    rejected_features: list[dict[str, Any]] = []

    for unit in seed_units:
        sample_ids = unit_sample_ids[int(unit)]
        cls_scores: list[dict[str, Any]] = []
        failed_sample_ids: list[int] = []
        for sample_id in sample_ids:
            vec = cls_cache.get(int(sample_id))
            if vec is None:
                failed_sample_ids.append(int(sample_id))
                continue
            score = float(vec[int(unit)])
            cls_scores.append({"sample_id": int(sample_id), "cls_activation": float(score)})
            if not (score > 0.0):
                failed_sample_ids.append(int(sample_id))

        payload = {
            "feature_key": f"block_{int(block_idx)}/feature_{int(unit)}",
            "block_idx": int(block_idx),
            "feature_id": int(unit),
            "seed_top5_rows": list(unit_rows[int(unit)]),
            "seed_sample_ids": list(sample_ids),
            "cls_activation_checks": cls_scores,
            "cls_activation_stats": {
                "min": float(min((row["cls_activation"] for row in cls_scores), default=0.0)),
                "max": float(max((row["cls_activation"] for row in cls_scores), default=0.0)),
                "mean": float(np.mean([row["cls_activation"] for row in cls_scores])) if cls_scores else 0.0,
            },
        }
        if failed_sample_ids:
            payload["rejected_sample_ids"] = failed_sample_ids
            rejected_features.append(payload)
        else:
            accepted_features.append(payload)

    accepted_features.sort(key=lambda row: (-float(row["cls_activation_stats"]["mean"]), int(row["feature_id"])))
    rejected_features.sort(key=lambda row: int(row["feature_id"]))
    return {
        "block_idx": int(block_idx),
        "seed_feature_count": int(len(seed_units)),
        "accepted_feature_count": int(len(accepted_features)),
        "rejected_feature_count": int(len(rejected_features)),
        "unique_seed_sample_count": int(len(all_sample_ids)),
        "accepted_features": accepted_features,
        "rejected_features": rejected_features,
    }


def _summary_text(payload: dict[str, Any]) -> str:
    lines = [
        "# CLS Feature Discovery",
        "",
        f"- Criterion: {payload['criterion']['seed']}",
        f"- Confirmation: {payload['criterion']['confirm']}",
        "",
    ]
    for block_key in sorted(payload["blocks"], key=lambda k: int(k)):
        block = payload["blocks"][block_key]
        lines.extend(
            [
                f"## Block {block_key}",
                "",
                f"- Seed features: {block['seed_feature_count']}",
                f"- Accepted CLS features: {block['accepted_feature_count']}",
                f"- Rejected seed features: {block['rejected_feature_count']}",
                f"- Unique seed samples forwarded: {block['unique_seed_sample_count']}",
                "",
            ]
        )
        preview = block["accepted_features"][:20]
        if preview:
            lines.append("Top accepted features:")
            lines.append("")
            for row in preview:
                stats = row["cls_activation_stats"]
                lines.append(
                    f"- `{row['feature_key']}`: min={stats['min']:.3f}, mean={stats['mean']:.3f}, max={stats['max']:.3f}"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover CLS-token features via ledger seeding and forward confirmation.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--session-name", default="clip50k_cls_feature_discovery_20260424")
    parser.add_argument("--vision-model-name", default="vit_base_patch16_clip_224")
    parser.add_argument("--blocks", nargs="*", type=int, default=[2, 6, 10])
    parser.add_argument("--deciles-root", required=True)
    parser.add_argument("--offline-meta-root", default="")
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = _build_config(args)
    out_dir = Path(args.workspace_root) / "outputs" / "manifests" / str(args.session_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    runtime = LegacyRuntime(config)
    try:
        blocks_payload: dict[str, Any] = {}
        for block_idx in [int(v) for v in args.blocks]:
            print(f"[discover] block {block_idx}", flush=True)
            blocks_payload[str(block_idx)] = _discover_block(
                con,
                runtime,
                config,
                block_idx=int(block_idx),
                batch_size=int(args.batch_size),
            )

        payload = {
            "session_name": str(args.session_name),
            "vision_model_name": str(args.vision_model_name),
            "criterion": {
                "seed": "feature top-5 ledger rows contain at least one CLS row (y=-1, x=0)",
                "confirm": "for every unique sample_id appearing in those top-5 rows, forward-pass CLS activation for the feature is > 0",
            },
            "blocks": blocks_payload,
        }
        _write_json(out_dir / "cls_feature_manifest.json", payload)
        _write_text(out_dir / "cls_feature_summary.md", _summary_text(payload))
        print(out_dir)
    finally:
        runtime.close()
        con.close()


if __name__ == "__main__":
    main()
