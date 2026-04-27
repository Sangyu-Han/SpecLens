from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(SCRIPT_DIR))
except ValueError:
    pass
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch

    try:
        import torchvision.ops  # noqa: F401
    except Exception:
        for _name in list(sys.modules):
            if _name == "torchvision" or _name.startswith("torchvision."):
                sys.modules.pop(_name, None)
        try:
            _tv_lib = torch.library.Library("torchvision", "DEF")
            _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            pass
except Exception:
    torch = None  # type: ignore[assignment]

from PIL import Image

from autolabel_eval.config import EvalConfig
from autolabel_eval.legacy import LegacyRuntime, token_record_from_row
from autolabel_eval.rendering import (
    CLIP_ZERO_RGB,
    save_feature_actmap_masked_image,
    save_support_mask_image,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _slug(value: str) -> str:
    return str(value).replace("/", "__")


def _feature_key(block_idx: int, feature_id: int) -> str:
    return f"block_{int(block_idx)}/feature_{int(feature_id)}"


def _parse_feature_key(value: str) -> tuple[int, int]:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty feature key")
    try:
        block_text, feature_text = text.split("/", 1)
        block_idx = int(block_text.replace("block_", ""))
        feature_id = int(feature_text.replace("feature_", ""))
    except Exception as exc:
        raise ValueError(f"Invalid feature key: {value!r}") from exc
    return block_idx, feature_id


def _build_config_from_args(args: argparse.Namespace) -> EvalConfig:
    config = replace(
        EvalConfig(),
        workspace_root=Path(args.workspace_root),
        model_name=str(args.vision_model_name),
        deciles_root_override=Path(args.deciles_root),
        checkpoints_root_override=Path(args.checkpoints_root),
        checkpoint_relpath_template=str(args.checkpoint_pattern),
        dataset_root_override=Path(args.dataset_root),
        erf_recovery_threshold=float(args.erf_threshold),
        erf_support_min_normalized_attribution=float(args.erf_support_min_attribution),
    )
    config.ensure_dirs()
    return config


def _collect_top_rows(
    runtime: LegacyRuntime,
    config: EvalConfig,
    *,
    block_idx: int,
    feature_id: int,
    top_k: int,
    unique_sample_ids: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = runtime.load_decile_frame(block_idx)
    feature_rows = frame[frame["unit"] == int(feature_id)].sort_values("score", ascending=False)
    if feature_rows.empty:
        raise ValueError(f"{_feature_key(block_idx, feature_id)} is absent from the decile frame")

    sample_ids = [int(sample_id) for sample_id in feature_rows["sample_id"].astype(int).unique().tolist()]
    sid_to_path = runtime.lookup_paths(sample_ids)
    accepted: list[dict[str, Any]] = []
    seen_sample_ids: set[int] = set()

    for row in feature_rows.itertuples(index=False):
        sample_id = int(row.sample_id)
        token_idx = runtime.row_x_to_token_idx(int(row.x))
        if token_idx < 0 or token_idx >= config.n_patches:
            continue
        if unique_sample_ids and sample_id in seen_sample_ids:
            continue
        image_path = sid_to_path.get(sample_id, "")
        if not image_path:
            continue
        validation = runtime.validate_feature_token(
            image_path,
            int(block_idx),
            int(feature_id),
            token_idx,
            float(row.score),
        )
        if validation is None:
            continue
        accepted.append(
            token_record_from_row(
                block_idx,
                int(feature_id),
                row,
                image_path,
                validation,
                token_idx=token_idx,
            )
        )
        seen_sample_ids.add(sample_id)
        if len(accepted) >= int(top_k):
            break

    stats = {
        "candidate_rows": int(len(feature_rows)),
        "requested_top_k": int(top_k),
        "accepted_top_k": int(len(accepted)),
        "unique_sample_ids": bool(unique_sample_ids),
    }
    return accepted, stats


def _render_feature(
    runtime: LegacyRuntime,
    *,
    session_dir: Path,
    block_idx: int,
    feature_id: int,
    top_rows: list[dict[str, Any]],
    selection_stats: dict[str, Any],
) -> dict[str, Any]:
    feature_key = _feature_key(block_idx, feature_id)
    feature_dir = session_dir / "assets" / _slug(feature_key)
    rendered_examples: list[dict[str, Any]] = []

    for rank, row in enumerate(top_rows):
        image_path = str(row["image_path"])
        token_idx = int(row["target_patch_idx"])

        sae_fire_path = feature_dir / f"example_{rank:02d}_sae_fire.png"
        erf_path = feature_dir / f"example_{rank:02d}_feature_erf_on_original.png"
        erf_json_path = feature_dir / f"example_{rank:02d}_feature_erf.json"

        actmap = runtime.feature_activation_map(image_path, int(block_idx), int(feature_id))
        save_feature_actmap_masked_image(
            image_path,
            actmap,
            sae_fire_path,
            token_idx=token_idx,
            background_color=CLIP_ZERO_RGB,
        )

        erf_payload = runtime.cautious_feature_erf(
            image_path,
            int(block_idx),
            token_idx,
            int(feature_id),
        )
        if not list(erf_payload.get("support_indices") or []) and not bool(
            erf_payload.get("recovery_threshold_reached", False)
        ):
            raise RuntimeError(
                f"Empty ERF support for {feature_key} rank={rank} sample_id={int(row['sample_id'])} "
                f"token_idx={token_idx}; threshold_reached={bool(erf_payload.get('recovery_threshold_reached', False))} "
                f"valid_ranking_len={len(erf_payload.get('valid_ranking') or [])}"
            )
        save_support_mask_image(
            image_path,
            erf_payload["support_indices"],
            erf_path,
            token_idx=token_idx,
            mode="masked_black",
            background_color=CLIP_ZERO_RGB,
            mask_resample=Image.NEAREST,
            token_marker_style="dot",
            token_marker_color=(40, 220, 80),
        )
        _write_json(erf_json_path, erf_payload)

        rendered_examples.append(
            {
                "rank": int(rank),
                "sample_id": int(row["sample_id"]),
                "token_idx": int(token_idx),
                "image_path": str(image_path),
                "ledger_score": float(row["ledger_score"]),
                "validation": dict(row["validation"]),
                "sae_fire": str(sae_fire_path.relative_to(session_dir)),
                "feature_erf_on_original": str(erf_path.relative_to(session_dir)),
                "feature_erf_json": str(erf_json_path.relative_to(session_dir)),
            }
        )

    return {
        "feature_key": feature_key,
        "block_idx": int(block_idx),
        "feature_id": int(feature_id),
        "selection_stats": dict(selection_stats),
        "label_examples": rendered_examples,
    }


def _build_source_review_html(out_path: Path, payload: dict[str, Any]) -> None:
    feature_blocks: list[str] = []
    for feature in payload["features"]:
        panels_html: list[str] = []
        for example in feature["label_examples"]:
            panels_html.append(
                f"""
      <div class="panel">
        <div class="meta">Example {int(example["rank"]) + 1} | sample_id={int(example["sample_id"])} | tok={int(example["token_idx"])}</div>
        <div class="pair">
          <figure>
            <img src="{html.escape(str(example["feature_erf_on_original"]))}" alt="ERF panel">
            <figcaption>ERF on original</figcaption>
          </figure>
          <figure>
            <img src="{html.escape(str(example["sae_fire"]))}" alt="SAE panel">
            <figcaption>SAE fire</figcaption>
          </figure>
        </div>
      </div>
"""
            )
        stats = dict(feature.get("selection_stats") or {})
        feature_blocks.append(
            f"""
    <section class="feature">
      <h2>{html.escape(str(feature["feature_key"]))}</h2>
      <div class="meta">candidate_rows={stats.get("candidate_rows", "")} | requested_top_k={stats.get("requested_top_k", "")} | accepted_top_k={stats.get("accepted_top_k", "")}</div>
      <div class="stack">
        {''.join(panels_html)}
      </div>
    </section>
"""
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blind Panel Source Review</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    .hero, .feature {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .meta {{ color:#6b645d; }}
    .pair {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin-top:10px; }}
    .panel {{ border:1px solid #ddd3c6; border-radius:12px; background:#fff; padding:12px; }}
    figure {{ margin:0; }}
    img {{ width:100%; border-radius:10px; border:1px solid #ddd3c6; background:#f0ebe2; }}
    figcaption {{ margin-top:8px; color:#5f574f; font-size:13px; }}
    .stack {{ display:grid; gap:12px; margin-top:14px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Blind Panel Source Review</h1>
      <div class="meta">Session: {html.escape(str(payload["session_name"]))} | feature_count={len(payload["features"])}</div>
      <div class="meta">skipped_features={len(payload.get("skipped_features") or [])}</div>
      <div class="meta">Panels: left=ERF on original, right=SAE fire</div>
    </section>
    {''.join(feature_blocks)}
  </div>
</body>
</html>
"""
    out_path.write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blind-panel source assets from decile-ledger top-k rows.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--feature-key", action="append", default=[])
    parser.add_argument("--feature-key-file", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-duplicate-sample-ids", action="store_true")
    parser.add_argument("--fail-on-insufficient-top-k", action="store_true")
    parser.add_argument("--vision-model-name", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    parser.add_argument(
        "--deciles-root",
        default="/home/sangyu/Desktop/Master/SpecLens/outputs/spec_lens_store/clip_50k_index/deciles",
    )
    parser.add_argument(
        "--checkpoints-root",
        default="/home/sangyu/Desktop/Master/SpecLens/outputs/spec_lens_store/clip_50k_sae",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="model.blocks.{block_idx}/step_0050000_tokens_204800000.pt",
    )
    parser.add_argument("--dataset-root", default="/data/datasets/imagenet/val")
    parser.add_argument("--erf-threshold", type=float, default=0.90)
    parser.add_argument("--erf-support-min-attribution", type=float, default=0.10)
    args = parser.parse_args()

    feature_keys = [str(v).strip() for v in list(args.feature_key) if str(v).strip()]
    if str(args.feature_key_file).strip():
        file_lines = Path(args.feature_key_file).read_text().splitlines()
        feature_keys.extend(str(line).strip() for line in file_lines if str(line).strip())
    feature_keys = list(dict.fromkeys(feature_keys))
    if not feature_keys:
        raise SystemExit("No feature keys provided")

    config = _build_config_from_args(args)
    session_dir = config.workspace_root / "outputs" / "review_sessions" / str(args.session_name)
    session_dir.mkdir(parents=True, exist_ok=True)

    runtime = LegacyRuntime(config)
    try:
        rendered_features: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        skipped_features: list[dict[str, Any]] = []
        for feature_key_text in feature_keys:
            block_idx, feature_id = _parse_feature_key(feature_key_text)
            top_rows, stats = _collect_top_rows(
                runtime,
                config,
                block_idx=block_idx,
                feature_id=feature_id,
                top_k=int(args.top_k),
                unique_sample_ids=not bool(args.allow_duplicate_sample_ids),
            )
            if len(top_rows) < int(args.top_k):
                reason = (
                    f"accepted_top_k={int(stats.get('accepted_top_k', 0))} "
                    f"< requested_top_k={int(stats.get('requested_top_k', args.top_k))}"
                )
                skipped_payload = {
                    "feature_key": _feature_key(block_idx, feature_id),
                    "block_idx": int(block_idx),
                    "feature_id": int(feature_id),
                    "reason": reason,
                    "selection_stats": dict(stats),
                }
                if bool(args.fail_on_insufficient_top_k):
                    raise RuntimeError(
                        f"Insufficient top-k rows for {_feature_key(block_idx, feature_id)}: {reason}"
                    )
                skipped_features.append(skipped_payload)
                print(f"[skipped] {_feature_key(block_idx, feature_id)} | {reason}", flush=True)
                continue
            rendered_features.append(
                _render_feature(
                    runtime,
                    session_dir=session_dir,
                    block_idx=block_idx,
                    feature_id=feature_id,
                    top_rows=top_rows,
                    selection_stats=stats,
                )
            )
            selection_rows.append(
                {
                    "feature_key": _feature_key(block_idx, feature_id),
                    "block_idx": int(block_idx),
                    "feature_id": int(feature_id),
                }
            )
            print(f"[rendered] {_feature_key(block_idx, feature_id)}", flush=True)
    finally:
        runtime.close()

    if not rendered_features:
        raise SystemExit("No features rendered successfully; all requested features were skipped.")

    payload = {
        "session_name": str(args.session_name),
        "source": "decile_ledger_topk",
        "workspace_root": str(config.workspace_root),
        "selection": selection_rows,
        "requested_feature_count": int(len(feature_keys)),
        "rendered_feature_count": int(len(rendered_features)),
        "skipped_features": skipped_features,
        "prompt": {
            "top_k": int(args.top_k),
            "unique_sample_ids": not bool(args.allow_duplicate_sample_ids),
            "vision_model_name": str(args.vision_model_name),
            "deciles_root": str(args.deciles_root),
            "checkpoints_root": str(args.checkpoints_root),
            "checkpoint_pattern": str(args.checkpoint_pattern),
            "dataset_root": str(args.dataset_root),
            "erf_recovery_threshold": float(args.erf_threshold),
            "erf_support_min_normalized_attribution": float(args.erf_support_min_attribution),
        },
        "features": rendered_features,
    }
    _write_json(session_dir / "selection_manifest.json", payload)
    _build_source_review_html(session_dir / "source_review.html", payload)
    _write_json(
        session_dir / "review_summary.json",
        {
            "session_name": str(args.session_name),
            "selection_manifest_json": str(session_dir / "selection_manifest.json"),
            "source_review_html": str(session_dir / "source_review.html"),
        },
    )
    print(session_dir)


if __name__ == "__main__":
    main()
