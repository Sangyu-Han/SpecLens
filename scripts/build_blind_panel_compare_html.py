from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _assert_panel_only_row(feature_key: str, row: dict[str, Any]) -> None:
    forbidden = [
        key
        for key in ("contact_sheet", "positive_contact_sheet", "negative_contact_sheet", "sheet_path")
        if key in row
    ]
    if forbidden:
        raise ValueError(
            f"{feature_key}: blind panel compare only supports panel-only sessions; found forbidden fields {forbidden}"
        )


def _metric_block(axis_summary: dict[str, Any] | None, variant_id: str) -> dict[str, Any]:
    if not axis_summary:
        return {}
    return dict(axis_summary.get(variant_id) or {})


def _supp_block(supp_summary: dict[str, Any] | None, variant_id: str) -> dict[str, Any]:
    if not supp_summary:
        return {}
    return dict(supp_summary.get(variant_id) or {})


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def _axis2_ndcg_key(axis2_overall: dict[str, Any]) -> str:
    for key in axis2_overall:
        if str(key).startswith("nDCG@"):
            return str(key)
    return "nDCG@16"


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _build_metric_cards(
    *,
    axis_summary: dict[str, Any] | None,
    supp_summary: dict[str, Any] | None,
    left_variant_id: str,
    right_variant_id: str,
    left_label: str,
    right_label: str,
) -> str:
    def _one(label: str, variant_id: str) -> str:
        axis1 = _metric_block((axis_summary or {}).get("axis1"), variant_id)
        axis2 = _metric_block((axis_summary or {}).get("axis2"), variant_id)
        supp = _supp_block((supp_summary or {}).get("supp_test"), variant_id)
        axis2_overall = dict(axis2.get("overall") or {})
        ndcg_key = _axis2_ndcg_key(axis2_overall)
        return f"""
        <div class="metric-card">
          <h3>{html.escape(label)}</h3>
          <div><span class="k">Axis 1 Top-1:</span> {_format_float(((axis1.get('overall') or {}).get('top1_accuracy')), 3)}</div>
          <div><span class="k">Axis 2 Top-1:</span> {_format_float(((axis2.get('overall') or {}).get('top1_accuracy')), 3)}</div>
          <div><span class="k">Axis 2 MRR:</span> {_format_float(((axis2.get('overall') or {}).get('mrr')), 3)}</div>
          <div><span class="k">Axis 2 {html.escape(ndcg_key)}:</span> {_format_float(axis2_overall.get(ndcg_key), 3)}</div>
          <div><span class="k">Supp Spearman:</span> {_format_float(((supp.get('overall') or {}).get('spearman_rho')), 3)}</div>
          <div><span class="k">Supp AUROC:</span> {_format_float(((supp.get('overall') or {}).get('auroc')), 3)}</div>
          <div><span class="k">Supp AP:</span> {_format_float(((supp.get('overall') or {}).get('average_precision')), 3)}</div>
          <div><span class="k">Supp MAE:</span> {_format_float(((supp.get('overall') or {}).get('mae')), 3)}</div>
        </div>
        """

    return f"""
    <div class="metric-grid">
      {_one(left_label, left_variant_id)}
      {_one(right_label, right_variant_id)}
    </div>
    """


def _resolve_review_image(session_dir: Path, review_image: str, out_dir: Path) -> str:
    resolved = (session_dir / review_image).resolve()
    return os.path.relpath(resolved, start=out_dir)


def _build_feature_rows(
    *,
    left_session_dir: Path,
    right_session_dir: Path,
    out_dir: Path,
    left_raw: dict[str, Any],
    right_raw: dict[str, Any],
) -> list[dict[str, Any]]:
    left_map = {str(row["feature_key"]): row for row in left_raw["features"]}
    right_map = {str(row["feature_key"]): row for row in right_raw["features"]}
    feature_keys = sorted(set(left_map) & set(right_map))
    rows: list[dict[str, Any]] = []
    for feature_key in feature_keys:
        left_row = left_map[feature_key]
        right_row = right_map[feature_key]
        _assert_panel_only_row(feature_key, left_row)
        _assert_panel_only_row(feature_key, right_row)
        rows.append(
            {
                "feature_key": feature_key,
                "block_idx": int(left_row["block_idx"]),
                "feature_id": int(left_row["feature_id"]),
                "left_examples": [
                    {
                        "rank": int(example["rank"]),
                        "sample_id": int(example["sample_id"]),
                        "token_idx": int(example["token_idx"]),
                        "image": _resolve_review_image(
                            left_session_dir,
                            str(example["review_image"]),
                            out_dir,
                        ),
                    }
                    for example in sorted(left_row.get("label_examples", []), key=lambda row: int(row["rank"]))
                ],
                "right_examples": [
                    {
                        "rank": int(example["rank"]),
                        "sample_id": int(example["sample_id"]),
                        "token_idx": int(example["token_idx"]),
                        "image": _resolve_review_image(
                            right_session_dir,
                            str(example["review_image"]),
                            out_dir,
                        ),
                    }
                    for example in sorted(right_row.get("label_examples", []), key=lambda row: int(row["rank"]))
                ],
                "left_output": dict(left_row.get("output") or {}),
                "right_output": dict(right_row.get("output") or {}),
                "left_elapsed_sec": float(left_row.get("elapsed_sec") or 0.0),
                "right_elapsed_sec": float(right_row.get("elapsed_sec") or 0.0),
                "left_returncode": int(left_row.get("returncode") or 0),
                "right_returncode": int(right_row.get("returncode") or 0),
            }
        )
    return rows


def _render_output_card(
    *,
    label: str,
    output: dict[str, Any],
    elapsed_sec: Any,
    returncode: Any,
) -> str:
    description = _first_text(
        output.get("description"),
        output.get("support_summary"),
        output.get("notes"),
        output.get("adjacent_context"),
        output.get("primary_locus"),
    )
    rationale = _first_text(output.get("rationale"))
    confidence = _format_float(output.get("confidence"), 2)
    detail_rows: list[str] = []
    if description:
        detail_rows.append(
            f"<div><span class='k'>description:</span> {html.escape(description)}</div>"
        )
    if rationale:
        detail_rows.append(
            f"<div><span class='k'>rationale:</span> {html.escape(rationale)}</div>"
        )
    if confidence:
        detail_rows.append(
            f"<div><span class='k'>confidence:</span> {html.escape(confidence)}</div>"
        )
    return f"""
    <div class="output-card">
      <h3>{html.escape(label)}</h3>
      <div class="label">{html.escape(str(output.get("canonical_label") or ""))}</div>
      {''.join(detail_rows)}
      <div class="muted">elapsed: {html.escape(_format_float(elapsed_sec, 1) + 's')}</div>
      <div class="muted">returncode: {html.escape(str(returncode))}</div>
    </div>
    """


def _build_html(
    *,
    payload: dict[str, Any],
    left_label: str,
    right_label: str,
    title: str,
) -> str:
    metric_cards = _build_metric_cards(
        axis_summary=payload.get("axis_summary"),
        supp_summary=payload.get("supp_summary"),
        left_variant_id=str(payload.get("left_metric_id") or ""),
        right_variant_id=str(payload.get("right_metric_id") or ""),
        left_label=left_label,
        right_label=right_label,
    )

    feature_chunks: list[str] = []
    for row in payload["features"]:
        def render_examples(examples: list[dict[str, Any]], side_label: str) -> str:
            cards = []
            for ex in examples:
                cards.append(
                    f"""
                    <div class="panel">
                      <div class="meta">Example {int(ex['rank']) + 1}</div>
                      <img src="{html.escape(str(ex['image']))}" alt="{html.escape(side_label)} example {int(ex['rank']) + 1}">
                    </div>
                    """
                )
            return "\n".join(cards)

        left_output = row["left_output"]
        right_output = row["right_output"]
        feature_chunks.append(
            f"""
            <section class="feature">
              <div class="feature-meta mono">{html.escape(row['feature_key'])}</div>
              <div class="panel-grid two-up">
                <div>
                  <h3>{html.escape(left_label)}</h3>
                  <div class="panel-grid five-up">
                    {render_examples(row['left_examples'], left_label)}
                  </div>
                </div>
                <div>
                  <h3>{html.escape(right_label)}</h3>
                  <div class="panel-grid five-up">
                    {render_examples(row['right_examples'], right_label)}
                  </div>
                </div>
              </div>
              <div class="output-grid">
                {_render_output_card(
                    label=left_label,
                    output=left_output,
                    elapsed_sec=row.get('left_elapsed_sec'),
                    returncode=row.get('left_returncode'),
                )}
                {_render_output_card(
                    label=right_label,
                    output=right_output,
                    elapsed_sec=row.get('right_elapsed_sec'),
                    returncode=row.get('right_returncode'),
                )}
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 2200px; margin: 0 auto; padding: 24px; }}
    .hero, .feature {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .meta, .muted {{ color:#6b645d; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, monospace; }}
    .metric-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:16px; margin-top:16px; }}
    .metric-card, .output-card, .panel {{ border:1px solid #ece3d6; border-radius:12px; padding:12px; background:#fff; }}
    .panel-grid {{ display:grid; gap:12px; }}
    .panel-grid.two-up {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top:16px; }}
    .panel-grid.five-up {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
    .panel img {{ width:100%; border-radius:10px; border:1px solid #ddd3c6; background:#efebe4; }}
    .output-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:16px; margin-top:16px; }}
    .label {{ font-size:22px; font-weight:700; margin-bottom:10px; }}
    .k {{ font-weight:700; }}
    h1,h2,h3,p,div {{ overflow-wrap:anywhere; }}
    @media (max-width: 1400px) {{
      .panel-grid.five-up {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    }}
    @media (max-width: 960px) {{
      .metric-grid, .output-grid, .panel-grid.two-up {{ grid-template-columns: 1fr; }}
      .panel-grid.five-up {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p class="meta">{html.escape(left_label)}: {html.escape(payload['left_session'])}</p>
      <p class="meta">{html.escape(right_label)}: {html.escape(payload['right_session'])}</p>
      <p class="meta">Model: {html.escape(str(payload['model']))} | Reasoning: {html.escape(str(payload['reasoning_effort']))} | Features: {len(payload['features'])}</p>
      <p class="meta">Each condition shows the five individual input panels passed directly to the labeler. No contact sheet is used.</p>
      {metric_cards}
    </section>
    {''.join(feature_chunks)}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-session-dir", required=True)
    parser.add_argument("--right-session-dir", required=True)
    parser.add_argument("--left-label", default="Blind ERF-only")
    parser.add_argument("--right-label", default="Blind SAE-only")
    parser.add_argument("--left-metric-id", default="blind_erf_only")
    parser.add_argument("--right-metric-id", default="blind_sae_only")
    parser.add_argument("--axis-summary", default="")
    parser.add_argument("--supp-summary", default="")
    parser.add_argument("--title", default="Blind Panel Compare")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-name", default="blind_panel_compare.html")
    args = parser.parse_args()

    left_session_dir = Path(args.left_session_dir)
    right_session_dir = Path(args.right_session_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_raw = _read_json(left_session_dir / "raw_predictions.json")
    right_raw = _read_json(right_session_dir / "raw_predictions.json")
    axis_summary = _read_json(Path(args.axis_summary)) if str(args.axis_summary).strip() else None
    supp_summary = _read_json(Path(args.supp_summary)) if str(args.supp_summary).strip() else None

    payload = {
        "left_session": str(left_raw["session_name"]),
        "right_session": str(right_raw["session_name"]),
        "model": str(left_raw["model"]),
        "reasoning_effort": str(left_raw["reasoning_effort"]),
        "left_metric_id": str(args.left_metric_id),
        "right_metric_id": str(args.right_metric_id),
        "axis_summary": axis_summary,
        "supp_summary": supp_summary,
        "features": _build_feature_rows(
            left_session_dir=left_session_dir,
            right_session_dir=right_session_dir,
            out_dir=out_dir,
            left_raw=left_raw,
            right_raw=right_raw,
        ),
    }

    (out_dir / args.out_name).write_text(
        _build_html(
            payload=payload,
            left_label=str(args.left_label),
            right_label=str(args.right_label),
            title=str(args.title),
        )
    )
    (out_dir / "compare_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_dir / args.out_name)


if __name__ == "__main__":
    main()
