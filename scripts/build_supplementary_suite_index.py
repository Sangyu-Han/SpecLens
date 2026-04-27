#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build suite-level supplementary HTML index.")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--repeat-count", type=int, default=10)
    return parser.parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rel(base: Path, path: Path) -> str:
    return os.path.relpath(path, start=base)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _metrics_cell(summary: dict | None, split: str, variant: str) -> str:
    if not summary:
        return "<td colspan='4'>pending</td>"
    blob = summary.get(split, {}).get(variant, {}).get("overall", {})
    return (
        f"<td>{_fmt(blob.get('spearman_rho'))}</td>"
        f"<td>{_fmt(blob.get('auroc'))}</td>"
        f"<td>{_fmt(blob.get('average_precision'))}</td>"
        f"<td>{_fmt(blob.get('mae'))}</td>"
    )


def main() -> None:
    args = _parse_args()
    workspace_root = args.workspace_root
    outputs_root = workspace_root / "outputs"
    review_root = outputs_root / "review_sessions" / "supplementary_pilot_sessions"
    supp_root = outputs_root / "supplementary_pilot_sessions"
    suite_root = outputs_root / "repeat_suites" / args.session_prefix
    suite_root.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    completed = 0

    for idx in range(1, args.repeat_count + 1):
        repeat = f"r{idx:02d}"
        session_name = f"{args.session_prefix}_{repeat}_supp_shortdesc"
        summary_path = supp_root / session_name / "summary.json"
        report_path = supp_root / session_name / "report.md"
        human_root = review_root / session_name / "human_eval"
        summary = _load_json(summary_path)
        done = summary is not None
        if done:
            completed += 1

        def _link(path: Path, label: str) -> str:
            if not path.exists():
                return "pending"
            return f"<a href='{html.escape(_rel(suite_root, path))}'>{html.escape(label)}</a>"

        links = " | ".join(
            [
                _link(human_root / "index.html", "human index"),
                _link(human_root / "erf_cyan_cross" / "supp_valid.html", "ERF valid"),
                _link(human_root / "erf_cyan_cross" / "supp_test.html", "ERF test"),
                _link(human_root / "sae_only" / "supp_valid.html", "SAE valid"),
                _link(human_root / "sae_only" / "supp_test.html", "SAE test"),
                _link(report_path, "report"),
            ]
        )

        rows.append(
            f"""
            <tr>
              <td>{repeat}</td>
              <td>{'done' if done else 'pending'}</td>
              { _metrics_cell(summary, 'supp_valid', 'erf_cyan_cross') }
              { _metrics_cell(summary, 'supp_valid', 'sae_only') }
              { _metrics_cell(summary, 'supp_test', 'erf_cyan_cross') }
              { _metrics_cell(summary, 'supp_test', 'sae_only') }
              <td>{links}</td>
            </tr>
            """
        )

    out_path = suite_root / "supplementary_human_eval_index.html"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Supplementary Human Eval Index</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #f6f2ea; color: #231f1a; }}
    .page {{ max-width: 1800px; margin: 0 auto; padding: 24px; }}
    .hero, .card {{ background: #fffdf9; border: 1px solid #e1d8cb; border-radius: 16px; padding: 18px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(0,0,0,0.04); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e7ded1; padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f2ece2; position: sticky; top: 0; }}
    a {{ color: #0f5c7a; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .muted {{ color: #6b645d; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Supplementary Human Eval Index</h1>
      <div class="muted">Suite: {html.escape(args.session_prefix)}</div>
      <div class="muted">Completed repeats: {completed}/{args.repeat_count}</div>
      <div class="muted">Metrics shown per repeat: rho, AUROC, AP, MAE for ERF and SAE on supp_valid and supp_test.</div>
    </div>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th rowspan="2">repeat</th>
            <th rowspan="2">status</th>
            <th colspan="4">supp_valid ERF</th>
            <th colspan="4">supp_valid SAE</th>
            <th colspan="4">supp_test ERF</th>
            <th colspan="4">supp_test SAE</th>
            <th rowspan="2">links</th>
          </tr>
          <tr>
            <th>rho</th><th>AUROC</th><th>AP</th><th>MAE</th>
            <th>rho</th><th>AUROC</th><th>AP</th><th>MAE</th>
            <th>rho</th><th>AUROC</th><th>AP</th><th>MAE</th>
            <th>rho</th><th>AUROC</th><th>AP</th><th>MAE</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
