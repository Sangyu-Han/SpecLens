#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class RepeatPaths:
    prefix: str
    selection_manifest: Path
    source_dir: Path
    source_manifest: Path
    cyan_dir: Path
    erf_dir: Path
    sae_dir: Path
    axis_dir: Path
    supp_dir: Path
    pipeline_dir: Path
    erf_raw: Path
    sae_raw: Path
    axis_summary: Path
    supp_summary: Path
    pipeline_manifest: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor repeat-suite progress.")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--heartbeat-interval", type=int, default=20)
    return parser.parse_args()


def _suite_root(workspace_root: Path, session_prefix: str) -> Path:
    return workspace_root / "outputs" / "repeat_suites" / session_prefix


def _repeat_paths(workspace_root: Path, session_prefix: str, repeat_idx: int) -> RepeatPaths:
    repeat_tag = f"{session_prefix}_r{repeat_idx:02d}"
    outputs_root = workspace_root / "outputs"
    return RepeatPaths(
        prefix=repeat_tag,
        selection_manifest=outputs_root / "manifests" / f"{repeat_tag}_selection.json",
        source_dir=outputs_root / "review_sessions" / f"{repeat_tag}_source_panel5_t90_attr10",
        source_manifest=outputs_root / "review_sessions" / f"{repeat_tag}_source_panel5_t90_attr10" / "selection_manifest.json",
        cyan_dir=outputs_root / "review_sessions" / f"{repeat_tag}_erfcyan_source",
        erf_dir=outputs_root / "review_sessions" / f"{repeat_tag}_erfcross_shortdesc",
        sae_dir=outputs_root / "review_sessions" / f"{repeat_tag}_sae_shortdesc",
        axis_dir=outputs_root / "axis_pilot_sessions" / f"{repeat_tag}_axis_shortdesc",
        supp_dir=outputs_root / "supplementary_pilot_sessions" / f"{repeat_tag}_supp_shortdesc",
        pipeline_dir=outputs_root / "pipeline_runs" / repeat_tag,
        erf_raw=outputs_root / "review_sessions" / f"{repeat_tag}_erfcross_shortdesc" / "raw_predictions.json",
        sae_raw=outputs_root / "review_sessions" / f"{repeat_tag}_sae_shortdesc" / "raw_predictions.json",
        axis_summary=outputs_root / "axis_pilot_sessions" / f"{repeat_tag}_axis_shortdesc" / "summary.json",
        supp_summary=outputs_root / "supplementary_pilot_sessions" / f"{repeat_tag}_supp_shortdesc" / "summary.json",
        pipeline_manifest=outputs_root / "pipeline_runs" / repeat_tag / "pipeline_manifest.json",
    )


def _safe_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _safe_dir_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir())


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _feature_total(paths: RepeatPaths) -> int:
    manifest = _load_json(paths.selection_manifest)
    if isinstance(manifest, dict):
        selected = manifest.get("selected_feature_keys")
        if isinstance(selected, list):
            return len(selected)
    return 0


def _prediction_json_count(session_dir: Path) -> int:
    predictions_dir = session_dir / "predictions"
    if not predictions_dir.exists():
        return 0
    return sum(1 for p in predictions_dir.glob("*.json"))


def _ps_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,etime,cmd"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.splitlines()


def _matching_lines(lines: Iterable[str], needle: str) -> list[str]:
    return [line.strip() for line in lines if needle in line]


def _stage_for_repeat(paths: RepeatPaths, ps_lines: list[str]) -> str:
    prefix = paths.prefix
    if paths.pipeline_manifest.exists():
        return "completed"
    if any("run_openai_style_supplementary_pilot.py" in line and prefix in line for line in ps_lines):
        return "supplementary_running"
    if paths.supp_summary.exists():
        return "supplementary_done"
    if any("run_prompt_axis_pilot.py" in line and prefix in line for line in ps_lines):
        return "axis_running"
    if paths.axis_summary.exists():
        return "axis_done"
    if any("run_blind_panel_labeling.py" in line and f"{prefix}_sae_shortdesc" in line for line in ps_lines):
        return "sae_labeling_running"
    if paths.sae_raw.exists():
        return "sae_label_done"
    if any("run_blind_panel_labeling.py" in line and f"{prefix}_erfcross_shortdesc" in line for line in ps_lines):
        return "erf_labeling_running"
    if paths.erf_raw.exists():
        return "erf_label_done"
    if any("build_erf_indicator_variants_source.py" in line and prefix in line for line in ps_lines):
        return "erf_cyan_source_running"
    if paths.cyan_dir.exists():
        return "erf_cyan_source_done"
    if any("build_blind_panel_source_from_ledger.py" in line and prefix in line for line in ps_lines):
        return "source_render_running"
    if paths.source_manifest.exists() or paths.source_dir.exists() or paths.selection_manifest.exists():
        return "source_render_done"
    return "pending"


def _snapshot(workspace_root: Path, session_prefix: str, repeat_count: int) -> dict:
    ps_lines = _ps_lines()
    suite_root = _suite_root(workspace_root, session_prefix)
    repeats: list[dict] = []
    completed = 0
    for idx in range(1, repeat_count + 1):
        paths = _repeat_paths(workspace_root, session_prefix, idx)
        stage = _stage_for_repeat(paths, ps_lines)
        feature_total = _feature_total(paths)
        if stage == "completed":
            completed += 1
        repeats.append(
            {
                "repeat": f"r{idx:02d}",
                "stage": stage,
                "feature_total": feature_total,
                "selection_manifest_exists": paths.selection_manifest.exists(),
                "source_feature_count": _safe_dir_count(paths.source_dir / "assets"),
                "cyan_feature_count": _safe_dir_count(paths.cyan_dir / "assets"),
                "erf_prediction_count": _prediction_json_count(paths.erf_dir),
                "sae_prediction_count": _prediction_json_count(paths.sae_dir),
                "axis_files": _safe_file_count(paths.axis_dir),
                "supp_files": _safe_file_count(paths.supp_dir),
                "pipeline_manifest_exists": paths.pipeline_manifest.exists(),
                "paths": {
                    "selection_manifest": str(paths.selection_manifest),
                    "axis_summary": str(paths.axis_summary),
                    "supp_summary": str(paths.supp_summary),
                },
            }
        )
    suite_ps = _matching_lines(ps_lines, session_prefix)
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_prefix": session_prefix,
        "suite_root": str(suite_root),
        "completed_repeats": completed,
        "repeat_count": repeat_count,
        "active_repeats": [r["repeat"] for r in repeats if r["stage"].endswith("_running")],
        "repeats": repeats,
        "suite_process_lines": suite_ps,
    }


def _timestamp(snapshot: dict) -> str:
    return f"[{snapshot['timestamp']}]"


def _compact_progress_line(snapshot: dict, item: dict) -> str:
    total = item["feature_total"] or "?"
    stage = item["stage"]
    if stage in {"source_render_running", "source_render_done"}:
        return f"{_timestamp(snapshot)} {item['repeat']} source {item['source_feature_count']}/{total}"
    if stage in {"erf_cyan_source_running", "erf_cyan_source_done"}:
        return f"{_timestamp(snapshot)} {item['repeat']} cyan_source {item['cyan_feature_count']}/{total}"
    if stage in {"erf_labeling_running", "erf_label_done"}:
        return f"{_timestamp(snapshot)} {item['repeat']} erf_label {item['erf_prediction_count']}/{total}"
    if stage in {"sae_labeling_running", "sae_label_done"}:
        return f"{_timestamp(snapshot)} {item['repeat']} sae_label {item['sae_prediction_count']}/{total}"
    return f"{_timestamp(snapshot)} {item['repeat']} {stage}"


def _axis_item_lines(snapshot: dict, axis_summary_path: Path, repeat: str) -> list[str]:
    summary = _load_json(axis_summary_path)
    if not isinstance(summary, dict):
        return []
    lines: list[str] = []
    for axis_name in ("axis1", "axis2"):
        axis_blob = summary.get(axis_name)
        if not isinstance(axis_blob, dict):
            continue
        for variant_id, variant_blob in axis_blob.items():
            if not isinstance(variant_blob, dict):
                continue
            overall = variant_blob.get("overall", {})
            top1 = overall.get("top1_accuracy")
            if top1 is not None:
                lines.append(f"{_timestamp(snapshot)} {repeat} {axis_name} {variant_id} top1={top1:.3f}")
            per_item = variant_blob.get("per_item")
            if not isinstance(per_item, list):
                continue
            for row in per_item:
                if not isinstance(row, dict):
                    continue
                feature_key = row.get("feature_key", "?")
                if axis_name == "axis1":
                    correct = row.get("correct")
                else:
                    correct = row.get("top1_correct")
                if correct is None:
                    continue
                lines.append(f"{_timestamp(snapshot)} {repeat} {axis_name} {variant_id} {feature_key} correct={int(correct)}")
    return lines


def _supp_summary_lines(snapshot: dict, supp_summary_path: Path, repeat: str) -> list[str]:
    summary = _load_json(supp_summary_path)
    if not isinstance(summary, dict):
        return []
    lines: list[str] = []
    for split_name in ("supp_valid", "supp_test"):
        split_blob = summary.get(split_name)
        if not isinstance(split_blob, dict):
            continue
        for variant_id, variant_blob in split_blob.items():
            if not isinstance(variant_blob, dict):
                continue
            overall = variant_blob.get("overall", {})
            rho = overall.get("spearman_rho")
            auroc = overall.get("auroc")
            ap = overall.get("average_precision")
            mae = overall.get("mae")
            if None not in (rho, auroc, ap, mae):
                lines.append(
                    f"{_timestamp(snapshot)} {repeat} {split_name} {variant_id} "
                    f"rho={rho:.3f} auroc={auroc:.3f} ap={ap:.3f} mae={mae:.3f}"
                )
    return lines


def _load_state(path: Path) -> dict:
    payload = _load_json(path)
    if isinstance(payload, dict):
        return payload
    return {"repeats": {}}


def _append_lines(progress_log: Path, lines: list[str]) -> None:
    if not lines:
        return
    with progress_log.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)
            fh.write("\n")


def main() -> None:
    args = _parse_args()
    suite_root = _suite_root(args.workspace_root, args.session_prefix)
    suite_root.mkdir(parents=True, exist_ok=True)
    progress_log = suite_root / "progress.log"
    progress_status = suite_root / "progress_status.json"
    progress_state = suite_root / "progress_state.json"

    while True:
        snapshot = _snapshot(args.workspace_root, args.session_prefix, args.repeat_count)
        progress_status.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        state = _load_state(progress_state)
        repeat_state = state.setdefault("repeats", {})
        log_lines: list[str] = []

        for item in snapshot["repeats"]:
            repeat = item["repeat"]
            prev = repeat_state.setdefault(
                repeat,
                {
                    "stage": None,
                    "source_feature_count": -1,
                    "cyan_feature_count": -1,
                    "erf_prediction_count": -1,
                    "sae_prediction_count": -1,
                    "axis_logged": False,
                    "supp_logged": False,
                },
            )

            if item["stage"] != prev["stage"]:
                log_lines.append(f"{_timestamp(snapshot)} {repeat} stage={item['stage']}")
                prev["stage"] = item["stage"]

            progress_key_by_stage = {
                "source_render_running": "source_feature_count",
                "source_render_done": "source_feature_count",
                "erf_cyan_source_running": "cyan_feature_count",
                "erf_cyan_source_done": "cyan_feature_count",
                "erf_labeling_running": "erf_prediction_count",
                "erf_label_done": "erf_prediction_count",
                "sae_labeling_running": "sae_prediction_count",
                "sae_label_done": "sae_prediction_count",
            }
            active_progress_key = progress_key_by_stage.get(item["stage"])
            for key in (
                "source_feature_count",
                "cyan_feature_count",
                "erf_prediction_count",
                "sae_prediction_count",
            ):
                if item[key] != prev[key]:
                    if key == active_progress_key and item[key] >= 0:
                        log_lines.append(_compact_progress_line(snapshot, item))
                    prev[key] = item[key]

            axis_summary = Path(item["paths"]["axis_summary"])
            if axis_summary.exists() and not prev["axis_logged"]:
                log_lines.extend(_axis_item_lines(snapshot, axis_summary, repeat))
                prev["axis_logged"] = True

            supp_summary = Path(item["paths"]["supp_summary"])
            if supp_summary.exists() and not prev["supp_logged"]:
                log_lines.extend(_supp_summary_lines(snapshot, supp_summary, repeat))
                prev["supp_logged"] = True

        _append_lines(progress_log, log_lines)
        progress_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
