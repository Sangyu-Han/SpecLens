from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    # Keep child processes on the GNU threading layer to avoid MKL/libgomp failures.
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    return env


def _extract_feature_keys_from_manifest(path: Path) -> list[str]:
    payload = _read_json(path)
    keys: list[str] = []
    if isinstance(payload, dict):
        if "selected_feature_keys" in payload:
            keys.extend(str(key).strip() for key in list(payload["selected_feature_keys"]) if str(key).strip())
        elif "features" in payload:
            for row in list(payload["features"]):
                if isinstance(row, str) and str(row).strip():
                    keys.append(str(row).strip())
                elif isinstance(row, dict) and str(row.get("feature_key") or "").strip():
                    keys.append(str(row["feature_key"]).strip())
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, str) and str(row).strip():
                keys.append(str(row).strip())
            elif isinstance(row, dict) and str(row.get("feature_key") or "").strip():
                keys.append(str(row["feature_key"]).strip())
    if not keys:
        raise ValueError(f"No feature keys found in manifest: {path}")
    return keys


def _collect_feature_keys(
    *,
    feature_keys: list[str],
    feature_key_file: str,
    feature_manifest_json: str,
) -> list[str]:
    out: list[str] = []
    for key in feature_keys:
        text = str(key).strip()
        if text:
            out.append(text)
    if str(feature_key_file).strip():
        for line in Path(feature_key_file).read_text().splitlines():
            text = str(line).strip()
            if text:
                out.append(text)
    if str(feature_manifest_json).strip():
        out.extend(_extract_feature_keys_from_manifest(Path(feature_manifest_json)))
    deduped: list[str] = []
    seen: set[str] = set()
    for key in out:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    if not deduped:
        raise SystemExit("No feature keys provided. Use --feature-key, --feature-key-file, or --feature-manifest-json.")
    return deduped


def _assert_absent(path: Path, *, label: str) -> None:
    if path.exists():
        raise SystemExit(f"{label} already exists: {path}")


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    label: str,
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    print(f"[start] {label}", flush=True)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    elapsed = time.time() - started
    record = {
        "label": label,
        "cmd": list(cmd),
        "cwd": str(cwd),
        "elapsed_sec": elapsed,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    command_log.append(record)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed with code {proc.returncode}\n"
            f"CMD: {shlex.join(cmd)}\n"
            f"STDOUT tail:\n{proc.stdout[-4000:]}\n"
            f"STDERR tail:\n{proc.stderr[-4000:]}"
        )
    print(f"[done] {label} ({elapsed:.1f}s)", flush=True)
    return record


def _config_cli_args(args: argparse.Namespace) -> list[str]:
    return [
        "--workspace-root",
        str(Path(args.workspace_root).resolve()),
        "--vision-model-name",
        str(args.vision_model_name),
        "--deciles-root",
        str(args.deciles_root),
        "--checkpoints-root",
        str(args.checkpoints_root),
        "--checkpoint-pattern",
        str(args.checkpoint_pattern),
        "--dataset-root",
        str(args.dataset_root),
        "--erf-threshold",
        str(float(args.erf_threshold)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--feature-key", action="append", default=[])
    parser.add_argument("--feature-key-file", default="")
    parser.add_argument("--feature-manifest-json", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--python-bin", default=sys.executable)
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
    parser.add_argument("--label-model", default="gpt-5.4")
    parser.add_argument("--label-reasoning-effort", default="xhigh")
    parser.add_argument("--label-prompt-style", default="label_shortdesc_where_v1")
    parser.add_argument("--jobs-label", type=int, default=6)
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--judge-reasoning-effort", default="xhigh")
    parser.add_argument("--jobs-axis", type=int, default=4)
    parser.add_argument("--jobs-supp", type=int, default=4)
    parser.add_argument("--axis2-candidate-count", type=int, default=16)
    parser.add_argument("--skip-axis", action="store_true")
    parser.add_argument("--skip-supp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    review_root = workspace_root / "outputs" / "review_sessions"
    axis_root = workspace_root / "outputs" / "axis_pilot_sessions"
    supp_root = workspace_root / "outputs" / "supplementary_pilot_sessions"
    pipeline_root = workspace_root / "outputs" / "pipeline_runs" / str(args.session_prefix)
    pipeline_root.mkdir(parents=True, exist_ok=True)

    feature_keys = _collect_feature_keys(
        feature_keys=list(args.feature_key),
        feature_key_file=str(args.feature_key_file),
        feature_manifest_json=str(args.feature_manifest_json),
    )
    _write_text(pipeline_root / "feature_keys.txt", "\n".join(feature_keys) + "\n")

    source_session = f"{args.session_prefix}_source_panel5_t90_attr10"
    cyan_source_session = f"{args.session_prefix}_erfcyan_source"
    erf_session = f"{args.session_prefix}_erfcross_shortdesc"
    sae_session = f"{args.session_prefix}_sae_shortdesc"
    axis_session = f"{args.session_prefix}_axis_shortdesc"
    supp_session = f"{args.session_prefix}_supp_shortdesc"

    source_dir = review_root / source_session
    cyan_source_dir = review_root / cyan_source_session
    erf_dir = review_root / erf_session
    sae_dir = review_root / sae_session
    axis_dir = axis_root / axis_session
    supp_dir = supp_root / supp_session

    if not bool(args.resume):
        paths_to_guard = {
            "source session": source_dir,
            "cyan source session": cyan_source_dir,
            "ERF label session": erf_dir,
            "SAE label session": sae_dir,
        }
        if not bool(args.skip_axis):
            paths_to_guard["axis session"] = axis_dir
        if not bool(args.skip_supp):
            paths_to_guard["supplementary session"] = supp_dir
        for label, path in paths_to_guard.items():
            _assert_absent(path, label=label)

    python_bin = str(args.python_bin or sys.executable).strip()
    if not python_bin:
        raise SystemExit("Could not resolve python interpreter. Pass --python-bin explicitly.")

    command_log: list[dict[str, Any]] = []

    source_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "build_blind_panel_source_from_ledger.py"),
        "--workspace-root",
        str(workspace_root),
        "--session-name",
        source_session,
        "--top-k",
        str(int(args.top_k)),
        "--vision-model-name",
        str(args.vision_model_name),
        "--deciles-root",
        str(args.deciles_root),
        "--checkpoints-root",
        str(args.checkpoints_root),
        "--checkpoint-pattern",
        str(args.checkpoint_pattern),
        "--dataset-root",
        str(args.dataset_root),
        "--erf-threshold",
        str(float(args.erf_threshold)),
        "--erf-support-min-attribution",
        str(float(args.erf_support_min_attribution)),
    ]
    for key in feature_keys:
        source_cmd.extend(["--feature-key", key])
    source_complete = (source_dir / "selection_manifest.json").exists()
    if not source_complete:
        _run_command(source_cmd, cwd=ROOT, label="build source panels", command_log=command_log)

    cyan_source_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "build_erf_indicator_variants_source.py"),
        "--source-session-dir",
        str(source_dir),
        "--out-session-dir",
        str(cyan_source_dir),
    ]
    cyan_source_complete = (cyan_source_dir / "selection_manifest.json").exists()
    if not cyan_source_complete:
        _run_command(cyan_source_cmd, cwd=ROOT, label="build cyan-cross ERF source", command_log=command_log)

    erf_label_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "run_blind_panel_labeling.py"),
        "--workspace-root",
        str(workspace_root),
        "--source-session",
        cyan_source_session,
        "--session-name",
        erf_session,
        "--panel-kind",
        "blind_erf_cyan_cross_v1",
        "--model",
        str(args.label_model),
        "--reasoning-effort",
        str(args.label_reasoning_effort),
        "--prompt-style",
        str(args.label_prompt_style),
        "--jobs",
        str(int(args.jobs_label)),
    ]
    erf_complete = (erf_dir / "raw_predictions.json").exists() and (erf_dir / "selection_manifest.json").exists()
    if not erf_complete:
        _run_command(erf_label_cmd, cwd=ROOT, label="label cyan-cross ERF panels", command_log=command_log)

    sae_label_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "run_blind_panel_labeling.py"),
        "--workspace-root",
        str(workspace_root),
        "--source-session",
        source_session,
        "--session-name",
        sae_session,
        "--panel-kind",
        "blind_sae_only_v1",
        "--model",
        str(args.label_model),
        "--reasoning-effort",
        str(args.label_reasoning_effort),
        "--prompt-style",
        str(args.label_prompt_style),
        "--jobs",
        str(int(args.jobs_label)),
    ]
    sae_complete = (sae_dir / "raw_predictions.json").exists() and (sae_dir / "selection_manifest.json").exists()
    if not sae_complete:
        _run_command(sae_label_cmd, cwd=ROOT, label="label SAE-only panels", command_log=command_log)

    compare_dir = pipeline_root / "compare_html"
    compare_html_name = "erf_vs_sae_shortdesc_compare.html"

    def build_compare(axis_summary: Path | None, supp_summary: Path | None, *, label: str) -> None:
        cmd = [
            python_bin,
            str(SCRIPTS_DIR / "build_blind_panel_compare_html.py"),
            "--left-session-dir",
            str(review_root / erf_session),
            "--right-session-dir",
            str(review_root / sae_session),
            "--left-label",
            "ERF cyan cross shortdesc",
            "--right-label",
            "SAE-only shortdesc",
            "--left-metric-id",
            "erf_cyan_cross",
            "--right-metric-id",
            "sae_only",
            "--title",
            "ERF cyan cross vs SAE-only shortdesc",
            "--out-dir",
            str(compare_dir),
            "--out-name",
            compare_html_name,
        ]
        if axis_summary is not None:
            cmd.extend(["--axis-summary", str(axis_summary)])
        if supp_summary is not None:
            cmd.extend(["--supp-summary", str(supp_summary)])
        _run_command(cmd, cwd=ROOT, label=label, command_log=command_log)

    build_compare(None, None, label="build initial compare HTML")

    axis_summary_path: Path | None = None
    supp_summary_path: Path | None = None

    if not bool(args.skip_axis):
        if (axis_dir / "summary.json").exists():
            axis_summary_path = axis_dir / "summary.json"
        else:
            axis_cmd = [
                python_bin,
                str(SCRIPTS_DIR / "run_prompt_axis_pilot.py"),
                "--session-name",
                axis_session,
                "--workspace-root",
                str(workspace_root),
                "--model",
                str(args.judge_model),
                "--reasoning-effort",
                str(args.judge_reasoning_effort),
                "--jobs",
                str(int(args.jobs_axis)),
                "--axis2-candidate-count",
                str(int(args.axis2_candidate_count)),
                "--variant",
                f"erf_cyan_cross={erf_session}",
                "--variant",
                f"sae_only={sae_session}",
                "--vision-model-name",
                str(args.vision_model_name),
                "--deciles-root",
                str(args.deciles_root),
                "--checkpoints-root",
                str(args.checkpoints_root),
                "--checkpoint-pattern",
                str(args.checkpoint_pattern),
                "--dataset-root",
                str(args.dataset_root),
                "--erf-threshold",
                str(float(args.erf_threshold)),
            ]
            _run_command(axis_cmd, cwd=ROOT, label="run axis pilot", command_log=command_log)
            axis_summary_path = axis_dir / "summary.json"

    if not bool(args.skip_supp):
        if (supp_dir / "summary.json").exists():
            supp_summary_path = supp_dir / "summary.json"
        else:
            supp_cmd = [
                python_bin,
                str(SCRIPTS_DIR / "run_openai_style_supplementary_pilot.py"),
                "--session-name",
                supp_session,
                "--workspace-root",
                str(workspace_root),
                "--model",
                str(args.judge_model),
                "--reasoning-effort",
                str(args.judge_reasoning_effort),
                "--jobs",
                str(int(args.jobs_supp)),
                "--variant",
                f"erf_cyan_cross={erf_session}",
                "--variant",
                f"sae_only={sae_session}",
                "--vision-model-name",
                str(args.vision_model_name),
                "--deciles-root",
                str(args.deciles_root),
                "--checkpoints-root",
                str(args.checkpoints_root),
                "--checkpoint-pattern",
                str(args.checkpoint_pattern),
                "--dataset-root",
                str(args.dataset_root),
                "--erf-threshold",
                str(float(args.erf_threshold)),
            ]
            _run_command(supp_cmd, cwd=ROOT, label="run supplementary pilot", command_log=command_log)
            supp_summary_path = supp_dir / "summary.json"

    if axis_summary_path is not None or supp_summary_path is not None:
        build_compare(axis_summary_path, supp_summary_path, label="rebuild compare HTML with metrics")

    manifest = {
        "session_prefix": str(args.session_prefix),
        "workspace_root": str(workspace_root),
        "feature_count": int(len(feature_keys)),
        "feature_keys_path": str(pipeline_root / "feature_keys.txt"),
        "feature_keys": feature_keys,
        "sessions": {
            "source_session": source_session,
            "cyan_source_session": cyan_source_session,
            "erf_session": erf_session,
            "sae_session": sae_session,
            "axis_session": axis_session if not bool(args.skip_axis) else "",
            "supp_session": supp_session if not bool(args.skip_supp) else "",
        },
        "paths": {
            "source_session_dir": str(review_root / source_session),
            "cyan_source_session_dir": str(cyan_source_dir),
            "erf_session_dir": str(review_root / erf_session),
            "sae_session_dir": str(review_root / sae_session),
            "erf_review_html": str(review_root / erf_session / "label_team_review.html"),
            "sae_review_html": str(review_root / sae_session / "label_team_review.html"),
            "compare_html": str(compare_dir / compare_html_name),
            "axis_summary_json": str(axis_summary_path) if axis_summary_path is not None else "",
            "supp_summary_json": str(supp_summary_path) if supp_summary_path is not None else "",
        },
        "config": {
            "top_k": int(args.top_k),
            "label_prompt_style": str(args.label_prompt_style),
            "label_model": str(args.label_model),
            "label_reasoning_effort": str(args.label_reasoning_effort),
            "judge_model": str(args.judge_model),
            "judge_reasoning_effort": str(args.judge_reasoning_effort),
            "jobs_label": int(args.jobs_label),
            "jobs_axis": int(args.jobs_axis),
            "jobs_supp": int(args.jobs_supp),
            "axis2_candidate_count": int(args.axis2_candidate_count),
            "vision_model_name": str(args.vision_model_name),
            "deciles_root": str(args.deciles_root),
            "checkpoints_root": str(args.checkpoints_root),
            "checkpoint_pattern": str(args.checkpoint_pattern),
            "dataset_root": str(args.dataset_root),
            "erf_threshold": float(args.erf_threshold),
            "erf_support_min_attribution": float(args.erf_support_min_attribution),
        },
        "commands": command_log,
    }
    _write_json(pipeline_root / "pipeline_manifest.json", manifest)

    commands_text = "\n".join(
        f"# {row['label']}\n{shlex.join(list(row['cmd']))}\n"
        for row in command_log
    )
    _write_text(pipeline_root / "commands.sh", commands_text)

    report_lines = [
        "# Cyan-Cross Shortdesc Pipeline",
        "",
        f"- session_prefix: `{args.session_prefix}`",
        f"- feature_count: `{len(feature_keys)}`",
        f"- label prompt: `{args.label_prompt_style}`",
        f"- compare html: `{compare_dir / compare_html_name}`",
        f"- ERF review html: `{review_root / erf_session / 'label_team_review.html'}`",
        f"- SAE review html: `{review_root / sae_session / 'label_team_review.html'}`",
    ]
    if axis_summary_path is not None:
        report_lines.append(f"- axis summary: `{axis_summary_path}`")
    if supp_summary_path is not None:
        report_lines.append(f"- supp summary: `{supp_summary_path}`")
    _write_text(pipeline_root / "report.md", "\n".join(report_lines) + "\n")

    print(pipeline_root / "pipeline_manifest.json")


if __name__ == "__main__":
    main()
