from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .axis1_c_conditioned import (
    build_axis1_c_conditioned_dataset,
    build_axis1_c_conditioned_evidence,
    compute_axis1_c_conditioned_metrics,
    export_axis1_c_conditioned_requests,
)
from .autolabel_loop import (
    apply_autolabel_phase_gate,
    advance_autolabel_round,
    build_autolabel_phase_gate,
    build_autolabel_round_packet,
    build_autolabel_session,
    ingest_autolabel_round,
    promote_autolabel_labels,
    refresh_autolabel_session_assets,
    serve_autolabel_phase_gate,
    serve_autolabel_session,
)
from .axis_experiments import build_axis_feature_subsets, summarize_axis_experiment
from .axis_orchestrator import (
    build_axis_experiment,
    build_axis_failure_audit,
    build_axis_label_demand,
    ingest_axis_failure_audit,
    review_axis_experiment,
    run_axis_experiment,
    run_dynamic_label_generation,
)
from .config import load_config
from .evidence import build_token_evidence, export_baseline_label_requests, export_pairwise_requests
from .feature_diagnostics import build_feature_binary_diagnostics, build_feature_mass_diagnostics
from .feature_bank import build_feature_bank, load_feature_bank
from .human_study import build_human_quiz, build_human_session, score_human_quiz
from .metrics import build_ground_truth, compute_metrics_summary
from .study_protocol import (
    build_study_axis1_session,
    build_study_axis2_session,
    build_study_label_session,
    ingest_study_label_session,
    load_label_registry,
    score_study_axis1_session,
    score_study_axis2_session,
)


def _print_summary(title: str, payload) -> None:
    print(f"=== {title} ===")
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
    else:
        print(payload)


def _parse_feature_specs(values) -> list[tuple[int, int]] | None:
    if not values:
        return None
    feature_specs: list[tuple[int, int]] = []
    for spec in values:
        parts = str(spec).split(":", 1)
        if len(parts) != 2:
            raise SystemExit(f"Invalid --diagnostic-feature value: {spec!r}; expected BLOCK:FEATURE")
        feature_specs.append((int(parts[0]), int(parts[1])))
    return feature_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Token-level autolabel evaluation pipeline.")
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vision-model-name", type=str, default=None)
    parser.add_argument("--blocks", nargs="*", type=int, default=None)
    parser.add_argument("--features-per-block", type=int, default=None)
    parser.add_argument("--train-per-feature", type=int, default=None)
    parser.add_argument("--holdout-per-feature", type=int, default=None)
    parser.add_argument("--erf-threshold", type=float, default=None)
    parser.add_argument("--deciles-root", type=Path, default=None)
    parser.add_argument("--offline-meta-root", type=Path, default=None)
    parser.add_argument("--checkpoints-root", type=Path, default=None)
    parser.add_argument("--checkpoint-pattern", type=str, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--session-name", type=str, default="pilot_v1")
    parser.add_argument("--session-seed", type=int, default=None)
    parser.add_argument("--human-features-per-block", type=int, default=None)
    parser.add_argument("--study-features-per-block", type=int, default=None)
    parser.add_argument("--response-json", type=Path, default=None)
    parser.add_argument("--spec-json", type=Path, default=None)
    parser.add_argument("--subsets-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--phase-gate-id", type=str, default=None)
    parser.add_argument("--variant-id", type=str, default=None)
    parser.add_argument("--provider-type", type=str, default="human")
    parser.add_argument("--round-index", type=int, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--diagnostic-feature", action="append", default=None)
    parser.add_argument("--diagnostic-threshold", action="append", type=float, default=None)
    parser.add_argument("--diagnostic-mass", action="append", type=float, default=None)
    parser.add_argument("--diagnostic-objective", action="append", default=None)
    parser.add_argument("--diagnostic-examples", type=int, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-feature-bank")
    sub.add_parser("export-label-requests")
    sub.add_parser("build-token-evidence")
    sub.add_parser("export-pairwise-requests")
    sub.add_parser("build-ground-truth")
    sub.add_parser("compute-metrics")
    sub.add_parser("build-axis1-dataset")
    sub.add_parser("build-axis1-evidence")
    sub.add_parser("export-axis1-requests")
    sub.add_parser("compute-axis1-metrics")
    sub.add_parser("build-axis-feature-subsets")
    sub.add_parser("summarize-axis-experiment")
    sub.add_parser("build-axis-label-demand")
    sub.add_parser("run-dynamic-label-generation")
    sub.add_parser("build-axis-experiment")
    sub.add_parser("run-axis-experiment")
    sub.add_parser("build-axis-failure-audit")
    sub.add_parser("ingest-axis-failure-audit")
    sub.add_parser("review-axis-experiment")
    sub.add_parser("build-human-session")
    sub.add_parser("build-human-quiz")
    sub.add_parser("score-human-quiz")
    sub.add_parser("build-feature-diagnostics")
    sub.add_parser("build-feature-mass-diagnostics")
    sub.add_parser("build-study-label-session")
    sub.add_parser("ingest-study-label-session")
    sub.add_parser("build-study-axis1-session")
    sub.add_parser("score-study-axis1-session")
    sub.add_parser("build-study-axis2-session")
    sub.add_parser("score-study-axis2-session")
    sub.add_parser("build-autolabel-session")
    sub.add_parser("build-autolabel-round-packet")
    sub.add_parser("ingest-autolabel-round")
    sub.add_parser("refresh-autolabel-session")
    sub.add_parser("serve-autolabel-session")
    sub.add_parser("advance-autolabel-round")
    sub.add_parser("build-autolabel-phase-gate")
    sub.add_parser("serve-autolabel-phase-gate")
    sub.add_parser("apply-autolabel-phase-gate")
    sub.add_parser("promote-autolabel-labels")
    sub.add_parser("show-label-registry")
    sub.add_parser("show-config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    overrides = {}
    if args.workspace_root is not None:
        overrides["workspace_root"] = args.workspace_root
    if args.device is not None:
        overrides["device"] = args.device
    if args.vision_model_name is not None:
        overrides["model_name"] = args.vision_model_name
    if args.blocks:
        overrides["blocks"] = tuple(int(v) for v in args.blocks)
    if args.features_per_block is not None:
        overrides["features_per_block"] = int(args.features_per_block)
    if args.train_per_feature is not None:
        overrides["train_examples_per_feature"] = int(args.train_per_feature)
    if args.holdout_per_feature is not None:
        overrides["holdout_examples_per_feature"] = int(args.holdout_per_feature)
    if args.erf_threshold is not None:
        overrides["erf_recovery_threshold"] = float(args.erf_threshold)
    if args.deciles_root is not None:
        overrides["deciles_root_override"] = args.deciles_root
    if args.offline_meta_root is not None:
        overrides["offline_meta_root_override"] = args.offline_meta_root
    if args.checkpoints_root is not None:
        overrides["checkpoints_root_override"] = args.checkpoints_root
    if args.checkpoint_pattern is not None:
        overrides["checkpoint_relpath_template"] = args.checkpoint_pattern
    if args.dataset_root is not None:
        overrides["dataset_root_override"] = args.dataset_root
    if overrides:
        config = replace(config, **overrides)
        config.ensure_dirs()
    if args.command == "build-feature-bank":
        payload = build_feature_bank(config)
        _print_summary("Feature Bank", {"feature_bank_json": config.feature_bank_json, "blocks": list(payload["blocks"].keys())})
        return 0
    if args.command == "export-label-requests":
        payload = export_baseline_label_requests(config)
        _print_summary("Baseline Label Requests", {"baseline_label_requests_json": config.baseline_label_requests_json, "n_requests": len(payload["requests"])})
        return 0
    if args.command == "build-token-evidence":
        payload = build_token_evidence(config)
        _print_summary("Token Evidence", {"token_evidence_json": config.token_evidence_json, "n_tokens": len(payload["tokens"])})
        return 0
    if args.command == "export-pairwise-requests":
        payload = export_pairwise_requests(config)
        _print_summary("Pairwise Requests", {"pairwise_requests_jsonl": config.pairwise_requests_jsonl, "n_requests": payload["n_requests"]})
        return 0
    if args.command == "build-ground-truth":
        payload = build_ground_truth(config)
        _print_summary("Ground Truth", {"ground_truth_json": config.ground_truth_json, "blocks": list(payload["blocks"].keys())})
        return 0
    if args.command == "compute-metrics":
        payload = compute_metrics_summary(config)
        _print_summary("Metrics", {"metrics_summary_json": config.metrics_summary_json, "overall": payload["overall"]})
        return 0
    if args.command == "build-axis1-dataset":
        payload = build_axis1_c_conditioned_dataset(config)
        _print_summary(
            "Axis1 Dataset",
            {"axis1_dataset_json": config.axis1_dataset_json, "n_items": len(payload["items"])},
        )
        return 0
    if args.command == "build-axis1-evidence":
        payload = build_axis1_c_conditioned_evidence(config)
        _print_summary(
            "Axis1 Evidence",
            {"axis1_evidence_json": config.axis1_evidence_json, "n_tokens": len(payload["tokens"])},
        )
        return 0
    if args.command == "export-axis1-requests":
        payload = export_axis1_c_conditioned_requests(config)
        _print_summary(
            "Axis1 Requests",
            {"axis1_requests_jsonl": config.axis1_requests_jsonl, "n_requests": payload["n_requests"]},
        )
        return 0
    if args.command == "compute-axis1-metrics":
        payload = compute_axis1_c_conditioned_metrics(config)
        _print_summary(
            "Axis1 Metrics",
            {"axis1_metrics_json": config.axis1_metrics_json, "overall": payload["overall"]},
        )
        return 0
    if args.command == "build-axis-feature-subsets":
        payload = build_axis_feature_subsets(
            config,
            output_json=args.output_json,
        )
        _print_summary(
            "Axis Feature Subsets",
            {
                "axis_feature_subsets_json": payload["output_json"],
                "n_features": len(payload["feature_subsets"]),
                "subsets": {key: len(value) for key, value in payload["subsets"].items()},
            },
        )
        return 0
    if args.command == "summarize-axis-experiment":
        if args.spec_json is None:
            raise SystemExit("--spec-json is required for summarize-axis-experiment")
        payload = summarize_axis_experiment(
            config,
            spec_json=args.spec_json,
            subsets_json=args.subsets_json,
            output_json=args.output_json,
            output_html=args.output_html,
        )
        _print_summary(
            "Axis Experiment Summary",
            {
                "output_json": payload["output_json"],
                "output_html": payload["output_html"],
                "baseline_variant": payload["baseline_variant"],
                "n_variants": len(payload["variants"]),
            },
        )
        return 0
    if args.command == "build-axis-label-demand":
        feature_specs = _parse_feature_specs(args.diagnostic_feature)
        payload = build_axis_label_demand(
            config,
            experiment_name=args.session_name,
            spec_json=args.spec_json,
            feature_specs=feature_specs,
            features_per_block=args.study_features_per_block,
            seed=args.session_seed,
        )
        _print_summary(
            "Axis Label Demand",
            {
                "output_json": payload["output_json"],
                "n_selected": payload["summary"]["n_selected"],
                "n_missing_label": payload["summary"]["n_missing_label"],
            },
        )
        return 0
    if args.command == "run-dynamic-label-generation":
        payload = run_dynamic_label_generation(
            config,
            experiment_name=args.session_name,
            response_json=args.response_json,
            provider_type=args.provider_type,
        )
        _print_summary("Dynamic Label Generation", payload)
        return 0
    if args.command == "build-axis-experiment":
        payload = build_axis_experiment(
            config,
            experiment_name=args.session_name,
            spec_json=args.spec_json,
        )
        _print_summary(
            "Axis Experiment Build",
            {
                "experiment_manifest_json": str(config.axis_runs_root / args.session_name / "experiment_manifest.json"),
                "n_variants": len(payload["variants"]),
                "frozen_labels_json": payload["frozen_labels_json"],
            },
        )
        return 0
    if args.command == "run-axis-experiment":
        payload = run_axis_experiment(
            config,
            experiment_name=args.session_name,
        )
        _print_summary("Axis Experiment Run", payload)
        return 0
    if args.command == "build-axis-failure-audit":
        if not args.variant_id:
            raise SystemExit("--variant-id is required for build-axis-failure-audit")
        payload = build_axis_failure_audit(
            config,
            experiment_name=args.session_name,
            variant_id=args.variant_id,
        )
        _print_summary("Axis Failure Audit", payload)
        return 0
    if args.command == "ingest-axis-failure-audit":
        if not args.variant_id:
            raise SystemExit("--variant-id is required for ingest-axis-failure-audit")
        payload = ingest_axis_failure_audit(
            config,
            experiment_name=args.session_name,
            variant_id=args.variant_id,
            response_json=args.response_json,
        )
        _print_summary("Axis Failure Audit Ingest", payload)
        return 0
    if args.command == "review-axis-experiment":
        payload = review_axis_experiment(
            config,
            experiment_name=args.session_name,
            response_json=args.response_json,
        )
        _print_summary("Axis Experiment Review", payload)
        return 0
    if args.command == "build-human-session":
        feature_specs = None
        if args.diagnostic_feature:
            feature_specs = []
            for spec in args.diagnostic_feature:
                parts = str(spec).split(":", 1)
                if len(parts) != 2:
                    raise SystemExit(f"Invalid --diagnostic-feature value: {spec!r}; expected BLOCK:FEATURE")
                feature_specs.append((int(parts[0]), int(parts[1])))
        payload = build_human_session(
            config,
            session_name=args.session_name,
            features_per_block=args.human_features_per_block,
            seed=args.session_seed,
            feature_specs=feature_specs,
            examples_per_feature=args.train_per_feature,
        )
        _print_summary(
            "Human Session",
            {
                "session_dir": str(config.human_study_root / args.session_name),
                "selected_features": len(payload["selected_features"]),
                "human_labels_json": payload["human_labels_json"],
            },
        )
        return 0
    if args.command == "build-human-quiz":
        payload = build_human_quiz(
            config,
            session_name=args.session_name,
            seed=args.session_seed,
        )
        _print_summary(
            "Human Quiz",
            {
                "session_dir": str(config.human_study_root / args.session_name),
                "n_questions": len(payload["questions"]),
                "quiz_answers_json": payload["quiz_answers_json"],
            },
        )
        return 0
    if args.command == "score-human-quiz":
        payload = score_human_quiz(
            config,
            session_name=args.session_name,
        )
        _print_summary("Human Quiz Score", payload)
        return 0
    if args.command == "build-feature-diagnostics":
        if not args.diagnostic_feature:
            raise SystemExit("--diagnostic-feature is required, e.g. --diagnostic-feature 10:2198 10:14484")
        feature_specs: list[tuple[int, int]] = []
        for spec in args.diagnostic_feature:
            parts = str(spec).split(":", 1)
            if len(parts) != 2:
                raise SystemExit(f"Invalid --diagnostic-feature value: {spec!r}; expected BLOCK:FEATURE")
            feature_specs.append((int(parts[0]), int(parts[1])))
        payload = build_feature_binary_diagnostics(
            config,
            session_name=args.session_name,
            feature_specs=feature_specs,
            thresholds=tuple(args.diagnostic_threshold or (0.75, 0.80, 0.85, 0.90)),
            examples_per_feature=int(args.diagnostic_examples or 5),
            objective_modes=tuple(args.diagnostic_objective or ("cosine",)),
        )
        _print_summary(
            "Feature Diagnostics",
            {
                "session_dir": str(config.human_study_root / args.session_name),
                "n_features": len(payload["features"]),
                "thresholds": payload["thresholds"],
                "objective_modes": payload["objective_modes"],
            },
        )
        return 0
    if args.command == "build-feature-mass-diagnostics":
        if not args.diagnostic_feature:
            raise SystemExit("--diagnostic-feature is required, e.g. --diagnostic-feature 2:2259 6:10765 10:19572")
        feature_specs: list[tuple[int, int]] = []
        for spec in args.diagnostic_feature:
            parts = str(spec).split(":", 1)
            if len(parts) != 2:
                raise SystemExit(f"Invalid --diagnostic-feature value: {spec!r}; expected BLOCK:FEATURE")
            feature_specs.append((int(parts[0]), int(parts[1])))
        payload = build_feature_mass_diagnostics(
            config,
            session_name=args.session_name,
            feature_specs=feature_specs,
            mass_thresholds=tuple(args.diagnostic_mass or (0.90, 0.95, 0.99)),
            examples_per_feature=int(args.diagnostic_examples or 5),
            objective_mode=str((args.diagnostic_objective or [config.erf_objective_mode])[0]),
        )
        _print_summary(
            "Feature Mass Diagnostics",
            {
                "session_dir": str(config.human_study_root / args.session_name),
                "n_features": len(payload["features"]),
                "mass_thresholds": payload["mass_thresholds"],
                "objective_mode": payload["objective_mode"],
            },
        )
        return 0
    if args.command == "build-study-label-session":
        payload = build_study_label_session(
            config,
            session_name=args.session_name,
            features_per_block=args.study_features_per_block,
            seed=args.session_seed,
        )
        _print_summary(
            "Study Label Session",
            {
                "session_dir": str(config.study_root / args.session_name),
                "selected_features": len(payload["selected_features"]),
                "html_path": payload["html_path"],
                "default_response_json": payload["default_response_json"],
            },
        )
        return 0
    if args.command == "ingest-study-label-session":
        payload = ingest_study_label_session(
            config,
            session_name=args.session_name,
            response_json=args.response_json,
            provider_type=args.provider_type,
        )
        _print_summary("Study Label Ingest", payload)
        return 0
    if args.command == "build-study-axis1-session":
        payload = build_study_axis1_session(
            config,
            session_name=args.session_name,
            features_per_block=args.study_features_per_block,
            seed=args.session_seed,
        )
        _print_summary(
            "Study Axis1 Session",
            {
                "session_dir": str(config.study_root / args.session_name),
                "n_items": len(payload["items"]),
                "html_path": payload["html_path"],
                "default_response_json": payload["default_response_json"],
            },
        )
        return 0
    if args.command == "score-study-axis1-session":
        payload = score_study_axis1_session(
            config,
            session_name=args.session_name,
            response_json=args.response_json,
        )
        _print_summary(
            "Study Axis1 Score",
            {"session_dir": str(config.study_root / args.session_name), "overall": payload["overall"]},
        )
        return 0
    if args.command == "build-study-axis2-session":
        payload = build_study_axis2_session(
            config,
            session_name=args.session_name,
            features_per_block=args.study_features_per_block,
            seed=args.session_seed,
        )
        _print_summary(
            "Study Axis2 Session",
            {
                "session_dir": str(config.study_root / args.session_name),
                "n_items": len(payload["items"]),
                "html_path": payload["html_path"],
                "default_response_json": payload["default_response_json"],
            },
        )
        return 0
    if args.command == "score-study-axis2-session":
        payload = score_study_axis2_session(
            config,
            session_name=args.session_name,
            response_json=args.response_json,
        )
        _print_summary(
            "Study Axis2 Score",
            {"session_dir": str(config.study_root / args.session_name), "overall": payload["overall"]},
        )
        return 0
    if args.command == "build-autolabel-session":
        feature_specs = None
        if args.diagnostic_feature:
            feature_specs = []
            for spec in args.diagnostic_feature:
                parts = str(spec).split(":", 1)
                if len(parts) != 2:
                    raise SystemExit(f"Invalid --diagnostic-feature value: {spec!r}; expected BLOCK:FEATURE")
                feature_specs.append((int(parts[0]), int(parts[1])))
        payload = build_autolabel_session(
            config,
            session_name=args.session_name,
            features_per_block=args.study_features_per_block,
            seed=args.session_seed,
            feature_specs=feature_specs,
        )
        _print_summary(
            "Autolabel Session",
            {
                "session_dir": payload["session_dir"],
                "feature_pool_json": payload["feature_pool_json"],
                "feature_state_jsonl": payload["feature_state_jsonl"],
                "current_prompt_config_json": payload["current_prompt_config_json"],
            },
        )
        return 0
    if args.command == "build-autolabel-round-packet":
        payload = build_autolabel_round_packet(
            config,
            session_name=args.session_name,
            round_index=args.round_index,
        )
        _print_summary("Autolabel Round Packet", payload)
        return 0
    if args.command == "ingest-autolabel-round":
        payload = ingest_autolabel_round(
            config,
            session_name=args.session_name,
            round_index=args.round_index,
            response_json=args.response_json,
        )
        _print_summary("Autolabel Round Ingest", payload)
        return 0
    if args.command == "refresh-autolabel-session":
        payload = refresh_autolabel_session_assets(
            config,
            session_name=args.session_name,
        )
        _print_summary("Autolabel Session Refresh", payload)
        return 0
    if args.command == "serve-autolabel-session":
        serve_autolabel_session(
            config,
            session_name=args.session_name,
            round_index=args.round_index,
            host=args.host,
            port=args.port,
        )
        return 0
    if args.command == "advance-autolabel-round":
        payload = advance_autolabel_round(
            config,
            session_name=args.session_name,
            round_index=args.round_index,
        )
        _print_summary("Autolabel Round Advance", payload)
        return 0
    if args.command == "build-autolabel-phase-gate":
        payload = build_autolabel_phase_gate(
            config,
            session_name=args.session_name,
            summary_json=args.response_json,
        )
        _print_summary("Autolabel Phase Gate", payload)
        return 0
    if args.command == "serve-autolabel-phase-gate":
        serve_autolabel_phase_gate(
            config,
            session_name=args.session_name,
            gate_id=args.phase_gate_id,
            host=args.host,
            port=args.port,
        )
        return 0
    if args.command == "apply-autolabel-phase-gate":
        payload = apply_autolabel_phase_gate(
            config,
            session_name=args.session_name,
            gate_id=args.phase_gate_id,
        )
        _print_summary("Autolabel Phase Gate Apply", payload)
        return 0
    if args.command == "promote-autolabel-labels":
        payload = promote_autolabel_labels(
            config,
            session_name=args.session_name,
        )
        _print_summary("Autolabel Promotion", payload)
        return 0
    if args.command == "show-label-registry":
        rows = load_label_registry(config)
        _print_summary(
            "Label Registry",
            {
                "label_registry_jsonl": str(config.label_registry_jsonl),
                "n_records": len(rows),
                "n_accepted": sum(int(str(row.get("status", "")) == "accepted") for row in rows),
            },
        )
        return 0
    if args.command == "show-config":
        _print_summary("Config", config.to_dict())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
