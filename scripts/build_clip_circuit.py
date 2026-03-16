#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

from src.core.indexing.decile_parquet_ledger import DecileParquetLedger
from src.core.indexing.offline_meta import build_offline_ledger
from src.core.circuits.topology_presets import (
    DEFAULT_CLIP_TOPOLOGY_DIR,
    build_topology,
    default_feature_graph_config,
    list_topology_presets,
    load_topology_spec,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _slugify(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {".", "-", "_"}:
            out.append(ch.replace(".", "_"))
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "run"


def _bool_arg(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _canonical_layer(layer: str) -> str:
    if "::" in layer:
        layer = layer.split("::", 1)[0]
    return layer


def _strip_model_prefix(layer: str) -> str:
    return layer[len("model.") :] if layer.startswith("model.") else layer


def _lookup_offline_paths(ledger: Any, sample_ids: Sequence[int]) -> Dict[int, str]:
    if ledger is None:
        return {}
    lookup = getattr(ledger, "lookup", None)
    if lookup is None:
        return {}
    try:
        return lookup(sample_ids)
    except Exception:
        return {}


def _resolve_stop_grad_map(entries: Sequence[str]) -> Dict[str, List[str]]:
    """
    Parse stop-grad overrides of the form 'dst=src1,src2'.
    """
    mapping: Dict[str, List[str]] = {}
    for raw in entries or []:
        if "=" not in raw:
            continue
        dst, srcs = raw.split("=", 1)
        dst = dst.strip()
        if not dst:
            continue
        src_list = [s.strip() for s in srcs.split(",") if s.strip()]
        if not src_list:
            continue
        mapping.setdefault(dst, [])
        for src in src_list:
            if src not in mapping[dst]:
                mapping[dst].append(src)
    return mapping


def _dedup(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_root_layer(raw: str, *, topology_layers: Dict[str, str], head_layer: str, head_module: str) -> str:
    if raw in topology_layers:
        return raw
    norm = _canonical_layer(raw)
    norm_no_model = _strip_model_prefix(norm)
    if norm in topology_layers:
        return norm
    if norm_no_model in topology_layers:
        return norm_no_model
    if raw == head_module or norm == _canonical_layer(head_module) or norm_no_model == _strip_model_prefix(_canonical_layer(head_module)):
        return head_layer
    for layer, module in topology_layers.items():
        module_norm = _canonical_layer(module)
        if module == raw or module_norm == norm or _strip_model_prefix(module_norm) == norm_no_model:
            return layer
    raise ValueError(f"Unknown target layer '{raw}'. Available layers: {sorted(topology_layers.keys())}")


def _choose_image_path(
    *,
    args: argparse.Namespace,
    index_cfg: Dict[str, Any],
    target_layer: str,
    target_unit: int,
) -> Path:
    if args.image:
        return Path(args.image).expanduser()

    ledger_root = Path(index_cfg.get("indexing", {}).get("out_dir", ""))
    offline_ledger = None
    if args.sample_id is not None or args.decile is not None:
        offline_ledger = build_offline_ledger(index_cfg)

    if args.sample_id is not None:
        path_map = _lookup_offline_paths(offline_ledger, [int(args.sample_id)])
        p = path_map.get(int(args.sample_id))
        if not p:
            raise FileNotFoundError(f"sample_id {args.sample_id} not found under offline meta")
        return Path(p)

    if args.decile is not None:
        ledger = DecileParquetLedger(ledger_root)
        decile_idx = int(args.decile)
        rank = max(0, int(args.rank_in_decile or 0))
        tbl = ledger.topn_for(
            layer=_canonical_layer(target_layer),
            unit=int(target_unit),
            decile=decile_idx,
            n=rank + 1,
        )
        if tbl is None or tbl.num_rows <= rank:
            raise RuntimeError(f"Ledger has no row for layer={target_layer} unit={target_unit} decile={decile_idx} rank>={rank}")
        sample_id = int(tbl.column("sample_id")[rank].as_py())
        path_map = _lookup_offline_paths(offline_ledger, [sample_id])
        p = path_map.get(sample_id)
        if not p:
            raise FileNotFoundError(f"Could not resolve image path for sample_id {sample_id}")
        return Path(p)

    return Path(args.default_image).expanduser()


def _build_runtime_block(
    *,
    target_module: str,
    target_unit: int,
    override_mode: str,
    objective_aggregation: str,
    backward_backend: str,
    forward_backend: str,
    ordered_layers: List[str],
    layer_to_module: Dict[str, str],
    head_layer: str,
    head_module: str,
    backward_ig_active: Sequence[str],
    libragrad: bool,
    libragrad_gamma: float | None,
) -> dict:
    root_layer = None
    for layer, module in layer_to_module.items():
        if module == target_module:
            root_layer = layer
            break
    root_layer = root_layer or _canonical_layer(target_module)

    idx = ordered_layers.index(root_layer) if root_layer in ordered_layers else -1
    prev_layer = ordered_layers[idx - 1] if idx > 0 else None
    next_layer = ordered_layers[idx + 1] if idx >= 0 and idx + 1 < len(ordered_layers) else None

    backward_anchor_modules = []
    backward_anchor_modules.append(layer_to_module.get(prev_layer, "")) if prev_layer else None
    backward_anchor_modules.append(target_module)
    backward_anchor_modules = ["model.patch_embed::pre@0"] + backward_anchor_modules
    backward_anchor_modules = [m for m in backward_anchor_modules if m]

    forward_anchor_modules = [head_module]
    if next_layer and next_layer != head_layer:
        forward_anchor_modules.append(layer_to_module[next_layer])

    runtime_cfg = {
        "target": {
            "layer": target_module,
            "unit": int(target_unit),
            "override_mode": override_mode,
            "objective_aggregation": objective_aggregation,
        },
        "backward": {
            "enabled": True,
            "method": backward_backend,
            "ig_steps": 32,
            "baseline": "zeros",
            "override_mode": override_mode,
            "backward_anchors": {
                "ig_active": list(backward_ig_active),
                "module": _dedup(backward_anchor_modules),
                "stop_grad": [],
            },
        },
        "forward": {
            "enabled": True,
            "method": forward_backend,
            "ig_steps": 32,
            "baseline": "zeros",
            "override_mode": override_mode,
            "forward_anchors": {
                "ig_active": [],
                "module": _dedup(forward_anchor_modules),
            },
        },
        "libragrad": bool(libragrad),
        "libragrad_gamma": float(libragrad_gamma) if libragrad_gamma is not None else None,
    }
    return runtime_cfg


def _build_heatmap_cfg(args: argparse.Namespace) -> dict:
    return {
        "reduce": args.heatmap_reduce,
        "log_q_lo": float(args.heatmap_log_q_lo),
        "log_q_hi": float(args.heatmap_log_q_hi),
        "save_overlays": bool(args.heatmap_save_overlays),
    }


def _write_yaml(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False)


def _maybe_run_circuit(cfg_path: Path, output_root: Path, *, viz_config: Path, viz_root: Path | None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.core.circuits.run_circuit",
        "--config",
        str(cfg_path),
        "--output_root",
        str(output_root),
        "--viz_config",
        str(viz_config),
    ]
    if viz_root:
        cmd.extend(["--viz_root", str(viz_root)])
    subprocess.run(cmd, check=True)


def _add_topology_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--topology",
        type=str,
        default="linear",
        help="Topology preset name (YAML under core/topologies/clip) or a path to a topology YAML.",
    )
    parser.add_argument(
        "--topology-dir",
        type=str,
        default=None,
        help="Custom directory to search for topology YAMLs (default: core/topologies/clip).",
    )
    parser.add_argument(
        "--stop-grad",
        action="append",
        default=None,
        help="Stop-grad override per destination layer, e.g. 'blocks.1=blocks.0.mlp'.",
    )
    parser.add_argument(
        "--forward-backend",
        type=str,
        default='input_x_grad',
        help="Override forward backend for edges (default comes from topology YAML).",
    )
    parser.add_argument(
        "--backward-backend",
        type=str,
        default='input_x_grad',
        help="Override backward backend for edges (default comes from topology YAML).",
    )
    parser.add_argument(
        "--backward-ig-active",
        action="append",
        default=None,
        help="Extra ig_active anchors for backward edges (preset default is patch_embed).",
    )
    parser.add_argument(
        "--no-head-terminal",
        type=_bool_arg,
        default=False,
        help="Do not mark the head edge as terminal (keeps logits as regular nodes).",
    )


def _add_image_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", type=str, default=None, help="Explicit image path to use for the circuit run.")
    parser.add_argument("--sample-id", type=int, default=None, help="Resolve image via offline meta using sample_id.")
    parser.add_argument(
        "--decile",
        type=int,
        default=None,
        help="Pick image from decile parquet (requires --rank-in-decile, default rank=0).",
    )
    parser.add_argument(
        "--rank-in-decile",
        type=int,
        default=0,
        dest="rank_in_decile",
        help="Rank within the chosen decile (default: 0).",
    )
    parser.add_argument(
        "--default-image",
        type=str,
        default="otter_head.png",
        help="Fallback image if no sample/decile is requested.",
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-layer",
        type=str,
        required=False,
        default="blocks.7",
        help="Target layer (e.g., blocks.10). Default: blocks.7 for VSCode debugging convenience.",
    )
    parser.add_argument(
        "--unit",
        type=int,
        required=False,
        default=6983,
        help="Target unit index. Default: 6983.",
    )
    parser.add_argument("--override-mode", type=str, default="all_tokens", help="Runtime override_mode.")
    parser.add_argument("--objective-aggregation", type=str, default="sum", help="Objective aggregation mode.")
    parser.add_argument(
        "--heatmap-reduce",
        type=str,
        default="l2",
        help="Heatmap reduction (default: l2).",
    )
    parser.add_argument(
        "--heatmap-log-q-lo",
        type=float,
        default=0.05,
        help="Heatmap log quantile low.",
    )
    parser.add_argument(
        "--heatmap-log-q-hi",
        type=float,
        default=1.0,
        help="Heatmap log quantile high.",
    )
    parser.add_argument(
        "--heatmap-save-overlays",
        type=_bool_arg,
        default=True,
        help="Save overlay images alongside JSON outputs.",
    )
    parser.add_argument(
        "--edge-weight-mode",
        type=str,
        default='target_conditioned',
        help="Override feature_graph.edge_weighting.mode (e.g., target_conditioned/none).",
    )
    parser.add_argument(
        "--edge-weight-backend",
        type=str,
        default=None,
        help="Override feature_graph.edge_weighting.weight_backend (e.g., input_x_grad/ig).",
    )
    parser.add_argument(
        "--edge-weight-positive-only",
        type=_bool_arg,
        default=None,
        help="Override feature_graph.edge_weighting.positive_only.",
    )
    parser.add_argument(
        "--edge-weight-score-mode",
        type=str,
        default=None,
        help="Override feature_graph.edge_weighting.score_mode (sign/magnitude).",
    )
    parser.add_argument(
        "--libragrad",
        type=_bool_arg,
        default=True,
        help="Apply FullGrad-style libragrad patching to the model/SAEs.",
    )
    parser.add_argument(
        "--libragrad-gamma",
        type=float,
        default=None,
        help="Optional gamma parameter for libragrad linear layers.",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP circuit config + run from simple inputs.")
    parser.add_argument(
        "--index-config",
        type=str,
        default="configs/clip_imagenet_index.yaml",
        help="Indexing config path (for model/SAE/ledger lookup).",
    )
    parser.add_argument(
        "--config-out",
        type=str,
        default=None,
        help="Where to write the generated circuit yaml. Default: <run_dir>/clip_circuit_auto.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Circuit output root (feature_tree JSON/HTML). Default: <run_dir>.",
    )
    parser.add_argument(
        "--viz-config",
        type=str,
        default="configs/clip_attr_viz.yaml",
        help="Viz config used when --run is set (for auto thumbnail generation).",
    )
    parser.add_argument(
        "--viz-root",
        type=str,
        default=None,
        help="Viz cache root for Sankey thumbnails. Default: <run_dir>/viz_cache.",
    )
    parser.add_argument("--run", type=_bool_arg, default=True, help="Run debug_make_circuit.py after writing the config.")
    parser.add_argument("--list-topologies", type=_bool_arg, default=False, help="List available clip topology presets and exit.")

    _add_topology_args(parser)
    _add_image_args(parser)
    _add_runtime_args(parser)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.list_topologies and (args.target_layer is None or args.unit is None):
        raise SystemExit("--target-layer and --unit are required unless --list-topologies is set.")

    if args.list_topologies:
        topo_dir = Path(args.topology_dir) if args.topology_dir else DEFAULT_CLIP_TOPOLOGY_DIR
        for name, desc in list_topology_presets(base_dir=topo_dir):
            print(f"{name:>20}: {desc}")
        return

    index_cfg = _load_yaml(Path(args.index_config))

    topo_dir = Path(args.topology_dir) if args.topology_dir else DEFAULT_CLIP_TOPOLOGY_DIR
    spec = load_topology_spec(args.topology, base_dir=topo_dir)
    topology = build_topology(
        spec,
        extra_stop_grad=_resolve_stop_grad_map(args.stop_grad or []),
        forward_backend=args.forward_backend,
        backward_backend=args.backward_backend,
        backward_ig_active_override=args.backward_ig_active,
        mark_head_terminal=not args.no_head_terminal,
    )
    root_layer = _resolve_root_layer(
        args.target_layer,
        topology_layers=topology.layer_to_module,
        head_layer=topology.head_layer,
        head_module=topology.head_module,
    )
    target_module = topology.layer_to_module[root_layer]
    image_path = _choose_image_path(
        args=args,
        index_cfg=index_cfg,
        target_layer=target_module,
        target_unit=args.unit,
    )

    # Per-run folder (based on target + image stem) for easy VSCode/debugging management.
    run_slug = _slugify(f"{root_layer}_u{args.unit}_{Path(image_path).stem}")
    run_dir = Path(args.output_root) if args.output_root else Path("outputs") / "circuit_runs" / run_slug
    config_out = Path(args.config_out) if args.config_out else run_dir / "clip_circuit_auto.yaml"
    viz_root = Path(args.viz_root) if args.viz_root else run_dir / "viz_cache"

    runtime_cfg = _build_runtime_block(
        target_module=target_module,
        target_unit=args.unit,
        override_mode=args.override_mode,
        objective_aggregation=args.objective_aggregation,
        backward_backend=topology.backward_backend,
        forward_backend=topology.forward_backend,
        ordered_layers=topology.ordered_layers,
        layer_to_module=topology.layer_to_module,
        head_layer=topology.head_layer,
        head_module=topology.head_module,
        backward_ig_active=topology.backward_ig_active,
        libragrad=bool(args.libragrad),
        libragrad_gamma=args.libragrad_gamma,
    )

    tree_cfg = {
        "root": {
            "layer": root_layer,
            "units": [int(args.unit)],
        },
        "feature_graph": default_feature_graph_config(),
        "nodes": topology.nodes,
        "edges": topology.edges,
    }

    # feature_graph overrides
    fg = tree_cfg.get("feature_graph", {})
    ew = fg.get("edge_weighting", {})
    if args.edge_weight_mode:
        ew["mode"] = args.edge_weight_mode
    if args.edge_weight_backend:
        ew["weight_backend"] = args.edge_weight_backend
    if args.edge_weight_positive_only is not None:
        ew["positive_only"] = bool(args.edge_weight_positive_only)
    if args.edge_weight_score_mode:
        ew["score_mode"] = args.edge_weight_score_mode
    fg["edge_weighting"] = ew
    tree_cfg["feature_graph"] = fg

    full_cfg = {
        "pack": "clip",
        "indexing": {"config": str(Path(args.index_config))},
        "runtime": runtime_cfg,
        "heatmap": _build_heatmap_cfg(args),
        "tree": tree_cfg,
        "output_root": str(run_dir),
        "image_path": str(image_path),
    }

    _write_yaml(full_cfg, config_out)
    print(f"[ok] wrote config to {config_out}")

    if args.run:
        _maybe_run_circuit(
            cfg_path=config_out,
            output_root=run_dir,
            viz_config=Path(args.viz_config),
            viz_root=viz_root,
        )
        print(f"[ok] built circuit under {run_dir}")


if __name__ == "__main__":
    main()
