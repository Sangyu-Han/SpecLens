from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from PIL import Image

from src.core.hooks.spec import parse_spec
from src.core.circuits.runtime import CircuitRuntime
from src.core.indexing.registry_utils import sanitize_layer_name
from src.packs.clip.dataset.builders import build_clip_transform
from src.packs.clip.models.adapters import CLIPVisionAdapter
from src.utils.utils import load_obj
from src.packs.clip.models.libragrad import apply_libragrad, enable_sae_libragrad


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_list(value: Optional[Iterable[str]]) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _candidate_layer_dirs(root: Path, layer: str) -> List[Path]:
    candidates = [root / sanitize_layer_name(layer)]
    parsed = parse_spec(layer)
    if parsed.method is not None:
        candidates.append(root / sanitize_layer_name(parsed.base_with_branch))
    return candidates


def _load_sae_for_layer(cfg: Dict[str, Any], layer: str, device: torch.device, *, libragrad: bool = False) -> torch.nn.Module:
    root = Path(cfg["output"]["save_path"])
    ckpts: List[Path] = []
    for candidate in _candidate_layer_dirs(root, layer):
        files = sorted(candidate.glob("*.pt"))
        if files:
            ckpts = files
            break
    if not ckpts:
        raise FileNotFoundError(f"No SAE checkpoints found under {root} for layer '{layer}'")
    pkg = torch.load(ckpts[-1], map_location="cpu")
    sae_cfg = dict(pkg.get("sae_config", {}))
    act_size = int(pkg.get("act_size") or sae_cfg.get("act_size", 0))
    if act_size <= 0:
        raise RuntimeError(f"SAE checkpoint at {ckpts[-1]} is missing act_size metadata")
    sae_cfg.update({"act_size": act_size, "device": str(device)})
    create_sae = load_obj(cfg["factory"])
    sae = create_sae(sae_cfg.get("sae_type", "batch-topk"), sae_cfg)
    state = pkg.get("sae_state", pkg.get("state_dict", {}))
    sae.load_state_dict(state, strict=False)
    sae.to(device).eval()
    for param in sae.parameters():
        param.requires_grad = False
    if libragrad:
        try:
            enable_sae_libragrad(sae)
        except Exception:
            pass
    return sae


def _build_image_batch(cfg_index: Dict[str, Any], image_path: Path) -> Dict[str, Any]:
    dataset_cfg = dict(cfg_index.get("dataset", {}))
    transform = build_clip_transform(dataset_cfg, is_train=False)
    tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    sample_id = torch.tensor([0], dtype=torch.long)
    return {
        "pixel_values": tensor,
        "label": None,
        "sample_id": sample_id,
        "path": [str(image_path)],
    }


def _make_forward_fn(adapter: CLIPVisionAdapter, batch_on_dev: Dict[str, Any]):
    def _forward():
        adapter.forward(batch_on_dev)

    return _forward


def _default_image_path(cfg: Dict[str, Any]) -> Path:
    runtime_img = (
        cfg.get("runtime", {}).get("image_path")
        if "runtime" in cfg
        else cfg.get("image_path")
    )
    if runtime_img:
        return Path(runtime_img)
    return Path("otter_head.png")


def _collect_specs_for_wrappers(cfg: Dict[str, Any]) -> List[str]:
    specs: set[str] = set()
    runtime_cfg = cfg.get("runtime") or {}
    backward_cfg = runtime_cfg.get("backward", {})
    forward_cfg = runtime_cfg.get("forward", {})
    backward_anchors = backward_cfg.get("backward_anchors", {})
    forward_anchors = forward_cfg.get("forward_anchors", {})
    specs.update(_ensure_list(backward_anchors.get("module")))
    specs.update(_ensure_list(forward_anchors.get("module")))
    tree_cfg = cfg.get("tree") or {}
    for node in tree_cfg.get("nodes", []):
        module = node.get("module")
        if module:
            specs.add(module)
    for edge in tree_cfg.get("edges", []):
        anchors = edge.get("anchors", {})
        for mod in _ensure_list(anchors.get("capture") or edge.get("anchor_modules")):
            specs.add(mod)
    target_layer = (runtime_cfg.get("target") or {}).get("layer")
    if target_layer:
        specs.discard(target_layer)
    return [s for s in specs if s]


class ClipCircuitRuntime(CircuitRuntime):
    """
    CLIP-specific constructor that wires model/adapter/sae loading for CircuitRuntime.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        device: Optional[str] = None,
        image_path: Optional[Path | str] = None,
    ) -> None:
        cfg_full = config or {}
        runtime_cfg = cfg_full.get("runtime") or cfg_full
        index_cfg_path = (
            (cfg_full.get("indexing") or {}).get("config")
            or (runtime_cfg.get("indexing") or {}).get("config")
            or runtime_cfg.get("indexing_config")
        )
        if not index_cfg_path:
            raise ValueError("indexing config path is required to initialise ClipCircuitRuntime")
        self.index_cfg = _load_yaml(Path(index_cfg_path))
        self.device = torch.device(device or self.index_cfg.get("model", {}).get("device", "cuda"))

        model_loader = load_obj(self.index_cfg["model"]["loader"])
        model = model_loader(self.index_cfg["model"], device=self.device).eval()
        libragrad_enabled = bool(runtime_cfg.get("libragrad", False))
        libragrad_gamma = runtime_cfg.get("libragrad_gamma", None)
        libragrad_alpha = runtime_cfg.get("libragrad_alpha", None)
        libragrad_beta = runtime_cfg.get("libragrad_beta", None)
        restore_libragrad = None
        if libragrad_enabled:
            restore_libragrad = apply_libragrad(
                model,
                gamma=libragrad_gamma,
                alpha=libragrad_alpha,
                beta=libragrad_beta,
            )
        adapter = CLIPVisionAdapter(model, device=self.device)

        img_path = Path(image_path or _default_image_path(cfg_full)).expanduser()
        if not img_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {img_path}")
        raw_sample = _build_image_batch(self.index_cfg, img_path)
        batch_on_dev = adapter.preprocess_input(raw_sample)
        forward_fn = _make_forward_fn(adapter, batch_on_dev)

        self._sae_cache: Dict[str, torch.nn.Module] = {}

        def _resolve_sae(spec: str) -> torch.nn.Module:
            if spec not in self._sae_cache:
                self._sae_cache[spec] = _load_sae_for_layer(
                    self.index_cfg["sae"],
                    spec,
                    self.device,
                    libragrad=libragrad_enabled,
                )
            return self._sae_cache[spec]

        wrap_specs = _collect_specs_for_wrappers(cfg_full)
        super().__init__(
            runtime_cfg=runtime_cfg,
            model=model,
            adapter=adapter,
            forward_fn=forward_fn,
            sae_resolver=_resolve_sae,
            wrap_specs=wrap_specs,
            device=self.device,
            allow_missing_anchor_grad=libragrad_enabled,
        )
        self._libragrad_restore = restore_libragrad

    def cleanup(self) -> None:
        try:
            super().cleanup()
        finally:
            if self._libragrad_restore is not None:
                try:
                    self._libragrad_restore()
                except Exception:
                    pass
                self._libragrad_restore = None


ClipAttrRuntime = ClipCircuitRuntime

__all__ = ["ClipCircuitRuntime", "ClipAttrRuntime"]
