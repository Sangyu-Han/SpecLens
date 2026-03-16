"""logit_prism_patches.py — stable edition
=================================================
Analyse a single Matryoshka‑SAE feature’s impact on OpenCLIP ViT‑L‑14
ImageNet logits with **logit‑prism** rules.

Key fixes
---------
* Multi‑head‑attention patch no longer crashes (only *q,k* cached).
* Fused‑QKV attention left untouched (OpenCLIP ViT‑L‑14 does **not** use it).
* Bias removal/restoration stores the *module object* directly ⇒ O(1).
* Direct‑grad path computes a single `vjp` instead of 1000 backward loops.
* Misc. type/shape guards and safer checkpoint loading.
"""
from __future__ import annotations

import argparse, contextlib, sys, types, weakref
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip
from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

# ---------------------------------------------------------------------------
# Matryoshka‑SAE helpers -----------------------------------------------------
# ---------------------------------------------------------------------------
try:
    from matryoshka_sae.config import get_default_cfg, post_init_cfg
    from matryoshka_sae.sae import GlobalBatchTopKMatryoshkaSAE
except ImportError:  # when the repo is absent we still allow import‑time success
    GlobalBatchTopKMatryoshkaSAE = object  # type: ignore

# ---------------------------------------------------------------------------
# NEW: Caption-weights helpers ----------------------------------------------
# ---------------------------------------------------------------------------
import os, faiss, numpy as np, math, gc

def load_caption_weights(
    path: str,
    device: torch.device | str = "cpu",
    *,               # named-only
    topk_from_img: int | None = None,
    model: nn.Module | None = None,
) -> torch.Tensor:
    """
    Load Conceptual-Captions sentence embeddings as W (C, D).

    Parameters
    ----------
    path : str
        Numpy `.npy` file with shape (N, D) & **already L2-normalised** rows.
    device : torch device
        Where to copy the resulting weight matrix / slice.
    topk_from_img : int, optional
        If given, compute image embedding with `model` and keep only top-K
        nearest captions (cos-sim) on CPU via FAISS, to save GPU memory.
    model : nn.Module, required when `topk_from_img` is not None
        The OpenCLIP model – only needed to embed the query image.
    """
    assert os.path.isfile(path), f"weights file not found: {path}"
    W_np = np.load(path)              # (N, D), float32
    if topk_from_img is None:
        # straight copy to GPU
        return torch.from_numpy(W_np).to(device, non_blocking=True)

    # ---------- memory-saving branch ------------------------------------------------
    assert model is not None, "model must be supplied when using top-K mode"
    dim = W_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(W_np)                   # CPU-only index; uses ~4 bytes/vec

    # The image will be set later by caller; we just provide a closure
    def _select(image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = model.encode_image(image).cpu().float().numpy()  # (1,D)
            faiss.normalize_L2(f)
            _, idx = index.search(f, topk_from_img)
        W_k = torch.from_numpy(W_np[idx[0]]).to(device, non_blocking=True)
        return W_k
    return _select  # caller will invoke with image tensor later

def load_sae(layer_name: str, sae_root: str, device: torch.device) -> "GlobalBatchTopKMatryoshkaSAE":
    """Load a Matryoshka‑SAE checkpoint. Raises helpful errors when missing."""
    from matryoshka_sae.config import get_default_cfg, post_init_cfg  # local import to fail clearly
    from matryoshka_sae.sae import GlobalBatchTopKMatryoshkaSAE

    cfg = get_default_cfg()
    cfg.update({
    "model_name": "ViT-L-14",
    "layer_name": layer_name,
    "aux_penalty": 1 / 32,
    "lr": 3e-4,
    "input_unit_norm": False,
    "dict_size": 16384,
    "l1_coeff": 0.0,
    "act_size": 1024,
    "device": str(device),
    "bandwidth": 0.001,
    "top_k_matryoshka": [10, 10, 10, 10, 10],
    "group_sizes": [1024 // 4, 1024 // 4, 1024 // 2, 1024, 1024 * 2, 1024 * 4, 1024 * 8],
    "num_tokens": int(519_045_120),
    "model_batch_size": 1024,
    "model_dtype": torch.bfloat16,
    "sae_type": "global-matryoshka-topk",
    })
    cfg = post_init_cfg(cfg)
    ckpt = Path(sae_root) / f"{cfg['name']}_126719" / "sae.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt}")

    sae = GlobalBatchTopKMatryoshkaSAE(cfg)
    sae.load_state_dict(torch.load(ckpt, map_location="cpu"))
    sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Utility -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _device_like(src, ref: torch.Tensor):
    if isinstance(src, tuple):
        return tuple(s.to(ref.device, dtype=ref.dtype) for s in src)
    return src.to(ref.device, dtype=ref.dtype)


class PrismPatcher:
    """Monkey‑patch sub‑modules to follow logit‑prism rules; bias toggle included."""

    def __init__(self, model: nn.Module, *, gelu_rule: str = "identity"):
        if gelu_rule not in {"identity", "difference"}:
            raise ValueError
        self.model = model
        self._gelu_rule = gelu_rule
        self._orig_fwd: Dict[nn.Module, types.FunctionType] = {}
        self._saved_input: Dict[int, torch.Tensor | Tuple[torch.Tensor, ...]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._orig_bias: Dict[nn.Module, Dict[str, Optional[torch.Tensor]]] = {}

    # ---------------- cache clean inputs ----------------------------------
    def _make_save_hook(self, key: int):
        def _hook(mod: nn.Module, inputs):
            # MultiheadAttention receives (q,k,v,...)
            if isinstance(mod, nn.MultiheadAttention):
                q, k, *_ = inputs
                self._saved_input[key] = (q.detach().cpu(), k.detach().cpu())
            else:  # LayerNorm, GELU, fused Attention
                self._saved_input[key] = inputs[0].detach().cpu()
        return _hook

    def register_save_hooks(self):
        """Run over *clean* pass to cache inputs of critical sub‑modules."""
        attn_types: Tuple[type, ...] = ()
        if hasattr(open_clip, "transformer"):
            from open_clip.transformer import Attention  # type: ignore
            attn_types = (Attention,)

        for _, mod in self.model.visual.named_modules():
            if isinstance(mod, (nn.LayerNorm, nn.GELU, nn.MultiheadAttention) + attn_types):
                h = mod.register_forward_pre_hook(self._make_save_hook(id(mod)), with_kwargs=False)
                self._handles.append(h)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---------------- bias toggle -----------------------------------------
    def _toggle_bias(self, remove: bool):
        for m in self.model.visual.modules():
            if isinstance(m, nn.Linear):
                if remove:
                    if m.bias is not None:
                        self._orig_bias.setdefault(m, {})["bias"] = m.bias
                        m.bias = None
                elif m in self._orig_bias and "bias" in self._orig_bias[m]:
                    m.bias = self._orig_bias[m]["bias"]
            elif isinstance(m, nn.MultiheadAttention):
                if remove:
                    if getattr(m, "in_proj_bias", None) is not None:
                        self._orig_bias.setdefault(m, {})["in_proj_bias"] = m.in_proj_bias  # type: ignore[attr-defined]
                        m.in_proj_bias = None  # type: ignore[attr-defined]
                    if m.out_proj.bias is not None:
                        self._orig_bias.setdefault(m, {})["out_bias"] = m.out_proj.bias
                        m.out_proj.bias = None
                elif m in self._orig_bias:
                    if "in_proj_bias" in self._orig_bias[m]:
                        m.in_proj_bias = self._orig_bias[m]["in_proj_bias"]  # type: ignore[attr-defined]
                    if "out_bias" in self._orig_bias[m]:
                        m.out_proj.bias = self._orig_bias[m]["out_bias"]
        if not remove:
            self._orig_bias.clear()

    # ---------------- patch / unpatch -------------------------------------
    def patch(self):
        self._toggle_bias(True)
        for m in self.model.visual.modules():
            m._prism_patcher = weakref.proxy(self)  # type: ignore[attr-defined]
            if isinstance(m, nn.LayerNorm):
                self._orig_fwd[m] = m.forward  # type: ignore[assignment]
                m.forward = types.MethodType(PrismPatcher._ln_forward, m)
            elif isinstance(m, nn.GELU):
                self._orig_fwd[m] = m.forward  # type: ignore[assignment]
                rule = PrismPatcher._gelu_identity if self._gelu_rule == "identity" else PrismPatcher._gelu_diff
                m.forward = types.MethodType(rule, m)
            elif isinstance(m, nn.MultiheadAttention):
                self._orig_fwd[m] = m.forward  # type: ignore[assignment]
                m.forward = types.MethodType(PrismPatcher._mha_forward, m)

    def unpatch(self):
        for m, fwd in self._orig_fwd.items():
            m.forward = fwd  # type: ignore[assignment]
            if hasattr(m, "_prism_patcher"):
                delattr(m, "_prism_patcher")
        self._orig_fwd.clear()
        self._toggle_bias(False)

    # ------------- patched forward implementations ------------------------
    @staticmethod
    def _ln_forward(self: nn.LayerNorm, x):  # type: ignore[override]
        pp = self._prism_patcher  # type: ignore[attr-defined]
        x_saved = _device_like(pp._saved_input[id(self)], x)
        mu_delta = x.mean(dim=-1, keepdim=True)
        var_saved = (x_saved - x_saved.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1, keepdim=True)
        std = (var_saved + self.eps).sqrt()
        y = (x - mu_delta) / std
        if self.weight is not None:
            y = y * self.weight
        return y

    @staticmethod
    def _gelu_identity(self: nn.GELU, x):  # type: ignore[override]
        pp = self._prism_patcher  # type: ignore[attr-defined]
        x_saved = _device_like(pp._saved_input[id(self)], x)
        return (F.gelu(x_saved) / (x_saved + 1e-6)) * x

    @staticmethod
    def _gelu_diff(self: nn.GELU, x):  # type: ignore[override]
        pp = self._prism_patcher  # type: ignore[attr-defined]
        x_saved = _device_like(pp._saved_input[id(self)], x)
        return F.gelu(x_saved) - F.gelu(x_saved - x)

    @staticmethod
    def _mha_forward(self: nn.MultiheadAttention, q, k, v, **kwargs):  # type: ignore[override]
        pp = self._prism_patcher  # type: ignore[attr-defined]
        q_saved, k_saved = _device_like(pp._saved_input[id(self)], q)
        return pp._orig_fwd[self](q_saved, k_saved, v, **kwargs)  # type: ignore[misc]

# ---------------------------------------------------------------------------
# ImageNet logits -----------------------------------------------------------
# ---------------------------------------------------------------------------

def imagenet_logits(model, img: torch.Tensor, W: torch.Tensor, *, no_grad: bool = True):
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        f = model.encode_image(img)
        f = f / f.norm(dim=-1, keepdim=True)
        return 100.0 * f @ W.T

# ---------------------------------------------------------------------------
# Zero‑shot ImageNet classifier ---------------------------------------------
# ---------------------------------------------------------------------------

def build_classifier(model, tokenizer, device):
    with torch.no_grad():
        weights = []
        for cname in IMAGENET_CLASSNAMES:
            tokens = tokenizer([tmpl(cname) for tmpl in OPENAI_IMAGENET_TEMPLATES]).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            weights.append(emb.mean(0))
        W = torch.stack(weights)
        W = W / W.norm(dim=-1, keepdim=True)
    return W

# ---------------------------------------------------------------------------
# Residual capture & δR -----------------------------------------------------
# ---------------------------------------------------------------------------

def capture_residual(model: nn.Module, img: torch.Tensor, layer: int) -> torch.Tensor:
    holder: Dict[str, torch.Tensor] = {}

    def _hook(_m, inputs):  # inputs = (residual,)
        holder["r"] = inputs[0].detach()

    handle = model.visual.transformer.resblocks[layer].register_forward_pre_hook(_hook, with_kwargs=False)
    with torch.no_grad():
        model.encode_image(img)
    handle.remove()
    return holder["r"]


def compute_delta_r(residual: torch.Tensor, sae: "GlobalBatchTopKMatryoshkaSAE", feat_idx: int) -> torch.Tensor:
    B, S, D = residual.shape
    acts = sae.encode(residual.view(-1, D)).view(B, S, -1)
    strength = acts[:, :, feat_idx:feat_idx + 1]  # (B,S,1)
    delta = strength @ sae.W_dec[feat_idx:feat_idx + 1].to(residual.device)  # (B,S,D)
    return delta

# ---------------------------------------------------------------------------
# Delta injection -----------------------------------------------------------
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def inject_delta(model: nn.Module, layer: int, delta: torch.Tensor):
    blk = model.visual.transformer.resblocks[layer]
    handle = blk.register_forward_pre_hook(lambda _m, _in: (delta,), with_kwargs=False)
    try:
        yield
    finally:
        handle.remove()

# ---------------------------------------------------------------------------
# Logit helpers -------------------------------------------------------------
# ---------------------------------------------------------------------------

def imagenet_logits(model, img: torch.Tensor, W: torch.Tensor, *, no_grad: bool = True):
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        feats = model.encode_image(img)
        return feats @ W.T

# ---------------------------------------------------------------------------
# Direct contributions ------------------------------------------------------
# ---------------------------------------------------------------------------

def direct_lens(model: nn.Module, delta: torch.Tensor, W: torch.Tensor):
    cls_delta = delta[:, 0]  # (B,D)
    z = model.visual.ln_post(cls_delta)
    if getattr(model.visual, "proj", None) is not None:
        z = z @ model.visual.proj  # type: ignore[operator]
    # z = z / z.norm(dim=-1, keepdim=True)
    return z @ W.T
    # (1000,)
def direct_logit_grad(model, img, layer, delta, W):
    delta_req = delta.clone().detach().requires_grad_(True)
    with inject_delta(model, layer, delta_req):
        logits = imagenet_logits(model, img, W, no_grad=False).squeeze(0)

    # 모든 클래스에 대해 같은 delta를 쓰므로 grad_outputs를 1-vector로
    grad, = torch.autograd.grad(logits, delta_req,
                                grad_outputs=torch.ones_like(logits),
                                retain_graph=False)
    # (B,S,D) → (1000,) : <grad, δR>  을 모든 클래스에 복제
    return (grad * delta).sum(dim=(1, 2)).expand_as(logits).detach()
# def direct_logit_grad(model: nn.Module, img: torch.Tensor, layer: int, delta: torch.Tensor, W: torch.Tensor):
#     """Compute δlogit ≈ ∂logits/∂res · δR via a single VJP (faster than 1000 loops)."""
#     delta_req = delta.clone().detach().requires_grad_(True)
#     with inject_delta(model, layer, delta_req):
#         logits = imagenet_logits(model, img, W, no_grad=False).squeeze(0)
#     v = torch.ones_like(logits)
#     (vjp,) = torch.autograd.grad(logits, delta_req, grad_outputs=v)
#     return (vjp * delta).sum(dim=(1, 2)).expand_as(logits)  # scalar repeated per class

# ---------------------------------------------------------------------------
# Composite ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def feature_contributions(model, img, W, *, layer: int, delta: torch.Tensor, gelu_rule="identity", direct_mode="simple"):
    baseline = imagenet_logits(model, img, W).squeeze(0)

    patcher = PrismPatcher(model, gelu_rule=gelu_rule)
    patcher.register_save_hooks(); _ = model.encode_image(img); patcher.remove_hooks()

    patcher.patch()
    with inject_delta(model, layer, delta):
        prism_logits = imagenet_logits(model, img, W).squeeze(0)
    patcher.unpatch()
    prism_delta = prism_logits

    direct_simple = direct_lens(model, delta, W).squeeze(0)
    direct_grad_vec = direct_logit_grad(model, img, layer, delta, W).squeeze(0) if direct_mode == "grad" else torch.zeros_like(direct_simple)

    return baseline, prism_delta, direct_simple, direct_grad_vec

# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _pretty(scores: torch.Tensor, k: int, labels: Sequence[str]) -> List[str]:
    """Return top-k (<label> (<score>)) strings."""
    vals, idx = scores.topk(k)
    return [f"{labels[i]} ({v:+.2f})" for i, v in zip(idx.tolist(), vals.tolist())]


def main():
    p = argparse.ArgumentParser("logit‑prism CLI")
    p.add_argument("--image", default="/home/sangyu/Pictures/13_415/4.JPEG")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--feat", type=int, default=392)
    p.add_argument("--sae_root", default="./matryoshka_sae/checkpoints")
    p.add_argument("--gelu_rule", choices=["identity", "difference"], default="difference")
    p.add_argument("--direct_mode", choices=["simple", "grad"], default="grad")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--caption_weights", default="emb_np.npy",
                   help="Path to Conceptual-Captions embedding .npy")
    p.add_argument("--topk_caps", type=int, default=None,
                   help="Top-K captions to keep per image (memory saver)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = p.parse_args()

    if args.layer == 0:
        sys.exit("layer 0 cannot be analysed (SAE uses layer‑1 add_2).")

    # Model + preprocess --------------------------------------------------
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b-s32b-b82k", device=dev)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    if args.caption_weights is None:
        W = build_classifier(model, tokenizer, dev)      # ImageNet default
        LABELS  = IMAGENET_CLASSNAMES 
    else:
        from datasets import load_dataset                      # ⟵ HF datasets
        captions_ds = load_dataset("conceptual_captions", split="train[:25%]")
        sentences   = captions_ds["caption"]                   # List[str] ≃700 k
        W_np        = np.load(args.caption_weights)            # (N, 768)
        W           = torch.from_numpy(W_np).to(dev)
        LABELS      = sentences        
    img = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(dev)
    
  
    # Residual & δR -------------------------------------------------------
    residual = capture_residual(model, img, args.layer)
    sae = load_sae(f"transformer_resblocks_{args.layer - 1}_add_2", args.sae_root, torch.device(dev))
    delta = compute_delta_r(residual, sae, args.feat)

    # Compute contributions ----------------------------------------------
    base, prism, d_simple, d_grad = feature_contributions(
        model, img, W, layer=args.layer, delta=delta, gelu_rule=args.gelu_rule, direct_mode=args.direct_mode
    )

    # Display -------------------------------------------------------------
    print("\nBaseline:");          [print(" ", s) for s in _pretty(base,    args.topk, LABELS)]
    print("\nPrism Δ:");           [print(" ▲", s) for s in _pretty(prism,  args.topk, LABELS)]
    print("\nDirect-lens Δ:");     [print(" ", s) for s in _pretty(d_simple,args.topk, LABELS)]
    if args.direct_mode == "grad":
        print("\nDirect grad×input Δ:")
        [print(" ", s) for s in _pretty(d_grad, args.topk, LABELS)]


if __name__ == "__main__":
    main()
