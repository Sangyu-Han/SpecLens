"""logit_prism_patches.py — *second‑lens* edition (SAE‑friendly)
===================================================================
Adds a faithful **second‑order‑lens (SoL)** implementation — à‑la
<https://arxiv.org/abs/2406.04341> official repo — on top of the
existing *logit‑prism* analysis pipeline, but specialised for **Matryoshka‑SAE
features**.

Highlights & design choices
---------------------------
* **Exact linear path** Replicates LayerNorm‑1 → V proj → soft‑max
  attention → O proj for *all* blocks *after* the injection layer.
* **Uses cached clean‑run statistics** We reuse the same `PrismPatcher`
  hooks to cache LayerNorm inputs (`ln_1`) and the *pre‑projection* Q/K
  tensors; from these we rebuild the attention maps so we never need to
  re‑run the network.
* **Bias‑free lens** Biases are stripped (identical to logit‑prism),
  matching the paper’s formulation that focuses purely on weight paths.
* **Device‑agnostic** Works on CPU or GPU depending on the *delta*
  tensor.
* **Optional inclusion in CLI** `--show_sol` prints the Top‑k ImageNet
  classes affected by the second‑order path.

Drop‑in‑compatible: just import & call `second_order_lens()` or use the
updated `feature_contributions()`.
"""
from __future__ import annotations

import argparse, contextlib, math, sys, types, weakref
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Union

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
import open_clip
from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

# ---------------------------------------------------------------------------
# Matryoshka‑SAE helpers (unchanged) ----------------------------------------
# ---------------------------------------------------------------------------
try:
    from matryoshka_sae.config import get_default_cfg, post_init_cfg
    from matryoshka_sae.sae import GlobalBatchTopKMatryoshkaSAE
except ImportError:  # graceful degradation when repo absent
    GlobalBatchTopKMatryoshkaSAE = object  # type: ignore
from logitprism_2_general import (
    PrismPatcher, load_caption_weights, load_sae, capture_residual,
    compute_delta_r, inject_delta, build_classifier,
    imagenet_logits, direct_lens, direct_logit_grad,
)
# ---------------------------------------------------------------------------
# Utility -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _device_like(src, ref: torch.Tensor):
    if isinstance(src, tuple):
        return tuple(s.to(ref.device, dtype=ref.dtype) for s in src)
    return src.to(ref.device, dtype=ref.dtype)

# ---------------------------------------------------------------------------
# ▼▼▼  SECOND‑ORDER‑LENS CORE  ▼▼▼ ------------------------------------------
# ---------------------------------------------------------------------------

class SecondOrderLens:
    """Standalone second‑order‑lens computer for a *single* residual delta.

    Parameters
    ----------
    model : nn.Module
        The OpenCLIP model (ViT‑L‑14) **already run once** in *clean* mode.
    patcher : PrismPatcher
        Must have run `register_save_hooks()` + a clean forward so its
        ``_saved_input`` dict is populated with LayerNorm inputs and Q/K
        tensors. *No need* to keep the bias/activation patches active; we
        only rely on the cached tensors.
    start_layer : int
        Layer index **where `delta` is injected** (1 ≤ L < n_layers).

    Notes
    -----
    *Follows exactly Eq.(5) of the paper*: we propagate the residual
    change only through the *attention* path of every subsequent block,
    omitting all later MLPs, then project with `ln_post` + (optional)
    `proj` into the 1024‑dim CLIP embedding space.
    """

    def __init__(self, model: nn.Module, patcher: "PrismPatcher", *, start_layer: int):
        self.model = model
        self.patcher = patcher
        self.start_layer = start_layer
        self.blocks: List[nn.Module] = list(model.visual.transformer.resblocks)
        self._sanity()

    # -------------------------------------------------------------------
    # public API ---------------------------------------------------------
    # -------------------------------------------------------------------
    def __call__(self, delta: torch.Tensor) -> torch.Tensor:
        """Return **δCLS residual** after all attention paths (shape : (B,D))."""
        assert delta.ndim == 3, "delta must be (B,S,D) residual tensor"
        B, S, D = delta.shape
        dtype, device = delta.dtype, delta.device
        delta_cur = delta  # will be mutated in‑place layer‑by‑layer
        # delta_cur[:,0] += delta.mean(dim=1) 
        # run over layers AFTER the injection point
        for l in range(self.start_layer, len(self.blocks)):
            blk = self.blocks[l]
            delta_cur = self._one_block(delta_cur, blk, device, dtype)

        # CLS only -------------------------------------------------------
        cls_delta = delta_cur[:,0] # (B,D)
        # cls_delta = delta_cur[:,1] 

        # ----- ln_post strictly linearised (bias‑free) ------------------
        ln_post = self.model.visual.ln_post
        x_saved = _device_like(self.patcher._saved_input[id(ln_post)], cls_delta)[:, 0]  # clean CLS input
        std = (x_saved.var(dim=-1, unbiased=False, keepdim=True) + ln_post.eps) ** 0.5
        if ln_post.weight is not None:
            z = (cls_delta / std) * ln_post.weight
        else:
            z = cls_delta / std  # rare: no weight param

        # optional projection to 1024‑D embedding
        if getattr(self.model.visual, "proj", None) is not None:
            z = z @ self.model.visual.proj
        return z  # (B,1024)

    # -------------------------------------------------------------------
    # inner helpers ------------------------------------------------------
    # -------------------------------------------------------------------
    def _one_block(self, delta: torch.Tensor, blk, device, dtype):
        """Propagate *one* residual delta through Attention of `blk`."""
        ln1 = blk.ln_1
        attn = blk.attn  # works for both OpenCLIP & torch MHA
        B,S,D = delta.shape
        # (1) -------- LayerNorm (use saved variance, fresh mean of delta)
        x_saved = _device_like(self.patcher._saved_input[id(ln1)], delta)
        var_saved = (x_saved - x_saved.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1, keepdim=True)
        std = (var_saved + ln1.eps).sqrt()
        # std = std.mean()
        # var = delta.pow(2).mean(dim=-1,keepdim=True)
        # std = (var + ln1.eps).sqrt()  # fresh std of delta
        delta_norm = (delta - delta.mean(dim=-1, keepdim=True)) / std
        if ln1.weight is not None:
            delta_norm = delta_norm * ln1.weight

        # (2) -------- Value projection (B,S,D) → (B,heads,S,H)
        W_V = _get_v_proj(attn).to(device=device, dtype=dtype)          # (D, D)
        delta_v = delta_norm @ W_V                                     # (B,S,D)
        nh, hd = attn.num_heads, attn.head_dim
        delta_v = delta_v.view(B, S, nh, hd).permute(0, 2, 1, 3)       # (B,nh,S,hd)

        # (3) -------- Attention map (re‑derived from cached Q & K)
        q_saved, k_saved = self.patcher._saved_input[id(attn)]          # each: (S,B,D) on CPU
        q = q_saved.to(device=device, dtype=dtype).permute(1, 0, 2)    # (B,S,D)
        k = k_saved.to(device=device, dtype=dtype).permute(1, 0, 2)    # (B,S,D)
        # q = delta_norm
        # k = delta_norm
        

        W_Q, W_K = _get_q_proj(attn).to(device, dtype), _get_k_proj(attn).to(device, dtype)
        q_proj = (q @ W_Q).view(B, S, nh, hd).permute(0, 2, 1, 3)       # (B,nh,S,hd)
        k_proj = (k @ W_K).view(B, S, nh, hd).permute(0, 2, 1, 3)       # (B,nh,S,hd)
        attn_scores = (q_proj @ k_proj.transpose(-2, -1)) / math.sqrt(hd)  # (B,nh,S,S)
        # attn_scores = torch.ones_like(attn_scores)
        
        # identity = torch.eye(S, device=attn_scores.device, dtype=attn_scores.dtype)  # (S, S)
        # attn_probs = identity.unsqueeze(0).unsqueeze(0).expand(B, nh, S, S) 
        # attn_scores = attn_scores - attn_probs
        
        attn_probs = attn_scores.softmax(dim=-1)

        # (4) -------- Context accumulation  (B,nh,S,hd)
        delta_ctx = attn_probs @ delta_v  # (B,nh,S,hd)
        # delta_ctx = 0.12 * delta_v
        delta_ctx = delta_ctx.permute(0, 2, 1, 3).contiguous().view(B, S, D)  # (B,S,D)

        # (5) -------- Out proj & residual add
        W_O = attn.out_proj.weight.T.to(device=device, dtype=dtype)  # (D,D)
        delta_out = delta_ctx @ W_O  # (B,S,D)
        return delta_out + delta  # residual connection

    # -------------------------------------------------------------------
    def _sanity(self):
        if not self.patcher._saved_input:
            raise RuntimeError("PrismPatcher must have run a clean forward first.")
        if self.start_layer >= len(self.blocks):
            raise ValueError("start_layer out of range")

# ---------------------------------------------------------------------------
# projection helpers --------------------------------------------------------
# ---------------------------------------------------------------------------

def _get_q_proj(attn):
    if hasattr(attn, "q_proj"):
        return attn.q_proj.weight.T  # (D,D)
    # torch.nn.MultiheadAttention path
    D = attn.embed_dim
    return attn.in_proj_weight[:D, :].T  # type: ignore[attr-defined]

def _get_k_proj(attn):
    if hasattr(attn, "k_proj"):
        return attn.k_proj.weight.T
    D = attn.embed_dim
    return attn.in_proj_weight[D:2 * D, :].T  # type: ignore[attr-defined]

def _get_v_proj(attn):
    if hasattr(attn, "v_proj"):
        return attn.v_proj.weight.T
    D = attn.embed_dim
    return attn.in_proj_weight[2 * D : 3 * D, :].T  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# PrismPatcher (original) ---------------------------------------------------
# ---------------------------------------------------------------------------
#  ... (UNCHANGED – the full PrismPatcher class from the base file goes here) ...
# Due to brevity, the original PrismPatcher implementation is omitted – keep
# it exactly as in the stable edition provided by the user.

# ---------------------------------------------------------------------------
# delta capture / SAE helpers  (UNCHANGED) ----------------------------------
# ---------------------------------------------------------------------------
#  ... capture_residual, compute_delta_r, etc. – copy verbatim ...

# ---------------------------------------------------------------------------
# Second‑lens contribution wrapper -----------------------------------------
# ---------------------------------------------------------------------------

def second_order_lens(model: nn.Module, patcher: "PrismPatcher", delta: torch.Tensor, *, layer: int, W: torch.Tensor):
    """Return ImageNet **δlogits** via second‑order lens (no grads)."""
    sol = SecondOrderLens(model, patcher, start_layer=layer)
    z_delta = sol(delta)                     # (B,1024)
    return z_delta @ W.T                     # (B,1000) logits delta

# ---------------------------------------------------------------------------
# Composite – extended with SoL --------------------------------------------
# ---------------------------------------------------------------------------

def feature_contributions(
    model,
    img,
    W,
    *,
    layer: int,
    delta: torch.Tensor,
    gelu_rule="identity",
    direct_mode="simple",
    show_sol: bool = True,
):
    """Return (baseline, prismΔ, directΔ, gradΔ, *[solΔ]*) tuple."""
    # with inject_delta(model, layer, delta):
    #     baseline = imagenet_logits(model, img, W).squeeze(0)
        
    baseline = imagenet_logits(model, img, W).squeeze(0)

    patcher = PrismPatcher(model, gelu_rule=gelu_rule)
    # 1️⃣ Clean run – fill caches
    patcher.register_save_hooks(); _ = model.encode_image(img); patcher.remove_hooks()

    # 2️⃣ Second‑order lens (no patches needed during propagation)
    sol_delta = second_order_lens(model, patcher, delta, layer=layer, W=W).squeeze(0) if show_sol else torch.zeros_like(baseline)

    # 3️⃣ Prism path (needs patches)
    patcher.patch()
    with inject_delta(model, layer, delta):
        prism_logits = imagenet_logits(model, img, W).squeeze(0)
    patcher.unpatch()
    prism_delta = prism_logits
    
    
    
    
    
    
    
    # 4️⃣ Direct paths
    direct_simple = direct_lens(model, delta, W).squeeze(0)
    direct_grad_vec = (
        direct_logit_grad(model, img, layer, delta, W).squeeze(0) if direct_mode == "grad" else torch.zeros_like(direct_simple)
    )

    return baseline, prism_delta, direct_simple, direct_grad_vec, sol_delta

# ---------------------------------------------------------------------------
# CLI (extended) ------------------------------------------------------------
# ---------------------------------------------------------------------------

def _pretty(scores: torch.Tensor, k: int, labels: Sequence[str]) -> List[str]:
    vals, idx = scores.topk(k)
    return [f"{labels[i]} ({v:+.2f})" for i, v in zip(idx.tolist(), vals.tolist())]

def _pretty_scores(scores: torch.Tensor, ext_idxs, labels: Sequence[str] ) -> List[str]:
    vals = scores[ext_idxs]
    return [f"{labels[i]} ({v:+.2f})" for i, v in zip(ext_idxs.tolist(), vals.tolist())]

def compute_delta_r(residual: torch.Tensor, sae: "GlobalBatchTopKMatryoshkaSAE", feat_idx: int) -> torch.Tensor:
    B, S, D = residual.shape
    acts = sae.encode(residual.view(-1, D)).view(B, S, -1)
    strength = acts[:, :, feat_idx:feat_idx + 1]  # (B,S,1)
    # zero = 4 * torch.ones_like(strength)  # this is for non-class features
    # zero[0,0] = 0.  # (B,S,1) --- IGNORE ---
    zero = torch.zeros_like(strength)  # (B,S,1) # this is for class features?
    zero[0,40] = 1.
    zero[0,41] = 1.
    zero[0,42] = 1.
    
    # strength = zero
    delta = strength @ sae.W_dec[feat_idx:feat_idx + 1].to(residual.device)  # (B,S,D)
    # delta = residual - delta
    return delta

def main():
    p = argparse.ArgumentParser("logit‑prism + second‑lens CLI")
    # p.add_argument("--image", default="/home/sangyu/Pictures/12_415/1.JPEG")
    p.add_argument("--image", default="./Sample_imgs/freedom.jpeg")
    p.add_argument("--layer", type=int, default=10) # L + 1
    p.add_argument("--feat", type=int, default=277)
    p.add_argument("--sae_root", default="./matryoshka_sae/checkpoints")
    p.add_argument("--gelu_rule", choices=["identity", "difference"], default="identity")
    p.add_argument("--direct_mode", choices=["simple", "grad"], default="simple")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--show_sol", action="store_true", help="print second‑order‑lens Δ logits")
    p.add_argument("--caption_weights", default="emb_np.npy",
                help="Path to Conceptual-Captions embedding .npy")
    p.add_argument("--topk_caps", type=int, default=None,
                   help="Top-K captions to keep per image (memory saver)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = p.parse_args()

    if args.layer == 0:
        sys.exit("layer 0 cannot be analysed (SAE uses layer‑1 add_2).")

    # ---- Model ----------------------------------------------------------
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

    # ---- Residual Δ -----------------------------------------------------
    residual = capture_residual(model, img, args.layer)
    sae = load_sae(f"transformer_resblocks_{args.layer - 1}_add_2", args.sae_root, torch.device(dev))
    delta = compute_delta_r(residual, sae, args.feat)

    # ---- Contributions --------------------------------------------------
    base, prism, d_simple, d_grad, sol = feature_contributions(
        model,
        img,
        W,
        layer=args.layer,
        delta=delta,
        gelu_rule=args.gelu_rule,
        direct_mode=args.direct_mode,
        show_sol=True,
    )

    # ---- Display --------------------------------------------------------
    print("\nBaseline:");            [print(" ", s) for s in _pretty(base,    args.topk, LABELS)]
    print("\nPrism Δ:");             [print(" ▲", s) for s in _pretty(prism,  args.topk, LABELS)]
    print("\nDirect‑lens Δ:");       [print(" ", s) for s in _pretty(d_simple,args.topk, LABELS)]
    if args.direct_mode == "grad":
        print("\nDirect grad×input Δ:")
        [print(" ", s) for s in _pretty(d_grad, args.topk, LABELS)]

    print("\nSecond‑order‑lens Δ:")
    [print(" ", s) for s in _pretty(sol, args.topk, LABELS)]
    _, idxs = torch.topk(base,args.topk)
    print("\nPrism Δ to base:");      [print(" ", s) for s in _pretty_scores(prism, idxs, LABELS)]
    print("\nSecond‑order Δ to base:"); [print(" ", s) for s in _pretty_scores(sol, idxs, LABELS)]


if __name__ == "__main__":
    main()
