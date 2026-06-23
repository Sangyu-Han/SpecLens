#!/usr/bin/env python3
"""NER frontier — genuine NON-self-patch (latent bottleneck): Perceiver.

Perceiver has 512 LEARNED latents with NO input position; routing = encoder cross-attn
[512 latents x 50176 pixels]. Self-patch is gone. Question: recover input (pixel-grid)
necessity cheaply.

Model-agnostic eval (Block0Runner is CLIP-only): mask a g x g input grid toward the
MEAN pixel (=normalized 0), full forward, prob target -> deletion/insertion AUC.

Brackets this pass (NER added next):
  oracle       greedy-conditional pixel-grid removal (gold)
  xattn_roll   decoder->latents->pixels cross-attn rollout (RAW routing = sufficiency)
  gradient     |d prob/d input| pooled to grid (fails-ref)
  input_fri    FRI-delonly soft-mask on the grid, full-model solve (honest cheap baseline)
LOWER deletion AUC = better. cuda:0.
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


class _Stop(Exception):
    """early-stop a forward right after the cross-attention (keeps NER cheap)."""


def up_grid(keep_g, g, dev):
    s = 224 // g
    return keep_g.view(1, 1, g, g).repeat_interleave(s, 2).repeat_interleave(s, 3).to(dev)


def up_grid_batch(keep_b, g, dev):
    B = keep_b.shape[0]; s = 224 // g
    return keep_b.view(B, 1, g, g).repeat_interleave(s, 2).repeat_interleave(s, 3).to(dev)


class PerceiverNec:
    def __init__(self, model, x, dev):
        self.model = model; self.x = x; self.dev = dev

    def prob(self, keep_g, c, g):
        up = up_grid(keep_g, g, self.dev)
        with torch.no_grad():
            lg = self.model(inputs=self.x * up).logits[0]
        return float(torch.softmax(lg, -1)[c])

    def prob_diff(self, keep_g, c, g):
        up = up_grid(keep_g, g, self.dev)
        lg = self.model(inputs=self.x * up).logits[0]
        return torch.softmax(lg, -1)[c]

    # ---- methods (each returns score[g*g], higher = more necessary) ----
    def gradient(self, c, g):
        x2 = self.x.clone().requires_grad_(True)
        p = torch.softmax(self.model(inputs=x2).logits[0], -1)[c]
        gr = torch.autograd.grad(p, x2)[0][0].abs().sum(0)        # [224,224]
        s = 224 // g
        return gr.view(g, s, g, s).sum(dim=(1, 3)).flatten().detach().cpu().numpy()

    def xattn_rollout(self, c, g):
        with torch.no_grad():
            out = self.model(inputs=self.x, output_attentions=True, return_dict=True)
        ca = out.cross_attentions
        enc = ca[0][0].mean(0)                                    # [512, 50176] avg heads
        # decoder cross-attn (query->latents); fall back to uniform latent weights
        lat_w = None
        if len(ca) > 1 and ca[1] is not None:
            dec = ca[1][0].mean(0)                                # [q, 512]
            lat_w = dec.mean(0)                                   # [512]
        if lat_w is None or lat_w.shape[0] != enc.shape[0]:
            lat_w = torch.ones(enc.shape[0], device=enc.device)
        pix = (lat_w[:, None] * enc).sum(0)                       # [50176]
        side = int(round(pix.shape[0] ** 0.5))
        s = 224 // g
        return pix.view(side, side)[:224, :224].view(g, s, g, s).sum(dim=(1, 3)).flatten().detach().cpu().numpy()

    def input_fri(self, c, g, steps=48, lr=0.4, lr_end=0.01, l1=0.004, seed=0):
        N = g * g; dev = self.dev
        full = self.prob(torch.ones(N, device=dev), c, g)
        base = self.prob(torch.zeros(N, device=dev), c, g)
        den = abs(full - base) + 1e-6
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        la = torch.zeros(N, device=dev); mv = torch.zeros(N, device=dev); vv = torch.zeros(N, device=dev)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
            la_req = la.clone().requires_grad_(True)
            p = torch.sigmoid(la_req)
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
            r_state = (p / (p.sum() + eps) * budget).clamp(max=1)
            keep = 1.0 - r_state
            rec = (self.prob_diff(keep, c, g) - base) / den
            loss = rec + l1 * p.sum()
            gg = torch.autograd.grad(loss, la_req)[0].detach()
            t = step + 1
            mv = b1 * mv + (1 - b1) * gg; vv = b2 * vv + (1 - b2) * gg * gg
            adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
            cmask = (adam * gg > 0).float(); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
            la = la - cur * adam * cmask
        return torch.sigmoid(la).detach().cpu().numpy()

    # ---- NER: routing-anchored causal solve (cross-attn-local target) ----
    def _ghat(self, c, g=None, keep=None):
        """ĝ = d logit_c / d(post-cross-attn latents). keep=None -> at clean input;
        else at the masked state (for conditional refresh)."""
        store = {}
        def hook(m, i, o):
            ca = o[0] if isinstance(o, tuple) else o
            ca.retain_grad(); store["ca"] = ca
        h = self.model.perceiver.encoder.cross_attention.register_forward_hook(hook)
        self.model.zero_grad(set_to_none=True)
        if keep is None:
            xin = self.x.clone().requires_grad_(True)
        else:
            xin = (self.x * up_grid(keep, g, self.dev)).detach().requires_grad_(True)
        logit = self.model(inputs=xin).logits[0, c]
        logit.backward()
        h.remove()
        return store["ca"].grad.detach()                          # [1,512,1024]

    def _ca_masked(self, keep_g, g):
        """differentiable post-cross-attn latents for masked input; early-stop after cross-attn."""
        store = {}
        def hook(m, i, o):
            store["ca"] = o[0] if isinstance(o, tuple) else o
            raise _Stop()
        h = self.model.perceiver.encoder.cross_attention.register_forward_hook(hook)
        up = up_grid(keep_g, g, self.dev)
        try:
            self.model(inputs=self.x * up)
        except _Stop:
            pass
        h.remove()
        return store["ca"]                                        # [1,512,1024] differentiable

    def ner(self, c, g, steps=48, lr=0.4, lr_end=0.01, l1=0.004, seed=0):
        ghat = self._ghat(c); N = g * g; dev = self.dev
        def T(keep_g):
            return (self._ca_masked(keep_g, g) * ghat).sum()
        with torch.no_grad():
            full = float(T(torch.ones(N, device=dev))); base = float(T(torch.zeros(N, device=dev)))
        den = abs(full - base) + 1e-6
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        la = torch.zeros(N, device=dev); mv = torch.zeros(N, device=dev); vv = torch.zeros(N, device=dev)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
            la_req = la.clone().requires_grad_(True)
            p = torch.sigmoid(la_req)
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
            r_state = (p / (p.sum() + eps) * budget).clamp(max=1)
            keep = 1.0 - r_state
            rec = (T(keep) - base) / den
            loss = rec + l1 * p.sum()
            gg = torch.autograd.grad(loss, la_req)[0].detach()
            t = step + 1
            mv = b1 * mv + (1 - b1) * gg; vv = b2 * vv + (1 - b2) * gg * gg
            adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
            cmask = (adam * gg > 0).float(); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
            la = la - cur * adam * cmask
        return torch.sigmoid(la).detach().cpu().numpy()

    @torch.no_grad()
    def _ca_batch(self, keep_b, g):
        """post-cross-attn latents for a BATCH of grid masks; early-stop. [B,512,1024]."""
        store = {}
        def hook(m, i, o):
            store["ca"] = o[0] if isinstance(o, tuple) else o
            raise _Stop()
        h = self.model.perceiver.encoder.cross_attention.register_forward_hook(hook)
        up = up_grid_batch(keep_b, g, self.dev)
        try:
            self.model(inputs=self.x * up)
        except _Stop:
            pass
        h.remove()
        return store["ca"]

    def ner_cond(self, c, g, chunk=8, refresh=0):
        """greedy-CONDITIONAL removal scored by the cheap cross-attn projection <ca, ghat>.
        refresh>0: recompute ghat at the current masked state every `refresh` rounds (fixes
        the stale-linear-surrogate drift). Returns order-score + #cross-attn forward-images."""
        ghat = self._ghat(c, g); N = g * g; dev = self.dev
        removed = np.zeros(N, np.float32); order = []; remaining = list(range(N)); ncaf = 0
        round_i = 0
        while remaining:
            if refresh and round_i > 0 and round_i % refresh == 0:
                ghat = self._ghat(c, g, keep=torch.as_tensor(1.0 - removed, device=dev))
            round_i += 1
            K = len(remaining)
            keep = np.tile(1.0 - removed, (K, 1)).astype(np.float32)
            for r, idx in enumerate(remaining):
                keep[r, idx] = 0.0
            keep_t = torch.as_tensor(keep, device=dev)
            Ts = []
            for s in range(0, K, chunk):
                ca = self._ca_batch(keep_t[s:s + chunk], g)
                Ts.append((ca * ghat).sum(dim=(1, 2)).detach()); del ca
            T = torch.cat(Ts); ncaf += K
            j = int(torch.argmin(T)); p = remaining[j]
            order.append(p); removed[p] = 1.0; remaining.remove(p)
        score = np.zeros(N)
        for rank, p in enumerate(order):
            score[p] = N - rank
        return score, ncaf

    @torch.no_grad()
    def _prob_batch(self, keep_b, g, c):
        up = up_grid_batch(keep_b, g, self.dev)
        lg = self.model(inputs=self.x * up).logits
        return torch.softmax(lg, -1)[:, c]

    def oracle(self, c, g, chunk=8):
        N = g * g; dev = self.dev
        full = float(self._prob_batch(torch.ones(1, N, device=dev), g, c))
        removed = np.zeros(N, np.float32); order = []; remaining = list(range(N)); probc = []
        while remaining:
            K = len(remaining)
            keep = np.tile(1.0 - removed, (K, 1)).astype(np.float32)
            for r, idx in enumerate(remaining):
                keep[r, idx] = 0.0
            keep_t = torch.as_tensor(keep, device=dev)
            vals = []
            for s in range(0, K, chunk):
                vals.append(self._prob_batch(keep_t[s:s + chunk], g, c))
            vals = torch.cat(vals).cpu().numpy()
            j = int(np.argmin(vals)); p = remaining[j]
            order.append(p); removed[p] = 1.0; remaining.remove(p); probc.append(float(vals[j]))
        marg = np.zeros(N); prev = full
        for k, p in enumerate(order):
            marg[p] = max(prev - probc[k], 0.0); prev = probc[k]
        # TRUE floor: the greedy deletion curve itself (not the re-ranked marginal score)
        greedy_auc = float(np.mean([full] + probc) / (full + 1e-9))
        return marg, greedy_auc


def del_auc(pnec, scores, c, g, chunk=8):
    """hard deletion AUC: remove top-score cells cumulatively (batched), norm by full prob."""
    N = g * g
    order = np.argsort(-np.asarray(scores))
    masks = np.ones((N + 1, N), np.float32)
    for k in range(N):
        masks[k + 1:, order[k]] = 0.0                            # row r: order[:r] removed
    keep_t = torch.as_tensor(masks, device=pnec.dev)
    probs = []
    for s in range(0, N + 1, chunk):
        probs.append(pnec._prob_batch(keep_t[s:s + chunk], g, c))
    probs = torch.cat(probs).cpu().numpy()
    return float(probs.mean() / (probs[0] + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--nimg", type=int, default=3)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--val-dir", default="/media/sangyu/Dataset/imagenet/val")
    args = ap.parse_args()
    from transformers import PerceiverForImageClassificationLearned, PerceiverImageProcessor
    from PIL import Image
    dev = args.device
    proc = PerceiverImageProcessor.from_pretrained("deepmind/vision-perceiver-learned")
    model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned").eval().to(dev)
    for p in model.parameters(): p.requires_grad = False

    vd = Path(args.val_dir); subdirs = sorted([d for d in vd.iterdir() if d.is_dir()])
    step = max(1, len(subdirs) // args.nimg)
    imgs = [sorted(d.glob("*.JPEG"))[0] for d in subdirs[::step][:args.nimg]]
    g = args.grid
    methods = ["gradient", "xattn_roll", "ner_cond", "ner_cond_R", "oracle"]
    acc = {m: [] for m in methods}
    cost = {"ner_cond": [], "oracle": []}
    tfwd = {}
    print(f"[setup] perceiver-learned  grid={g}x{g}={g*g}  n={len(imgs)}", flush=True)
    t0 = time.time()
    for ii, path in enumerate(imgs):
        pil = Image.open(path).convert("RGB")
        x = proc(images=pil, return_tensors="pt").pixel_values.to(dev)
        with torch.no_grad():
            c = int(model(inputs=x).logits[0].argmax())
        pn = PerceiverNec(model, x, dev)
        if ii == 0:  # one-time per-forward cost: full vs early-stopped cross-attn
            ones = torch.ones(1, g * g, device=dev)
            torch.cuda.synchronize(); a = time.time()
            for _ in range(10): pn.prob(ones[0], c, g)
            torch.cuda.synchronize(); tfwd["full"] = (time.time() - a) / 10
            a = time.time()
            for _ in range(10): pn._ca_batch(ones, g)
            torch.cuda.synchronize(); tfwd["ca"] = (time.time() - a) / 10
        sc = {"gradient": pn.gradient(c, g), "xattn_roll": pn.xattn_rollout(c, g)}
        score_nc, ncaf = pn.ner_cond(c, g, refresh=0)
        sc["ner_cond"] = score_nc
        sc["ner_cond_R"], _ = pn.ner_cond(c, g, refresh=4)
        _, oracle_auc = pn.oracle(c, g)
        line = [f"[{ii+1}/{len(imgs)}] {path.name} c={c}"]
        for m in ["gradient", "xattn_roll", "ner_cond", "ner_cond_R"]:
            d = del_auc(pn, sc[m], c, g); acc[m].append(d); line.append(f"{m}:{d:.3f}")
        acc["oracle"].append(oracle_auc); line.append(f"oracle:{oracle_auc:.3f}")
        cost["ner_cond"].append(ncaf); cost["oracle"].append(g * g * (g * g + 1) // 2)
        print("  " + " | ".join(line), flush=True)
        del pn; torch.cuda.empty_cache()
    print(f"\n=== MEAN del AUC over {len(acc['oracle'])} imgs (LOWER=better) ===", flush=True)
    for m in methods:
        v = np.mean(acc[m])
        tag = "  <-- GOLD" if m == "oracle" else ("  <-- NER (cheap conditional via routing)" if m == "ner_cond" else "")
        print(f"  {m:<11} {v:.3f}{tag}", flush=True)
    if tfwd:
        gap = (np.mean(acc["xattn_roll"]) - np.mean(acc["ner_cond"])) / (np.mean(acc["xattn_roll"]) - np.mean(acc["oracle"]) + 1e-9)
        print(f"\n  NER-cond closes {gap*100:.0f}% of best-static->oracle gap", flush=True)
        print(f"  cost: ner_cond {np.mean(cost['ner_cond']):.0f} cross-attn fwd/img @ {tfwd['ca']*1e3:.0f}ms"
              f"  vs oracle {np.mean(cost['oracle']):.0f} FULL fwd/img @ {tfwd['full']*1e3:.0f}ms"
              f"  -> {tfwd['full']/max(tfwd['ca'],1e-9):.1f}x cheaper per forward", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
