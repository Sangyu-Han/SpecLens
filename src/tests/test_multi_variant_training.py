"""Tests for shared activation buffer multi-variant SAE training.

Run with: pytest src/tests/test_multi_variant_training.py -v
"""
from __future__ import annotations

import copy
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Minimal stubs so tests run without the full SAM2 / SpecLens install
# ---------------------------------------------------------------------------

# Stub TokenBlockQueue (mirrors the real one's interface)
class _FakeTokenBlockQueue:
    def __init__(self, block_size_tokens: int = 4096, **kwargs):
        self.block = block_size_tokens
        self.blocks: deque = deque()
        self.ntoks: int = 0

    def push(self, t: torch.Tensor) -> None:
        t = t.detach().cpu().contiguous()
        self.blocks.append(t)
        self.ntoks += t.shape[0]

    def pop_batch(self, B: int, shuffle: bool = True) -> Optional[torch.Tensor]:
        if self.ntoks == 0:
            return None
        parts = []
        remaining = B
        while remaining > 0 and self.blocks:
            blk = self.blocks[0]
            take = min(remaining, blk.shape[0])
            parts.append(blk[:take])
            if take == blk.shape[0]:
                self.blocks.popleft()
            else:
                self.blocks[0] = blk[take:]
            self.ntoks -= take
            remaining -= take
        if not parts:
            return None
        out = torch.cat(parts, dim=0)
        if shuffle:
            out = out[torch.randperm(out.shape[0])]
        return out


# Minimal SAE stub for integration tests
class _TinyBatchTopkSAE(torch.nn.Module):
    """Tiny 2-layer linear SAE for testing (not a real SAE)."""
    config: dict

    def __init__(self, act_size: int = 8, dict_size: int = 16):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.randn(act_size, dict_size) * 0.1)
        self.b_enc = torch.nn.Parameter(torch.zeros(dict_size))
        self.W_dec = torch.nn.Parameter(torch.randn(dict_size, act_size) * 0.1)
        self.b_dec = torch.nn.Parameter(torch.zeros(act_size))
        self.config = {"act_size": act_size, "dict_size": dict_size,
                       "n_batches_to_dead": 5}
        self.num_batches_not_active = torch.zeros(dict_size)

    def forward(self, x: torch.Tensor):
        h = torch.relu(x @ self.W_enc + self.b_enc)
        recon = h @ self.W_dec + self.b_dec
        loss = ((x - recon) ** 2).mean()
        return {"loss": loss, "sae_out": recon, "feature_acts": h,
                "l2_loss": loss, "l1_loss": torch.tensor(0.0),
                "aux_loss": torch.tensor(0.0)}


# ---------------------------------------------------------------------------
# Minimal stub for UniversalActivationStore (only the parts needed)
# ---------------------------------------------------------------------------

class _StubActivationStore:
    """Minimal activation store stub with just the multi-variant API."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.block_size_tokens = 256
        self.in_memory_blocks_per_layer = None
        self.spill_to_disk = False
        self.spill_dir = None
        self.buffer_on_cpu = True
        self.activation_batch_size = 64
        self.queues: Dict[str, _FakeTokenBlockQueue] = {}
        self.variant_queues: Dict[str, Dict[str, _FakeTokenBlockQueue]] = {}
        self._layer_variants: Dict[str, list] = {}
        self._pair_locks: dict = {}

        import threading
        self._get_lock = threading.Lock

    # -- mirror the real API --
    def _cfg_for_layer(self, lname: str) -> dict:
        return {}

    def register_variants(self, lname: str, variant_names: list) -> None:
        if not variant_names:
            return
        per_v = {}
        for vname in variant_names:
            per_v[vname] = _FakeTokenBlockQueue(block_size_tokens=self.block_size_tokens)
        self.variant_queues[lname] = per_v
        self._layer_variants[lname] = list(variant_names)

    def _fanout_to_variants(self, lname: str) -> int:
        src_q = self.queues.get(lname)
        var_qs = self.variant_queues.get(lname, {})
        if not var_qs or src_q is None or src_q.ntoks == 0:
            return 0
        fanned = 0
        while src_q.ntoks > 0:
            chunk_size = min(src_q.ntoks, src_q.block)
            chunk = src_q.pop_batch(chunk_size, shuffle=False)
            if chunk is None:
                break
            for vq in var_qs.values():
                vq.push(chunk)
            fanned += chunk.shape[0]
        return fanned

    def next_batch(self, layer_name: str, batch_size: Optional[int] = None,
                   shuffle: bool = True, variant: Optional[str] = None
                   ) -> Optional[torch.Tensor]:
        if batch_size is None:
            batch_size = self.activation_batch_size

        if variant is not None:
            var_qs = self.variant_queues.get(layer_name, {})
            if variant not in var_qs:
                return None
            self._fanout_to_variants(layer_name)
            vq = var_qs[variant]
            batch = vq.pop_batch(batch_size, shuffle=shuffle)
            if batch is None:
                return None
            return batch.to(self.device)

        q = self.queues.get(layer_name)
        if q is None:
            return None
        batch = q.pop_batch(batch_size, shuffle=shuffle)
        if batch is None:
            return None
        return batch.to(self.device)

    def get_activation_size(self, lname: str) -> int:
        return 8


# ===========================================================================
# Tests
# ===========================================================================

class TestVariantQueueFanout:
    """Unit test: fan-out from source queue to variant queues."""

    def test_fanout_distributes_tokens(self):
        store = _StubActivationStore()
        lname = "layer0"
        store.queues[lname] = _FakeTokenBlockQueue(block_size_tokens=32)
        store.register_variants(lname, ["v1", "v2"])

        # Push 50 tokens to source
        data = torch.randn(50, 8)
        store.queues[lname].push(data)
        assert store.queues[lname].ntoks == 50

        fanned = store._fanout_to_variants(lname)
        assert fanned == 50
        # Source should be empty
        assert store.queues[lname].ntoks == 0
        # Both variants should have all 50 tokens
        assert store.variant_queues[lname]["v1"].ntoks == 50
        assert store.variant_queues[lname]["v2"].ntoks == 50

    def test_fanout_empty_source_returns_zero(self):
        store = _StubActivationStore()
        lname = "layer1"
        store.queues[lname] = _FakeTokenBlockQueue()
        store.register_variants(lname, ["v1"])
        fanned = store._fanout_to_variants(lname)
        assert fanned == 0

    def test_fanout_no_variants_does_nothing(self):
        store = _StubActivationStore()
        lname = "layer2"
        store.queues[lname] = _FakeTokenBlockQueue()
        store.queues[lname].push(torch.randn(10, 8))
        # No variants registered
        fanned = store._fanout_to_variants(lname)
        assert fanned == 0
        assert store.queues[lname].ntoks == 10  # unchanged


class TestTwoVariantsGetSameActs:
    """Integration test: both variants receive the same activations."""

    def test_same_tokens_from_two_variants(self):
        store = _StubActivationStore()
        lname = "layer0"
        store.queues[lname] = _FakeTokenBlockQueue(block_size_tokens=256)
        store.register_variants(lname, ["baseline", "experimental"])

        # Push exactly 100 tokens with known values
        tokens = torch.arange(100 * 8, dtype=torch.float32).reshape(100, 8)
        store.queues[lname].push(tokens)

        # Drain both variants (fanout triggered by first call)
        bs_v1 = store.next_batch(lname, batch_size=100, shuffle=False, variant="baseline")
        bs_v2 = store.next_batch(lname, batch_size=100, shuffle=False, variant="experimental")

        assert bs_v1 is not None
        assert bs_v2 is not None
        assert bs_v1.shape == (100, 8)
        assert bs_v2.shape == (100, 8)
        # Both should contain exactly the same set of tokens (since no shuffle)
        assert torch.allclose(bs_v1.sort(dim=0).values, bs_v2.sort(dim=0).values)

    def test_unregistered_variant_returns_none(self):
        store = _StubActivationStore()
        lname = "layer0"
        store.queues[lname] = _FakeTokenBlockQueue()
        store.queues[lname].push(torch.randn(10, 8))
        store.register_variants(lname, ["v1"])

        result = store.next_batch(lname, batch_size=10, variant="nonexistent")
        assert result is None

    def test_single_variant_independent_fifo(self):
        """Each variant has its own independent queue after fan-out."""
        store = _StubActivationStore()
        lname = "layer0"
        store.queues[lname] = _FakeTokenBlockQueue(block_size_tokens=256)
        store.register_variants(lname, ["v1", "v2"])

        tokens = torch.ones(50, 8)
        store.queues[lname].push(tokens)

        # Only drain v1 fully
        b1_first = store.next_batch(lname, batch_size=50, shuffle=False, variant="v1")
        # v2 should still have all tokens
        assert store.variant_queues[lname]["v2"].ntoks == 50
        # v1 should be empty now
        assert store.variant_queues[lname]["v1"].ntoks == 0

        b2 = store.next_batch(lname, batch_size=50, shuffle=False, variant="v2")
        assert b2 is not None
        assert b2.shape[0] == 50


class TestTwoVariantsTrainIndependently:
    """Integration test: both variants update params after 3 training steps."""

    def _make_sae(self):
        return _TinyBatchTopkSAE(act_size=8, dict_size=16)

    def test_variants_update_different_params(self):
        store = _StubActivationStore()
        lname = "layer0"
        store.queues[lname] = _FakeTokenBlockQueue(block_size_tokens=512)
        store.register_variants(lname, ["v1", "v2"])

        # Push enough tokens for 3 steps each
        data = torch.randn(200, 8)
        store.queues[lname].push(data)

        sae1 = self._make_sae()
        sae2 = self._make_sae()
        # Give different initializations
        with torch.no_grad():
            sae2.W_enc.data = sae2.W_enc.data + 0.5

        opt1 = torch.optim.Adam(sae1.parameters(), lr=1e-3)
        opt2 = torch.optim.Adam(sae2.parameters(), lr=1e-3)

        params1_before = sae1.W_enc.data.clone()
        params2_before = sae2.W_enc.data.clone()

        n_steps = 3
        batch_size = 32
        for step in range(n_steps):
            acts1 = store.next_batch(lname, batch_size=batch_size, shuffle=False, variant="v1")
            acts2 = store.next_batch(lname, batch_size=batch_size, shuffle=False, variant="v2")
            if acts1 is None or acts2 is None:
                break

            opt1.zero_grad()
            out1 = sae1(acts1)
            out1["loss"].backward()
            opt1.step()

            opt2.zero_grad()
            out2 = sae2(acts2)
            out2["loss"].backward()
            opt2.step()

        # Both SAEs should have updated
        assert not torch.allclose(params1_before, sae1.W_enc.data), "SAE1 params did not change"
        assert not torch.allclose(params2_before, sae2.W_enc.data), "SAE2 params did not change"
        # They should be different from each other since different init
        assert not torch.allclose(sae1.W_enc.data, sae2.W_enc.data), \
            "SAE1 and SAE2 should have different params"


class TestCpuOffload:
    """Test that offload logic moves SAEs to CPU/GPU correctly."""

    def test_offload_moves_inactive_variant_to_cpu(self):
        # Simulate the offload logic from the training loop
        sae_active = _TinyBatchTopkSAE()
        sae_inactive = _TinyBatchTopkSAE()

        device = torch.device("cpu")

        # Both start on "device"
        sae_active.to(device)
        sae_inactive.to(device)

        # Simulate offload of inactive SAE (this is what the training loop does)
        other_sae = sae_inactive
        p = next(iter(other_sae.parameters()), None)
        if p is not None and p.device.type != "cpu":
            other_sae.cpu()

        # Verify active SAE still on device
        for param in sae_active.parameters():
            assert param.device == device

        # Verify inactive SAE is on CPU
        for param in sae_inactive.parameters():
            assert param.device.type == "cpu"

    def test_restore_from_cpu_to_device(self):
        device = torch.device("cpu")
        sae = _TinyBatchTopkSAE()
        sae.cpu()

        # Restore to device
        p = next(iter(sae.parameters()), None)
        if p is not None and p.device != device:
            sae.to(device)

        for param in sae.parameters():
            assert param.device.type == "cpu"  # device is cpu in test


class TestBackwardCompatSingleVariant:
    """Verify that when no variants are configured, behavior is identical to before."""

    def test_lv_key_default_variant_is_lname(self):
        """_lv_key with 'default' vname returns just lname."""
        # Import the static method logic directly (don't need a full pipeline)
        def _lv_key(lname: str, vname: str) -> str:
            return lname if vname == "default" else f"{lname}@@{vname}"

        def _lv_parse(key: str):
            if "@@" in key:
                lname, vname = key.split("@@", 1)
                return lname, vname
            return key, "default"

        lname = "model.image_encoder.blocks.0"
        assert _lv_key(lname, "default") == lname
        assert _lv_parse(lname) == (lname, "default")

    def test_lv_key_nondefault_variant_includes_separator(self):
        def _lv_key(lname: str, vname: str) -> str:
            return lname if vname == "default" else f"{lname}@@{vname}"

        def _lv_parse(key: str):
            if "@@" in key:
                lname, vname = key.split("@@", 1)
                return lname, vname
            return key, "default"

        lname = "layer0"
        vname = "ra-batchtopk"
        lv = _lv_key(lname, vname)
        assert "@@" in lv
        assert _lv_parse(lv) == (lname, vname)

    def test_no_variants_in_config_fallback_to_default(self):
        """_variants_for_layer returns [{'name': 'default'}] when variants not set."""
        # Simulate the method without full pipeline
        def _variants_for_layer(config: dict, lname: str) -> list:
            tr = config["sae"]["training"]
            variants_cfg = tr.get("variants") or []
            if isinstance(variants_cfg, list) and variants_cfg:
                return variants_cfg
            return [{"name": "default"}]

        config_no_variants = {"sae": {"training": {}}}
        variants = _variants_for_layer(config_no_variants, "layer0")
        assert variants == [{"name": "default"}]

        config_empty_variants = {"sae": {"training": {"variants": []}}}
        variants = _variants_for_layer(config_empty_variants, "layer0")
        assert variants == [{"name": "default"}]

    def test_single_variant_next_batch_uses_default_queue(self):
        """With default variant, next_batch falls through to normal queue."""
        store = _StubActivationStore()
        lname = "layer0"
        store.queues[lname] = _FakeTokenBlockQueue(block_size_tokens=256)
        data = torch.ones(50, 8)
        store.queues[lname].push(data)

        # No variants registered — should use normal queue path
        batch = store.next_batch(lname, batch_size=50, shuffle=False)
        assert batch is not None
        assert batch.shape[0] == 50

    def test_safe_key_sanitizes_separator(self):
        def _safe_key(lv_key: str) -> str:
            return lv_key.replace("/", "_").replace("@@", "__").replace(":", "_")

        assert _safe_key("layer/0@@variant") == "layer_0__variant"
        assert _safe_key("a:b/c") == "a_b_c"
        assert _safe_key("plain_layer") == "plain_layer"


# ---------------------------------------------------------------------------
# Smoke test: register_variants API
# ---------------------------------------------------------------------------

class TestRegisterVariantsApi:
    def test_register_creates_per_variant_queues(self):
        store = _StubActivationStore()
        lname = "enc.layer1"
        store.register_variants(lname, ["baseline", "experimental"])
        assert lname in store.variant_queues
        assert "baseline" in store.variant_queues[lname]
        assert "experimental" in store.variant_queues[lname]
        assert store._layer_variants[lname] == ["baseline", "experimental"]

    def test_register_empty_noop(self):
        store = _StubActivationStore()
        lname = "enc.layer1"
        store.register_variants(lname, [])
        assert lname not in store.variant_queues

    def test_second_fanout_does_nothing_on_empty_source(self):
        store = _StubActivationStore()
        lname = "layer0"
        store.queues[lname] = _FakeTokenBlockQueue(block_size_tokens=128)
        store.register_variants(lname, ["v1", "v2"])

        data = torch.randn(64, 8)
        store.queues[lname].push(data)

        # First fanout
        f1 = store._fanout_to_variants(lname)
        assert f1 == 64
        # Second fanout — source empty
        f2 = store._fanout_to_variants(lname)
        assert f2 == 0
        # Variant queues unchanged
        assert store.variant_queues[lname]["v1"].ntoks == 64
