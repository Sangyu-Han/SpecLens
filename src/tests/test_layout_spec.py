"""Unit tests for LayoutSpec provenance engine."""
from __future__ import annotations

import unittest

import torch

from src.core.base.layout import (
    DimRole,
    LayoutSpec,
    build_provenance_from_layout,
    infer_layout,
    parse_layout_spec,
)


class TestDimRole(unittest.TestCase):
    def test_all_roles_exist(self):
        for name in ("BATCH", "TIME", "LANE", "HEIGHT", "WIDTH", "TOKEN", "FEATURE"):
            self.assertIsNotNone(getattr(DimRole, name))


class TestLayoutSpec(unittest.TestCase):
    def test_feature_axis_nchw(self):
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
        self.assertEqual(spec.feature_axis, 1)

    def test_feature_axis_last(self):
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.TOKEN, DimRole.FEATURE))
        self.assertEqual(spec.feature_axis, 2)

    def test_prefix_roles(self):
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
        self.assertEqual(spec.prefix_roles, (DimRole.BATCH, DimRole.HEIGHT, DimRole.WIDTH))

    def test_provenance_columns_spatial(self):
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
        self.assertEqual(spec.provenance_columns(), ("sample_id", "y", "x"))

    def test_provenance_columns_token(self):
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.TOKEN, DimRole.FEATURE))
        # TOKEN expands to (y, x)
        self.assertEqual(spec.provenance_columns(), ("sample_id", "y", "x"))

    def test_provenance_columns_with_time(self):
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.TIME, DimRole.TOKEN, DimRole.FEATURE))
        self.assertEqual(spec.provenance_columns(), ("sample_id", "frame_idx", "y", "x"))

    def test_provenance_columns_with_lane(self):
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.LANE, DimRole.FEATURE))
        self.assertEqual(spec.provenance_columns(), ("sample_id", "uid"))

    def test_col_name_override(self):
        spec = LayoutSpec(
            dims=(DimRole.BATCH, DimRole.TIME, DimRole.FEATURE),
            col_name_overrides={DimRole.TIME: "t"},
        )
        self.assertEqual(spec.provenance_columns(), ("sample_id", "t"))


class TestInferLayout(unittest.TestCase):
    def test_2d(self):
        spec = infer_layout(torch.randn(4, 64))
        self.assertIsNotNone(spec)
        self.assertEqual(spec.dims, (DimRole.BATCH, DimRole.FEATURE))

    def test_3d(self):
        spec = infer_layout(torch.randn(2, 10, 64))
        self.assertIsNotNone(spec)
        self.assertEqual(spec.dims, (DimRole.BATCH, DimRole.TOKEN, DimRole.FEATURE))

    def test_4d(self):
        spec = infer_layout(torch.randn(2, 64, 8, 8))
        self.assertIsNotNone(spec)
        self.assertEqual(spec.dims, (DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))

    def test_5d_returns_none(self):
        spec = infer_layout(torch.randn(2, 3, 64, 4, 4))
        self.assertIsNone(spec)


class TestParseLayoutSpec(unittest.TestCase):
    def test_basic(self):
        spec = parse_layout_spec(["batch", "feature", "h", "w"])
        self.assertEqual(spec.dims, (DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))

    def test_aliases(self):
        spec = parse_layout_spec(["b", "c", "y", "x"])
        self.assertEqual(spec.dims, (DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))

    def test_5d(self):
        spec = parse_layout_spec(["batch", "time", "feature", "h", "w"])
        self.assertEqual(spec.dims, (DimRole.BATCH, DimRole.TIME, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_layout_spec(["batch", "unknown_role"])


class TestBuildProvenanceFromLayout(unittest.TestCase):
    def test_4d_nchw(self):
        """4D (B=2, C=64, H=4, W=4) -> N=32, cols=(sample_id, y, x)."""
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
        tensor = torch.randn(2, 64, 4, 4)
        sids = torch.tensor([100, 200])
        prov = build_provenance_from_layout(spec, tensor, sample_ids=sids)

        self.assertEqual(prov.shape, (32, 3))
        # First 16 tokens belong to batch 0 (sample_id=100)
        self.assertTrue((prov[:16, 0] == 100).all())
        # Next 16 tokens belong to batch 1 (sample_id=200)
        self.assertTrue((prov[16:, 0] == 200).all())
        # First token: y=0, x=0
        self.assertEqual(prov[0, 1].item(), 0)  # y
        self.assertEqual(prov[0, 2].item(), 0)  # x
        # Token at (y=1, x=2): index = 1*4+2 = 6
        self.assertEqual(prov[6, 1].item(), 1)  # y
        self.assertEqual(prov[6, 2].item(), 2)  # x

    def test_3d_token(self):
        """3D (B=2, L=5, C=64) -> N=10, cols=(sample_id, y, x) with y=-1."""
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.TOKEN, DimRole.FEATURE))
        tensor = torch.randn(2, 5, 64)
        sids = torch.tensor([10, 20])
        prov = build_provenance_from_layout(spec, tensor, sample_ids=sids)

        self.assertEqual(prov.shape, (10, 3))
        # All y values should be -1 for TOKEN
        self.assertTrue((prov[:, 1] == -1).all())
        # x values should be token indices 0..4 repeated
        self.assertEqual(prov[0, 2].item(), 0)
        self.assertEqual(prov[4, 2].item(), 4)
        self.assertEqual(prov[5, 2].item(), 0)  # second batch

    def test_2d_batch_only(self):
        """2D (B=3, C=64) -> N=3, cols=(sample_id,)."""
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE))
        tensor = torch.randn(3, 64)
        sids = torch.tensor([1, 2, 3])
        prov = build_provenance_from_layout(spec, tensor, sample_ids=sids)

        self.assertEqual(prov.shape, (3, 1))
        self.assertEqual(prov[:, 0].tolist(), [1, 2, 3])

    def test_5d_with_time(self):
        """5D (B=2, T=3, C=64, H=2, W=2) -> N=24, cols=(sample_id, frame_idx, y, x)."""
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.TIME, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
        tensor = torch.randn(2, 3, 64, 2, 2)
        sids = torch.tensor([100, 200])
        prov = build_provenance_from_layout(spec, tensor, sample_ids=sids)

        self.assertEqual(prov.shape, (24, 4))
        # First 12 tokens = batch 0 (sample_id=100)
        self.assertTrue((prov[:12, 0] == 100).all())
        # First 4 tokens = batch 0, time 0
        self.assertTrue((prov[:4, 1] == 0).all())
        # Next 4 tokens = batch 0, time 1
        self.assertTrue((prov[4:8, 1] == 1).all())

    def test_enrich_fn(self):
        """enrich_fn should be able to add/modify columns."""
        def my_enrich(columns, dim_indices):
            if DimRole.LANE in dim_indices:
                columns["uid"] = dim_indices[DimRole.LANE] * 1000

        spec = LayoutSpec(
            dims=(DimRole.BATCH, DimRole.LANE, DimRole.FEATURE),
            enrich_fn=my_enrich,
        )
        tensor = torch.randn(2, 5, 64)
        sids = torch.tensor([10, 20])
        prov = build_provenance_from_layout(spec, tensor, sample_ids=sids)

        self.assertEqual(prov.shape, (10, 2))  # (sample_id, uid)
        # lane=2 for batch=0 -> uid=2000
        self.assertEqual(prov[2, 1].item(), 2000)

    def test_ndim_mismatch_raises(self):
        """Mismatched spec dims and tensor ndim should raise ValueError."""
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
        tensor = torch.randn(2, 3, 64)  # 3D tensor with 4D spec
        with self.assertRaises(ValueError):
            build_provenance_from_layout(spec, tensor)

    def test_sample_ids_none(self):
        """Without sample_ids, should default to arange."""
        spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
        tensor = torch.randn(3, 64, 2, 2)
        prov = build_provenance_from_layout(spec, tensor, sample_ids=None)

        self.assertEqual(prov.shape, (12, 3))
        # Batch 0 → sample_id=0, Batch 1 → sample_id=1, etc.
        self.assertTrue((prov[:4, 0] == 0).all())
        self.assertTrue((prov[4:8, 0] == 1).all())
        self.assertTrue((prov[8:12, 0] == 2).all())


class TestAdapterIntegration(unittest.TestCase):
    def test_base_class_generic_provenance(self):
        """Base ModelAdapter should produce provenance via LayoutSpec."""
        from src.core.base.adapters import ModelAdapter

        class DummyAdapter(ModelAdapter):
            def preprocess_input(self, raw_batch):
                return raw_batch
            def forward(self, batch):
                pass

        adapter = DummyAdapter()
        adapter._current_sample_ids = torch.tensor([100, 200])

        raw = torch.randn(2, 64, 4, 4)
        flat = torch.randn(32, 64)
        prov = adapter.build_token_provenance(
            act_name="test", raw_output=raw, flattened_tokens=flat,
        )
        self.assertEqual(prov.shape, (32, 3))
        self.assertTrue((prov[:16, 0] == 100).all())
        self.assertTrue((prov[16:, 0] == 200).all())

    def test_fallback_for_non_tensor(self):
        """Non-tensor input should produce zero provenance."""
        from src.core.base.adapters import ModelAdapter

        class DummyAdapter(ModelAdapter):
            def preprocess_input(self, raw_batch):
                return raw_batch
            def forward(self, batch):
                pass

        adapter = DummyAdapter()
        prov = adapter.build_token_provenance(
            act_name="x", raw_output="not_a_tensor",
            flattened_tokens=torch.randn(5, 32),
        )
        self.assertEqual(prov.shape, (5, 3))
        self.assertTrue((prov == 0).all())


class TestHookHelperLayoutSpec(unittest.TestCase):
    def test_5d_with_spec(self):
        from src.core.sae.activation_stores.hook_helper import flatten_tensor_for_sae
        spec = parse_layout_spec(["batch", "time", "feature", "h", "w"])
        t = torch.randn(2, 3, 64, 4, 4)
        flat, meta = flatten_tensor_for_sae(t, layout_spec=spec)
        self.assertIsNotNone(flat)
        self.assertEqual(flat.shape, (96, 64))
        self.assertEqual(meta["permute"], (0, 1, 3, 4, 2))

    def test_5d_without_spec_returns_none(self):
        from src.core.sae.activation_stores.hook_helper import flatten_tensor_for_sae
        t = torch.randn(2, 3, 64, 4, 4)
        flat, meta = flatten_tensor_for_sae(t)
        self.assertIsNone(flat)

    def test_4d_unchanged(self):
        from src.core.sae.activation_stores.hook_helper import flatten_tensor_for_sae
        t = torch.randn(2, 64, 4, 4)
        flat, meta = flatten_tensor_for_sae(t)
        self.assertEqual(flat.shape, (32, 64))
        self.assertEqual(meta["permute"], (0, 2, 3, 1))


if __name__ == "__main__":
    unittest.main()
