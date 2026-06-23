import unittest

import numpy as np

from src.core.attribution.fri import (
    LocalProbeConfig,
    deletion_drops,
    local_probe_readout_from_values,
    make_local_probe_masks,
    score_local_probe_readouts,
    select_local_probe_readout,
)


class FriLocalProbeTest(unittest.TestCase):
    def test_deletion_drops_relative_and_absolute(self):
        vals = np.asarray([0.8, 1.1, np.nan, np.inf], dtype=np.float32)
        np.testing.assert_allclose(
            deletion_drops(full_value=1.0, deleted_values=vals, mode="relative"),
            np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            deletion_drops(full_value=2.0, deleted_values=np.asarray([1.5, 2.5]), mode="absolute"),
            np.asarray([0.5, 0.0], dtype=np.float32),
            rtol=1e-6,
        )

    def test_make_local_probe_masks_deletes_local_groups(self):
        cfg = LocalProbeConfig(probe_calls=2, radii=(0, 1), coverage_radius=0)
        prior = np.arange(16, dtype=np.float32)
        masks = make_local_probe_masks(prior, n_patches=16, grid_size=4, config=cfg)
        self.assertEqual(masks.masks.shape, (2, 16))
        self.assertEqual(masks.centers[0], 15)
        self.assertEqual(masks.groups[0].tolist(), [15])
        self.assertTrue(np.all(masks.masks[0, masks.groups[0]] == 0.0))
        self.assertTrue(np.all(masks.masks[1, masks.groups[1]] == 0.0))

    def test_sparse_content_selector_branches(self):
        cfg = LocalProbeConfig(low_drop=0.015, mid_drop=0.035, sparse_drop_mean=0.004)
        local = {
            "local_gate": np.full(4, 1.0, dtype=np.float32),
            "local_gate_g2": np.full(4, 2.0, dtype=np.float32),
            "local_x_prior_content_g2": np.full(4, 3.0, dtype=np.float32),
        }
        fallback = {
            "final": np.full(4, 4.0, dtype=np.float32),
            "grad_rank": np.full(4, 5.0, dtype=np.float32),
        }

        low = select_local_probe_readout(
            readout="adaptive_sparse_content",
            local_scores=local,
            fallback_scores=fallback,
            diagnostics={"drop_max": 0.01, "drop_mean": 0.001},
            config=cfg,
            low_fallback="final",
            mid_fallback="grad_rank",
        )
        np.testing.assert_allclose(low, np.full(4, 4.0, dtype=np.float32))

        sparse_mid = select_local_probe_readout(
            readout="adaptive_sparse_content",
            local_scores=local,
            fallback_scores=fallback,
            diagnostics={"drop_max": 0.02, "drop_mean": 0.003},
            config=cfg,
            low_fallback="final",
            mid_fallback="grad_rank",
        )
        np.testing.assert_allclose(sparse_mid, np.full(4, 3.0, dtype=np.float32))

        dense_mid = select_local_probe_readout(
            readout="adaptive_sparse_content",
            local_scores=local,
            fallback_scores=fallback,
            diagnostics={"drop_max": 0.02, "drop_mean": 0.006},
            config=cfg,
            low_fallback="final",
            mid_fallback="grad_rank",
        )
        np.testing.assert_allclose(dense_mid, np.full(4, 5.0, dtype=np.float32))

        reliable = select_local_probe_readout(
            readout="adaptive_sparse_content",
            local_scores=local,
            fallback_scores=fallback,
            diagnostics={"drop_max": 0.10, "drop_mean": 0.02},
            config=cfg,
            low_fallback="final",
            mid_fallback="grad_rank",
        )
        np.testing.assert_allclose(reliable, np.full(4, 2.0, dtype=np.float32))

    def test_score_local_probe_readouts_exposes_content_variant(self):
        out = score_local_probe_readouts(
            prior=np.asarray([1.0, 0.5, 0.2, 0.1], dtype=np.float32),
            drops=np.asarray([0.2, 0.1], dtype=np.float32),
            groups=(np.asarray([0, 1]), np.asarray([2, 3])),
            content_scores=np.asarray([0.1, 0.2, 1.0, 0.5], dtype=np.float32),
        )
        self.assertIn("local_x_prior_content_g2", out.scores)
        self.assertIn("drop_max", out.diagnostics)
        self.assertEqual(out.scores["local_x_prior_content_g2"].shape, (4,))

    def test_local_probe_readout_from_values_runs_full_pipeline(self):
        cfg = LocalProbeConfig(low_drop=0.015, mid_drop=0.035, sparse_drop_mean=0.03)
        selected, readouts = local_probe_readout_from_values(
            prior=np.asarray([1.0, 0.8, 0.1, 0.05], dtype=np.float32),
            full_value=1.0,
            deleted_values=np.asarray([0.98, 0.975], dtype=np.float32),
            groups=(np.asarray([0, 1]), np.asarray([2, 3])),
            fallback_scores={
                "final": np.zeros(4, dtype=np.float32),
                "grad_rank": np.ones(4, dtype=np.float32),
            },
            readout="adaptive_sparse_content",
            config=cfg,
            drop_mode="relative",
            content_scores=np.asarray([0.1, 0.2, 1.0, 0.5], dtype=np.float32),
            low_fallback="final",
            mid_fallback="grad_rank",
        )
        np.testing.assert_allclose(selected, readouts.scores["local_x_prior_content_g2"])


if __name__ == "__main__":
    unittest.main()
