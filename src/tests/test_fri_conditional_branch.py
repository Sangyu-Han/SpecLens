import unittest

import numpy as np

from src.core.attribution.fri import (
    BandReplacementConfig,
    ConditionalRandomBranchConfig,
    PrefixShapeGateConfig,
    PrefixShapeProbe,
    band_replacement_order,
    band_replacement_scores,
    conditional_deletion_effects,
    conditional_effects,
    fill_band_order,
    prefix_shape_decision,
    probe_prefix_shape,
    rank_scores_from_order,
    rank_values_like,
    rerank_band_order,
    run_conditional_random_branch,
    scores_from_order_like,
    stable_case_seed,
)


class FriConditionalBranchTest(unittest.TestCase):
    def test_conditional_deletion_effects_ignore_additive_constant(self):
        design = np.asarray(
            [
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
        deleted_values = np.asarray([0.20, 0.35, 0.50, 0.70], dtype=np.float64)
        np.testing.assert_allclose(
            conditional_deletion_effects(design, deleted_values),
            conditional_effects(design, 10.0 - deleted_values),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_stable_case_seed_is_key_stable(self):
        self.assertEqual(stable_case_seed(123, "00032327"), stable_case_seed(123, "00032327"))
        self.assertNotEqual(stable_case_seed(123, "00032327"), stable_case_seed(123, "00003508"))

    def test_rank_scores_from_order_matches_legacy_shape(self):
        scores = rank_scores_from_order([2, 0, 1, 3], n_tokens=4)
        np.testing.assert_array_equal(scores, np.asarray([0.75, 0.5, 1.0, 0.25], dtype=np.float32))

    def test_band_fill_and_rerank_orders_are_rank_only(self):
        base_order = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        cand_order = np.asarray([5, 4, 3, 2, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(
            fill_band_order(base_order, cand_order, keep=2, end=4),
            np.asarray([0, 1, 5, 4, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            rerank_band_order(base_order, cand_order, start=2, end=5),
            np.asarray([0, 1, 4, 3, 2, 5], dtype=np.int64),
        )

    def test_band_replacement_scores_preserve_base_mass_profile(self):
        base_scores = np.asarray([0.6, 0.4, 0.2, 0.1, 0.0, 0.0], dtype=np.float32)
        cand_scores = np.asarray([0.0, 0.0, 0.1, 0.2, 0.4, 0.6], dtype=np.float32)
        out = band_replacement_scores(
            base_scores=base_scores,
            candidate_scores=cand_scores,
            selected=True,
            config=BandReplacementConfig(kind="fill", start=2, end=4),
        )
        expected_order = band_replacement_order(
            base_scores=base_scores,
            candidate_scores=cand_scores,
            selected=True,
            config=BandReplacementConfig(kind="fill", start=2, end=4),
        )
        np.testing.assert_array_equal(out, scores_from_order_like(expected_order, base_scores))
        np.testing.assert_allclose(
            np.sort(out)[::-1],
            rank_values_like(base_scores).astype(np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_prefix_shape_probe_and_decision(self):
        scores = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)

        def value_fn(masks: np.ndarray) -> np.ndarray:
            return masks.mean(axis=1)

        probe = probe_prefix_shape(value_fn=value_fn, scores=scores, budgets=(1, 3))
        self.assertEqual(probe.budgets, (0, 1, 3))
        np.testing.assert_allclose(probe.ratios, (1.0, 0.75, 0.25), rtol=1e-6)

        late_collapse = PrefixShapeProbe(
            budgets=(0, 32, 64),
            values=(1.0, 0.98, 0.05),
            ratios=(1.0, 0.98, 0.05),
            full_value=1.0,
            auc_ratio=0.5,
            last_ratio=0.05,
            max_ratio=1.0,
            argmax_budget=0,
            argmax_frac=0.0,
        )
        selected = prefix_shape_decision(late_collapse, PrefixShapeGateConfig())
        self.assertTrue(selected.selected)

        early_collapse = PrefixShapeProbe(
            budgets=(0, 32, 64),
            values=(1.0, 0.50, 0.05),
            ratios=(1.0, 0.50, 0.05),
            full_value=1.0,
            auc_ratio=0.5,
            last_ratio=0.05,
            max_ratio=1.0,
            argmax_budget=0,
            argmax_frac=0.0,
        )
        rejected = prefix_shape_decision(early_collapse, PrefixShapeGateConfig())
        self.assertFalse(rejected.selected)

    def test_conditional_random_branch_is_deterministic(self):
        base_scores = np.linspace(1.0, 0.0, 12, endpoint=False, dtype=np.float32)
        seed_scores = np.roll(base_scores, 2)
        weights = np.linspace(0.1, 1.2, 12, dtype=np.float64)

        def value_fn(masks: np.ndarray) -> np.ndarray:
            deleted = 1.0 - np.asarray(masks, dtype=np.float64)
            return 1.0 - deleted @ weights / weights.sum()

        cfg = ConditionalRandomBranchConfig(
            head_k=2,
            pool_k=8,
            prefix_k=5,
            group_size=3,
            n_groups=6,
            mix_alpha=0.25,
            seed=1234,
        )
        first = run_conditional_random_branch(
            value_fn=value_fn,
            base_scores=base_scores,
            seed_scores=seed_scores,
            case_key="case-a",
            config=cfg,
        )
        second = run_conditional_random_branch(
            value_fn=value_fn,
            base_scores=base_scores,
            seed_scores=seed_scores,
            case_key="case-a",
            config=cfg,
        )
        np.testing.assert_array_equal(first.scores, second.scores)
        np.testing.assert_array_equal(first.order, second.order)
        self.assertEqual(first.design.shape, (6, len(first.pool)))
        self.assertEqual(first.values.shape, (6,))


if __name__ == "__main__":
    unittest.main()
