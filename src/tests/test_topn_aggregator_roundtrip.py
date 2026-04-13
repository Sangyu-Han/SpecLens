from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import torch

from src.core.indexing.decile_aggregator import RunFingerprint
from src.core.indexing.decile_parquet_ledger import DECILES_SCHEMA, DecileParquetLedger
from src.core.indexing.topn_aggregator import TopNAggregator


class TopNAggregatorRoundTripTest(unittest.TestCase):
    def test_repeated_batches_keep_accumulating_until_heap_is_full(self) -> None:
        acts = torch.tensor([[0.75, 0.0, 0.0]], dtype=torch.float32)
        prov = torch.tensor([[101, 0, 4, 7, 9001, 42]], dtype=torch.int64)
        prov_cols = ("sample_id", "frame_idx", "y", "x", "prompt_id", "uid")

        with TemporaryDirectory() as tmpdir:
            ledger = DecileParquetLedger(tmpdir, M_part=4)
            agg = TopNAggregator(
                dict_size=3,
                top_n=4,
                layer_name="layer0",
                fp=RunFingerprint(
                    model_name="test",
                    model_yaml="",
                    model_ckpt="",
                    model_ckpt_sha="",
                    sae_ckpt="",
                    sae_ckpt_sha="",
                    dataset_name="synthetic",
                    run_id="repeat-fill",
                ),
                ledger=ledger,
                prov_cols=prov_cols,
                track_frequency=True,
                rank=0,
            )

            for _ in range(3):
                agg.update(acts, prov, stride_step=2, batch_max=acts.max(dim=0).values)

            wrote = agg.finalize_and_write(progress_cb=None)
            self.assertEqual(wrote, 3)

            table = ledger.topn_for(layer="layer0", unit=0, decile=0, n=10)
            self.assertEqual(table.num_rows, 3)
            self.assertEqual(table.column("rank_in_decile").to_pylist(), [0, 1, 2])
            self.assertEqual(table.column("score").to_pylist(), [0.75, 0.75, 0.75])
            self.assertEqual(table.column("sample_id").to_pylist(), [101, 101, 101])

    def test_topn_roundtrip_preserves_activation_and_schema(self) -> None:
        acts_batches = [
            torch.tensor(
                [
                    [0.1, 0.0, 0.5],
                    [0.4, 0.3, 0.0],
                    [0.0, 0.8, 0.2],
                    [0.7, 0.0, 0.6],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [0.9, 0.2, 0.0],
                    [0.0, 0.1, 0.4],
                    [0.5, 0.0, 0.3],
                    [0.2, 0.9, 0.8],
                ],
                dtype=torch.float32,
            ),
        ]
        prov_batches = [
            torch.tensor(
                [
                    [10, 0, 0, 0, 1000, 2000],
                    [11, 0, 0, 1, 1001, 2001],
                    [12, 0, 1, 0, 1002, 2002],
                    [13, 0, 1, 1, 1003, 2003],
                ],
                dtype=torch.int64,
            ),
            torch.tensor(
                [
                    [14, 1, 2, 0, 1004, 2004],
                    [15, 1, 2, 1, 1005, 2005],
                    [16, 1, 3, 0, 1006, 2006],
                    [17, 1, 3, 1, 1007, 2007],
                ],
                dtype=torch.int64,
            ),
        ]
        prov_cols = ("sample_id", "frame_idx", "y", "x", "prompt_id", "uid")

        expected_by_feature: dict[int, list[dict[str, int | float]]] = {0: [], 1: [], 2: []}
        activation_lookup: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for acts, prov in zip(acts_batches, prov_batches):
            for token_idx in range(acts.shape[0]):
                prov_tuple = tuple(int(v) for v in prov[token_idx].tolist())
                for feature_idx in torch.nonzero(acts[token_idx], as_tuple=True)[0].tolist():
                    score = float(acts[token_idx, feature_idx].item())
                    activation_lookup[(prov_tuple[0], int(feature_idx))] = (score, prov_tuple)
                    expected_by_feature[int(feature_idx)].append(
                        {
                            "sample_id": prov_tuple[0],
                            "frame_idx": prov_tuple[1],
                            "y": prov_tuple[2],
                            "x": prov_tuple[3],
                            "prompt_id": prov_tuple[4],
                            "uid": prov_tuple[5],
                            "score": score,
                        }
                    )

        for feature_idx, rows in expected_by_feature.items():
            rows.sort(key=lambda row: row["score"], reverse=True)
            expected_by_feature[feature_idx] = rows[:2]

        with TemporaryDirectory() as tmpdir:
            ledger = DecileParquetLedger(tmpdir, M_part=4)
            agg = TopNAggregator(
                dict_size=3,
                top_n=2,
                layer_name="layer0",
                fp=RunFingerprint(
                    model_name="test",
                    model_yaml="",
                    model_ckpt="",
                    model_ckpt_sha="",
                    sae_ckpt="",
                    sae_ckpt_sha="",
                    dataset_name="synthetic",
                    run_id="roundtrip",
                ),
                ledger=ledger,
                prov_cols=prov_cols,
                track_frequency=True,
                rank=0,
            )

            for acts, prov in zip(acts_batches, prov_batches):
                agg.update(acts, prov, stride_step=3, batch_max=acts.max(dim=0).values)

            wrote = agg.finalize_and_write(progress_cb=None)
            self.assertEqual(wrote, 6)

            for feature_idx in range(3):
                table = ledger.topn_for(layer="layer0", unit=feature_idx, decile=0, n=5)
                self.assertEqual(table.num_rows, 2)
                observed = []
                for row_idx in range(table.num_rows):
                    observed.append(
                        {
                            "sample_id": int(table.column("sample_id")[row_idx].as_py()),
                            "frame_idx": int(table.column("frame_idx")[row_idx].as_py()),
                            "y": int(table.column("y")[row_idx].as_py()),
                            "x": int(table.column("x")[row_idx].as_py()),
                            "prompt_id": int(table.column("prompt_id")[row_idx].as_py()),
                            "uid": int(table.column("uid")[row_idx].as_py()),
                            "score": float(table.column("score")[row_idx].as_py()),
                            "rank_in_decile": int(
                                table.column("rank_in_decile")[row_idx].as_py()
                            ),
                            "stride_step": int(table.column("stride_step")[row_idx].as_py()),
                        }
                    )

                expected_rows = expected_by_feature[feature_idx]
                for rank_in_decile, (expected, actual) in enumerate(
                    zip(expected_rows, observed, strict=True)
                ):
                    self.assertEqual(actual["rank_in_decile"], rank_in_decile)
                    self.assertEqual(actual["stride_step"], 3)
                    self.assertEqual(actual["sample_id"], expected["sample_id"])
                    self.assertEqual(actual["frame_idx"], expected["frame_idx"])
                    self.assertEqual(actual["y"], expected["y"])
                    self.assertEqual(actual["x"], expected["x"])
                    self.assertEqual(actual["prompt_id"], expected["prompt_id"])
                    self.assertEqual(actual["uid"], expected["uid"])
                    self.assertAlmostEqual(actual["score"], expected["score"], places=6)

                    restored_score, restored_prov = activation_lookup[
                        (actual["sample_id"], feature_idx)
                    ]
                    self.assertAlmostEqual(actual["score"], restored_score, places=6)
                    self.assertEqual(
                        (
                            actual["sample_id"],
                            actual["frame_idx"],
                            actual["y"],
                            actual["x"],
                            actual["prompt_id"],
                            actual["uid"],
                        ),
                        restored_prov,
                    )

            dset = ledger.as_dataset()
            self.assertEqual(
                dset.schema.names,
                list(DECILES_SCHEMA.names) + ["layer_part", "decile_part", "part"],
            )


if __name__ == "__main__":
    unittest.main()
