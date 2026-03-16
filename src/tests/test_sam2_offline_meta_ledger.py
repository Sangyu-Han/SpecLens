from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from src.packs.sam2.offline.offline_meta_ledger import OfflineMetaParquetLedger


class Sam2OfflineMetaParquetLedgerTest(unittest.TestCase):
    def test_find_prompts_resolves_recurrent_frame_idx(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ledger = OfflineMetaParquetLedger(tmpdir, part_modulus=128)

            sample_rows = [
                {
                    "sample_id": 1001,
                    "dict_key": "train",
                    "name": "vid_a",
                    "seq_full": [0, 1, 2, 3],
                    "image_h": 1024,
                    "image_w": 1024,
                    "epoch_idx": 0,
                    "run_seed": 0,
                    "batch_sig": "sig",
                    "prompt_sets": [{"frame_idx": 0, "prompt_id": 777}],
                    "extra_json": "",
                }
            ]
            prompt_rows = [
                {
                    "sample_id": 1001,
                    "prompt_id": 777,
                    "frame_idx": 0,
                    "uid": 42,
                    "points_x": [10.0],
                    "points_y": [12.0],
                    "point_labels": [1],
                    "point_steps": [0],
                    "version": 1,
                }
            ]

            ledger.write_samples(
                sample_rows,
                dict_key="train",
                epoch_idx=0,
                skip_if_exists=False,
                use_claims=False,
            )
            ledger.write_prompts(
                prompt_rows,
                dict_key="train",
                epoch_idx=0,
                skip_if_exists=False,
                use_claims=False,
            )

            exact = ledger.find_prompts(1001, 777, 0)
            self.assertEqual(exact.num_rows, 1)

            # Recurrent provenance frame (e.g. t=2) should resolve to prompt frame (t=0).
            resolved = ledger.find_prompts(1001, 777, 2)
            self.assertEqual(resolved.num_rows, 1)
            self.assertEqual(int(resolved.column("frame_idx")[0].as_py()), 0)

            resolved_uid = ledger.find_prompt_for_uid(1001, 777, 3, 42)
            self.assertEqual(resolved_uid.num_rows, 1)
            self.assertEqual(int(resolved_uid.column("frame_idx")[0].as_py()), 0)

    def test_find_prompts_falls_back_without_prompt_sets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ledger = OfflineMetaParquetLedger(tmpdir, part_modulus=128)

            sample_rows = [
                {
                    "sample_id": 2002,
                    "dict_key": "train",
                    "name": "vid_b",
                    "seq_full": [0, 1, 2],
                    "image_h": 512,
                    "image_w": 512,
                    "epoch_idx": 0,
                    "run_seed": 0,
                    "batch_sig": "sig2",
                    "prompt_sets": [],
                    "extra_json": "",
                }
            ]
            prompt_rows = [
                {
                    "sample_id": 2002,
                    "prompt_id": 888,
                    "frame_idx": 1,
                    "uid": 77,
                    "points_x": [3.0],
                    "points_y": [4.0],
                    "point_labels": [1],
                    "point_steps": [0],
                    "version": 1,
                }
            ]

            ledger.write_samples(
                sample_rows,
                dict_key="train",
                epoch_idx=0,
                skip_if_exists=False,
                use_claims=False,
            )
            ledger.write_prompts(
                prompt_rows,
                dict_key="train",
                epoch_idx=0,
                skip_if_exists=False,
                use_claims=False,
            )

            resolved = ledger.find_prompts(2002, 888, 0)
            self.assertEqual(resolved.num_rows, 1)
            self.assertEqual(int(resolved.column("frame_idx")[0].as_py()), 1)


if __name__ == "__main__":
    unittest.main()
