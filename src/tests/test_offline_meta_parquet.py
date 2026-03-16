from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from src.packs.clip.offline.offline_meta_parquet import OfflineMetaParquetLedger


class OfflineMetaParquetLedgerTest(unittest.TestCase):
    def test_write_and_lookup_roundtrip(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ledger = OfflineMetaParquetLedger(
                tmpdir, part_modulus=4, run_id="runA", partition_by_run_id=True
            )
            batch = {
                "sample_id": [1, 6],
                "path": ["a.png", "b.png"],
                "label": [5, 7],
                "meta_json": [{"foo": 1}, "bar"],
            }
            written = ledger.write_from_batch(batch)
            self.assertEqual(written, 2)

            resolved = ledger.lookup([1, 6, 999])
            self.assertEqual(resolved.get(1), "a.png")
            self.assertEqual(resolved.get(6), "b.png")
            self.assertNotIn(999, resolved)

            dset = ledger.as_dataset()
            tbl = dset.to_table(columns=["sample_id", "part", "run_id"])
            self.assertEqual(tbl.num_rows, 2)
            self.assertEqual(set(tbl.column("part").to_pylist()), {1, 2})
            self.assertEqual(set(tbl.column("run_id").to_pylist()), {"runA"})

            # run_id filter should prune results when it does not match
            other = OfflineMetaParquetLedger(
                tmpdir, part_modulus=4, run_id="other", partition_by_run_id=True
            )
            self.assertEqual(other.lookup([1, 6]), {})


if __name__ == "__main__":
    unittest.main()
