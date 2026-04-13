from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.indexing.index_runner import (
    _cp_paths,
    _find_latest_sae_ckpt_path,
    _lv_key,
    _lv_parse,
    _variant_names_for_layer,
    _variant_out_dir,
)
from src.core.indexing.registry_utils import sanitize_layer_name


class MultiVariantIndexingHelpersTest(unittest.TestCase):
    def test_auto_discovers_default_and_named_variants(self) -> None:
        cfg = {"indexing": {}, "sae": {}}
        layer = "model.image_encoder.trunk@3"

        with TemporaryDirectory() as tmpdir:
            sae_root = Path(tmpdir)
            layer_dir = sae_root / sanitize_layer_name(layer)
            layer_dir.mkdir(parents=True)
            (layer_dir / "step_0000001_tokens_10.pt").touch()
            (layer_dir / "step_0000002_tokens_20.pt").touch()

            variant_a = layer_dir / "variant-a"
            variant_a.mkdir()
            (variant_a / "step_0000003_tokens_30.pt").touch()

            variant_b = layer_dir / "variant-b"
            variant_b.mkdir()
            (variant_b / "step_0000004_tokens_40.pt").touch()

            self.assertEqual(
                _variant_names_for_layer(cfg, sae_root, layer),
                ["default", "variant-a", "variant-b"],
            )
            self.assertEqual(
                _find_latest_sae_ckpt_path(sae_root, layer, "default").name,
                "step_0000002_tokens_20.pt",
            )
            self.assertEqual(
                _find_latest_sae_ckpt_path(sae_root, layer, "variant-a").name,
                "step_0000003_tokens_30.pt",
            )

    def test_configured_variants_override_auto_discovery(self) -> None:
        cfg = {
            "indexing": {},
            "sae": {
                "training": {
                    "variants": [
                        {"name": "batchtopk"},
                        {"name": "ra-all"},
                    ]
                }
            },
        }

        with TemporaryDirectory() as tmpdir:
            sae_root = Path(tmpdir)
            self.assertEqual(
                _variant_names_for_layer(cfg, sae_root, "layer0"),
                ["batchtopk", "ra-all"],
            )

    def test_variant_paths_preserve_single_variant_compat(self) -> None:
        base_out = Path("/tmp/index-out")
        cp_dir = Path("/tmp/index-state")
        layer = "model.memory_attention.layers.3"

        self.assertEqual(_lv_parse(_lv_key(layer, "default")), (layer, "default"))
        self.assertEqual(_lv_parse(_lv_key(layer, "ra-all")), (layer, "ra-all"))

        self.assertEqual(_variant_out_dir(base_out, "default", False), base_out)
        self.assertEqual(
            _variant_out_dir(base_out, "ra-all", True),
            base_out / "variants" / "ra-all",
        )

        self.assertEqual(
            _cp_paths(cp_dir, "index", layer),
            cp_dir / f"index.{sanitize_layer_name(layer)}.pt",
        )
        self.assertEqual(
            _cp_paths(cp_dir, "index", _lv_key(layer, "ra-all")),
            cp_dir / f"index.{sanitize_layer_name(layer)}__ra-all.pt",
        )


if __name__ == "__main__":
    unittest.main()
