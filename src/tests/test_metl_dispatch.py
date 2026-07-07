import builtins
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("src"))

from pepe.model_errors import (
    METL3DNotSupportedError,
    METLPackageRequiredError,
    ModelSelectionError,
)
from pepe.model_selecter import select_model


class TestMETLDispatch(unittest.TestCase):
    """Mocked unit tests for METL model dispatch (no downloads)."""

    def test_select_metl_1d_returns_metl_embedder(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=AssertionError(
                "AutoConfig should not be called for METL idents"
            ),
        ):
            embedder_cls = select_model("metl-g-20m-1d")

        from pepe.embedders.metl_embedder import METLEmbedder

        self.assertIs(embedder_cls, METLEmbedder)

    def test_select_metl_3d_raises_typed_error(self):
        with self.assertRaises(METL3DNotSupportedError) as ctx:
            select_model("metl-l-2m-3d-gb1")

        msg = str(ctx.exception).lower()
        self.assertIn("3d", msg)
        self.assertIn("metl-g-20m-1d", msg)

    def test_select_gitter_lab_metl_rejected_early(self):
        with patch("transformers.AutoConfig.from_pretrained") as mock_config:
            with self.assertRaises(ModelSelectionError) as ctx:
                select_model("gitter-lab/METL")

        msg = str(ctx.exception).lower()
        self.assertIn("gitter-lab/metl", msg)
        self.assertIn("metl-g-20m-1d", msg)
        mock_config.assert_not_called()

    def test_select_gitter_lab_metl_pretrained_rejected_early(self):
        with patch("transformers.AutoConfig.from_pretrained") as mock_config:
            with self.assertRaises(ModelSelectionError) as ctx:
                select_model("gitter-lab/metl-pretrained")

        msg = str(ctx.exception).lower()
        self.assertIn("gitter-lab/metl", msg)
        self.assertIn("metl-g-20m-1d", msg)
        mock_config.assert_not_called()

    def test_import_metl_missing_package_has_install_hint(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "metl":
                raise ImportError("No module named 'metl'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from pepe.embedders.metl_embedder import _import_metl

            with self.assertRaises(METLPackageRequiredError) as ctx:
                _import_metl()

        msg = str(ctx.exception)
        self.assertIn("metl-pretrained", msg)
        self.assertIn("pip install", msg)

    def test_gitter_lab_metl_init_requires_specific_ident(self):
        from pepe.embedders.metl_embedder import (
            _validate_supported_output_types,
            resolve_metl_ident,
        )

        with self.assertRaises(ModelSelectionError) as ctx:
            resolve_metl_ident("gitter-lab/METL")

        msg = str(ctx.exception).lower()
        self.assertIn("gitter-lab/metl", msg)
        self.assertIn("metl-g-20m-1d", msg)

        with self.assertRaises(ModelSelectionError):
            _validate_supported_output_types(["logits"])

        with self.assertRaises(ModelSelectionError):
            _validate_supported_output_types(["per_token", "attention_layer"])


if __name__ == "__main__":
    unittest.main()
