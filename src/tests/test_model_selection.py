import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("src"))

from pepe.model_selecter import select_model


def _config(model_type):
    cfg = MagicMock()
    cfg.model_type = model_type
    return cfg


class TestHuggingFaceDispatch(unittest.TestCase):
    """Config-first dispatch for HuggingFace repo ids (username/model-name)."""

    def _select_with_type(self, model_type, model_name="someuser/model-x"):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=_config(model_type),
        ):
            return select_model(model_name)

    def test_bert_routes_to_generic_embedder(self):
        from pepe.embedders.huggingface_embedder import GenericHuggingFaceEmbedder

        self.assertIs(self._select_with_type("bert"), GenericHuggingFaceEmbedder)

    def test_unknown_architecture_routes_to_generic_embedder(self):
        from pepe.embedders.huggingface_embedder import GenericHuggingFaceEmbedder

        self.assertIs(
            self._select_with_type("some_new_arch"), GenericHuggingFaceEmbedder
        )

    def test_esm_config_routes_to_esm2_embedder(self):
        from pepe.embedders.huggingface_embedder import ESM2Embedder

        self.assertIs(self._select_with_type("esm"), ESM2Embedder)

    def test_t5_config_routes_to_t5_embedder(self):
        from pepe.embedders.huggingface_embedder import T5Embedder

        self.assertIs(self._select_with_type("t5"), T5Embedder)

    def test_roformer_config_routes_to_antiberta2_embedder(self):
        from pepe.embedders.huggingface_embedder import Antiberta2Embedder

        self.assertIs(self._select_with_type("roformer"), Antiberta2Embedder)

    def test_config_overrides_misleading_slug(self):
        # A repo whose slug says "esm2" but whose config is a BERT must dispatch on
        # the config, not the name.
        from pepe.embedders.huggingface_embedder import GenericHuggingFaceEmbedder

        self.assertIs(
            self._select_with_type("bert", model_name="labX/my-esm2-finetune"),
            GenericHuggingFaceEmbedder,
        )

    def test_config_load_failure_raises_actionable_error(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=ValueError("Unrecognized model_type foo"),
        ):
            with self.assertRaises(ValueError) as ctx:
                select_model("someuser/mystery-model")
        # Not-found / unrecognized configs surface a clear message rather than a
        # generic crash.
        self.assertIn("PyTorch models only", str(ctx.exception))


class TestBareWeightNames(unittest.TestCase):
    """Bare ESM weight names (no slash) are matched by name without a config call."""

    def test_bare_esm2_name(self):
        from pepe.embedders.huggingface_embedder import ESM2Embedder

        # Must not touch the network: AutoConfig would raise if called.
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=AssertionError("AutoConfig should not be called for bare names"),
        ):
            self.assertIs(select_model("esm2_t6_8M_UR50D"), ESM2Embedder)

    def test_bare_esm1_name(self):
        from pepe.embedders.esm_embedder import ESMEmbedder

        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=AssertionError("AutoConfig should not be called for bare names"),
        ):
            self.assertIs(select_model("esm1b_t33_650M_UR50S"), ESMEmbedder)

    def test_unsupported_bare_name_raises(self):
        with self.assertRaises(ValueError):
            select_model("not-a-real-model")


if __name__ == "__main__":
    unittest.main()
