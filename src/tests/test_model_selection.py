import os
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("src"))

from pepe.model_errors import (
    ESMCForkRequiredError,
    GatedModelError,
    ModelEnvironmentError,
    ModelNotFoundError,
    ModelSelectionError,
    RemoteCodeRequiredError,
    UnsupportedArchitectureError,
)
from pepe.model_selecter import report_model, select_model


def _config(model_type):
    cfg = MagicMock()
    cfg.model_type = model_type
    return cfg


def _mock_hf_response():
    """huggingface_hub >=1.0 requires a response with headers."""
    response = MagicMock()
    response.headers = {}
    return response


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
            with self.assertRaises(UnsupportedArchitectureError) as ctx:
                select_model("someuser/mystery-model")
        self.assertIn("PyTorch models only", str(ctx.exception))


class TestTypedConfigErrors(unittest.TestCase):
    """Each upstream HuggingFace failure maps to a typed ModelSelectionError."""

    MODEL = "someuser/mystery-model"

    def test_each_upstream_exception_maps_to_typed_error(self):
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        cases = [
            (
                RepositoryNotFoundError(
                    "404 Client Error", response=_mock_hf_response()
                ),
                ModelNotFoundError,
                "not found",
            ),
            (
                GatedRepoError("403 gated", response=_mock_hf_response()),
                GatedModelError,
                "gated",
            ),
            (
                OSError("Connection reset by peer"),
                ModelEnvironmentError,
                "network",
            ),
            (
                ValueError(
                    "Loading someuser/mystery-model requires you to execute the "
                    "configuration file in that repo. Pass `trust_remote_code=True`."
                ),
                RemoteCodeRequiredError,
                "trust_remote_code",
            ),
            (
                ValueError("Unrecognized model_type foo"),
                UnsupportedArchitectureError,
                "PyTorch models only",
            ),
        ]
        for upstream, expected_cls, msg_fragment in cases:
            with self.subTest(expected=expected_cls.__name__):
                with patch(
                    "transformers.AutoConfig.from_pretrained",
                    side_effect=upstream,
                ):
                    with self.assertRaises(expected_cls) as ctx:
                        select_model(self.MODEL)
                self.assertIn(msg_fragment.lower(), str(ctx.exception).lower())

    def test_esmc_unrecognized_model_type_raises_fork_error(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=ValueError("Unrecognized model_type esmc"),
        ):
            with self.assertRaises(ESMCForkRequiredError) as ctx:
                select_model("biohub/ESMC-300M")
        self.assertIn("Biohub/transformers", str(ctx.exception))

    def test_gated_esmc_raises_fork_error(self):
        from huggingface_hub.errors import GatedRepoError

        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=GatedRepoError("403 gated", response=_mock_hf_response()),
        ):
            with self.assertRaises(ESMCForkRequiredError) as ctx:
                select_model("biohub/ESMC-300M")
        self.assertIn("Biohub/transformers", str(ctx.exception))

    def test_trust_remote_code_skips_remote_code_error_when_flag_set(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=ValueError("Pass `trust_remote_code=True` to load this model."),
        ):
            with self.assertRaises(ModelSelectionError):
                select_model(self.MODEL, trust_remote_code=True)


class TestBareWeightNames(unittest.TestCase):
    """Bare ESM weight names (no slash) are matched by name without a config call."""

    def test_bare_esm2_name(self):
        from pepe.embedders.huggingface_embedder import ESM2Embedder

        # Must not touch the network: AutoConfig would raise if called.
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=AssertionError(
                "AutoConfig should not be called for bare names"
            ),
        ):
            self.assertIs(select_model("esm2_t6_8M_UR50D"), ESM2Embedder)

    def test_bare_esm1_name(self):
        from pepe.embedders.esm_embedder import ESMEmbedder

        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=AssertionError(
                "AutoConfig should not be called for bare names"
            ),
        ):
            self.assertIs(select_model("esm1b_t33_650M_UR50S"), ESMEmbedder)

    def test_unsupported_bare_name_raises(self):
        with self.assertRaises(ValueError):
            select_model("not-a-real-model")


class TestTrustRemoteCode(unittest.TestCase):
    def test_select_model_forwards_trust_remote_code_to_autoconfig(self):
        with patch("transformers.AutoConfig.from_pretrained") as mock_config:
            mock_config.return_value = _config("bert")
            select_model("someuser/model-x", trust_remote_code=True)
        mock_config.assert_called_once_with("someuser/model-x", trust_remote_code=True)

    def test_generic_embedder_passes_trust_remote_code_to_from_pretrained(self):
        import torch

        from pepe.embedders.huggingface_embedder import GenericHuggingFaceEmbedder

        recorded = {"tokenizer": None, "model": None}

        def tok_from_pretrained(link, **kwargs):
            recorded["tokenizer"] = kwargs
            mock = MagicMock()
            mock.pad_token = "<pad>"
            mock.all_special_ids = [0]
            return mock

        def model_from_pretrained(link, **kwargs):
            recorded["model"] = kwargs
            mock = MagicMock()
            mock.config = MagicMock(
                num_attention_heads=4,
                num_hidden_layers=2,
                hidden_size=32,
            )
            mock.to.return_value = mock
            mock.eval.return_value = None
            return mock

        embedder = GenericHuggingFaceEmbedder.__new__(GenericHuggingFaceEmbedder)
        embedder.trust_remote_code = True
        embedder.return_contacts = False
        embedder.device = torch.device("cpu")

        with patch(
            "pepe.embedders.huggingface_embedder._import_transformers",
            return_value=(None,) * 5
            + (
                MagicMock(from_pretrained=model_from_pretrained),
                MagicMock(from_pretrained=tok_from_pretrained),
                MagicMock(from_pretrained=model_from_pretrained),
                MagicMock(from_pretrained=model_from_pretrained),
            ),
        ):
            embedder._initialize_model("lab/custom-model")

        self.assertEqual(
            recorded["tokenizer"], {"use_fast": True, "trust_remote_code": True}
        )
        self.assertTrue(recorded["model"]["trust_remote_code"])


class TestReportModel(unittest.TestCase):
    def test_report_names_embedder_and_flags_subword_tokenizer(self):
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_config.max_position_embeddings = 512

        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 512
        mock_tokenizer.encode.return_value = [1, 2, 3]

        with patch(
            "transformers.AutoConfig.from_pretrained", return_value=mock_config
        ), patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ), patch(
            "pepe.model_selecter.select_model",
            return_value=MagicMock(__name__="GenericHuggingFaceEmbedder"),
        ), patch("pepe.utils.is_character_tokenizer", return_value=False):
            buf = StringIO()
            with patch("sys.stdout", buf):
                report_model("someuser/protbert", trust_remote_code=True)

        output = buf.getvalue()
        self.assertIn("Architecture:    bert", output)
        self.assertIn("Embedder:        GenericHuggingFaceEmbedder", output)
        self.assertIn("Max length:      512", output)
        self.assertIn("subword", output)
        self.assertIn("Logits:          no", output)


if __name__ == "__main__":
    unittest.main()
