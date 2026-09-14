import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("src"))

from pepe.model_selecter import select_model


class TestESMCModelSelection(unittest.TestCase):
    def test_select_esmc_returns_esmc_embedder(self):
        mock_config = MagicMock()
        mock_config.model_type = "esmc"

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            embedder_cls = select_model("biohub/ESMC-300M")

        from pepe.embedders.huggingface_embedder import ESMCEmbedder

        self.assertIs(embedder_cls, ESMCEmbedder)

    def test_select_esmc_missing_fork_error(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=ValueError("Unrecognized model_type esmc"),
        ):
            with self.assertRaises(ValueError) as ctx:
                select_model("biohub/ESMC-300M")

        self.assertIn("Biohub/transformers", str(ctx.exception))

    def test_select_esmc_inspect_logits_available(self):
        import io
        import pepe.model_selecter

        mock_config = MagicMock()
        mock_config.model_type = "esmc"
        mock_config.num_hidden_layers = 30
        mock_config.num_attention_heads = 15
        mock_config.hidden_size = 960
        mock_config.vocab_size = 64
        mock_config.max_position_embeddings = 2048
        mock_config.n_positions = 2048

        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 2048
        mock_tokenizer.get_vocab.return_value = {"A": 4, "C": 5}

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            captured = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = captured
                pepe.model_selecter.report_model("biohub/ESMC-300M")
            finally:
                sys.stdout = old_stdout
            output = captured.getvalue()
            self.assertIn("Logits:          yes", output)

    def test_esmc_compute_outputs_returns_logits(self):
        import torch
        from pepe.embedders.huggingface_embedder import ESMCEmbedder

        embedder = object.__new__(ESMCEmbedder)
        embedder.layers = [30]
        embedder.precision = "32"
        embedder.return_contacts = False

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 10, 64)
        mock_outputs.hidden_states = [torch.randn(2, 10, 960)] * 31
        mock_model.return_value = mock_outputs

        toks = torch.zeros((2, 10), dtype=torch.long)
        attention_mask = torch.ones((2, 10), dtype=torch.long)

        logits, representations, attention_matrices = embedder._compute_outputs(
            model=mock_model,
            toks=toks,
            attention_mask=attention_mask,
            return_embeddings=True,
            return_contacts=False,
            return_logits=True,
        )

        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape, (2, 10, 64))
        self.assertIn(30, representations)
        self.assertEqual(representations[30].shape, (2, 10, 960))
        self.assertIsNone(attention_matrices)


@unittest.skipUnless(
    os.environ.get("ESMC_TEST") == "1",
    "Set ESMC_TEST=1 to run ESMC integration test (requires Biohub transformers fork)",
)
class TestESMCIntegration(unittest.TestCase):
    def test_esmc_mean_pooled(self):
        import pepe

        sequences = {
            "seq1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        }
        results = pepe.embed(
            model_name="biohub/ESMC-300M",
            sequences=sequences,
            extract_embeddings=["mean_pooled"],
            layers=[[-1]],
            device="cpu",
        )

        self.assertIn("mean_pooled", results)
        layer_outputs = results["mean_pooled"]
        layer_key = next(iter(layer_outputs))
        self.assertEqual(len(layer_outputs[layer_key]), 1)
        self.assertEqual(layer_outputs[layer_key][0].shape[0], 960)

    def test_esmc_logits(self):
        import pepe

        sequences = {
            "seq1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        }
        results = pepe.embed(
            model_name="biohub/ESMC-300M",
            sequences=sequences,
            extract_embeddings=["logits"],
            layers=[[-1]],
            device="cpu",
        )

        self.assertIn("logits", results)
        layer_outputs = results["logits"]
        layer_key = next(iter(layer_outputs))
        self.assertEqual(len(layer_outputs[layer_key]), 1)
        self.assertEqual(layer_outputs[layer_key][0].shape[1], 64)


if __name__ == "__main__":
    unittest.main()
