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

        self.assertIn("pepe-cli[esmc]", str(ctx.exception))


@unittest.skipUnless(
    os.environ.get("ESMC_TEST") == "1",
    "Set ESMC_TEST=1 to run ESMC integration test (requires pepe-cli[esmc])",
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


if __name__ == "__main__":
    unittest.main()
