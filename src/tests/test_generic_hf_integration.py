"""End-to-end integration test for the generic HuggingFace embedder path.

Mocked dispatch tests live in test_model_selection.py. This file downloads a
real tiny encoder from the Hub and runs pepe.embed() end-to-end.

Run the gated integration test locally:

    GENERIC_HF_TEST=1 python -m pytest src/tests/test_generic_hf_integration.py -v

Requires network access and transformers/torch (installed with pepe-cli).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("src"))


@unittest.skipUnless(
    os.environ.get("GENERIC_HF_TEST") == "1",
    "Set GENERIC_HF_TEST=1 to run generic HuggingFace integration test "
    "(downloads hf-internal-testing/tiny-random-BertModel from the Hub)",
)
class TestGenericHFIntegration(unittest.TestCase):
    MODEL = "hf-internal-testing/tiny-random-BertModel"

    def test_tiny_bert_mean_pooled_end_to_end(self):
        from transformers import AutoConfig

        import pepe

        config = AutoConfig.from_pretrained(self.MODEL)
        expected_hidden = config.hidden_size

        sequences = {
            # tiny-random-BertModel uses a multilingual WordPiece vocab; pick tokens
            # that exist as single-character entries so check_input_tokens passes.
            "seq1": '!"#$%',
        }
        results = pepe.embed(
            model_name=self.MODEL,
            sequences=sequences,
            extract_embeddings=["mean_pooled"],
            layers=[[-1]],
            device="cpu",
            streaming_output=False,
        )

        self.assertIn("mean_pooled", results)
        layer_outputs = results["mean_pooled"]
        layer_key = next(iter(layer_outputs))
        self.assertEqual(len(layer_outputs[layer_key]), 1)
        self.assertEqual(layer_outputs[layer_key][0].shape, (expected_hidden,))


if __name__ == "__main__":
    unittest.main()
