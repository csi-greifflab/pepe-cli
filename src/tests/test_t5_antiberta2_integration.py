"""End-to-end integration tests for T5 (ProtT5 path) and AntiBERTa2 embedders.

Dispatch is unit-tested in test_model_selection.py. These tests download tiny
Hub checkpoints and run pepe.embed() end-to-end.

Run locally (requires network for first model download):

    T5_ANTIBERTA2_TEST=1 python -m pytest src/tests/test_t5_antiberta2_integration.py -v

CI runs this file in the integration job when T5_ANTIBERTA2_TEST=1 is set.
"""

import os
import sys
import unittest

import pytest
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.abspath("src"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


def _single_char_sequence(tokenizer, count: int = 6) -> str:
    """Build a short sequence from single-character vocab entries."""
    valid = set()
    for token in tokenizer.get_vocab().keys():
        valid.add(token[1:] if token.startswith("▁") else token)
    chars = [c for c in valid if len(c) == 1][:count]
    assert chars, "tokenizer has no single-character entries for integration test"
    return "".join(chars)


@unittest.skipUnless(
    os.environ.get("T5_ANTIBERTA2_TEST") == "1",
    "Set T5_ANTIBERTA2_TEST=1 to run T5 / AntiBERTa2 integration tests "
    "(downloads tiny Hub checkpoints)",
)
class TestT5AntiBERTa2Integration(unittest.TestCase):
    T5_MODEL = "hf-internal-testing/tiny-random-t5-v1.1"
    ANTIBERTA2_MODEL = "hf-internal-testing/tiny-random-RoFormerModel"

    def test_t5_mean_pooled_end_to_end(self):
        import pepe

        config = AutoConfig.from_pretrained(self.T5_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(self.T5_MODEL)
        sequence = _single_char_sequence(tokenizer)

        results = pepe.embed(
            model_name=self.T5_MODEL,
            sequences={"seq1": sequence},
            extract_embeddings=["mean_pooled"],
            layers=[[-1]],
            device="cpu",
            streaming_output=False,
        )

        self.assertIn("mean_pooled", results)
        layer_outputs = results["mean_pooled"]
        layer_key = next(iter(layer_outputs))
        self.assertEqual(len(layer_outputs[layer_key]), 1)
        self.assertEqual(layer_outputs[layer_key][0].shape, (config.hidden_size,))

    def test_antiberta2_mean_pooled_end_to_end(self):
        import pepe

        config = AutoConfig.from_pretrained(self.ANTIBERTA2_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(self.ANTIBERTA2_MODEL)
        sequence = _single_char_sequence(tokenizer)

        results = pepe.embed(
            model_name=self.ANTIBERTA2_MODEL,
            sequences={"seq1": sequence},
            extract_embeddings=["mean_pooled"],
            layers=[[-1]],
            device="cpu",
            streaming_output=False,
        )

        self.assertIn("mean_pooled", results)
        layer_outputs = results["mean_pooled"]
        layer_key = next(iter(layer_outputs))
        self.assertEqual(len(layer_outputs[layer_key]), 1)
        self.assertEqual(layer_outputs[layer_key][0].shape, (config.hidden_size,))


if __name__ == "__main__":
    unittest.main()
