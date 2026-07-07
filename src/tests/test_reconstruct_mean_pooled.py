"""Regression: mean_pooled-only + long-sequence splitting (in-memory).

Before the fix, requesting only ``mean_pooled`` (without ``per_token``) together
with ``--split_long_sequences`` in in-memory mode raised ``KeyError: 'per_token'``
inside ``_reconstruct_chunks`` (mean-pooled reconstruction reads the per-token
scratch buffer, which was never built because per_token was not a requested
output). The engine now retains per_token internally for the duration of the run
so the pooled vector can be stitched back from chunks, without exporting it.

Uses the CustomEmbedder path so no model download is required. The wrapper
returns random hidden states, so values are not comparable across runs; the test
asserts structure (count/shape/non-zero) and, above all, that reconstruction does
not raise.
"""

import json
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath("src"))

import pepe


class TestReconstructMeanPooledOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = tempfile.mkdtemp()
        cls.config = {"num_layers": 2, "num_attention_heads": 4, "hidden_size": 32}
        with open(os.path.join(cls.model_dir, "config.json"), "w") as f:
            json.dump(cls.config, f)
        cls.model_path = os.path.join(cls.model_dir, "model.pt")
        torch.save({"config": cls.config, "state_dict": {}}, cls.model_path)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.model_dir, ignore_errors=True)

    def _embed(self, sequences, overlap):
        return pepe.embed(
            model_name=self.model_path,
            sequences=sequences,
            extract_embeddings=["mean_pooled"],  # deliberately NOT per_token
            layers=[[-1]],
            device="cpu",
            streaming_output=False,
            split_long_sequences=True,
            force_split_length=8,
            split_overlap=overlap,
        )

    def _assert_pooled(self, results, expected_count):
        self.assertIn("mean_pooled", results)
        layer_outputs = results["mean_pooled"]
        layer_key = next(iter(layer_outputs))
        vectors = layer_outputs[layer_key]
        self.assertEqual(len(vectors), expected_count)
        for vec in vectors:
            self.assertEqual(tuple(vec.shape), (self.config["hidden_size"],))
            self.assertTrue(bool((vec != 0).any()))
        # per_token was scratch-only and must not leak into the results.
        self.assertNotIn("per_token", results)

    def test_split_sequence_no_overlap(self):
        # 30-char sequence exceeds force_split_length (8) → gets split.
        results = self._embed({"long": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"}, overlap=0)
        self._assert_pooled(results, expected_count=1)

    def test_split_sequence_with_overlap(self):
        results = self._embed({"long": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"}, overlap=2)
        self._assert_pooled(results, expected_count=1)

    def test_mixed_split_and_unsplit(self):
        # One long (split) + one short (copied directly) exercises both branches.
        results = self._embed(
            {"long": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL", "short": "ACDEF"}, overlap=0
        )
        self._assert_pooled(results, expected_count=2)


if __name__ == "__main__":
    unittest.main()
