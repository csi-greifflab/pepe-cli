"""Smoke tests for ESMC embedding modes. Run with ESMC_TEST=1."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("src"))

import pepe


@unittest.skipUnless(
    os.environ.get("ESMC_TEST") == "1",
    "Set ESMC_TEST=1 to run ESMC mode tests",
)
class TestESMCEmbeddingModes(unittest.TestCase):
    MODEL = "biohub/ESMC-300M"
    EMBED_DIM = 960
    NUM_LAYERS = 30
    NUM_HEADS = 15

    def _embed(self, **kwargs):
        defaults = {
            "model_name": self.MODEL,
            "device": "cpu",
            "layers": [[-1]],
            "streaming_output": False,
        }
        defaults.update(kwargs)
        return pepe.embed(**defaults)

    def test_per_token(self):
        sequences = {"seq1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"}
        results = self._embed(
            sequences=sequences,
            extract_embeddings=["per_token"],
        )
        self.assertIn("per_token", results)
        layer = self.NUM_LAYERS
        self.assertIn(layer, results["per_token"])
        arr = results["per_token"][layer][0]
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.shape[1], self.EMBED_DIM)
        self.assertGreater(arr.shape[0], 0)

    def test_substring_pooled(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        fasta = os.path.join(repo_root, "src/tests/test_files/test.fasta")
        substring = os.path.join(repo_root, "src/tests/test_files/test_substring.csv")
        results = self._embed(
            fasta_path=fasta,
            substring_path=substring,
            extract_embeddings=["substring_pooled"],
        )
        self.assertIn("substring_pooled", results)
        layer = self.NUM_LAYERS
        self.assertIn(layer, results["substring_pooled"])
        outputs = results["substring_pooled"][layer]
        self.assertEqual(len(outputs), 10)
        for arr in outputs:
            self.assertEqual(arr.shape, (self.EMBED_DIM,))

    def test_attention_head(self):
        sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY"}
        results = self._embed(
            sequences=sequences,
            extract_embeddings=["attention_head"],
        )
        self.assertIn("attention_head", results)
        layer = self.NUM_LAYERS
        self.assertIn(layer, results["attention_head"])
        head_outputs = results["attention_head"][layer]
        self.assertEqual(len(head_outputs), self.NUM_HEADS)
        for head in range(self.NUM_HEADS):
            arr = head_outputs[head][0]
            self.assertEqual(arr.ndim, 2)
            self.assertEqual(arr.shape[0], arr.shape[1])

    def test_attention_layer(self):
        sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY"}
        results = self._embed(
            sequences=sequences,
            extract_embeddings=["attention_layer"],
        )
        self.assertIn("attention_layer", results)
        layer = self.NUM_LAYERS
        self.assertIn(layer, results["attention_layer"])
        arr = results["attention_layer"][layer][0]
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.shape[0], arr.shape[1])


if __name__ == "__main__":
    unittest.main()
