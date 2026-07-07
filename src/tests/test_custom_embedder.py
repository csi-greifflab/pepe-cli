"""Integration coverage for the CustomEmbedder (.pt) path."""
import json
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath("src"))

import pepe


class TestCustomEmbedderRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = tempfile.mkdtemp()
        cls.config = {
            "num_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
        }
        config_path = os.path.join(cls.model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(cls.config, f)

        cls.model_path = os.path.join(cls.model_dir, "model.pt")
        # Save weights-only payload so torch.load(weights_only=True) works on PyTorch 2.6+.
        torch.save({"config": cls.config, "state_dict": {}}, cls.model_path)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.model_dir, ignore_errors=True)

    def test_embed_custom_pt_model_in_memory(self):
        results = pepe.embed(
            model_name=self.model_path,
            sequences={"seq1": "ACDEFGHIK"},
            extract_embeddings=["mean_pooled"],
            device="cpu",
            streaming_output=False,
        )

        self.assertIn("mean_pooled", results)
        mean_pooled = results["mean_pooled"]
        self.assertIn(2, mean_pooled)
        self.assertEqual(len(mean_pooled[2]), 1)
        self.assertEqual(mean_pooled[2][0].shape, (self.config["hidden_size"],))


if __name__ == "__main__":
    unittest.main()
