import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath("src"))


@unittest.skipUnless(
    os.environ.get("METL_TEST") == "1",
    "Set METL_TEST=1 to run METL integration test (requires metl-pretrained)",
)
class TestMETLIntegration(unittest.TestCase):
    def test_metl_mean_pooled(self):
        import pepe

        sequences = {
            "seq1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        }
        results = pepe.embed(
            model_name="metl-g-20m-1d",
            sequences=sequences,
            extract_embeddings=["mean_pooled"],
            device="cpu",
        )

        self.assertIn("mean_pooled", results)
        layer_outputs = results["mean_pooled"]
        layer_key = next(iter(layer_outputs))
        embedding = layer_outputs[layer_key][0]
        self.assertEqual(embedding.shape, (512,))
        if hasattr(embedding, "detach"):
            values = embedding.detach().cpu().numpy()
        else:
            values = np.asarray(embedding)
        self.assertTrue(np.any(values != 0))


if __name__ == "__main__":
    unittest.main()
