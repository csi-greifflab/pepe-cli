import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath("src"))


def _as_numpy(embedding):
    if hasattr(embedding, "detach"):
        return embedding.detach().cpu().numpy()
    return np.asarray(embedding)


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
        self.assertTrue(np.any(_as_numpy(embedding) != 0))

    def test_metl_batch_matches_single(self):
        """Batching must not corrupt per-sequence embeddings (batch_first path).

        Two equal-length sequences are used so no padding is introduced, isolating
        the batch dimension from any padding/attention effects: each sequence's
        batched embedding must match its single-sequence embedding.
        """
        import pepe

        seq_a = "MVLSPADKTNVKAAWGKVGA"  # length 20
        seq_b = "ACDEFGHIKLMNPQRSTVWY"  # length 20

        def embed(seqs):
            return pepe.embed(
                model_name="metl-g-20m-1d",
                sequences=seqs,
                extract_embeddings=["mean_pooled"],
                layers=[[-1]],
                device="cpu",
            )

        single_a = embed({"a": seq_a})["mean_pooled"]
        ref_a = _as_numpy(single_a[next(iter(single_a))][0])
        single_b = embed({"b": seq_b})["mean_pooled"]
        ref_b = _as_numpy(single_b[next(iter(single_b))][0])

        batched = embed({"a": seq_a, "b": seq_b})["mean_pooled"]
        rows = [_as_numpy(x) for x in batched[next(iter(batched))]]

        self.assertEqual(len(rows), 2)
        # Order-independent: each single-sequence reference matches some batched row.
        self.assertTrue(any(np.allclose(ref_a, r, atol=1e-4) for r in rows))
        self.assertTrue(any(np.allclose(ref_b, r, atol=1e-4) for r in rows))
        # Distinct sequences must yield distinct embeddings (sanity).
        self.assertFalse(np.allclose(ref_a, ref_b, atol=1e-4))

    def test_metl_multiple_layers_distinct(self):
        """Requesting multiple layers returns distinct per-layer representations."""
        import pepe

        results = pepe.embed(
            model_name="metl-g-20m-1d",
            sequences={"a": "ACDEFGHIKLMNPQRSTVWY"},
            extract_embeddings=["mean_pooled"],
            layers=[[-1, -2]],
            device="cpu",
        )["mean_pooled"]

        self.assertEqual(len(results), 2)  # two separate layers captured
        keys = list(results)
        v0 = _as_numpy(results[keys[0]][0])
        v1 = _as_numpy(results[keys[1]][0])
        self.assertFalse(np.allclose(v0, v1))  # not aliased to the same layer


if __name__ == "__main__":
    unittest.main()
