"""End-to-end integration test for the ESM-1 (fair-esm) embedder path.

Model dispatch is unit-tested in test_model_selection.py. This file loads a
real fair-esm checkpoint and runs pepe.embed() end-to-end when fair-esm is
installed.

Run locally (requires fair-esm and network for first model download):

    python -m pytest src/tests/test_esm1_integration.py -v

CI installs fair-esm in the integration job and runs this file.
"""

import importlib.util
import os
import sys
import unittest

import pytest

sys.path.insert(0, os.path.abspath("src"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.usefixtures("esm1_model_cache"),
]

FAIR_ESM_AVAILABLE = importlib.util.find_spec("esm") is not None
SKIP_REASON = (
    "fair-esm is not installed (pip install fair-esm); "
    "ESM-1 integration tests require the esm package"
)

MODEL_NAME = "esm1_t6_43M_UR50S"

SHORT_SEQUENCES = {
    "seq1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "seq2": "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK",
}


@unittest.skipUnless(FAIR_ESM_AVAILABLE, SKIP_REASON)
class TestESM1Integration(unittest.TestCase):
    def test_esm1_mean_pooled_in_memory(self):
        import pepe

        results = pepe.embed(
            model_name=MODEL_NAME,
            sequences=SHORT_SEQUENCES,
            extract_embeddings=["mean_pooled"],
            layers=[[-1]],
            device="cpu",
            streaming_output=False,
        )

        self.assertIn("mean_pooled", results)
        layer_outputs = results["mean_pooled"]
        layer_key = next(iter(layer_outputs))
        self.assertEqual(len(layer_outputs[layer_key]), 2)
        self.assertGreater(layer_outputs[layer_key][0].shape[0], 0)


if __name__ == "__main__":
    unittest.main()
