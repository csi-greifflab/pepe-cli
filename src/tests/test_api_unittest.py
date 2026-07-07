import os
import sys
import tempfile
import unittest

import pytest

# Add src to sys.path
sys.path.insert(0, os.path.abspath("src"))

import pepe

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.usefixtures("esm2_model_cache"),
]


class TestPepeAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_sequences = {
            "seq1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
            "seq2": "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK",
        }
        cls.model_name = "esm2_t6_8M_UR50D"

    def test_embed_sequences_dict_in_memory(self):
        """Test embedding from a dictionary of sequences with in-memory results."""
        results = pepe.embed(
            model_name=self.model_name,
            sequences=self.test_sequences,
            extract_embeddings=["mean_pooled"],
            device="cpu",
        )

        self.assertIn("mean_pooled", results)
        self.assertIn("output_path", results)

        # ESM2-8M has 6 layers, default is last layer (-1 -> 6)
        mean_pooled = results["mean_pooled"]
        self.assertIn(6, mean_pooled)
        self.assertEqual(len(mean_pooled[6]), 2)
        self.assertEqual(mean_pooled[6][0].shape[0], 320)  # Embedding dim

    def test_embed_sequences_list_in_memory(self):
        """Test embedding from a list of sequences."""
        seq_list = list(self.test_sequences.values())
        results = pepe.embed(
            model_name=self.model_name,
            sequences=seq_list,
            extract_embeddings=["mean_pooled"],
            device="cpu",
        )

        self.assertIn("mean_pooled", results)
        mean_pooled = results["mean_pooled"]
        self.assertEqual(len(mean_pooled[6]), 2)

    def test_embed_to_disk(self):
        """Test embedding with output_path provided."""
        with tempfile.TemporaryDirectory() as tmp_out:
            results = pepe.embed(
                model_name=self.model_name,
                sequences=self.test_sequences,
                output_path=tmp_out,
                extract_embeddings=["mean_pooled"],
                device="cpu",
                streaming_output=False,
            )

            self.assertEqual(results["output_path"], tmp_out)
            # Verify file exists
            # Output structured as: {output_path}/{model_name_basename}/mean_pooled/...
            model_basename = os.path.basename(self.model_name)
            expected_dir = os.path.join(tmp_out, model_basename, "mean_pooled")
            self.assertTrue(os.path.exists(expected_dir))

            files = os.listdir(expected_dir)
            self.assertTrue(any(f.endswith(".npy") for f in files))

    def test_embed_discard_padding(self):
        """Test discard_padding through the API."""
        results = pepe.embed(
            model_name=self.model_name,
            sequences=self.test_sequences,
            extract_embeddings=["per_token"],
            device="cpu",
            discard_padding=True,
            streaming_output=False,
        )

        self.assertIn("per_token", results)
        per_token = results["per_token"][6]

        # Lengths should be different because padding is discarded
        len1 = per_token[0].shape[0]
        len2 = per_token[1].shape[0]
        self.assertNotEqual(len1, len2)

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with self.assertRaises(ValueError):
            pepe.embed(model_name=self.model_name, device="cpu")

    def test_embed_streaming_no_path(self):
        """Test that streaming_output is disabled if no output_path is provided."""
        with self.assertLogs("pepe.api", level="WARNING") as cm:
            results = pepe.embed(
                model_name=self.model_name,
                sequences=self.test_sequences,
                extract_embeddings=["mean_pooled"],
                device="cpu",
                streaming_output=True,
            )
            # Ensure warning was logged
            self.assertTrue(
                any("No output_path provided" in line for line in cm.output)
            )
            # Ensure results were returned (streaming was disabled)
            self.assertIn("mean_pooled", results)


if __name__ == "__main__":
    unittest.main()
