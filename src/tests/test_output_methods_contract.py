"""Contract tests across all output methods and modes in PEPE.

Ensures that:
1. Shape contract: Declared disk preallocation shapes in `_set_output_objects()`
   match actual runtime shapes on disk and in memory across all output methods
   (`mean_pooled`, `per_token`, `logits`, `attention_head`, `attention_layer`, `attention_model`).
2. Dimensionality invariants: Logits retain full vocabulary dimensions (3D unflattened, 2D flattened),
   per-token retains embedding dimension, and attention retains sequence-by-sequence dimensions.
3. Multi-layer indexing integrity: Requesting multiple layers does not cross-contaminate,
   slice vocabulary/batch as layers, or corrupt layer dictionaries.
4. Flattening contract: `flatten=True` produces correct 2D matrices across all output types.
5. Padding discard contract: `discard_padding=True` correctly strips padding tokens without
   dropping dimensions.
6. Splitting & chunk reconstruction contract: `split_long_sequences=True` correctly stitches
   and reconstructs `per_token`, `logits`, and `mean_pooled` without shape collapse or stack failures.
"""

import glob
import os
import sys
import tempfile
import unittest

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("src"))

import pepe
from pepe.embedders.base_embedder import BaseEmbedder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.usefixtures("esm2_model_cache"),
]

MODEL_NAME = "esm2_t6_8M_UR50D"
EMBED_DIM = 320
VOCAB_SIZE = 33
NUM_HEADS = 20
LAYER = 1

SEQUENCES = {
    "seq1": "ACDEFGHIKLMNPQRSTVWY",  # 20 AAs -> 22 tokens with BOS/EOS
    "seq2": "ACDEFGH",  # 7 AAs  -> 9 tokens with BOS/EOS, padded to 22
}
MAX_LEN = 22  # 20 + 2 special tokens


class TestOutputMethodsContract(unittest.TestCase):
    def test_shape_and_preallocation_contract_unflattened(self):
        """Verify preallocated shape matches written array shape for all outputs (unflattened)."""
        output_types = [
            "mean_pooled",
            "per_token",
            "logits",
            "attention_head",
            "attention_layer",
            "attention_model",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            res_mem = pepe.embed(
                model_name=MODEL_NAME,
                sequences=SEQUENCES,
                extract_embeddings=output_types,
                device="cpu",
                layers=[[LAYER]],
                flatten=False,
                discard_padding=False,
                streaming_output=False,
            )
            pepe.embed(
                model_name=MODEL_NAME,
                sequences=SEQUENCES,
                extract_embeddings=output_types,
                device="cpu",
                layers=[[LAYER]],
                flatten=False,
                discard_padding=False,
                streaming_output=True,
                output_path=tmpdir,
            )

            # Expected shapes: (num_sequences=2, ...)
            expected_shapes = {
                "mean_pooled": (2, EMBED_DIM),
                "per_token": (2, MAX_LEN, EMBED_DIM),
                "logits": (2, MAX_LEN, VOCAB_SIZE),
                "attention_layer": (2, MAX_LEN, MAX_LEN),
                "attention_model": (2, MAX_LEN, MAX_LEN),
                "attention_head": (2, MAX_LEN, MAX_LEN),
            }

            for out_type, exp_shape in expected_shapes.items():
                out_dir = os.path.join(tmpdir, MODEL_NAME, out_type)
                npy_files = glob.glob(os.path.join(out_dir, "*.npy"))
                self.assertGreater(
                    len(npy_files),
                    0,
                    f"No npy file saved for {out_type} in streaming mode",
                )

                for fpath in npy_files:
                    arr = np.load(fpath)
                    self.assertEqual(
                        arr.shape,
                        exp_shape,
                        f"Streaming array shape {arr.shape} mismatch for {out_type}, expected {exp_shape}",
                    )
                    self.assertTrue(
                        np.any(arr != 0),
                        f"Streaming output for {out_type} is all zeros",
                    )

            # Check in-memory shapes
            mean_pooled_arr = np.stack(res_mem["mean_pooled"][LAYER])
            self.assertEqual(mean_pooled_arr.shape, expected_shapes["mean_pooled"])

            per_token_arr = np.stack([t.numpy() for t in res_mem["per_token"][LAYER]])
            self.assertEqual(per_token_arr.shape, expected_shapes["per_token"])

            logits_arr = np.stack([t.numpy() for t in res_mem["logits"][LAYER]])
            self.assertEqual(logits_arr.shape, expected_shapes["logits"])

            attn_layer_arr = np.stack(
                [t.numpy() for t in res_mem["attention_layer"][LAYER]]
            )
            self.assertEqual(attn_layer_arr.shape, expected_shapes["attention_layer"])

            attn_model_arr = np.stack([t.numpy() for t in res_mem["attention_model"]])
            self.assertEqual(attn_model_arr.shape, expected_shapes["attention_model"])

    def test_shape_and_preallocation_contract_flattened(self):
        """Verify preallocated shape matches written array shape for all outputs when flatten=True."""
        output_types = [
            "mean_pooled",
            "per_token",
            "logits",
            "attention_head",
            "attention_layer",
            "attention_model",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            res_mem = pepe.embed(
                model_name=MODEL_NAME,
                sequences=SEQUENCES,
                extract_embeddings=output_types,
                device="cpu",
                layers=[[LAYER]],
                flatten=True,
                discard_padding=False,
                streaming_output=False,
            )
            pepe.embed(
                model_name=MODEL_NAME,
                sequences=SEQUENCES,
                extract_embeddings=output_types,
                device="cpu",
                layers=[[LAYER]],
                flatten=True,
                discard_padding=False,
                streaming_output=True,
                output_path=tmpdir,
            )

            expected_flattened_shapes = {
                "mean_pooled": (2, EMBED_DIM),
                "per_token": (2, MAX_LEN * EMBED_DIM),
                "logits": (2, MAX_LEN * VOCAB_SIZE),
                "attention_layer": (2, MAX_LEN * MAX_LEN),
                "attention_model": (2, MAX_LEN * MAX_LEN),
                "attention_head": (2, MAX_LEN * MAX_LEN),
            }

            for out_type, exp_shape in expected_flattened_shapes.items():
                out_dir = os.path.join(tmpdir, MODEL_NAME, out_type)
                npy_files = glob.glob(os.path.join(out_dir, "*.npy"))
                self.assertGreater(
                    len(npy_files),
                    0,
                    f"No npy file saved for {out_type} in streaming mode with flatten=True",
                )

                for fpath in npy_files:
                    arr = np.load(fpath)
                    self.assertEqual(
                        arr.shape,
                        exp_shape,
                        f"Flattened streaming array shape {arr.shape} mismatch for {out_type}, expected {exp_shape}",
                    )
                    self.assertEqual(
                        arr.ndim, 2, f"Flattened array for {out_type} should be 2D"
                    )
                    self.assertTrue(
                        np.any(arr != 0),
                        f"Flattened streaming output for {out_type} is all zeros",
                    )

            # In-memory flattened checks
            self.assertEqual(
                np.stack([t.numpy() for t in res_mem["logits"][LAYER]]).shape,
                expected_flattened_shapes["logits"],
            )
            self.assertEqual(
                np.stack([t.numpy() for t in res_mem["per_token"][LAYER]]).shape,
                expected_flattened_shapes["per_token"],
            )

    def test_multi_layer_indexing_integrity(self):
        """Verify that extracting multiple layers preserves separate layer outputs without dimension corruption."""
        requested_layers = [1, 2]
        res = pepe.embed(
            model_name=MODEL_NAME,
            sequences=SEQUENCES,
            extract_embeddings=[
                "mean_pooled",
                "per_token",
                "logits",
                "attention_head",
                "attention_layer",
            ],
            device="cpu",
            layers=[requested_layers],
            flatten=False,
            discard_padding=False,
            streaming_output=False,
        )

        for l_idx in requested_layers:
            self.assertIn(
                l_idx, res["mean_pooled"], f"Layer {l_idx} missing from mean_pooled"
            )
            self.assertIn(
                l_idx, res["per_token"], f"Layer {l_idx} missing from per_token"
            )
            self.assertIn(l_idx, res["logits"], f"Layer {l_idx} missing from logits")
            self.assertIn(
                l_idx,
                res["attention_layer"],
                f"Layer {l_idx} missing from attention_layer",
            )
            self.assertIn(
                l_idx,
                res["attention_head"],
                f"Layer {l_idx} missing from attention_head",
            )

            # Sequence count
            self.assertEqual(len(res["mean_pooled"][l_idx]), 2)
            self.assertEqual(len(res["per_token"][l_idx]), 2)
            self.assertEqual(len(res["logits"][l_idx]), 2)

            # Ensure logits retains full (seq_len, vocab_size) and is not sliced as a layer index
            for seq_logits in res["logits"][l_idx]:
                self.assertEqual(seq_logits.ndim, 2)
                self.assertEqual(seq_logits.shape[0], MAX_LEN)
                self.assertEqual(seq_logits.shape[1], VOCAB_SIZE)
                self.assertTrue(bool((seq_logits != 0).any()))

            # Ensure per_token retains full (seq_len, embedding_size)
            for seq_pt in res["per_token"][l_idx]:
                self.assertEqual(seq_pt.ndim, 2)
                self.assertEqual(seq_pt.shape[0], MAX_LEN)
                self.assertEqual(seq_pt.shape[1], EMBED_DIM)
                self.assertTrue(bool((seq_pt != 0).any()))

    def test_discard_padding_contract(self):
        """Verify that discard_padding=True strips padding tokens correctly without corrupting dimensions."""
        res = pepe.embed(
            model_name=MODEL_NAME,
            sequences=SEQUENCES,
            extract_embeddings=["mean_pooled", "per_token", "logits"],
            device="cpu",
            layers=[[LAYER]],
            discard_padding=True,
            streaming_output=False,
        )

        # seq1 has 20 AAs; seq2 has 7 AAs (unpadded)
        expected_lengths = [20, 7]

        # Mean pooled is (EMBED_DIM,) per sequence
        for vec in res["mean_pooled"][LAYER]:
            self.assertEqual(vec.shape, (EMBED_DIM,))
            self.assertTrue(bool((vec != 0).any()))

        # Per token: each sequence length must match unpadded AA count
        for i, pt in enumerate(res["per_token"][LAYER]):
            self.assertEqual(
                pt.shape,
                (expected_lengths[i], EMBED_DIM),
                f"per_token length mismatch at sequence {i}",
            )
            self.assertTrue(bool((pt != 0).any()))

        # Logits: each sequence must match unpadded AA count and full VOCAB_SIZE
        for i, logit_seq in enumerate(res["logits"][LAYER]):
            self.assertEqual(
                logit_seq.shape,
                (expected_lengths[i], VOCAB_SIZE),
                f"logits shape mismatch at sequence {i}",
            )
            self.assertTrue(bool((logit_seq != 0).any()))

    def test_split_long_sequences_reconstruction_contract(self):
        """Verify that sequence chunking & reconstruction restores correct token lengths for per_token, logits, and mean_pooled."""
        seqs = {
            "long_seq": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",  # 30 AAs
            "short_seq": "ACDEF",  # 5 AAs
        }

        # 1. With discard_padding=True
        res_discard = pepe.embed(
            model_name=MODEL_NAME,
            sequences=seqs,
            extract_embeddings=["mean_pooled", "per_token", "logits"],
            device="cpu",
            layers=[[LAYER]],
            split_long_sequences=True,
            force_split_length=10,
            split_overlap=2,
            discard_padding=True,
            streaming_output=False,
        )

        # Expected unpadded lengths: 30 and 5
        self.assertEqual(res_discard["mean_pooled"][LAYER][0].shape, (EMBED_DIM,))
        self.assertEqual(res_discard["mean_pooled"][LAYER][1].shape, (EMBED_DIM,))

        self.assertEqual(res_discard["per_token"][LAYER][0].shape, (30, EMBED_DIM))
        self.assertEqual(res_discard["per_token"][LAYER][1].shape, (5, EMBED_DIM))

        self.assertEqual(res_discard["logits"][LAYER][0].shape, (30, VOCAB_SIZE))
        self.assertEqual(res_discard["logits"][LAYER][1].shape, (5, VOCAB_SIZE))

        # 2. With discard_padding=False (includes 2 special tokens: BOS & EOS)
        res_no_discard = pepe.embed(
            model_name=MODEL_NAME,
            sequences=seqs,
            extract_embeddings=["mean_pooled", "per_token", "logits"],
            device="cpu",
            layers=[[LAYER]],
            split_long_sequences=True,
            force_split_length=10,
            split_overlap=2,
            discard_padding=False,
            streaming_output=False,
        )

        # Long sequence: 30 AAs + 2 special tokens = 32 tokens
        self.assertEqual(res_no_discard["per_token"][LAYER][0].shape, (32, EMBED_DIM))
        self.assertEqual(res_no_discard["logits"][LAYER][0].shape, (32, VOCAB_SIZE))

        # Check non-zero
        self.assertTrue(bool((res_no_discard["logits"][LAYER][0] != 0).any()))
        self.assertTrue(bool((res_no_discard["per_token"][LAYER][0] != 0).any()))

    def test_substring_pooled_contract(self):
        """Verify substring_pooled outputs correct embedding dimensions per sequence."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        fasta = os.path.join(repo_root, "src/tests/test_files/test.fasta")
        substring = os.path.join(repo_root, "src/tests/test_files/test_substring.csv")

        res = pepe.embed(
            model_name=MODEL_NAME,
            fasta_path=fasta,
            substring_path=substring,
            extract_embeddings=["substring_pooled"],
            device="cpu",
            layers=[[LAYER]],
            streaming_output=False,
        )
        self.assertIn("substring_pooled", res)
        vectors = res["substring_pooled"][LAYER]
        self.assertEqual(len(vectors), 10)
        for vec in vectors:
            self.assertEqual(vec.shape, (EMBED_DIM,))
            self.assertTrue(bool((vec != 0).any()))

    def test_vocab_size_resolution_invariants(self):
        """Verify _get_vocab_size helper resolves across diverse model attributes."""
        embedder = object.__new__(BaseEmbedder)

        # 1. Has explicit vocab_size
        embedder.vocab_size = 42
        self.assertEqual(embedder._get_vocab_size(), 42)

        # 2. Has tokenizer with vocab_size
        delattr(embedder, "vocab_size")

        class DummyTokenizer:
            vocab_size = 128

        embedder.tokenizer = DummyTokenizer()
        self.assertEqual(embedder._get_vocab_size(), 128)

        # 3. Has tokenizer with get_vocab()
        class DummyVocabTokenizer:
            def get_vocab(self):
                return {f"tok_{i}": i for i in range(64)}

        embedder.tokenizer = DummyVocabTokenizer()
        self.assertEqual(embedder._get_vocab_size(), 64)

        # 4. Has alphabet with all_toks
        delattr(embedder, "tokenizer")

        class DummyAlphabet:
            all_toks = ["<pad>", "A", "C", "D"]

        embedder.alphabet = DummyAlphabet()
        self.assertEqual(embedder._get_vocab_size(), 4)

        # 5. Fallback
        delattr(embedder, "alphabet")
        self.assertEqual(embedder._get_vocab_size(), 0)


if __name__ == "__main__":
    unittest.main()
