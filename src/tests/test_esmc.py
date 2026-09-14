import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, os.path.abspath("src"))

from pepe.embedders.huggingface_embedder import ESMCEmbedder
from pepe.model_selecter import select_model


class TestESMCModelSelection(unittest.TestCase):
    def test_select_esmc_returns_esmc_embedder(self):
        mock_config = MagicMock()
        mock_config.model_type = "esmc"

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            embedder_cls = select_model("biohub/ESMC-300M")

        from pepe.embedders.huggingface_embedder import ESMCEmbedder

        self.assertIs(embedder_cls, ESMCEmbedder)

    def test_select_esmc_missing_fork_error(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=ValueError("Unrecognized model_type esmc"),
        ):
            with self.assertRaises(ValueError) as ctx:
                select_model("biohub/ESMC-300M")

        self.assertIn("Biohub/transformers", str(ctx.exception))

    def test_select_esmc_inspect_logits_available(self):
        import io

        import pepe.model_selecter

        mock_config = MagicMock()
        mock_config.model_type = "esmc"
        mock_config.num_hidden_layers = 30
        mock_config.num_attention_heads = 15
        mock_config.hidden_size = 960
        mock_config.vocab_size = 64
        mock_config.max_position_embeddings = 2048
        mock_config.n_positions = 2048

        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 2048
        mock_tokenizer.get_vocab.return_value = {"A": 4, "C": 5}

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            captured = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = captured
                pepe.model_selecter.report_model("biohub/ESMC-300M")
            finally:
                sys.stdout = old_stdout
            output = captured.getvalue()
            self.assertIn("Logits:          yes", output)

    def test_esmc_compute_outputs_returns_logits(self):
        import torch

        from pepe.embedders.huggingface_embedder import ESMCEmbedder

        embedder = object.__new__(ESMCEmbedder)
        embedder.layers = [30]
        embedder.precision = "32"
        embedder.return_contacts = False

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 10, 64)
        mock_outputs.hidden_states = [torch.randn(2, 10, 960)] * 31
        mock_model.return_value = mock_outputs

        toks = torch.zeros((2, 10), dtype=torch.long)
        attention_mask = torch.ones((2, 10), dtype=torch.long)

        logits, representations, attention_matrices = embedder._compute_outputs(
            model=mock_model,
            toks=toks,
            attention_mask=attention_mask,
            return_embeddings=True,
            return_contacts=False,
            return_logits=True,
        )

        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape, (2, 10, 64))
        self.assertIn(30, representations)
        self.assertEqual(representations[30].shape, (2, 10, 960))
        self.assertIsNone(attention_matrices)

    def test_esmc_preallocation_shape(self):
        from pepe.embedders.huggingface_embedder import ESMCEmbedder

        embedder = object.__new__(ESMCEmbedder)
        embedder.layers = [30]
        embedder.num_sequences = 5
        embedder.max_input_length = 25
        embedder.vocab_size = 64
        embedder.embedding_size = 960
        embedder.num_heads = 15
        embedder.flatten = False
        embedder.output_path = "/tmp"

        embedder._set_output_objects()
        self.assertEqual(embedder.logits["shape"], (5, 25, 64))

        embedder.flatten = True
        embedder._set_output_objects()
        self.assertEqual(embedder.logits["shape"], (5, 25 * 64))

    def test_safe_compute_oom_retry_with_dict_representations_and_logits(self):
        import torch

        from pepe.embedders.huggingface_embedder import ESMCEmbedder

        embedder = object.__new__(ESMCEmbedder)
        embedder.model = MagicMock()
        embedder.layers = [30]
        embedder.return_embeddings = True
        embedder.return_contacts = False
        embedder.return_logits = True

        call_count = 0

        def fake_compute_outputs(
            model,
            toks,
            attention_mask,
            return_embeddings,
            return_contacts,
            return_logits,
        ):
            nonlocal call_count
            call_count += 1
            B = toks.size(0)
            if call_count == 1:
                raise torch.OutOfMemoryError("CUDA OOM")
            logits = torch.ones((B, toks.size(1), 64))
            reps = {30: torch.ones((B, toks.size(1), 960))}
            return logits, reps, None

        embedder._compute_outputs = fake_compute_outputs

        toks = torch.zeros((4, 10), dtype=torch.long)
        attention_mask = torch.ones((4, 10), dtype=torch.long)

        logits, reps, attn = embedder._safe_compute(toks, attention_mask)
        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape, (4, 10, 64))
        self.assertIn(30, reps)
        self.assertEqual(reps[30].shape, (4, 10, 960))
        self.assertIsNone(attn)

    def test_reconstruct_chunks_discard_padding_and_flatten_logits(self):
        emb = ESMCEmbedder.__new__(ESMCEmbedder)
        emb.chunks_mapping = {"seq": ["seq_chunk_0", "seq_chunk_1"]}
        emb.streaming_output = False
        emb.layers = [1]
        emb.sequence_labels = ["seq_chunk_0", "seq_chunk_1"]
        emb.output_types = ["logits", "per_token"]
        emb.split_overlap = 2
        emb.chunk_payload_lengths = {"seq_chunk_0": 5, "seq_chunk_1": 5}
        emb.discard_padding = True
        emb.flatten = True

        vocab_size = 64
        hidden_size = 16
        c0_logits = torch.randn(5, vocab_size)
        c1_logits = torch.randn(5, vocab_size)
        c0_reps = torch.randn(5, hidden_size)
        c1_reps = torch.randn(5, hidden_size)

        emb.logits = {"output_data": {1: [c0_logits, c1_logits]}}
        emb.per_token = {"output_data": {1: [c0_reps, c1_reps]}}

        emb._reconstruct_chunks()

        self.assertEqual(emb.sequence_labels, ["seq"])
        reconstructed_logits = emb.logits["output_data"][1][0]
        reconstructed_reps = emb.per_token["output_data"][1][0]
        self.assertEqual(reconstructed_logits.shape, (8 * vocab_size,))
        self.assertEqual(reconstructed_reps.shape, (8 * hidden_size,))

        expected_logits = torch.cat([c0_logits, c1_logits[2:]], dim=0).flatten()
        self.assertTrue(torch.equal(reconstructed_logits, expected_logits))
        expected_reps = torch.cat([c0_reps, c1_reps[2:]], dim=0).flatten()
        self.assertTrue(torch.equal(reconstructed_reps, expected_reps))

    def test_custom_model_wrapper_vocab_size_consistency(self):
        from pepe.embedders.custom_embedder import CustomModelWrapper

        # Default fallback is 21
        wrapper = CustomModelWrapper({}, {})
        self.assertEqual(wrapper.vocab_size, 21)
        out = wrapper(torch.zeros(2, 5, dtype=torch.long))
        self.assertEqual(out.logits.shape, (2, 5, 21))

        # Configured vocab_size overrides default
        wrapper_custom = CustomModelWrapper({}, {"vocab_size": 35})
        self.assertEqual(wrapper_custom.vocab_size, 35)
        out_custom = wrapper_custom(torch.zeros(2, 5, dtype=torch.long))
        self.assertEqual(out_custom.logits.shape, (2, 5, 35))


@unittest.skipUnless(
    os.environ.get("ESMC_TEST") == "1",
    "Set ESMC_TEST=1 to run ESMC integration test (requires Biohub transformers fork)",
)
class TestESMCIntegration(unittest.TestCase):
    def test_esmc_mean_pooled(self):
        import pepe

        sequences = {
            "seq1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        }
        results = pepe.embed(
            model_name="biohub/ESMC-300M",
            sequences=sequences,
            extract_embeddings=["mean_pooled"],
            layers=[[-1]],
            device="cpu",
        )

        self.assertIn("mean_pooled", results)
        layer_outputs = results["mean_pooled"]
        layer_key = next(iter(layer_outputs))
        self.assertEqual(len(layer_outputs[layer_key]), 1)
        self.assertEqual(layer_outputs[layer_key][0].shape[0], 960)

    def test_esmc_logits(self):
        import tempfile

        import numpy as np

        import pepe

        sequences = {
            "seq1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        }
        in_memory = pepe.embed(
            model_name="biohub/ESMC-300M",
            sequences=sequences,
            extract_embeddings=["logits"],
            layers=[[-1]],
            device="cpu",
            streaming_output=False,
        )

        self.assertIn("logits", in_memory)
        layer_key = next(iter(in_memory["logits"]))
        in_mem_arr = in_memory["logits"][layer_key][0]
        self.assertEqual(in_mem_arr.ndim, 2)
        self.assertEqual(in_mem_arr.shape[1], 64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pepe.embed(
                model_name="biohub/ESMC-300M",
                sequences=sequences,
                extract_embeddings=["logits"],
                layers=[[-1]],
                device="cpu",
                output_path=tmp_dir,
                streaming_output=True,
            )
            # Find and load the saved memmap file
            pattern = os.path.join(tmp_dir, "*", "logits", "*.npy")
            import glob

            files = glob.glob(pattern)
            self.assertTrue(len(files) > 0, "No streaming logits memmap file found")
            streamed_arr = np.load(files[0])
            self.assertEqual(streamed_arr.ndim, 3)
            self.assertEqual(streamed_arr.shape[-1], 64)
            # Verify shapes and non-trivial values
            self.assertEqual(streamed_arr.shape[0], 1)
            np.testing.assert_allclose(
                in_mem_arr,
                streamed_arr[0, : in_mem_arr.shape[0], :],
                rtol=1e-4,
                atol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
