import os
import sys
import unittest
import torch
import numpy as np
import subprocess
import shutil

# Add src to sys.path
sys.path.append(os.path.abspath("src"))

from pepe.model_selecter import select_model

class TestSplittingIntegrated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a long sequence for testing (Antiberta2 limit is 256)
        cls.test_dir = "src/tests/splitting_test_assets"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.fasta_path = os.path.join(cls.test_dir, "long.fasta")
        cls.short_fasta_path = os.path.join(cls.test_dir, "short.fasta")
        
        # 300 AAs -> 300 + 2 special tokens = 302 tokens ( > 256)
        long_seq = "M" * 300
        with open(cls.fasta_path, "w") as f:
            f.write(f">long_prot\n{long_seq}\n")
            
        short_seq = "M" * 50
        with open(cls.short_fasta_path, "w") as f:
            f.write(f">short_prot\n{short_seq}\n")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_library_reconstruction(self):
        """Test that the library correctly reconstructs embeddings in memory."""
        from pepe.model_selecter import select_model
        
        class DummyArgs:
            def __init__(self, fasta, split=True):
                self.fasta_path = fasta
                self.output_path = "tmp_out"
                self.experiment_name = "test"
                self.model_name = "alchemab/antiberta2-cssp"
                self.tokenizer_from = None
                self.disable_special_tokens = False
                self.substring_path = None
                self.context = 0
                self.layers = [[-1]]
                self.batch_size = 1024
                self.max_input_length = "max_input_length"
                self.device = "cpu"
                self.extract_embeddings = ["per_token", "mean_pooled"]
                self.discard_padding = False
                self.flatten = False
                self.streaming_output = False
                self.precision = "32"
                self.flush_batches_after = 128
                self.split_long_sequences = split
                self.split_overlap = 50
                self.force_split_length = None
                self.num_workers = 1

        # Test long sequence splitting
        args = DummyArgs(self.fasta_path, split=True)
        EmbedderClass = select_model(args.model_name)
        embedder = EmbedderClass(args)
        
        self.assertTrue(len(embedder.chunks_mapping) > 0)
        embedder.embed()
        
        # Verify reconstruction
        self.assertEqual(len(embedder.sequence_labels), 1)
        self.assertEqual(embedder.sequence_labels[0], "long_prot")
        
        per_token = embedder.per_token["output_data"][embedder.layers[0]][0]
        
        # Calculate expected length: sequence length + special tokens count
        special_tokens_count = len(embedder.tokenizer.encode("", add_special_tokens=True))
        expected_len = 300 + special_tokens_count
        self.assertEqual(per_token.shape[0], expected_len)



    def test_force_split_length(self):
        """Test that the library overrides model defaults when force_split_length is supplied."""
        from pepe.model_selecter import select_model
        
        class DummyArgs:
            def __init__(self, fasta, split=True):
                self.fasta_path = fasta
                self.output_path = "tmp_out"
                self.experiment_name = "test"
                self.model_name = "alchemab/antiberta2-cssp"
                self.tokenizer_from = None
                self.disable_special_tokens = False
                self.substring_path = None
                self.context = 0
                self.layers = [[-1]]
                self.batch_size = 1024
                self.max_input_length = "max_input_length"
                self.device = "cpu"
                self.extract_embeddings = ["per_token", "mean_pooled"]
                self.discard_padding = False
                self.flatten = False
                self.streaming_output = False
                self.precision = "32"
                self.flush_batches_after = 128
                self.split_long_sequences = split
                self.split_overlap = 50
                self.force_split_length = 150
                self.num_workers = 1

        args = DummyArgs(self.fasta_path, split=True)
        EmbedderClass = select_model(args.model_name)
        embedder = EmbedderClass(args)
        
        self.assertTrue(len(embedder.chunks_mapping) > 0)
        
        # Since force_split_length is 150, and our protein is 300 characters, it should be chunked into more than 2 pieces!
        self.assertTrue(len(embedder.chunks_mapping['long_prot']) > 2)

        embedder.embed()
        
        # Verify reconstruction
        self.assertEqual(len(embedder.sequence_labels), 1)
        self.assertEqual(embedder.sequence_labels[0], "long_prot")
        
        per_token = embedder.per_token["output_data"][embedder.layers[0]][0]
        
        # Calculate expected length: sequence length + special tokens count
        special_tokens_count = len(embedder.tokenizer.encode("", add_special_tokens=True))
        expected_len = 300 + special_tokens_count
        self.assertEqual(per_token.shape[0], expected_len)



    def test_cli_reconstruction_no_streaming(self):
        """Test that CLI with streaming_output=False reconstructs files on disk."""
        out_dir = os.path.join(self.test_dir, "cli_no_streaming")
        cmd = [
            sys.executable, "-m", "pepe",
            "--model_name", "alchemab/antiberta2-cssp",
            "--fasta_path", self.fasta_path,
            "--output_path", out_dir,
            "--split_long_sequences",
            "--split_overlap", "50",
            "--device", "cpu",
            "--streaming_output", "False",
            "--extract_embeddings", "per_token"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        subprocess.run(cmd, env=env, check=True)
        
        # Index should show original labels
        idx_path = os.path.join(out_dir, "antiberta2-cssp", "long_idx.csv")
        with open(idx_path, "r") as f:
            lines = f.readlines()
        
        # header + 1 sequence
        self.assertEqual(len(lines), 2)
        self.assertIn("long_prot", lines[1])
        
        # Check tensor shape on disk (non-streaming uses .npy)
        npy_path = os.path.join(out_dir, "antiberta2-cssp", "per_token", "long_antiberta2-cssp_per_token_layer_16.npy")
        data = np.load(npy_path)
        self.assertEqual(data.shape[0], 1)
        # Check the sequence length (reconstructed)
        # For AntiBERTa2, 300 AA + [CLS] + [SEP] = 302
        self.assertEqual(data[0].shape[0], 302)


    def test_cli_streaming_remains_chunked(self):
        """Test that CLI with streaming_output=True (default) exports chunks."""
        out_dir = os.path.join(self.test_dir, "cli_streaming")
        cmd = [
            sys.executable, "-m", "pepe",
            "--model_name", "alchemab/antiberta2-cssp",
            "--fasta_path", self.fasta_path,
            "--output_path", out_dir,
            "--split_long_sequences",
            "--split_overlap", "50",
            "--device", "cpu",
            "--streaming_output", "True",
            "--extract_embeddings", "per_token"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        subprocess.run(cmd, env=env, check=True)
        
        # Index should show chunks
        idx_path = os.path.join(out_dir, "antiberta2-cssp", "long_idx.csv")
        with open(idx_path, "r") as f:
            content = f.read()
        
        self.assertIn("long_prot_chunk_0", content)
        self.assertIn("long_prot_chunk_1", content)

if __name__ == "__main__":
    unittest.main()
