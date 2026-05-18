import unittest
from unittest.mock import MagicMock, patch
import torch
import os
import sys
from types import SimpleNamespace

# Add src to sys.path
sys.path.insert(0, os.path.abspath("src"))

from pepe.embedders.base_embedder import BaseEmbedder

class TestDeviceLogic(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model_name="esm2_t6_8M_UR50D",
            fasta_path="dummy.fasta",
            output_path="tmp_out",
            extract_embeddings=["mean_pooled"],
            layers=[[-1]],
            batch_size=1024,
            device="cuda",
            precision="32",
            streaming_output=False,
            discard_padding=False,
            max_input_length="max_length",
            substring_path=None,
            num_workers=1,
            disable_special_tokens=False,
            experiment_name=None,
            context=0,
            flatten=False,
            flush_batches_after=128
        )

    @patch("torch.cuda.is_available", return_value=True)
    def test_base_device_logic_gpu_available(self, mock_cuda):
        # Mocking file system calls and substring loading
        with patch("os.path.exists", return_value=True), \
             patch("os.makedirs"), \
             patch.object(BaseEmbedder, "_load_substrings", return_value=None):
            
            # 1. GPU requested and available
            self.args.device = "cuda"
            base = BaseEmbedder(self.args)
            self.assertEqual(base.device.type, "cuda")
            
            # 2. CPU specifically requested even if GPU available
            self.args.device = "cpu"
            base = BaseEmbedder(self.args)
            self.assertEqual(base.device.type, "cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_base_device_logic_gpu_specific(self, mock_cuda):
        with patch("os.path.exists", return_value=True), \
             patch("os.makedirs"), \
             patch.object(BaseEmbedder, "_load_substrings", return_value=None):
            
            # GPU cuda:1 requested and available
            self.args.device = "cuda:1"
            base = BaseEmbedder(self.args)
            self.assertEqual(base.device.type, "cuda")
            self.assertEqual(base.device.index, 1)

    @patch("torch.cuda.is_available", return_value=False)
    def test_base_device_logic_gpu_not_available(self, mock_cuda):
        with patch("os.path.exists", return_value=True), \
             patch("os.makedirs"), \
             patch.object(BaseEmbedder, "_load_substrings", return_value=None):
            
            # GPU requested but NOT available -> should fallback to CPU
            self.args.device = "cuda"
            base = BaseEmbedder(self.args)
            self.assertEqual(base.device.type, "cpu")

    def test_get_bracket_type_robustness(self):
        from pepe.utils import get_bracket_type
        
        # Test unknown bracket (should default to square)
        mock_tokenizer = MagicMock()
        mock_tokenizer.all_special_tokens = ["__special__"]
        del mock_tokenizer.special_tokens_map
        
        self.assertEqual(get_bracket_type(mock_tokenizer), "square")
        
        # Test no special tokens (should default to square)
        mock_tokenizer.all_special_tokens = []
        self.assertEqual(get_bracket_type(mock_tokenizer), "square")

        # Test angle bracket
        mock_tokenizer.all_special_tokens = ["<special>"]
        self.assertEqual(get_bracket_type(mock_tokenizer), "angle")

    def test_subclass_device_type_usage(self):
        # This test verifies that we are using .type == "cuda" in subclasses
        # by checking the source code of the relevant files
        from pepe.embedders.esm_embedder import ESMEmbedder
        import inspect
        source = inspect.getsource(ESMEmbedder._initialize_model)
        self.assertIn('self.device.type == "cuda"', source)
        
        from pepe.embedders.huggingface_embedder import Antiberta2Embedder
        source = inspect.getsource(Antiberta2Embedder._initialize_model)
        self.assertIn('self.device.type == "cuda"', source)

if __name__ == "__main__":
    unittest.main()
