import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("src"))

from pepe.embedders.huggingface_embedder import HuggingfaceEmbedder


class _StubEmbedder(HuggingfaceEmbedder):
    def __init__(self, num_layers=30):
        self.num_layers = num_layers


class TestLoadLayersDefault(unittest.TestCase):
    def test_none_returns_all_layers(self):
        embedder = _StubEmbedder(num_layers=6)
        self.assertEqual(embedder._load_layers(None), [1, 2, 3, 4, 5, 6])

    def test_empty_returns_last_layer(self):
        embedder = _StubEmbedder(num_layers=30)
        self.assertEqual(embedder._load_layers([]), [30])

    def test_omitted_normalized_to_last_layer(self):
        embedder = _StubEmbedder(num_layers=30)
        self.assertEqual(embedder._load_layers([-1]), [30])


if __name__ == "__main__":
    unittest.main()
