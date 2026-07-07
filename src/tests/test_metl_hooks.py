import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("src"))

import torch
import torch.nn as nn

from pepe.embedders.metl_embedder import METLEmbedder


class _MarkBlock(nn.Module):
    """Transformer-block stand-in that adds a distinct scalar per block.

    Chaining the blocks yields a strictly increasing cumulative value, so the
    tensor captured by a forward hook uniquely identifies which block produced
    it (guards against the layer-0 -> layers[-1] aliasing bug).
    """

    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, x):
        return x + self.increment


class _MarkNorm(nn.Module):
    """Final-norm stand-in that scales its input by 10 (a recognizable marker)."""

    def forward(self, x):
        return x * 10.0


class _FakeEncoder(nn.Module):
    """Minimal batch_first TransformerEncoder analogue for METL."""

    def __init__(self, num_layers, with_norm):
        super().__init__()
        self.layers = nn.ModuleList([_MarkBlock(i + 1) for i in range(num_layers)])
        self.norm = _MarkNorm() if with_norm else None

    def forward(self, src):
        x = src
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class _FakeMETLInner(nn.Module):
    def __init__(self, num_layers, with_norm):
        super().__init__()
        self.tr_encoder = _FakeEncoder(num_layers, with_norm)


class _FakeMETLModel(nn.Module):
    def __init__(self, num_layers, with_norm):
        super().__init__()
        self.model = _FakeMETLInner(num_layers, with_norm)


def _make_embedder(num_layers, layers, with_norm=True):
    """Build a METLEmbedder instance without running the real __init__."""
    emb = METLEmbedder.__new__(METLEmbedder)
    emb.model = _FakeMETLModel(num_layers, with_norm)
    emb.num_layers = num_layers
    emb.layers = layers
    emb._repr_outputs = {}
    emb._hook_handles = []
    return emb


class TestMETLLoadLayers(unittest.TestCase):
    """Layer-selection semantics must match the HuggingFace convention."""

    def test_default_selects_all_transformer_layers(self):
        emb = METLEmbedder.__new__(METLEmbedder)
        emb.num_layers = 4
        # Regression for finding #3: default must be every layer, not just the last.
        self.assertEqual(emb._load_layers(None), [1, 2, 3, 4])

    def test_negative_and_zero_indexing(self):
        emb = METLEmbedder.__new__(METLEmbedder)
        emb.num_layers = 4
        self.assertEqual(emb._load_layers([-1]), [4])
        self.assertEqual(emb._load_layers([0]), [0])
        self.assertEqual(emb._load_layers([]), [4])


class TestMETLRegisterReprHooks(unittest.TestCase):
    """Each requested layer must capture the correct module's output."""

    def _run(self, emb):
        src = torch.zeros((2, 3, 5))  # (batch, seq, embed)
        emb._register_repr_hooks()
        emb.model.model.tr_encoder(src)
        return src

    def test_layer_zero_captures_encoder_input(self):
        # Regression for finding #2: layer 0 is the input embeddings, not layers[-1].
        emb = _make_embedder(num_layers=4, layers=[0])
        src = self._run(emb)
        captured = emb._repr_outputs[0]
        self.assertEqual(tuple(captured.shape), (2, 3, 5))
        self.assertTrue(torch.equal(captured, src))  # all zeros == the raw input

    def test_middle_layer_captures_its_own_block(self):
        emb = _make_embedder(num_layers=4, layers=[2])
        self._run(emb)
        # After blocks 0 and 1: 0 + 1 + 2 = 3 everywhere. Distinct from any other
        # layer's value, so it cannot be silently aliased to the last block.
        self.assertTrue(torch.all(emb._repr_outputs[2] == 3.0))

    def test_final_layer_uses_norm_when_present(self):
        emb = _make_embedder(num_layers=4, layers=[4], with_norm=True)
        self._run(emb)
        # Cumulative through all 4 blocks: 1+2+3+4 = 10, then norm scales by 10 -> 100.
        self.assertTrue(torch.all(emb._repr_outputs[4] == 100.0))

    def test_final_layer_falls_back_when_norm_is_none(self):
        # Regression: norm-less METL models must not crash and must use the last block.
        emb = _make_embedder(num_layers=4, layers=[4], with_norm=False)
        self._run(emb)
        self.assertTrue(torch.all(emb._repr_outputs[4] == 10.0))  # 1+2+3+4, no scaling

    def test_all_layers_captured_with_distinct_values(self):
        emb = _make_embedder(num_layers=3, layers=[0, 1, 2, 3], with_norm=True)
        self._run(emb)
        self.assertEqual(set(emb._repr_outputs), {0, 1, 2, 3})
        # 0=input(0), 1=block0(1), 2=block1(1+2=3), 3=norm((1+2+3)*10=60)
        self.assertTrue(torch.all(emb._repr_outputs[0] == 0.0))
        self.assertTrue(torch.all(emb._repr_outputs[1] == 1.0))
        self.assertTrue(torch.all(emb._repr_outputs[2] == 3.0))
        self.assertTrue(torch.all(emb._repr_outputs[3] == 60.0))


if __name__ == "__main__":
    unittest.main()
