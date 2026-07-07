"""Regression tests: streaming disk output must match in-memory results."""

import glob
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath("src"))

import pepe
from pepe.embedders.base_embedder import BaseEmbedder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.usefixtures("esm2_model_cache"),
]

MODEL_NAME = "esm2_t6_8M_UR50D"
LAYER = 6
NUM_HEADS = 8
EXPERIMENT_NAME = "roundtrip"

SHORT_SEQUENCES = {
    "seq1": (
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    ),
    "seq2": (
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK"
    ),
}

OUTPUT_TYPES = [
    "mean_pooled",
    "per_token",
    "attention_head",
    "attention_layer",
    "attention_model",
    "logits",
]


def _stack_tensors(items):
    if isinstance(items[0], torch.Tensor):
        return torch.stack(items).numpy()
    return np.stack(items)


def in_memory_to_numpy(output_data, output_type, layer=LAYER, num_heads=NUM_HEADS):
    if output_type == "attention_model":
        return _stack_tensors(output_data)
    if output_type == "attention_head":
        return {
            head: _stack_tensors(output_data[layer][head]) for head in range(num_heads)
        }
    return _stack_tensors(output_data[layer])


def _output_dir(output_path, output_type):
    return os.path.join(output_path, MODEL_NAME, output_type)


def load_streaming_array(output_path, output_type, layer=LAYER, head=None):
    output_dir = _output_dir(output_path, output_type)
    if output_type == "attention_model":
        pattern = os.path.join(
            output_dir, f"{EXPERIMENT_NAME}_{MODEL_NAME}_attention_model.npy"
        )
        files = (
            [pattern]
            if os.path.exists(pattern)
            else glob.glob(os.path.join(output_dir, "*.npy"))
        )
    elif output_type == "attention_head":
        pattern = os.path.join(
            output_dir,
            f"{EXPERIMENT_NAME}_{MODEL_NAME}_attention_head_layer_{layer}_head_{head + 1}.npy",
        )
        files = (
            [pattern]
            if os.path.exists(pattern)
            else glob.glob(
                os.path.join(output_dir, f"*_layer_{layer}_head_{head + 1}.npy")
            )
        )
    else:
        pattern = os.path.join(
            output_dir,
            f"{EXPERIMENT_NAME}_{MODEL_NAME}_{output_type}_layer_{layer}.npy",
        )
        files = (
            [pattern]
            if os.path.exists(pattern)
            else glob.glob(
                os.path.join(output_dir, f"*_{output_type}_layer_{layer}.npy")
            )
        )
    assert len(files) == 1, f"Expected one file for {output_type}, found {files}"
    return np.load(files[0])


def load_streaming_output(output_path, output_type, layer=LAYER, num_heads=NUM_HEADS):
    if output_type == "attention_head":
        return {
            head: load_streaming_array(output_path, output_type, layer, head)
            for head in range(num_heads)
        }
    return load_streaming_array(output_path, output_type, layer)


def _assert_allclose(reference, streaming, output_type):
    if output_type == "attention_head":
        for head in reference:
            np.testing.assert_allclose(
                reference[head], streaming[head], rtol=1e-4, atol=1e-4
            )
            assert np.any(streaming[head] != 0), (
                f"{output_type} head {head} is all zeros"
            )
    else:
        np.testing.assert_allclose(reference, streaming, rtol=1e-4, atol=1e-4)
        assert np.any(streaming != 0), f"{output_type} is all zeros"


def test_streaming_matches_in_memory(tmp_path):
    common_kwargs = {
        "model_name": MODEL_NAME,
        "sequences": SHORT_SEQUENCES,
        "extract_embeddings": OUTPUT_TYPES,
        "device": "cpu",
        "experiment_name": EXPERIMENT_NAME,
        "layers": [[-1]],
    }

    in_memory = pepe.embed(streaming_output=False, **common_kwargs)
    for output_type in OUTPUT_TYPES:
        assert output_type in in_memory

    pepe.embed(
        streaming_output=True,
        output_path=str(tmp_path),
        **common_kwargs,
    )

    for output_type in OUTPUT_TYPES:
        reference = in_memory_to_numpy(in_memory[output_type], output_type)
        streaming = load_streaming_output(str(tmp_path), output_type)
        _assert_allclose(reference, streaming, output_type)


def test_get_substring_positions_missing_raises():
    embedder = object.__new__(BaseEmbedder)
    embedder.sequences = {"seq1": "ACDEFG"}
    embedder.substring_dict = {}
    with pytest.raises(SystemExit, match="No matching substring found for seq1"):
        embedder.get_substring_positions("seq1", special_tokens=1)
