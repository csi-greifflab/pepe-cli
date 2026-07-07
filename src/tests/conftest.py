"""Shared pytest fixtures for PEPE tests."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath("src"))

ESM2_MODEL_NAME = "esm2_t6_8M_UR50D"
ESM1_MODEL_NAME = "esm1_t6_43M_UR50S"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests that download/load real models"
    )
    config.addinivalue_line("markers", "slow: tests that take significant time")


@pytest.fixture(scope="session")
def esm2_model_cache():
    """Load ESM2-8M once per session for integration tests."""
    from pepe.embedders.huggingface_embedder import (
        _import_transformers,
        _resolve_esm2_model_link,
    )

    model_link = _resolve_esm2_model_link(ESM2_MODEL_NAME)
    device = torch.device("cpu")

    (
        _T5EncoderModel,
        _T5Tokenizer,
        _RoFormerTokenizer,
        _RoFormerModel,
        _RoFormerSinusoidalPositionalEmbedding,
        _AutoModel,
        AutoTokenizer,
        _AutoModelForCausalLM,
        AutoModelForMaskedLM,
    ) = _import_transformers()

    tokenizer = AutoTokenizer.from_pretrained(model_link)
    # MaskedLM + eager attention covers logits, attention, and embedding outputs.
    model = AutoModelForMaskedLM.from_pretrained(
        model_link, attn_implementation="eager"
    ).to(device)
    model.eval()

    config = model.config
    yield {
        "model": model,
        "tokenizer": tokenizer,
        "num_heads": config.num_attention_heads,
        "num_layers": config.num_hidden_layers,
        "embedding_size": config.hidden_size,
    }


@pytest.fixture(autouse=True)
def _esm2_model_cache_patch(request):
    """Patch ESM2 only for tests that request esm2_model_cache."""
    if "esm2_model_cache" not in request.fixturenames:
        yield
        return

    from pepe.embedders.huggingface_embedder import ESM2Embedder

    bundle = request.getfixturevalue("esm2_model_cache")
    original = ESM2Embedder._initialize_model

    def _cached_initialize(self, model_name):
        return (
            bundle["model"],
            bundle["tokenizer"],
            bundle["num_heads"],
            bundle["num_layers"],
            bundle["embedding_size"],
        )

    ESM2Embedder._initialize_model = _cached_initialize
    yield
    ESM2Embedder._initialize_model = original


@pytest.fixture(scope="session")
def esm1_model_cache():
    """Load ESM-1 43M once per session and patch embedder to reuse it."""
    import importlib.util

    if importlib.util.find_spec("esm") is None:
        pytest.skip(
            "fair-esm is not installed (pip install fair-esm); "
            "ESM-1 integration tests require the esm package"
        )

    from esm import pretrained

    from pepe.embedders.esm_embedder import ESMEmbedder

    model, alphabet = pretrained.load_model_and_alphabet_hub(ESM1_MODEL_NAME)
    model.eval()
    model.prepend_bos = True
    model.append_eos = False  # ESM-1 does not use EOS by default

    num_heads = model.layers[0].self_attn.num_heads
    num_layers = len(model.layers)
    embedding_size = model.embed_tokens.embedding_dim

    bundle = {
        "model": model,
        "alphabet": alphabet,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "embedding_size": embedding_size,
        "prepend_bos": model.prepend_bos,
        "append_eos": model.append_eos,
    }

    original = ESMEmbedder._initialize_model

    def _cached_initialize(self, model_name):
        return (
            bundle["model"],
            bundle["alphabet"],
            bundle["num_heads"],
            bundle["num_layers"],
            bundle["embedding_size"],
            bundle["prepend_bos"],
            bundle["append_eos"],
        )

    ESMEmbedder._initialize_model = _cached_initialize
    yield bundle
    ESMEmbedder._initialize_model = original
