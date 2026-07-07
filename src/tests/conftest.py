"""Shared pytest fixtures for PEPE tests."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath("src"))

ESM2_MODEL_NAME = "esm2_t6_8M_UR50D"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests that download/load real models"
    )
    config.addinivalue_line("markers", "slow: tests that take significant time")


@pytest.fixture(scope="session")
def esm2_model_cache():
    """Load ESM2-8M once per session and patch embedder to reuse it."""
    from pepe.embedders.huggingface_embedder import (
        ESM2Embedder,
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
    bundle = {
        "model": model,
        "tokenizer": tokenizer,
        "num_heads": config.num_attention_heads,
        "num_layers": config.num_hidden_layers,
        "embedding_size": config.hidden_size,
    }

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
    yield bundle
    ESM2Embedder._initialize_model = original
