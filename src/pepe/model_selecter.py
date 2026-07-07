from pepe.embedders.custom_embedder import CustomEmbedder
from pepe.model_errors import translate_hf_config_error

import os


def _get_esm_embedder():
    """Lazy import of ESM embedder to avoid loading heavy dependencies."""
    from pepe.embedders.esm_embedder import ESMEmbedder

    return ESMEmbedder


def _get_esm2_embedder():
    """Lazy import of ESM-2 embedder (HuggingFace transformers)."""
    from pepe.embedders.huggingface_embedder import ESM2Embedder

    return ESM2Embedder


def _get_esmc_embedder():
    """Lazy import of ESMC embedder (Biohub transformers fork)."""
    from pepe.embedders.huggingface_embedder import ESMCEmbedder

    return ESMCEmbedder


def _get_huggingface_embedders():
    """Lazy import of HuggingFace embedders to avoid loading heavy dependencies."""
    from pepe.embedders.huggingface_embedder import T5Embedder, Antiberta2Embedder

    return T5Embedder, Antiberta2Embedder


def _get_generic_hf_embedder():
    """Lazy import of the generic AutoModel-based HuggingFace fallback embedder."""
    from pepe.embedders.huggingface_embedder import GenericHuggingFaceEmbedder

    return GenericHuggingFaceEmbedder


def select_model(model_name, trust_remote_code=False):
    # 1. Local checkpoints / directories take precedence over any name heuristic,
    #    so a local file or folder is never mistaken for a HuggingFace repo id or a
    #    bare weight name (e.g. a directory called "my_esm2_run/").
    if (
        model_name.endswith(".pt")
        or model_name.endswith(".pth")
        or model_name.startswith("custom:")
        or (
            os.path.exists(model_name)
            and (os.path.isfile(model_name) or os.path.isdir(model_name))
        )
    ):
        return CustomEmbedder

    # 2. Anything shaped like a HuggingFace repo id (username/model-name) is
    #    dispatched by inspecting its config, not its name. This is the primary
    #    signal: a fine-tune whose slug says "esm2" but is really a BERT no longer
    #    gets mis-routed.
    if "/" in model_name:
        return _select_hf_model(model_name, trust_remote_code=trust_remote_code)

    # 3. Bare weight names (no slash) are fair-esm / facebook ESM weight sets, which
    #    have no downloadable config to inspect. Match on the name for these only.
    if "esm2" in model_name.lower():
        return _get_esm2_embedder()
    if "esm1" in model_name.lower():
        return _get_esm_embedder()

    raise ValueError(
        f"Model {model_name} not supported. Pass a HuggingFace repo id "
        f"(username/model-name), a local .pt/.pth path or directory, or a bare ESM "
        f"weight name (e.g. esm2_t6_8M_UR50D)."
    )


def _select_hf_model(model_name, trust_remote_code=False):
    """Dispatch a HuggingFace repo id to an embedder by inspecting its config.

    Known architectures get their specialized embedder; everything else (BERT-like
    encoders and any architecture without a dedicated embedder) falls back to the
    generic AutoModel-based ``GenericHuggingFaceEmbedder``. The forward pass is
    architecture-agnostic, so standard encoders such as ProtBert, AntiBERTy and
    IgBert work through the fallback with no engine changes.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except Exception as e:
        translate_hf_config_error(
            model_name, e, trust_remote_code=trust_remote_code
        )

    model_type = (getattr(config, "model_type", "") or "").lower()

    if model_type in ["t5", "mt5"]:
        T5Embedder, Antiberta2Embedder = _get_huggingface_embedders()
        return T5Embedder
    if model_type in ["roformer"]:
        T5Embedder, Antiberta2Embedder = _get_huggingface_embedders()
        return Antiberta2Embedder
    if model_type in ["esm"]:
        return _get_esm2_embedder()
    if model_type in ["esmc"]:
        return _get_esmc_embedder()

    # Fallback: BERT-like and any other architecture route to the generic embedder
    # instead of raising.
    import logging

    logging.getLogger("pepe.model_selecter").info(
        f"No specialized embedder for architecture '{model_type or 'unknown'}'; "
        f"using the generic HuggingFace embedder for {model_name}."
    )
    return _get_generic_hf_embedder()


_MODEL_MAX_LENGTH_SENTINEL = 1_000_000_000


def _format_max_length(config, tokenizer):
    """Return a human-readable max sequence length, ignoring HF sentinel values."""
    if hasattr(config, "max_position_embeddings") and config.max_position_embeddings:
        val = config.max_position_embeddings
        if val < _MODEL_MAX_LENGTH_SENTINEL:
            return str(val)

    if hasattr(tokenizer, "model_max_length"):
        val = tokenizer.model_max_length
        if val < _MODEL_MAX_LENGTH_SENTINEL:
            return str(val)
        return "unknown (no enforced limit)"

    if hasattr(config, "n_positions") and config.n_positions:
        val = config.n_positions
        if val < _MODEL_MAX_LENGTH_SENTINEL:
            return str(val)

    return "unknown"


def report_model(model_name, trust_remote_code=False):
    """Load config + tokenizer only and print a compatibility summary."""
    import pepe.utils

    from transformers import AutoConfig, AutoTokenizer

    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except Exception as e:
        translate_hf_config_error(
            model_name, e, trust_remote_code=trust_remote_code
        )

    embedder_cls = select_model(model_name, trust_remote_code=trust_remote_code)
    embedder_name = embedder_cls.__name__
    model_type = getattr(config, "model_type", None) or "unknown"
    max_length = _format_max_length(config, tokenizer)
    char_level = pepe.utils.is_character_tokenizer(tokenizer)
    if char_level:
        tokenizer_type = "character-level"
    else:
        tokenizer_type = (
            "subword (per_token/substring_pooled unreliable; prefer mean_pooled)"
        )

    logits_available = embedder_name == "ESM2Embedder"

    if embedder_name in ("ESMEmbedder",):
        attention = "yes (fair-esm contacts)"
    elif embedder_name == "GenericHuggingFaceEmbedder":
        attention = "yes (attn_implementation=eager may be required)"
    elif embedder_name in (
        "ESM2Embedder",
        "Antiberta2Embedder",
        "ESMCEmbedder",
        "T5Embedder",
    ):
        attention = "yes"
    elif embedder_name == "CustomEmbedder":
        attention = "depends on model"
    else:
        attention = "unknown"

    print(f"Model:           {model_name}")
    print(f"Architecture:    {model_type}")
    print(f"Embedder:        {embedder_name}")
    print(f"Max length:      {max_length}")
    print(f"Tokenizer:       {tokenizer_type}")
    print(f"Logits:          {'yes' if logits_available else 'no'}")
    print(f"Attention:       {attention}")


supported_models = [
    # ESM models
    "esm1_t34_670M_UR50S",
    "esm1_t34_670M_UR50D",
    "esm1_t34_670M_UR100",
    "esm1_t12_85M_UR50S",
    "esm1_t6_43M_UR50S",
    "esm1b_t33_650M_UR50S",
    #'esm_msa1_t12_100M_UR50S',
    #'esm_msa1b_t12_100M_UR50S',
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
    #'esm_if1_gvp4_t16_142M_UR50',
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
    # Pre-defined Hugging Face models
    "Rostlab/prot_t5_xl_half_uniref50-enc",
    "Rostlab/ProstT5",
    "alchemab/antiberta2-cssp",
    "alchemab/antiberta2",
    # ESMC models (requires Biohub transformers fork; see README)
    "biohub/ESMC-300M",
    "biohub/ESMC-600M",
    "biohub/ESMC-6B",
    # Custom models examples:
    # - PyTorch models: "/path/to/model.pt", "/path/to/model_directory/", "custom:/path/to/model.pt"
    # - Hugging Face models: "username/model-name", "./local_hf_model"
    # - See documentation for details on custom model requirements
]
