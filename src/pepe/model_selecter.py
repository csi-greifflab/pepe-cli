from pepe.embedders.custom_embedder import CustomEmbedder

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


def select_model(model_name):
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
        return _select_hf_model(model_name)

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


def _select_hf_model(model_name):
    """Dispatch a HuggingFace repo id to an embedder by inspecting its config.

    Known architectures get their specialized embedder; everything else (BERT-like
    encoders and any architecture without a dedicated embedder) falls back to the
    generic AutoModel-based ``GenericHuggingFaceEmbedder``. The forward pass is
    architecture-agnostic, so standard encoders such as ProtBert, AntiBERTy and
    IgBert work through the fallback with no engine changes.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        _raise_hf_config_error(model_name, e)

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

    logging.getLogger("src.model_selecter").info(
        f"No specialized embedder for architecture '{model_type or 'unknown'}'; "
        f"using the generic HuggingFace embedder for {model_name}."
    )
    return _get_generic_hf_embedder()


def _raise_hf_config_error(model_name, error):
    """Translate an AutoConfig load failure into an actionable ValueError."""
    error_msg = str(error)
    if "esmc" in model_name.lower() or "esmc" in error_msg.lower():
        raise ValueError(
            f"ESMC models require Biohub's transformers fork (ESM1 fair-esm is unaffected). "
            f"Install with: pip install git+https://github.com/Biohub/transformers.git@main"
        ) from error
    if "Unrecognized model" in error_msg or "model_type" in error_msg:
        raise ValueError(
            f"Model {model_name} appears to be a Keras/TensorFlow model or has an unsupported architecture. PEPE currently supports PyTorch models only. Consider using a PyTorch version or converting the model."
        ) from error
    raise ValueError(
        f"Could not load model config for {model_name}: {error}"
    ) from error


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
