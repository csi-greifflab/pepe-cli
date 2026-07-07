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


def select_model(model_name):
    if "esm2" in model_name.lower():
        return _get_esm2_embedder()
    elif "esm1" in model_name.lower():
        return _get_esm_embedder()
    # elif "antiberta2" in model_name.lower() and model_name.startswith("alchemab"):
    #    T5Embedder, Antiberta2Embedder = _get_huggingface_embedders()
    #    return Antiberta2Embedder
    # elif "t5" in model_name.lower() and model_name.startswith("Rostlab"):
    #    T5Embedder, Antiberta2Embedder = _get_huggingface_embedders()
    #    return T5Embedder
    elif (
        model_name.endswith(".pt")
        or model_name.endswith(".pth")
        or model_name.startswith("custom:")
        or (
            os.path.exists(model_name)
            and (os.path.isfile(model_name) or os.path.isdir(model_name))
        )
    ):
        return CustomEmbedder
    elif "/" in model_name:
        # Assume it's a Hugging Face model (username/model-name format)
        # Try to determine the architecture automatically
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name)
            model_type = config.model_type.lower()

            if model_type in ["t5", "mt5"]:
                T5Embedder, Antiberta2Embedder = _get_huggingface_embedders()
                return T5Embedder
            elif model_type in ["roformer"]:
                T5Embedder, Antiberta2Embedder = _get_huggingface_embedders()
                return Antiberta2Embedder
            elif model_type in ["esm"]:
                return _get_esm2_embedder()
            elif model_type in ["esmc"]:
                return _get_esmc_embedder()
            elif model_type in ["bert"]:
                # For BERT-like models, we could potentially use a generic embedder
                # but for now, suggest using CustomEmbedder or creating a specific one
                raise ValueError(
                    f"BERT-like models are not yet directly supported. Consider using a PyTorch version of the model with CustomEmbedder."
                )
            else:
                # For other architectures, you might want to add more specific embedders
                # or use a generic HuggingfaceEmbedder
                raise ValueError(
                    f"Model architecture '{model_type}' not yet supported for custom Hugging Face models"
                )
        except Exception as e:
            error_msg = str(e)
            if "esmc" in model_name.lower() or "esmc" in error_msg.lower():
                raise ValueError(
                    f"ESMC models require Biohub's transformers fork (ESM1 fair-esm is unaffected). "
                    f"Install with: pip install git+https://github.com/Biohub/transformers.git@main"
                ) from e
            if "Unrecognized model" in error_msg or "model_type" in error_msg:
                raise ValueError(
                    f"Model {model_name} appears to be a Keras/TensorFlow model or has an unsupported architecture. PEPE currently supports PyTorch models only. Consider using a PyTorch version or converting the model."
                ) from e
            raise ValueError(
                f"Could not determine model architecture for {model_name}: {e}"
            ) from e
    else:
        raise ValueError(f"Model {model_name} not supported")


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
