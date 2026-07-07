"""Typed errors for HuggingFace model selection and loading."""


class ModelSelectionError(ValueError):
    """Base class for model dispatch / config load failures."""


class ModelNotFoundError(ModelSelectionError):
    """HuggingFace repository id does not exist on the Hub."""


class GatedModelError(ModelSelectionError):
    """HuggingFace repository is gated and requires authentication."""


class RemoteCodeRequiredError(ModelSelectionError):
    """Model ships custom code that must be explicitly trusted."""


class UnsupportedArchitectureError(ModelSelectionError):
    """Model architecture is not supported by PEPE (e.g. Keras/TensorFlow)."""


class ModelEnvironmentError(ModelSelectionError):
    """Network, cache, or filesystem failure while loading model metadata."""


class ESMCForkRequiredError(ModelSelectionError):
    """ESMC models require Biohub's transformers fork."""


class METL3DNotSupportedError(ModelSelectionError):
    """METL 3D models require PDB structures and are not supported by PEPE."""


class METLPackageRequiredError(ModelSelectionError):
    """METL models require the metl-pretrained package."""


def _import_hf_hub_errors():
    """Lazy import of huggingface_hub exception types (version-tolerant)."""
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        return RepositoryNotFoundError, GatedRepoError
    except ImportError:
        try:
            from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

            return RepositoryNotFoundError, GatedRepoError
        except ImportError:
            return None, None


_REMOTE_CODE_MARKERS = (
    "trust_remote_code",
    "requires you to execute",
    "custom code",
)


def requires_remote_code(error):
    """Return True when upstream signals that trust_remote_code is required."""
    msg = str(error).lower()
    return any(marker in msg for marker in _REMOTE_CODE_MARKERS)


def _is_unsupported_architecture_error(error):
    if not isinstance(error, ValueError):
        return False
    msg = str(error)
    return "Unrecognized model" in msg or (
        "model_type" in msg and "esmc" not in msg.lower()
    )


def _is_esmc_fork_error(model_name, error):
    if "esmc" not in model_name.lower():
        return False
    if _is_unsupported_architecture_error(error):
        return True
    return "esmc" in str(error).lower()


def _esmc_fork_message():
    return (
        "ESMC models require Biohub's transformers fork (ESM1 fair-esm is unaffected). "
        "Install with: pip install git+https://github.com/Biohub/transformers.git@main"
    )


def translate_hf_config_error(model_name, error, *, trust_remote_code=False):
    """Map an AutoConfig load failure to a typed, actionable ModelSelectionError."""
    if _is_esmc_fork_error(model_name, error):
        raise ESMCForkRequiredError(_esmc_fork_message()) from error

    repository_not_found_error, gated_repo_error = _import_hf_hub_errors()

    if gated_repo_error is not None and isinstance(error, gated_repo_error):
        if "esmc" in model_name.lower():
            raise ESMCForkRequiredError(_esmc_fork_message()) from error
        raise GatedModelError(
            f"HuggingFace model '{model_name}' is gated and requires authentication. "
            f"Log in with `huggingface-cli login` and accept the model's terms on the Hub, "
            f"then retry."
        ) from error

    if repository_not_found_error is not None and isinstance(
        error, repository_not_found_error
    ):
        raise ModelNotFoundError(
            f"HuggingFace model '{model_name}' was not found. Check the repo id for typos "
            f"and confirm it exists on https://huggingface.co/models."
        ) from error

    if requires_remote_code(error) and not trust_remote_code:
        raise RemoteCodeRequiredError(
            f"Model '{model_name}' requires custom modeling code from the Hub. "
            f"Re-run with --trust_remote_code (CLI) or trust_remote_code=True (embed()). "
            f"Only enable this for repos you trust — it executes remote Python code."
        ) from error

    if _is_unsupported_architecture_error(error):
        raise UnsupportedArchitectureError(
            f"Model {model_name} appears to be a Keras/TensorFlow model or has an "
            f"unsupported architecture. PEPE currently supports PyTorch models only. "
            f"Consider using a PyTorch version or converting the model."
        ) from error

    if isinstance(error, (OSError, EnvironmentError)):
        raise ModelEnvironmentError(
            f"Could not reach HuggingFace to load config for '{model_name}' "
            f"(network, cache, or filesystem issue): {error}"
        ) from error

    raise ModelSelectionError(
        f"Could not load model config for {model_name}: {error}"
    ) from error
