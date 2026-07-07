import logging
import os
import torch
import pepe.utils
from pepe.embedders.base_embedder import BaseEmbedder


# Lazy imports to avoid loading heavy dependencies at import time
def _import_transformers():
    """Lazy import of transformers components to avoid loading issues."""
    try:
        from transformers import T5EncoderModel, T5Tokenizer
        from transformers import RoFormerTokenizer, RoFormerModel
        from transformers.models.roformer.modeling_roformer import (
            RoFormerSinusoidalPositionalEmbedding,
        )
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        from transformers import AutoModelForMaskedLM

        return (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
        )
    except ImportError as e:
        logger.error(f"Failed to import transformers: {e}")
        raise ImportError(
            "Failed to import transformers. Please ensure transformers is installed: pip install transformers"
        ) from e


# Set max_split_size_mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

logger = logging.getLogger("pepe.embedders.huggingface_embedder")


class HuggingfaceEmbedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        if self.return_logits:
            logger.warning(
                "Warning: Logits are not supported for this model. Setting to False."
            )
            self.return_logits = False
            self.output_types.remove("logits")

    def _load_layers(self, layers):
        """Check if the specified representation layers are valid."""
        num_layers = self.num_layers  # type: ignore
        if layers is None:
            return list(range(1, num_layers + 1))
        if not layers:
            layers = [-1]
        assert all(-(num_layers + 1) <= i <= num_layers for i in layers)
        layers = [(i + num_layers + 1) % (num_layers + 1) for i in layers]
        return layers

    def _load_data(self, sequences, substring_dict, bracket_type):
        """Tokenize sequences and create a DataLoader."""
        # Tokenize sequences
        dataset = pepe.utils.HuggingFaceDataset(
            sequences,
            substring_dict,
            self.context,
            bracket_type,
            self.tokenizer,  # type: ignore
            self.max_input_length,
            add_special_tokens=not self.disable_special_tokens,
        )
        logger.info("Batching sequences...")
        batch_sampler = pepe.utils.TokenBudgetBatchSampler(
            dataset=dataset, token_budget=self.batch_size
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=dataset.safe_collate
        )
        max_length = dataset.get_max_encoded_length()
        logger.info("Finished tokenizing and batching sequences")

        return data_loader, max_length

    def _compute_outputs(
        self,
        model,
        toks,
        attention_mask,
        return_embeddings,
        return_contacts,
        return_logits=False,
    ):
        outputs = model(
            input_ids=toks,
            attention_mask=attention_mask,
            output_hidden_states=return_embeddings,
            output_attentions=return_contacts,
        )
        if return_contacts:
            attention_matrices = (
                torch.stack(outputs.attentions)  # type: ignore
                .to(self._precision_to_dtype(self.precision, "torch"))  # type: ignore
                .cpu()
            )  # stack attention matrices across layers
            torch.cuda.empty_cache()
        else:
            attention_matrices = None
        if return_embeddings:
            dtype = self._precision_to_dtype(self.precision, "torch")
            hidden_states = outputs.hidden_states
            if isinstance(hidden_states, torch.Tensor):
                representations = {
                    layer: hidden_states[layer].to(dtype).cpu()
                    for layer in self.layers  # type: ignore
                }
            else:
                representations = {
                    layer: hidden_states[layer].to(dtype).cpu()
                    for layer in self.layers  # type: ignore
                }
            torch.cuda.empty_cache()
        else:
            representations = None
        logits = None  # Model doesn't return logits
        return logits, representations, attention_matrices


class Antiberta2Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_link)
        self.valid_tokens = set(self.tokenizer.get_vocab().keys())
        self.bracket_type = pepe.utils.get_bracket_type(self.tokenizer)
        self._check_max_input_length()
        pepe.utils.check_input_tokens(
            self.valid_tokens,
            self.sequences,
            self.model_name,
            split_long_sequences=self.split_long_sequences,
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_input_length = self._load_data(
            self.sequences, self.substring_dict, self.bracket_type
        )
        self._set_output_objects()
        if not self.split_long_sequences:
            assert self.max_input_length <= 256, "AntiBERTa2 only supports max_length <= 256. Use --split_long_sequences to process longer sequences."

    def _initialize_model(self, model_link="alchemab/antiberta2-cssp"):
        """Initialize the model, tokenizer, and device."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            device = torch.device("cuda")
            logger.info("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        # Lazy import transformers components
        (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
        ) = _import_transformers()

        tokenizer = RoFormerTokenizer.from_pretrained(model_link, use_fast=True)
        model = RoFormerModel.from_pretrained(model_link).to(device)  # type: ignore
        model.eval()
        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size


class T5Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = self.fasta_to_dict(args.fasta_path)  # type: ignore
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_link)
        self.valid_tokens = self.get_valid_tokens()
        self.bracket_type = pepe.utils.get_bracket_type(self.tokenizer)
        pepe.utils.check_input_tokens(
            self.valid_tokens, self.sequences, self.model_name, self.bracket_type
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_input_length = self._load_data(
            self.sequences, self.substring_dict, self.bracket_type
        )
        self._set_output_objects()

    def get_valid_tokens(self):
        valid_tokens = set(
            k[1:] if k.startswith("▁") else k
            for k in set(self.tokenizer.get_vocab().keys())
        )
        return valid_tokens

    def _initialize_model(self, model_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
        """Initialize the model, tokenizer, and device."""

        if torch.cuda.is_available() and self.device.type == "cuda":
            device = torch.device("cuda")
            logger.info("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        # Lazy import transformers components
        (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
        ) = _import_transformers()

        tokenizer = T5Tokenizer.from_pretrained(
            model_link, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        model = T5EncoderModel.from_pretrained(
            model_link, trust_remote_code=self.trust_remote_code
        ).to(device)  # type: ignore
        model.eval()
        num_heads = model.config.num_heads
        num_layers = model.config.num_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size


def _resolve_esm2_model_link(model_name):
    if "/" in model_name:
        return model_name
    return f"facebook/{model_name}"


class ESM2Embedder(HuggingfaceEmbedder):
    """ESM-2 embedder using HuggingFace transformers instead of fair-esm."""

    def __init__(self, args):
        BaseEmbedder.__init__(self, args)
        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_name)
        self.valid_tokens = self._get_valid_tokens()
        self.bracket_type = pepe.utils.get_bracket_type(self.tokenizer)
        self._check_max_input_length()
        pepe.utils.warn_if_non_character_tokenizer(self.tokenizer, self.model_name)
        pepe.utils.check_input_tokens(
            self.valid_tokens,
            self.sequences,
            self.model_name,
            split_long_sequences=self.split_long_sequences,
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_input_length = self._load_data(
            self.sequences, self.substring_dict, self.bracket_type
        )
        self._set_output_objects()

    def _get_valid_tokens(self):
        return {tok for tok in self.tokenizer.get_vocab().keys() if len(tok) == 1}

    def _load_data(self, sequences, substring_dict, bracket_type):
        dataset = pepe.utils.HuggingFaceDataset(
            sequences,
            substring_dict,
            self.context,
            bracket_type,
            self.tokenizer,
            self.max_input_length,
            add_special_tokens=not self.disable_special_tokens,
        )
        dataset.pad_token_id = self.tokenizer.pad_token_id
        logger.info("Batching sequences...")
        batch_sampler = pepe.utils.TokenBudgetBatchSampler(
            dataset=dataset, token_budget=self.batch_size
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=dataset.safe_collate
        )
        max_length = dataset.get_max_encoded_length()
        logger.info("Finished tokenizing and batching sequences")
        return data_loader, max_length

    def _initialize_model(self, model_name):
        model_link = _resolve_esm2_model_link(model_name)
        if torch.cuda.is_available() and self.device.type == "cuda":
            device = torch.device("cuda")
            logger.info("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
        ) = _import_transformers()

        logger.info(f"Loading ESM-2 model from HuggingFace: {model_link}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_link, trust_remote_code=self.trust_remote_code
        )
        model_kwargs = {"trust_remote_code": self.trust_remote_code}
        if self.return_contacts:
            model_kwargs["attn_implementation"] = "eager"

        if self.return_logits:
            model = AutoModelForMaskedLM.from_pretrained(
                model_link, **model_kwargs
            ).to(device)
        else:
            model = AutoModel.from_pretrained(model_link, **model_kwargs).to(device)
        model.eval()

        config = model.config
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers
        embedding_size = config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size

    def _compute_outputs(
        self,
        model,
        toks,
        attention_mask,
        return_embeddings,
        return_contacts,
        return_logits=False,
    ):
        outputs = model(
            input_ids=toks,
            attention_mask=attention_mask,
            output_hidden_states=return_embeddings,
            output_attentions=return_contacts,
        )
        if return_logits:
            logits = (
                outputs.logits
                .to(dtype=self._precision_to_dtype(self.precision, "torch"))
                .permute(2, 0, 1)
                .cpu()
            )
            torch.cuda.empty_cache()
        else:
            logits = None

        if return_contacts:
            attention_matrices = (
                torch.stack(outputs.attentions)  # type: ignore
                .to(self._precision_to_dtype(self.precision, "torch"))  # type: ignore
                .cpu()
            )
            torch.cuda.empty_cache()
        else:
            attention_matrices = None

        if return_embeddings:
            representations = {
                layer: outputs.hidden_states[layer]
                .to(self._precision_to_dtype(self.precision, "torch"))
                .cpu()
                for layer in self.layers  # type: ignore
            }
            torch.cuda.empty_cache()
        else:
            representations = None

        return logits, representations, attention_matrices


def _get_config_attr(config, *names, default=None):
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


class ESMCEmbedder(HuggingfaceEmbedder):
    """ESMC embedder using Biohub transformers fork (model_type esmc)."""

    def __init__(self, args):
        BaseEmbedder.__init__(self, args)
        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_link)
        self.valid_tokens = self._get_valid_tokens()
        self.bracket_type = pepe.utils.get_bracket_type(self.tokenizer)
        self._check_max_input_length()
        pepe.utils.check_input_tokens(
            self.valid_tokens,
            self.sequences,
            self.model_name,
            split_long_sequences=self.split_long_sequences,
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_input_length = self._load_data(
            self.sequences, self.substring_dict, self.bracket_type
        )
        self._set_output_objects()

    def _get_valid_tokens(self):
        return {tok for tok in self.tokenizer.get_vocab().keys() if len(tok) == 1}

    def _load_data(self, sequences, substring_dict, bracket_type):
        dataset = pepe.utils.HuggingFaceDataset(
            sequences,
            substring_dict,
            self.context,
            bracket_type,
            self.tokenizer,
            self.max_input_length,
            add_special_tokens=not self.disable_special_tokens,
            gapped_sequences=False,
        )
        dataset.pad_token_id = self.tokenizer.pad_token_id
        logger.info("Batching sequences...")
        batch_sampler = pepe.utils.TokenBudgetBatchSampler(
            dataset=dataset, token_budget=self.batch_size
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=dataset.safe_collate
        )
        max_length = dataset.get_max_encoded_length()
        logger.info("Finished tokenizing and batching sequences")
        return data_loader, max_length

    def _initialize_model(self, model_link):
        if torch.cuda.is_available() and self.device.type == "cuda":
            device = torch.device("cuda")
            logger.info("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
        ) = _import_transformers()

        logger.info(f"Loading ESMC model from HuggingFace: {model_link}")
        tokenizer = AutoTokenizer.from_pretrained(model_link)
        model_kwargs = {}
        if self.return_contacts:
            model_kwargs["attn_implementation"] = "eager"

        if self.return_logits:
            model = AutoModelForMaskedLM.from_pretrained(
                model_link, **model_kwargs
            ).to(device)
        else:
            model = AutoModel.from_pretrained(model_link, **model_kwargs).to(device)
        model.eval()

        config = model.config
        num_heads = _get_config_attr(
            config, "num_attention_heads", "n_heads", "num_heads"
        )
        num_layers = _get_config_attr(
            config, "num_hidden_layers", "n_layers", "num_layers"
        )
        embedding_size = _get_config_attr(
            config, "hidden_size", "d_model", "embed_dim"
        )
        return model, tokenizer, num_heads, num_layers, embedding_size


class GenericHuggingFaceEmbedder(HuggingfaceEmbedder):
    """Generic HuggingFace embedder that can handle models with unknown architectures using AutoModel and AutoTokenizer."""

    def __init__(self, args):
        super().__init__(args)
        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_link)
        self.valid_tokens = self._get_valid_tokens()
        self.bracket_type = pepe.utils.get_bracket_type(self.tokenizer)
        self._check_max_input_length()
        pepe.utils.warn_if_non_character_tokenizer(self.tokenizer, self.model_name)
        pepe.utils.check_input_tokens(
            self.valid_tokens,
            self.sequences,
            self.model_name,
            split_long_sequences=self.split_long_sequences,
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_input_length = self._load_data(
            self.sequences, self.substring_dict, self.bracket_type
        )
        self._set_output_objects()

    def _get_valid_tokens(self):
        """Get valid tokens from the tokenizer."""
        if hasattr(self.tokenizer, "get_vocab"):
            vocab = self.tokenizer.get_vocab()
            # Handle different tokenizer types
            if hasattr(self.tokenizer, "decoder") and self.tokenizer.decoder:
                # T5-style tokenizer
                valid_tokens = set(
                    k[1:] if k.startswith("▁") else k for k in vocab.keys()
                )
            else:
                # Standard tokenizer
                valid_tokens = set(vocab.keys())
            return valid_tokens
        return set()

    def _initialize_model(self, model_link):
        """Initialize the model, tokenizer, and device using AutoModel and AutoTokenizer."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            device = torch.device("cuda")
            logger.info("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        # Lazy import transformers components
        (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
        ) = _import_transformers()

        from pepe.model_errors import translate_hf_config_error

        model_kwargs = {"trust_remote_code": self.trust_remote_code}
        if self.return_contacts:
            model_kwargs["attn_implementation"] = "eager"
        attn_fallback_attempted = False

        def _load_pretrained(model_cls, link, **extra_kwargs):
            nonlocal attn_fallback_attempted
            kwargs = {**model_kwargs, **extra_kwargs}
            try:
                return model_cls.from_pretrained(link, **kwargs).to(device)
            except (TypeError, ValueError) as load_error:
                if (
                    model_kwargs.get("attn_implementation")
                    and not attn_fallback_attempted
                ):
                    attn_fallback_attempted = True
                    logger.warning(
                        "Model rejected attn_implementation='eager' (%s); "
                        "attention extraction may be unavailable.",
                        load_error,
                    )
                    model_kwargs.pop("attn_implementation", None)
                    kwargs = {**model_kwargs, **extra_kwargs}
                    return model_cls.from_pretrained(link, **kwargs).to(device)
                raise

        def _load_model(link, **extra_kwargs):
            try:
                return _load_pretrained(AutoModel, link, **extra_kwargs)
            except ValueError:
                return _load_pretrained(AutoModelForCausalLM, link, **extra_kwargs)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_link, use_fast=True, trust_remote_code=self.trust_remote_code
            )
            model = _load_model(model_link)
        except Exception as e:
            translate_hf_config_error(
                model_link, e, trust_remote_code=self.trust_remote_code
            )

        # Handle tokenizer padding token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                # Add a special padding token
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

        model.eval()

        # Get model configuration
        config = model.config
        num_heads = getattr(
            config,
            "num_attention_heads",
            getattr(config, "num_heads", getattr(config, "n_head", 12)),
        )
        num_layers = getattr(
            config,
            "num_hidden_layers",
            getattr(config, "num_layers", getattr(config, "n_layer", 12)),
        )
        embedding_size = getattr(
            config,
            "hidden_size",
            getattr(config, "d_model", getattr(config, "embed_dim", 768)),
        )

        logger.info(f"Loaded generic HuggingFace model: {model_link}")
        logger.info(f"Model type: {config.model_type}")
        logger.info(
            f"Number of heads: {num_heads}, layers: {num_layers}, embedding size: {embedding_size}"
        )

        return model, tokenizer, num_heads, num_layers, embedding_size
