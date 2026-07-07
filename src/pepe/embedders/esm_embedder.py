import logging
import torch
from typing import Any, Dict, List, Optional, Tuple
from pepe.embedders.base_embedder import BaseEmbedder
import pepe.utils


# Lazy imports to avoid loading heavy dependencies at import time
def _import_esm() -> Any:
    """Lazy import of ESM components to avoid loading issues."""
    try:
        from esm import pretrained

        return pretrained
    except ImportError as e:
        logger.error(f"Failed to import ESM: {e}")
        raise ImportError(
            "Failed to import ESM. Please ensure fair-esm is installed: pip install fair-esm"
        ) from e


logger = logging.getLogger("pepe.embedders.esm_embedder")


class ESMEmbedder(BaseEmbedder):
    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.alphabet,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
            self.prepend_bos,
            self.append_eos,
        ) = self._initialize_model(self.model_name)
        self.valid_tokens = set(self.alphabet.all_toks)
        self._check_max_input_length()
        pepe.utils.check_input_tokens(
            self.valid_tokens,
            self.sequences,
            self.model_name,
            split_long_sequences=self.split_long_sequences,
        )
        self.special_tokens = self.get_special_tokens()
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_input_length = self._load_data(
            self.sequences, self.substring_dict
        )  # tokenize and batch sequences and update max_input_length
        self._set_output_objects()

    def _initialize_model(
        self,
        model_link: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> Tuple[Any, ...]:
        """Initialize the model, tokenizer"""
        assert model_link is not None
        model_name = model_link
        #  Loading the pretrained model and alphabet for tokenization
        logger.info("Loading model...")

        # Lazy import ESM components
        pretrained = _import_esm()

        # model, alphabet = pretrained.load_model_and_alphabet(model_name)
        model, alphabet = pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # Setting the model to evaluation mode
        if not self.disable_special_tokens:
            model.append_eos = True if not model_name.startswith("esm1") else False
            model.prepend_bos = True
        else:
            model.append_eos = False
            model.prepend_bos = False

        num_heads = model.layers[0].self_attn.num_heads
        num_layers = len(model.layers)
        embedding_size = (
            model.embed_tokens.embedding_dim
            if model_name.startswith("esm1")
            else model.embed_dim
        )

        # Moving the model to GPU if available for faster processing
        if torch.cuda.is_available() and self.device.type == "cuda":
            model = model.cuda()
            logger.info("Transferred model to GPU")
        else:
            logger.info("No GPU available, using CPU")
        return (
            model,
            alphabet,
            num_heads,
            num_layers,
            embedding_size,
            model.prepend_bos,
            model.append_eos,
        )

    def get_special_tokens(self) -> torch.Tensor:
        special_tokens = self.alphabet.all_special_tokens
        special_token_ids = torch.tensor(
            [self.alphabet.tok_to_idx[tok] for tok in special_tokens],
            device=self.device,
            dtype=torch.int8,
        )
        return special_token_ids

    def _load_layers(self, layers: Optional[List[int]] = None) -> List[int]:
        if layers is None:
            return list(range(1, self.model.num_layers + 1))
        if not layers:
            layers = [-1]
        # Checking if the specified representation layers are valid
        assert all(
            -(self.model.num_layers + 1) <= i <= self.model.num_layers for i in layers
        )
        layers = [
            (i + self.model.num_layers + 1) % (self.model.num_layers + 1)
            for i in layers
        ]
        return layers

    def _load_data(
        self,
        sequences: Optional[Dict[str, str]] = None,
        substring_dict: Optional[Dict[str, str]] = None,
        bracket_type: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        assert sequences is not None
        # Creating a dataset from the input fasta file
        dataset = pepe.utils.ESMDataset(
            sequences,
            substring_dict,
            self.context,
            self.alphabet,
            self.max_input_length,
            self.prepend_bos,
            self.append_eos,
        )
        # Generating batch indices based on token count
        logger.info("Generating batches...")
        batches = pepe.utils.TokenBudgetBatchSampler(dataset, self.batch_size)
        # DataLoader to iterate through batches efficiently
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batches, collate_fn=dataset.safe_collate
        )
        logger.info("Data loaded")
        # Getting the maximum sequence length from the dataset
        max_length = dataset.get_max_encoded_length()
        return data_loader, max_length

    def _compute_outputs(
        self,
        model: Any,
        toks: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        return_embeddings: bool,
        return_contacts: bool,
        return_logits: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        outputs = model(
            toks,
            repr_layers=self.layers,
            return_contacts=return_contacts,
        )
        if return_logits:
            logits = (
                outputs["logits"]
                .to(dtype=self._precision_to_dtype(self.precision, "torch"))
                .permute(2, 0, 1)
                .cpu()
            )  # permute to match the shape of the representations
            torch.cuda.empty_cache()
        else:
            logits = None

        if return_contacts:
            attention_matrices = (
                outputs["attentions"]
                .to(dtype=self._precision_to_dtype(self.precision, "torch"))
                .permute(1, 0, 2, 3, 4)
            ).cpu()  # permute to match the shape of the representations
            torch.cuda.empty_cache()
        else:
            attention_matrices = None
        # Extracting layer representations and moving them to CPU
        if return_embeddings:
            representations = {
                layer: t.to(
                    dtype=self._precision_to_dtype(self.precision, "torch")
                ).cpu()
                for layer, t in outputs["representations"].items()
            }
            torch.cuda.empty_cache()
        else:
            representations = None
        return logits, representations, attention_matrices
