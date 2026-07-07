import logging

import torch

import pepe.utils
from pepe.embedders.base_embedder import BaseEmbedder
from pepe.model_errors import METLPackageRequiredError, ModelSelectionError


def _import_metl():
    """Lazy import of metl-pretrained to avoid loading issues."""
    try:
        import metl

        return metl
    except ImportError as e:
        raise METLPackageRequiredError(
            "METL models require metl-pretrained. Install with: "
            "pip install git+https://github.com/gitter-lab/metl-pretrained.git"
        ) from e


def resolve_metl_ident(model_name):
    """Map a PEPE model name to a metl-pretrained identifier."""
    normalized = model_name.strip()
    if normalized.lower() in ("gitter-lab/metl", "gitter-lab/metl-pretrained"):
        raise ModelSelectionError(
            "gitter-lab/METL is the HuggingFace wrapper and is not supported by PEPE. "
            "Use a metl-pretrained identifier instead (e.g. metl-g-20m-1d)."
        )
    return normalized.lower()


logger = logging.getLogger("pepe.embedders.metl_embedder")


class METLEmbedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        if self.return_logits:
            logger.warning(
                "Warning: Logits are not supported for METL models. Setting to False."
            )
            self.return_logits = False
            if "logits" in self.output_types:
                self.output_types.remove("logits")
        if self.return_contacts:
            logger.warning(
                "Warning: Attention matrices are not supported for METL models. Setting to False."
            )
            self.return_contacts = False
            for output_type in ("attention_head", "attention_layer", "attention_model"):
                if output_type in self.output_types:
                    self.output_types.remove(output_type)

        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.data_encoder,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_link)
        self.valid_tokens = set(self.data_encoder.chars[1:])
        self._check_max_input_length()
        pepe.utils.check_input_tokens(
            self.valid_tokens,
            self.sequences,
            self.model_name,
            split_long_sequences=self.split_long_sequences,
        )
        self.special_tokens = torch.tensor([0], device=self.device, dtype=torch.int8)
        self.layers = self._load_layers(self.layers)
        self._repr_outputs = {}
        self._hook_handles = []
        self._register_repr_hooks()
        self.data_loader, self.max_input_length = self._load_data(
            self.sequences, self.substring_dict
        )
        self._set_output_objects()

    def _initialize_model(self, model_link):
        logger.info("Loading METL model...")
        metl = _import_metl()
        ident = resolve_metl_ident(model_link)
        model, data_encoder = metl.get_from_ident(ident)
        model.eval()

        tr_encoder = model.model.tr_encoder
        num_tr_layers = len(tr_encoder.layers)
        num_layers = num_tr_layers
        embedding_size = (
            getattr(model, "embedding_len", None) or model.model.embedding_len
        )
        num_heads = 1

        if torch.cuda.is_available() and self.device.type == "cuda":
            model = model.cuda()
            logger.info("Transferred model to GPU")
        else:
            logger.info(f"Using device: {self.device.type}")

        return model, data_encoder, num_heads, num_layers, embedding_size

    def _register_repr_hooks(self):
        tr_encoder = self.model.model.tr_encoder
        num_tr_layers = len(tr_encoder.layers)
        final_norm = getattr(tr_encoder, "norm", None)

        def make_hook(captured_layer):
            def hook(_module, _input, output):
                self._repr_outputs[captured_layer] = output

            return hook

        def make_pre_hook(captured_layer):
            def pre_hook(_module, args):
                self._repr_outputs[captured_layer] = args[0]

            return pre_hook

        # Layer index semantics match the HuggingFace path (hidden_states[k]):
        # layer 0 = input embeddings, layer k (1..num_tr_layers) = k-th block output.
        for layer in self.layers:
            if layer == 0:
                # Input embeddings (+ positional encoding): the tensor fed into the
                # encoder, captured via a forward_pre_hook so it is agnostic to whether
                # the model uses absolute or relative positional encoding.
                handle = tr_encoder.register_forward_pre_hook(make_pre_hook(layer))
            elif layer == num_tr_layers and final_norm is not None:
                # Post-norm final representation when the encoder has a final norm
                # (norm_first architectures); otherwise fall through to the last block.
                handle = final_norm.register_forward_hook(make_hook(layer))
            else:
                handle = tr_encoder.layers[layer - 1].register_forward_hook(
                    make_hook(layer)
                )
            self._hook_handles.append(handle)

    def _load_layers(self, layers):
        if layers is None:
            return list(range(1, self.num_layers + 1))
        if not layers:
            layers = [-1]
        assert all(-(self.num_layers + 1) <= i <= self.num_layers for i in layers)
        layers = [(i + self.num_layers + 1) % (self.num_layers + 1) for i in layers]
        return layers

    def _load_data(self, sequences, substring_dict=None):
        dataset = pepe.utils.METLDataset(
            sequences,
            substring_dict,
            self.context,
            self.data_encoder,
            self.max_input_length,
        )
        logger.info("Generating batches...")
        batches = pepe.utils.TokenBudgetBatchSampler(dataset, self.batch_size)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batches, collate_fn=dataset.safe_collate
        )
        logger.info("Data loaded")
        max_length = dataset.get_max_encoded_length()
        return data_loader, max_length

    def _compute_outputs(
        self,
        model,
        toks,
        attention_mask,
        return_embeddings,
        return_contacts,
        return_logits,
    ):
        self._repr_outputs.clear()
        model(toks)

        if return_embeddings:
            dtype = self._precision_to_dtype(self.precision, "torch")
            representations = {
                layer: self._repr_outputs[layer].to(dtype=dtype).cpu()
                for layer in self.layers
            }
            torch.cuda.empty_cache()
        else:
            representations = None

        return None, representations, None

    def _get_model_max_allowed(self):
        if hasattr(self, "force_split_length") and self.force_split_length is not None:
            return self.force_split_length
        return self.model.aa_seq_len

    def _check_max_input_length(self):
        """Check max length without BOS/EOS adjustment (METL has no special tokens)."""
        max_allowed = self._get_model_max_allowed()
        if max_allowed is None:
            return

        sequences_too_long = any(len(s) > max_allowed for s in self.sequences.values())
        needs_splitting = (
            isinstance(self.max_input_length, int)
            and self.max_input_length > max_allowed
        ) or sequences_too_long

        if needs_splitting:
            if self.split_long_sequences:
                logger.info(
                    f"Splitting sequences because they exceed model limit ({max_allowed})."
                )
                self._handle_sequence_splitting(max_allowed)
            else:
                logger.warning(
                    f"Warning: Sequences exceed the model's maximum allowed length ({max_allowed})."
                )

    def _handle_sequence_splitting(self, max_allowed):
        """Split sequences without reserving space for BOS/EOS tokens."""
        new_sequences = {}
        self.chunks_mapping = {}
        chunk_size = max_allowed
        overlap = self.split_overlap

        if chunk_size <= overlap:
            logger.error(
                f"chunk_size ({chunk_size}) must be greater than overlap ({overlap}). Disabling splitting."
            )
            return

        for label, sequence in self.sequences.items():
            if len(sequence) <= chunk_size:
                new_sequences[label] = sequence
                continue

            self.original_sequences[label] = sequence
            chunks = []
            start = 0
            chunk_idx = 0
            while start < len(sequence):
                end = min(start + chunk_size, len(sequence))
                chunk_payload = sequence[start:end]
                chunk_label = f"{label}_chunk_{chunk_idx}"

                new_sequences[chunk_label] = chunk_payload
                self.chunk_payload_lengths[chunk_label] = len(chunk_payload)
                chunks.append(chunk_label)

                if end == len(sequence):
                    break
                start = end - overlap
                chunk_idx += 1

            self.chunks_mapping[label] = chunks

        self.sequences = new_sequences
        if hasattr(self, "num_sequences"):
            self.num_sequences = len(self.sequences)
        self.max_input_length = chunk_size
