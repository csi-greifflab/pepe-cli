import csv
import inspect
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import torch
from alive_progress import alive_bar
from numpy.lib.format import open_memmap

from pepe.utils import MultiIODispatcher, check_disk_free_space

logger = logging.getLogger("pepe.embedders.base_embedder")


class BaseEmbedder:
    # Subclass contract — set before _set_output_objects() / embed().
    num_sequences: int
    embedding_size: int
    num_heads: int
    sequences: Dict[str, str]
    data_loader: Iterable[Any]
    model: Any
    special_tokens: torch.Tensor
    tokenizer: Any
    layers: Optional[List[int]]
    num_layers: int
    substring_dict: Optional[Dict[str, str]]
    memmap_registry: Dict[Tuple[Any, Any, Any], Any]
    io_dispatcher: MultiIODispatcher
    sequence_labels: List[str]
    logits: Dict[str, Any]
    mean_pooled: Dict[str, Any]
    per_token: Dict[str, Any]
    attention_head: Dict[str, Any]
    attention_layer: Dict[str, Any]
    attention_model: Dict[str, Any]
    substring_pooled: Dict[str, Any]

    def __init__(self, args: Any) -> None:
        self.fasta_path = args.fasta_path
        self.model_link = args.model_name
        self.disable_special_tokens = args.disable_special_tokens
        if (
            self.model_link.endswith(".pt")
            or self.model_link.endswith(".pth")
            or self.model_link.startswith("custom:")
            or (
                os.path.exists(self.model_link)
                and (os.path.isfile(self.model_link) or os.path.isdir(self.model_link))
            )
        ):
            self.model_name = Path(
                self.model_link
            ).stem  # Use the stem of the model link as the model name
        else:
            self.model_name = re.sub(r"^.*?/", "", self.model_link)
        self.output_path = os.path.join(args.output_path, self.model_name)
        # Check if output directory exists and creates it if it's missing
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not args.experiment_name:
            self.output_prefix = os.path.splitext(os.path.basename(self.fasta_path))[
                0
            ]  # get filename without extension and path
        else:
            self.output_prefix = args.experiment_name
        self.substring_path = args.substring_path
        self.context = args.context
        self.layers: Optional[List[int]] = (
            [j for i in args.layers for j in i] if args.layers != [None] else None
        )
        self.substring_dict = (
            self._load_substrings(args.substring_path) if args.substring_path else None
        )
        self.batch_size = args.batch_size
        self.max_input_length = args.max_input_length
        self.split_long_sequences = getattr(args, "split_long_sequences", False)
        self.split_overlap = getattr(args, "split_overlap", 0)
        self.force_split_length = getattr(args, "force_split_length", None)
        self.trust_remote_code = getattr(args, "trust_remote_code", False)
        self.chunks_mapping: Dict[str, List[str]] = {}
        self.chunk_payload_lengths: Dict[str, int] = {}
        self.original_sequences: Dict[str, str] = {}
        # Set in embed() when mean_pooled must be stitched back from chunks but
        # per_token was not itself requested: per-token reps are then retained
        # internally so mean-pooled reconstruction has the data it needs.
        self._retain_per_token = False
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        self.output_types = args.extract_embeddings
        self.discard_padding = args.discard_padding
        self.streaming_output = args.streaming_output
        if self.discard_padding and self.streaming_output:
            logger.warning(
                "Warning: --discard_padding is not compatible with --streaming_output. Streaming output will be disabled."
            )
            self.streaming_output = False
        self.return_embeddings = False
        self.return_contacts = False
        self.return_logits = False
        for output_type in self.output_types:
            if "pooled" in output_type or "per_token" in output_type:
                self.return_embeddings = True
            if "attention" in output_type:
                self.return_contacts = True
            if output_type == "logits":
                self.return_logits = True
        self.flatten = args.flatten
        self.num_workers = args.num_workers if self.streaming_output else 1
        self.max_in_flight = self.num_workers * 2
        self.flush_batches_after = args.flush_batches_after * 1024**2  # in bytes
        self.precision = args.precision
        # self.log_memory = args.log_memory # TODO implement memory logging
        self.verbose = getattr(args, "verbose", False)
        self.total_gpu_time = 0.0
        self.total_backpressure_time = 0.0
        self.total_io_enqueue_time = 0.0

        # Set up checkpoint directory for crash recovery
        self.checkpoint_dir = self.output_path

    def _precision_to_dtype(self, precision: str, framework: str) -> Any:
        half_precision = ["float16", "16", "half"]
        full_precision = ["float32", "32", "full"]
        if precision in half_precision:
            if framework == "torch":
                return torch.float16
            elif framework == "numpy":
                return np.float16
        elif precision in full_precision:
            if framework == "torch":
                return torch.float32
            elif framework == "numpy":
                return np.float32
        else:
            raise ValueError(
                f"Unsupported precision: {precision}. Supported values are {half_precision} or {full_precision}."
            )

    def _set_output_objects(self) -> None:
        """Initialize output objects."""
        assert self.layers is not None
        self.sequence_labels = []
        self.logits = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_logits,
            "output_dir": os.path.join(self.output_path, "logits"),
            "shape": (
                self.num_sequences,
                self.max_input_length,
            ),
        }
        self.mean_pooled = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_mean_pooled,
            "output_dir": os.path.join(self.output_path, "mean_pooled"),
            "shape": (self.num_sequences, self.embedding_size),
        }
        self.per_token = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_per_token,
            "output_dir": os.path.join(self.output_path, "per_token"),
            "shape": (
                (
                    self.num_sequences,
                    self.max_input_length,
                    self.embedding_size,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_input_length * self.embedding_size,
                )
            ),
        }
        self.substring_pooled = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_substring_pooled,
            "output_dir": os.path.join(self.output_path, "substring_pooled"),
            "shape": (self.num_sequences, self.embedding_size),
        }
        self.attention_head = {
            "output_data": {
                layer: {head: [] for head in range(self.num_heads)}
                for layer in self.layers
            },
            "method": self._extract_attention_head,
            "output_dir": os.path.join(self.output_path, "attention_head"),
            "shape": (
                (
                    self.num_sequences,
                    self.max_input_length,
                    self.max_input_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_input_length**2,
                )
            ),
        }
        self.attention_layer = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_attention_layer,
            "output_dir": os.path.join(self.output_path, "attention_layer"),
            "shape": (
                (
                    self.num_sequences,
                    self.max_input_length,
                    self.max_input_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_input_length**2,
                )
            ),
        }
        self.attention_model = {
            "output_data": [],
            "method": self._extract_attention_model,
            "output_dir": os.path.join(self.output_path, "attention_model"),
            "shape": (
                (
                    self.num_sequences,
                    self.max_input_length,
                    self.max_input_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_input_length**2,
                )
            ),
        }

    # When changes made here, also update base_embedder.py BaseEmbedder.set_output_objects() method.
    def _get_output_types(self, args):
        output_types = []

        options_mapping = {
            "per_token": "per_token",
            "mean_pooled": "mean_pooled",
            "substring_pooled": "substring_pooled",
            "attention_head": "attention_head",
            "attention_layer": "attention_layer",
            "attention_model": "attention_model",
            "logits": "logits",
        }

        for option in args.extract_embeddings:
            if option in options_mapping:
                output_type = options_mapping[option]
                if output_type not in output_types:
                    output_types.append(output_type)

        return output_types

    def _make_output_filepath(self, output_type, output_dir, layer=None, head=None):
        base = f"{self.output_prefix}_{self.model_name}_{output_type}"
        if layer is not None:
            base += f"_layer_{layer}"
        if head is not None:
            base += f"_head_{head + 1}"
        return os.path.join(output_dir, base + ".npy")

    def preallocate_disk_space(self) -> Dict[Tuple[Any, Any, Any], Any]:
        assert self.layers is not None
        memmap_registry: Dict[Tuple[Any, Any, Any], Any] = {}
        total_bytes = 0
        for output_type in self.output_types:
            output_data = getattr(self, output_type)["output_data"]
            shape = getattr(self, output_type)["shape"]
            output_dir = getattr(self, output_type)["output_dir"]
            np_dtype = self._precision_to_dtype(self.precision, "numpy")
            bytes_per_array = np.dtype(np_dtype).itemsize * np.prod(shape)

            if isinstance(output_data, dict):
                for layer in self.layers:
                    if isinstance(output_data[layer], dict):  # e.g., all_heads
                        for head in range(self.num_heads):
                            file_path = self._make_output_filepath(
                                output_type, output_dir, layer, head
                            )
                            mode = "r+" if os.path.exists(file_path) else "w+"
                            output_data[layer][head] = open_memmap(
                                file_path, mode=mode, dtype=np_dtype, shape=shape
                            )
                            memmap_registry[(output_type, layer, head)] = output_data[
                                layer
                            ][head]
                            total_bytes += bytes_per_array
                    else:
                        file_path = self._make_output_filepath(
                            output_type, output_dir, layer
                        )
                        mode = "r+" if os.path.exists(file_path) else "w+"
                        output_data[layer] = open_memmap(
                            file_path, mode=mode, dtype=np_dtype, shape=shape
                        )
                        memmap_registry[(output_type, layer, None)] = output_data[layer]
                        total_bytes += bytes_per_array
            else:
                file_path = self._make_output_filepath(output_type, output_dir)
                mode = "r+" if os.path.exists(file_path) else "w+"
                output_array = open_memmap(
                    file_path, mode=mode, dtype=np_dtype, shape=shape
                )
                getattr(self, output_type)["output_data"] = output_array
                memmap_registry[(output_type, None, None)] = output_array
                total_bytes += bytes_per_array

        logger.info(f"Preparing to write {total_bytes / 1024**3:.2f} GB to disk.")
        check_disk_free_space(self.output_path, total_bytes)
        return memmap_registry

    def _load_substrings(self, substring_path):
        """Load substrings and store in a dictionary."""
        if substring_path:
            with open(substring_path) as f:
                reader = csv.reader(f)  # skip header
                next(reader)
                substring_dict = {rows[0]: rows[1] for rows in reader}
            return substring_dict
        else:
            return None

    def _safe_compute(
        self, toks: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """
        Try to run compute_outputs; on OOM, empty cache, split in half,
        recurse on each half, then concatenate.
        """
        try:
            return self._compute_outputs(
                self.model,
                toks,
                attention_mask,
                self.return_embeddings,
                self.return_contacts,
                self.return_logits,
            )
        except torch.OutOfMemoryError:
            logger.error("[GPU memory overflow] Decreasing batch size and retrying...")
            torch.cuda.empty_cache()
            B = toks.size(0)
            if B == 1:
                # can’t split anymore
                logger.error("OOM on single sample!")
                raise
            # split into two roughly equal chunks
            half = B // 2
            toks_chunks = torch.split(toks, [half, B - half], dim=0)
            mask_chunks: Any
            if attention_mask is not None:
                mask_chunks = torch.split(attention_mask, [half, B - half], dim=0)
            else:
                mask_chunks = (None, None)

            outs = [
                self._safe_compute(tc, mc) for tc, mc in zip(toks_chunks, mask_chunks)
            ]
            # outs is list of (logits, reps, attn)
            logits = (
                torch.cat([cast(torch.Tensor, o[0]) for o in outs], dim=0)
                if self.return_logits
                else None
            )
            representations = (
                torch.cat([cast(torch.Tensor, o[1]) for o in outs], dim=0)
                if self.return_embeddings
                else None
            )
            attention_matrices = (
                torch.cat([cast(torch.Tensor, o[2]) for o in outs], dim=0)
                if self.return_contacts
                else None
            )
            return logits, representations, attention_matrices

    def _active_output_types(self) -> List[str]:
        """Output types to extract per batch.

        Normally exactly the requested outputs, but when mean-pooled results
        must be reconstructed from long-sequence chunks we also extract
        per_token internally (it is not exported) so the stitched-together
        per-residue tensors are available to recompute the pooled vector.
        """
        if self._retain_per_token and "per_token" not in self.output_types:
            return list(self.output_types) + ["per_token"]
        return list(self.output_types)

    def embed(self) -> None:
        # Long-sequence mean-pooled reconstruction (in-memory only) needs the
        # per-residue representations of each chunk. If the user asked for
        # mean_pooled but not per_token, retain per_token internally for the
        # duration of the run so _reconstruct_chunks can stitch and re-pool.
        self._retain_per_token = (
            not self.streaming_output
            and bool(self.chunks_mapping)
            and "mean_pooled" in self.output_types
            and "per_token" not in self.output_types
        )
        if self._retain_per_token:
            logger.info(
                "Retaining per_token representations internally to reconstruct "
                "mean_pooled outputs for split sequences (not exported)."
            )
        if self.streaming_output:
            # Start centralized I/O dispatcher with checkpoint support
            self.io_dispatcher = MultiIODispatcher(
                self.memmap_registry,
                num_workers=self.num_workers,
                flush_bytes_limit=self.flush_batches_after,
                heavy_output_type="per_token",
                checkpoint_dir=self.checkpoint_dir,
            )

            # Check if we're resuming from a checkpoint
            resume_info = self.io_dispatcher.get_resume_info()
            if resume_info:
                logger.info(f"Resuming from checkpoint: {resume_info}")

        with (
            alive_bar(
                len(self.sequences),
                title=f"{self.model_name}: Generating embeddings ...",
            ) as bar,
            torch.no_grad(),
        ):
            offset = 0
            for (
                labels,
                strs,
                toks,
                attention_mask,
                substring_mask,
            ) in self.data_loader:
                toks = toks.to(self.device, non_blocking=True)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=True)
                pooling_mask = self._mask_special_tokens(
                    toks, self.special_tokens
                ).cpu()  # mask special tokens to avoid diluting signal when pooling embeddings
                t0_gpu = time.time()
                logits, representations, attention_matrices = self._safe_compute(
                    toks, attention_mask
                )
                self.total_gpu_time += time.time() - t0_gpu

                output_bundle = {
                    "logits": logits,
                    "attention_matrices": attention_matrices,
                    "representations": representations,
                    "batch_labels": labels,
                    "pooling_mask": pooling_mask,
                    "substring_mask": substring_mask,
                    "offset": offset,
                    "special_tokens": not self.disable_special_tokens,
                }
                if self.streaming_output:
                    # Apply backpressure if write queue is too full
                    t0_bp = time.time()
                    backpressure_triggered = False
                    while self.io_dispatcher.queue_fullness() > 0.6:
                        if not backpressure_triggered:
                            logger.warning(
                                f"[embed] Backpressure: queue fullness {self.io_dispatcher.queue_fullness():.2f}. Waiting for IOFlushWorker to catch up..."
                            )
                            backpressure_triggered = True
                        time.sleep(0.05)
                    if backpressure_triggered:
                        self.total_backpressure_time += time.time() - t0_bp

                t0_io = time.time()
                self._extract_batch(output_bundle)
                self.total_io_enqueue_time += time.time() - t0_io

                del logits, representations, attention_matrices

                offset += len(toks)

                self.sequence_labels.extend(labels)
                bar(len(toks))

            if self.split_long_sequences and self.chunks_mapping:
                self._reconstruct_chunks()

            if self.streaming_output:
                self.io_dispatcher.stop()

            logger.info("Finished extracting embeddings")
            if self.verbose:
                logger.info("--- Profiling Results ---")
                logger.info(f"Total GPU compute time: {self.total_gpu_time:.2f}s")
                logger.info(
                    f"Total Backpressure wait time: {self.total_backpressure_time:.2f}s"
                )
                logger.info(f"Total IO Enqueue time: {self.total_io_enqueue_time:.2f}s")
                if self.total_gpu_time > 0:
                    overhead = (
                        self.total_backpressure_time + self.total_io_enqueue_time
                    ) / self.total_gpu_time
                    logger.info(f"IO Overhead ratio: {overhead:.2f}x")

        # After successful completion, clean up the checkpoint file
        if self.streaming_output:
            self._cleanup_checkpoint()

    def _cleanup_checkpoint(self) -> None:
        """Clean up the checkpoint file after successful completion."""
        checkpoint_file = os.path.join(self.checkpoint_dir, "global_checkpoint.json")
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                logger.info(f"Cleaned up checkpoint file: {checkpoint_file}")
            except Exception as e:
                logger.error(
                    f"Warning: Could not remove checkpoint file {checkpoint_file}: {e}"
                )
        else:
            logger.info("No checkpoint file found to clean up.")

    def _load_data(
        self,
        sequences: Optional[Dict[str, str]] = None,
        substring_dict: Optional[Dict[str, str]] = None,
        bracket_type: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def _initialize_model(
        self,
        model_link: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> Tuple[Any, ...]:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def _load_layers(self, layers: Optional[List[int]] = None) -> List[int]:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def _compute_outputs(
        self,
        model: Any,
        toks: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        return_embeddings: bool,
        return_contacts: bool,
        return_logits: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def get_substring_positions(
        self, label: str, special_tokens: int, context: int = 0
    ) -> Tuple[int, int]:
        """Get the start and end positions of the substring in the full sequence."""
        full_sequence = self.sequences[label]

        if self.substring_dict is None:
            raise SystemExit(f"No matching substring found for {label}")
        try:
            substring = self.substring_dict[label]
        except KeyError:
            raise SystemExit(f"No matching substring found for {label}")
        # remove '-' from substring
        substring = substring.replace("-", "")

        # get position of substring in sequence
        start = max(full_sequence.find(substring) - context, 0) + int(special_tokens)
        end = min(start + len(substring) + context, len(full_sequence)) + special_tokens

        return start, end

    def _extract_batch(
        self,
        output_bundle: Dict[str, Any],
    ) -> None:
        for output_type in self._active_output_types():
            sig = inspect.signature(getattr(self, output_type)["method"])
            needed_args = {
                k: v for k, v in output_bundle.items() if k in sig.parameters
            }
            getattr(self, output_type)["method"](**needed_args)
        # clear the output bundle to free up memory
        output_bundle.clear()
        del output_bundle

    def _mask_special_tokens(
        self,
        input_tensor: torch.Tensor,
        special_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create a boolean mask for special tokens in the input tensor.

        """
        if (
            special_tokens is not None
        ):  # Create a boolean mask: True where the value is not in special_tokens.
            mask = ~torch.isin(input_tensor, special_tokens)
        else:  # Create a boolean mask: True where the value is not 0, 1, or 2.
            mask = (input_tensor != 0) & (input_tensor != 1) & (input_tensor != 2)
        # Convert and return the boolean mask to boolean type.
        return mask

    def _extract_logits(
        self,
        logits: Any,
        offset: int,
    ) -> None:
        assert self.layers is not None
        for layer in self.layers:
            tensor = logits[layer - 1]
            if self.streaming_output:
                # output_file = self.logits["output_data"][layer]
                # self.write_batch_to_disk(output_file, tensor, offset)

                self.io_dispatcher.enqueue(
                    output_type="logits",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.logits["output_data"][layer].extend(tensor)

    def _extract_mean_pooled(
        self,
        representations: Any,
        batch_labels: List[str],
        pooling_mask: torch.Tensor,
        offset: int,
    ) -> None:
        assert self.layers is not None
        for layer in self.layers:
            tensor = torch.stack(
                [
                    (
                        (pooling_mask[i].unsqueeze(-1) * representations[layer][i]).sum(
                            0
                        )
                        / pooling_mask[i].unsqueeze(-1).sum(0)
                    )
                    for i in range(len(batch_labels))
                ]
            )
            if self.streaming_output:
                # output_file = self.embeddings["output_data"][layer]
                # self.write_batch_to_disk(output_file, tensor, offset)
                self.io_dispatcher.enqueue(
                    output_type="mean_pooled",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.mean_pooled["output_data"][layer].extend(tensor)

    def _extract_per_token(
        self,
        representations: Any,
        batch_labels: List[str],
        pooling_mask: torch.Tensor,
        offset: int,
    ) -> None:
        assert self.layers is not None
        if not self.discard_padding:
            for layer in self.layers:
                tensor = torch.stack(
                    [representations[layer][i] for i in range(len(batch_labels))]
                )
                if self.flatten:
                    tensor = tensor.flatten(start_dim=1)
                if self.streaming_output:
                    # output_file = self.per_token["output_data"][layer]
                    # self.write_batch_to_disk(output_file, tensor, offset)
                    self.io_dispatcher.enqueue(
                        output_type="per_token",
                        layer=layer,
                        head=None,
                        offset=offset,
                        array=np.ascontiguousarray(
                            tensor.cpu().numpy()
                        ),  # Ensure it's on CPU and NumPy
                    )
                else:
                    self.per_token["output_data"][layer].extend(tensor)
        else:
            for layer in self.layers:
                if self.flatten:
                    self.per_token["output_data"][layer].extend(
                        [
                            representations[layer][i][pooling_mask[i]].flatten()
                            for i in range(len(batch_labels))
                        ]
                    )
                else:
                    self.per_token["output_data"][layer].extend(
                        [
                            representations[layer][i][pooling_mask[i]]
                            for i in range(len(batch_labels))
                        ]
                    )

    def _extract_attention_head(
        self,
        attention_matrices: Any,
        batch_labels: List[str],
        offset: int,
    ) -> None:
        assert self.layers is not None
        for layer in self.layers:
            for head in range(self.num_heads):
                tensor = torch.stack(
                    [
                        attention_matrices[layer - 1, i, head]
                        for i in range(len(batch_labels))
                    ]
                )
                if self.flatten:
                    tensor = tensor.flatten(start_dim=1)
                if self.streaming_output:
                    # output_file = self.attention_matrices_all_heads["output_data"][
                    #    layer
                    # ][head]
                    # self.write_batch_to_disk(output_file, tensor, offset)
                    self.io_dispatcher.enqueue(
                        output_type="attention_head",
                        layer=layer,
                        head=head,
                        offset=offset,
                        array=np.ascontiguousarray(
                            tensor.cpu().numpy()
                        ),  # Ensure it's on CPU and NumPy
                    )
                else:
                    self.attention_head["output_data"][layer][head].extend(tensor)

    def _extract_attention_layer(
        self,
        attention_matrices: Any,
        batch_labels: List[str],
        offset: int,
    ) -> None:
        assert self.layers is not None
        for layer in self.layers:
            tensor = torch.stack(
                [
                    attention_matrices[layer - 1, i].mean(0)
                    for i in range(len(batch_labels))
                ]
            )
            if self.flatten:
                tensor = tensor.flatten(start_dim=1)
            if self.streaming_output:
                # output_file = self.attention_matrices_average_layers["output_data"][
                #    layer
                # ]
                # self.write_batch_to_disk(output_file, tensor, offset)
                self.io_dispatcher.enqueue(
                    output_type="attention_layer",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.attention_layer["output_data"][layer].extend(tensor)

    def _extract_attention_model(
        self,
        attention_matrices: Any,
        batch_labels: List[str],
        offset: int,
    ) -> None:
        tensor = torch.stack(
            [
                attention_matrices[:, i].mean(dim=(0, 1))
                for i in range(len(batch_labels))
            ]
        )
        if self.flatten:
            tensor = tensor.flatten(start_dim=1)
        if self.streaming_output:
            # output_file = self.attention_matrices_average_all["output_data"]
            # self.write_batch_to_disk(output_file, tensor, offset)
            self.io_dispatcher.enqueue(
                output_type="attention_model",
                layer=None,
                head=None,
                offset=offset,
                array=np.ascontiguousarray(
                    tensor.cpu().numpy()
                ),  # Ensure it's on CPU and NumPy
            )
        else:
            self.attention_model["output_data"].extend(tensor)

    def _extract_substring_pooled(
        self,
        representations: Any,
        substring_mask: Any,
        offset: int,
    ) -> None:
        assert self.layers is not None
        for layer in self.layers:
            tensor = torch.stack(
                [
                    (
                        (mask.unsqueeze(-1) * representations[layer][i]).sum(0)
                        / mask.unsqueeze(-1).sum(0)
                    )
                    for i, mask in enumerate(substring_mask)
                ]
            )
            if self.streaming_output:
                # output_file = self.substring_pooled["output_data"][layer]
                # self.write_batch_to_disk(output_file, tensor, offset)
                self.io_dispatcher.enqueue(
                    output_type="substring_pooled",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.substring_pooled["output_data"][layer].extend(tensor)

    def _prepare_tensor(self, data_list: Any, flatten: bool) -> Any:
        if self.discard_padding:
            # Handle variable-length sequences by returning an object array of numpy arrays
            return np.array([t.numpy() for t in data_list], dtype=object)

        tensor = torch.stack(data_list, dim=0)
        if flatten:
            tensor = tensor.flatten(start_dim=1)
        return tensor.numpy()

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().contiguous().numpy()

    def export_to_disk(self) -> None:
        assert self.layers is not None
        for output_type in self.output_types:
            logger.info(f"Saving {output_type} representations...")

            output_data = getattr(self, output_type)["output_data"]
            output_dir = getattr(self, output_type)["output_dir"]

            if isinstance(output_data, dict):
                for layer in self.layers:
                    if isinstance(output_data[layer], dict):  # e.g., attention_head
                        for head in range(self.num_heads):
                            tensor = self._prepare_tensor(
                                output_data[layer][head], self.flatten
                            )
                            file_path = self._make_output_filepath(
                                output_type, output_dir, layer, head
                            )
                            np.save(file_path, tensor)
                            logger.info(
                                f"Saved {output_type} layer {layer} head {head + 1} to {file_path}"
                            )
                    else:
                        # Handle layer-based outputs (mean_pooled, per_token, substring_pooled, attention_layer, logits)
                        flatten = self.flatten and output_type == "per_token"
                        tensor = self._prepare_tensor(output_data[layer], flatten)
                        file_path = self._make_output_filepath(
                            output_type, output_dir, layer
                        )
                        np.save(file_path, tensor)
                        logger.info(f"Saved {output_type} layer {layer} to {file_path}")
            else:
                # Handle model-level outputs (attention_model)
                tensor = self._prepare_tensor(output_data, self.flatten)
                file_path = self._make_output_filepath(output_type, output_dir)
                np.save(file_path, tensor)
                logger.info(f"Saved {output_type} to {file_path}")

    def export_sequence_indices(self) -> None:
        """Save sequence indices to a CSV file."""
        input_file_name = os.path.basename(self.fasta_path)
        # replace file extension with _idx.csv regardless of pattern
        output_file_name = os.path.splitext(input_file_name)[0] + "_idx.csv"
        output_file_idx = os.path.join(self.output_path, output_file_name)
        with open(output_file_idx, "w") as f:
            f.write("index,sequence_id\n")
            for i, label in enumerate(self.sequence_labels):
                f.write(f"{i},{label}\n")
        logger.info(f"Saved sequence indices to {output_file_idx}")

    def _create_output_dirs(self) -> None:
        for output_type in self.output_types:
            output_type_path = os.path.join(self.output_path, output_type)
            if not os.path.exists(output_type_path):
                os.makedirs(output_type_path)

    def run(self) -> None:
        self._create_output_dirs()
        if self.streaming_output:
            logger.info("Preallocating disk space...")
            self.memmap_registry = self.preallocate_disk_space()
            logger.info("Preallocated disk space")
        logger.info("Created output directories")

        logger.info("Start embedding extraction")
        self.embed()
        logger.info("Finished embedding extraction")

        logger.info("Saving embeddings...")
        if not self.streaming_output:
            self.export_to_disk()

        self.export_sequence_indices()

        # Final cleanup of checkpoint file (in case embed() didn't handle it)
        if self.streaming_output:
            self._cleanup_checkpoint()

        logger.info("Pipeline completed successfully!")

    def _check_max_input_length(self) -> None:
        """Check if max_input_length exceeds the model's allowed maximum length and handle splitting."""
        max_allowed = self._get_model_max_allowed()
        if max_allowed is None:
            return

        # Check if any sequence exceeds the model limit (-2 for cls/eos special tokens)
        sequences_too_long = any(
            len(s) > (max_allowed - 2) for s in self.sequences.values()
        )
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

    _UNKNOWN_MAX_LENGTH_THRESHOLD = 1e9

    @classmethod
    def _is_unknown_max_length(cls, value: Any) -> bool:
        """Treat HuggingFace sentinel model_max_length (~1e30) as unknown."""
        try:
            return float(value) >= cls._UNKNOWN_MAX_LENGTH_THRESHOLD
        except (TypeError, ValueError):
            return False

    def _get_model_max_allowed(self) -> Optional[int]:
        """Estimate the maximum allowed sequence length for the model."""
        if hasattr(self, "force_split_length") and self.force_split_length is not None:
            return self.force_split_length

        max_allowed = None
        max_source = None
        if hasattr(self, "model"):
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "max_position_embeddings"
            ):
                max_allowed = self.model.config.max_position_embeddings
                max_source = "config.max_position_embeddings"
            elif hasattr(self, "tokenizer") and hasattr(
                self.tokenizer, "model_max_length"
            ):
                max_allowed = self.tokenizer.model_max_length
                max_source = "tokenizer.model_max_length"

            if max_allowed is None and hasattr(self.model, "max_positions"):
                max_allowed = getattr(self.model, "max_positions", None)
                if callable(max_allowed):
                    max_allowed = max_allowed()
                elif hasattr(self.model, "max_positions") and isinstance(
                    self.model.max_positions, int
                ):
                    max_allowed = self.model.max_positions
                max_source = "model.max_positions"

            if (
                max_allowed is None
                and hasattr(self.model, "config")
                and hasattr(self.model.config, "n_positions")
            ):
                max_allowed = self.model.config.n_positions
                max_source = "config.n_positions"

        if max_allowed is not None and self._is_unknown_max_length(max_allowed):
            logger.info(
                "Model maximum sequence length is unknown (%s=%s); "
                "length limits will not be enforced automatically.",
                max_source or "max_length",
                max_allowed,
            )
            return None
        return max_allowed

    def _handle_sequence_splitting(self, max_allowed: int) -> None:
        """Split sequences that exceed max_allowed into chunks."""
        new_sequences = {}
        self.chunks_mapping = {}
        special_tokens_count = 2  # cls + eos (conservative default)
        chunk_size = max_allowed - special_tokens_count
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
        # Update num_sequences
        if hasattr(self, "num_sequences"):
            self.num_sequences = len(self.sequences)

        # Update max_input_length to chunk size
        self.max_input_length = chunk_size

    def _reconstruct_chunks(self) -> None:
        """Reconstruct original sequences from chunks in memory."""
        if not self.chunks_mapping or self.streaming_output:
            return

        assert self.layers is not None
        logger.info("Reconstructing original sequences from chunks...")

        label_to_idx = {label: i for i, label in enumerate(self.sequence_labels)}

        # For each output type, we need to rebuild the data
        rebuild_logits = "logits" in self.output_types
        per_token_requested = "per_token" in self.output_types
        rebuild_mean_pooled = "mean_pooled" in self.output_types
        # Mean-pooled reconstruction is derived from the stitched per-residue
        # tensors, so per_token must be (re)built as scratch whenever mean_pooled
        # needs it — even if per_token itself was not a requested output. In that
        # scratch-only case embed() has retained per_token via _retain_per_token.
        build_per_token = per_token_requested or rebuild_mean_pooled

        if rebuild_mean_pooled and not (per_token_requested or self._retain_per_token):
            # embed() sets _retain_per_token for exactly this case; if it is unset
            # the per-token reps were dropped and mean_pooled cannot be stitched.
            # Fail loudly rather than raising an opaque KeyError mid-loop.
            raise RuntimeError(
                "Cannot reconstruct mean_pooled for split sequences without "
                "per_token representations (internal per-token retention was "
                "not enabled)."
            )

        output_type_map = []
        if rebuild_logits:
            output_type_map.append(("logits", self.logits))
        if build_per_token:
            output_type_map.append(("per_token", self.per_token))
        if rebuild_mean_pooled:
            output_type_map.append(("mean_pooled", self.mean_pooled))

        # Map original labels to their new index in the final list
        new_sequence_labels = []
        labels_processed = set()

        # Temporary storage for reconstructed results keyed by output type
        reconstructed_data: Dict[str, Dict[int, List[Any]]] = {
            output_type: {layer: [] for layer in self.layers}
            for output_type, _ in output_type_map
        }

        for label in self.sequence_labels:
            # Find the original label
            orig_label = label
            is_chunk = False
            for parent, chunks in self.chunks_mapping.items():
                if label in chunks:
                    orig_label = parent
                    is_chunk = True
                    break

            if orig_label in labels_processed:
                continue

            labels_processed.add(orig_label)
            new_sequence_labels.append(orig_label)

            if not is_chunk:
                # Just copy the existing data
                idx = label_to_idx[label]
                for output_type, obj in output_type_map:
                    for layer in self.layers:
                        reconstructed_data[output_type][layer].append(
                            obj["output_data"][layer][idx]
                        )
                continue

            # Reconstruct from chunks
            chunk_labels = self.chunks_mapping[orig_label]

            # 1. Concatenate per-token and logits
            if build_per_token or rebuild_logits:
                for output_type, obj, flag in [
                    ("per_token", self.per_token, build_per_token),
                    ("logits", self.logits, rebuild_logits),
                ]:
                    if flag:
                        for layer in self.layers:
                            parts = []
                            for i, cl in enumerate(chunk_labels):
                                idx = label_to_idx[cl]
                                full_tensor = obj["output_data"][layer][idx]
                                payload_len = self.chunk_payload_lengths[cl]

                                # Identify indices for extraction
                                start_idx = 1
                                if i > 0:
                                    start_idx += self.split_overlap

                                end_idx = 1 + payload_len
                                meat = full_tensor[start_idx:end_idx]

                                if i == 0:
                                    meat = torch.cat([full_tensor[0:1], meat], dim=0)

                                if i == len(chunk_labels) - 1:
                                    expected_unpadded_len = 1 + payload_len
                                    # Safe bet: if there's no EOS, it's either padding or out of bounds.
                                    # To be robust during reconstruction of standard models, we should append if `add_special_tokens` was True and the tokenizer adds EOS.
                                    # The simplest heuristic: the original tokenizer encoded "" into >1 token or it has an EOS token
                                    eos_count = (
                                        len(
                                            self.tokenizer.encode(
                                                "", add_special_tokens=True
                                            )
                                        )
                                        - 1
                                        if hasattr(self, "tokenizer")
                                        and hasattr(self.tokenizer, "encode")
                                        else 0
                                    )

                                    if eos_count > 0:
                                        appended = full_tensor[
                                            expected_unpadded_len : expected_unpadded_len
                                            + 1
                                        ]
                                        meat = torch.cat([meat, appended], dim=0)

                                parts.append(meat)

                            reconstructed = torch.cat(parts, dim=0)
                            reconstructed_data[output_type][layer].append(reconstructed)

            # 2. Handle Mean Pooled
            if rebuild_mean_pooled:
                for layer in self.layers:
                    full_per_token = reconstructed_data["per_token"][layer][-1]
                    reconstructed_mean = full_per_token.mean(0)
                    reconstructed_data["mean_pooled"][layer].append(reconstructed_mean)

        # Replace original data with reconstructed data
        self.sequence_labels = new_sequence_labels
        self.num_sequences = len(self.sequence_labels)
        for output_type, obj in output_type_map:
            # per_token may have been rebuilt only as scratch for mean-pooling;
            # don't overwrite (or export) it unless the user asked for it.
            if output_type == "per_token" and not per_token_requested:
                continue
            for layer in self.layers:
                obj["output_data"][layer] = reconstructed_data[output_type][layer]

        logger.info(
            f"Reconstruction complete. Final sequence count: {self.num_sequences}"
        )
