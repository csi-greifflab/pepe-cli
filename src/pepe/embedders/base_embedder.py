import os
import csv
import torch
import re
import numpy as np
from numpy.lib.format import open_memmap
import inspect
import gc
from pepe.utils import MultiIODispatcher, check_disk_free_space
from alive_progress import alive_bar
import time
from pathlib import Path
import logging

logger = logging.getLogger("src.embedders.base_embedder")


class BaseEmbedder:
    def __init__(self, args):
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
        self.layers = (
            [j for i in args.layers for j in i] if args.layers != [None] else None
        )
        self.substring_dict = (
            self._load_substrings(args.substring_path) if args.substring_path else None
        )
        self.batch_size = args.batch_size
        self.max_input_length = args.max_input_length
        self.split_long_sequences = getattr(args, "split_long_sequences", False)
        self.split_overlap = getattr(args, "split_overlap", 0)
        self.chunks_mapping = {}  # Map original label to list of chunk labels
        self.chunk_payload_lengths = {} # Map chunk label to its payload length
        self.original_sequences = {} # Store original sequence for reference
        if torch.cuda.is_available() and args.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.output_types = args.extract_embeddings
        self.discard_padding = args.discard_padding
        self.flatten = args.flatten
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
        self.streaming_output = args.streaming_output
        self.num_workers = args.num_workers if self.streaming_output else 1
        self.max_in_flight = self.num_workers * 2
        self.flush_batches_after = args.flush_batches_after * 1024**2  # in bytes
        self.precision = args.precision
        # self.log_memory = args.log_memory # TODO implement memory logging

        # Set up checkpoint directory for crash recovery
        self.checkpoint_dir = self.output_path

    def _precision_to_dtype(self, precision, framework):
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

    def _set_output_objects(self):
        """Initialize output objects."""
        self._check_max_input_length()
        self.sequence_labels = []
        self.logits = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_logits,
            "output_dir": os.path.join(self.output_path, "logits"),
            "shape": (
                self.num_sequences,  # type: ignore
                self.max_input_length,
            ),
        }
        self.mean_pooled = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_mean_pooled,
            "output_dir": os.path.join(self.output_path, "mean_pooled"),
            "shape": (self.num_sequences, self.embedding_size),  # type: ignore
        }
        self.per_token = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_per_token,
            "output_dir": os.path.join(self.output_path, "per_token"),
            "shape": (
                (
                    self.num_sequences,  # type: ignore
                    self.max_input_length,
                    self.embedding_size,  # type: ignore
                )
                if not self.flatten
                else (
                    self.num_sequences,  # type: ignore
                    self.max_input_length * self.embedding_size,  # type: ignore
                )
            ),
        }
        self.substring_pooled = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_substring_pooled,
            "output_dir": os.path.join(self.output_path, "substring_pooled"),
            "shape": (self.num_sequences, self.embedding_size),  # type: ignore
        }
        self.attention_head = {
            "output_data": {
                layer: {head: [] for head in range(self.num_heads)}  # type: ignore
                for layer in self.layers  # type: ignore
            },
            "method": self._extract_attention_head,
            "output_dir": os.path.join(self.output_path, "attention_head"),
            "shape": (
                (
                    self.num_sequences,  # type: ignore
                    self.max_input_length,
                    self.max_input_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,  # type: ignore
                    self.max_input_length**2,
                )
            ),
        }
        self.attention_layer = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_attention_layer,
            "output_dir": os.path.join(self.output_path, "attention_layer"),
            "shape": (
                (
                    self.num_sequences,  # type: ignore
                    self.max_input_length,
                    self.max_input_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,  # type: ignore
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
                    self.num_sequences,  # type: ignore
                    self.max_input_length,
                    self.max_input_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,  # type: ignore
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

    def preallocate_disk_space(self):
        memmap_registry = {}
        total_bytes = 0
        for output_type in self.output_types:
            output_data = getattr(self, output_type)["output_data"]
            shape = getattr(self, output_type)["shape"]
            output_dir = getattr(self, output_type)["output_dir"]
            np_dtype = self._precision_to_dtype(self.precision, "numpy")
            bytes_per_array = np.dtype(np_dtype).itemsize * np.prod(shape)  # type: ignore

            if isinstance(output_data, dict):
                for layer in self.layers:  # type: ignore
                    if isinstance(output_data[layer], dict):  # e.g., all_heads
                        for head in range(self.num_heads):  # type: ignore
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
                setattr(getattr(self, output_type), "output_data", output_array)
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

    def _safe_compute(self, toks, attention_mask):
        """
        Try to run compute_outputs; on OOM, empty cache, split in half,
        recurse on each half, then concatenate.
        """
        try:
            return self._compute_outputs(  # type: ignore
                self.model,  # type: ignore
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
            if attention_mask is not None:
                mask_chunks = torch.split(attention_mask, [half, B - half], dim=0)
            else:
                mask_chunks = [None, None]

            outs = [
                self._safe_compute(tc, mc) for tc, mc in zip(toks_chunks, mask_chunks)
            ]
            # outs is list of (logits, reps, attn)
            logits = (
                torch.cat([o[0] for o in outs], dim=0) if self.return_logits else None  # type: ignore
            )
            representations = (
                torch.cat([o[1] for o in outs], dim=0)  # type: ignore
                if self.return_embeddings
                else None
            )
            attention_matrices = (
                torch.cat([o[2] for o in outs], dim=0) if self.return_contacts else None  # type: ignore
            )
            return logits, representations, attention_matrices

    def embed(self):
        if self.streaming_output:
            # Start centralized I/O dispatcher with checkpoint support
            self.io_dispatcher = MultiIODispatcher(
                self.memmap_registry,
                flush_bytes_limit=self.flush_batches_after,
                heavy_output_type="per_token",
                checkpoint_dir=self.checkpoint_dir,
            )

            # Check if we're resuming from a checkpoint
            resume_info = self.io_dispatcher.get_resume_info()
            if resume_info:
                logger.info(f"Resuming from checkpoint: {resume_info}")

        with alive_bar(
            len(self.sequences),  # type: ignore
            title=f"{self.model_name}: Generating embeddings ...",
        ) as bar, torch.no_grad():
            offset = 0
            for (
                labels,
                strs,
                toks,
                attention_mask,
                substring_mask,
            ) in self.data_loader:  # type: ignore
                toks = toks.to(self.device, non_blocking=True)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=True)
                pooling_mask = self._mask_special_tokens(
                    toks, self.special_tokens  # type: ignore
                ).cpu()  # mask special tokens to avoid diluting signal when pooling embeddings
                logits, representations, attention_matrices = self._safe_compute(
                    toks, attention_mask
                )
                torch.cuda.empty_cache()

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
                    while self.io_dispatcher.queue_fullness() > 0.6:
                        logger.warning(
                            "[embed] Backpressure: waiting for IOFlushWorker to catch up..."
                        )
                        time.sleep(0.05)
                self._extract_batch(output_bundle)

                del logits, representations, attention_matrices
                gc.collect()
                torch.cuda.empty_cache()

                offset += len(toks)

                self.sequence_labels.extend(labels)
                bar(len(toks))

            if self.split_long_sequences and self.chunks_mapping:
                self._reconstruct_chunks()

            if self.streaming_output:
                self.io_dispatcher.stop()

            logger.info("Finished extracting embeddings")

        # After successful completion, clean up the checkpoint file
        if self.streaming_output:
            self._cleanup_checkpoint()

    def _cleanup_checkpoint(self):
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

    def _load_data(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def _initialize_model(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def _load_layers(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def get_substring_positions(self, label, special_tokens, context=0):
        """Get the start and end positions of the substring in the full sequence."""
        full_sequence = self.sequences[label]  # type: ignore

        try:
            substring = self.substring_dict[label]  # type: ignore
        except KeyError:
            SystemExit(f"No matching substring found for {label}")
        # remove '-' from substring
        substring = substring.replace("-", "")

        # get position of substring in sequence
        start = max(full_sequence.find(substring) - context, 0) + int(
            special_tokens
        )
        end = (
            min(start + len(substring) + context, len(full_sequence))
            + special_tokens
        )

        return start, end

    def _extract_batch(
        self,
        output_bundle,
    ):
        for output_type in self.output_types:
            sig = inspect.signature(getattr(self, output_type)["method"])
            needed_args = {
                k: v for k, v in output_bundle.items() if k in sig.parameters
            }
            getattr(self, output_type)["method"](**needed_args)
        # clear the output bundle to free up memory
        output_bundle.clear()
        del output_bundle
        torch.cuda.empty_cache()

    def _mask_special_tokens(self, input_tensor, special_tokens=None):
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
        logits,
        offset,
    ):
        for layer in self.layers:  # type: ignore
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
        representations,
        batch_labels,
        pooling_mask,
        offset,
    ):
        for layer in self.layers:  # type: ignore
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
        representations,
        batch_labels,
        offset,
    ):
        if not self.discard_padding:
            for layer in self.layers:  # type: ignore
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
        else:  # TODO remove padding tokens
            logger.warning("Feature not implemented yet")
            pass
            for layer in self.layers:  # type: ignore
                if self.flatten:
                    self.per_token["output_data"][layer].extend(
                        [
                            representations[layer][i].flatten(start_dim=1)
                            for i in range(len(batch_labels))
                        ]
                    )
                else:
                    self.per_token["output_data"][layer].extend(
                        [representations[layer][i] for i in range(len(batch_labels))]
                    )

    def _extract_attention_head(
        self,
        attention_matrices,
        batch_labels,
        offset,
    ):
        for layer in self.layers:  # type: ignore
            for head in range(self.num_heads):  # type: ignore
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
                        output_type="attention_matrices_all_heads",
                        layer=layer,
                        head=head,
                        offset=offset,
                        array=np.ascontiguousarray(
                            tensor.cpu().numpy()
                        ),  # Ensure it's on CPU and NumPy
                    )
                else:
                    self.attention_head["output_data"][layer][
                        head
                    ].extend(tensor)

    def _extract_attention_layer(
        self,
        attention_matrices,
        batch_labels,
        offset,
    ):
        for layer in self.layers:  # type: ignore
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
                    output_type="attention_matrices_average_layers",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.attention_layer["output_data"][layer].extend(
                    tensor
                )

    def _extract_attention_model(
        self,
        attention_matrices,
        batch_labels,
        offset,
    ):
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
                output_type="attention_matrices_average_all",
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
        representations,
        substring_mask,
        offset,
    ):
        for layer in self.layers:  # type: ignore
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

    def _prepare_tensor(self, data_list, flatten):
        tensor = torch.stack(data_list, dim=0)
        if flatten:
            tensor = tensor.flatten(start_dim=1)
        return tensor.numpy()

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().contiguous().numpy()

    def export_to_disk(self):
        for output_type in self.output_types:
            logger.info(f"Saving {output_type} representations...")

            output_data = getattr(self, output_type)["output_data"]
            output_dir = getattr(self, output_type)["output_dir"]

            if isinstance(output_data, dict):
                for layer in self.layers:  # type: ignore
                    if isinstance(output_data[layer], dict):  # e.g., attention_head
                        for head in range(self.num_heads):  # type: ignore
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

    def export_sequence_indices(self):
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

    def _create_output_dirs(self):
        for output_type in self.output_types:
            output_type_path = os.path.join(self.output_path, output_type)
            if not os.path.exists(output_type_path):
                os.makedirs(output_type_path)

    def run(self):
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
    def _check_max_input_length(self):
        """Check if max_input_length exceeds the model's allowed maximum length and handle splitting."""
        max_allowed = self._get_model_max_allowed()
        if max_allowed is None:
            return

        # Check if splitting is needed (either by explicit max_input_length or by reviewing sequences)
        needs_splitting = False
        if isinstance(self.max_input_length, int) and self.max_input_length > max_allowed:
            needs_splitting = True
        elif self.split_long_sequences:
            # Check if any individual sequence exceeds the limit
            if any(len(s) > (max_allowed - 2) for s in self.sequences.values()): # -2 for cls/eos
                needs_splitting = True

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

    def _get_model_max_allowed(self):
        """Estimate the maximum allowed sequence length for the model."""
        max_allowed = None
        if hasattr(self, "model"):
            if hasattr(self.model, "config") and hasattr(self.model.config, "max_position_embeddings"):
                max_allowed = self.model.config.max_position_embeddings
            elif hasattr(self, "tokenizer") and hasattr(self.tokenizer, "model_max_length"):
                max_allowed = self.tokenizer.model_max_length
            
            if max_allowed is None and hasattr(self.model, "max_positions"):
                max_allowed = getattr(self.model, "max_positions", None)
                if callable(max_allowed):
                    max_allowed = max_allowed()
                elif hasattr(self.model, "max_positions") and isinstance(self.model.max_positions, int):
                    max_allowed = self.model.max_positions
                    
            if max_allowed is None and hasattr(self.model, "config") and hasattr(self.model.config, "n_positions"):
                max_allowed = self.model.config.n_positions
        return max_allowed

    def _handle_sequence_splitting(self, max_allowed):
        """Split sequences that exceed max_allowed into chunks."""
        new_sequences = {}
        self.chunks_mapping = {}
        special_tokens_count = 2 # cls + eos (conservative default)
        chunk_size = max_allowed - special_tokens_count
        overlap = self.split_overlap

        if chunk_size <= overlap:
            logger.error(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap}). Disabling splitting.")
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
        if hasattr(self, 'num_sequences'):
            self.num_sequences = len(self.sequences)
        
        # Update max_input_length to chunk size
        self.max_input_length = chunk_size

    def _reconstruct_chunks(self):
        """Reconstruct original sequences from chunks in memory."""
        if not self.chunks_mapping or self.streaming_output:
            return

        logger.info("Reconstructing original sequences from chunks...")
        
        label_to_idx = {label: i for i, label in enumerate(self.sequence_labels)}
        
        # For each output type, we need to rebuild the data
        outputs_to_rebuild = []
        rebuild_logits = self.return_logits and hasattr(self, "logits")
        rebuild_per_token = self.return_embeddings and hasattr(self, "per_token")
        rebuild_mean_pooled = self.return_embeddings and hasattr(self, "mean_pooled")
        
        if rebuild_logits: outputs_to_rebuild.append(self.logits)
        if rebuild_per_token: outputs_to_rebuild.append(self.per_token)
        if rebuild_mean_pooled: outputs_to_rebuild.append(self.mean_pooled)
        
        # Map original labels to their new index in the final list
        new_sequence_labels = []
        labels_processed = set()
        
        # Temporary storage for reconstructed results
        reconstructed_data = {id(obj): {layer: [] for layer in self.layers} for obj in outputs_to_rebuild}

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
                for obj in outputs_to_rebuild:
                    for layer in self.layers:
                        reconstructed_data[id(obj)][layer].append(obj["output_data"][layer][idx])
                continue
            
            # Reconstruct from chunks
            chunk_labels = self.chunks_mapping[orig_label]
            
            # 1. Concatenate per-token and logits
            if rebuild_per_token or rebuild_logits:
                for obj, flag in [(self.per_token, rebuild_per_token), (self.logits, rebuild_logits)]:
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
                                    meat = torch.cat([meat, full_tensor[1 + payload_len : 2 + payload_len]], dim=0)

                                parts.append(meat)

                            reconstructed = torch.cat(parts, dim=0)
                            reconstructed_data[id(obj)][layer].append(reconstructed)

            # 2. Handle Mean Pooled
            if rebuild_mean_pooled:
                for layer in self.layers:
                    full_per_token = reconstructed_data[id(self.per_token)][layer][-1]
                    reconstructed_mean = full_per_token.mean(0)
                    reconstructed_data[id(self.mean_pooled)][layer].append(reconstructed_mean)

        # Replace original data with reconstructed data
        self.sequence_labels = new_sequence_labels
        self.num_sequences = len(self.sequence_labels)
        for obj in outputs_to_rebuild:
            for layer in self.layers:
                obj["output_data"][layer] = reconstructed_data[id(obj)][layer]
        
        logger.info(f"Reconstruction complete. Final sequence count: {self.num_sequences}")
