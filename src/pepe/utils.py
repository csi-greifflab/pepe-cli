import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from torch.utils.data import Dataset
import re
import torch
import time
import gc
import os
import json
from transformers import RoFormerTokenizer, T5Tokenizer
import threading, queue
from alive_progress import alive_bar
import shutil

logger = logging.getLogger("pepe.utils")


class TokenBudgetBatchSampler:
    def __init__(self, dataset: Any, token_budget: int) -> None:
        self.dataset = dataset
        self.token_budget = token_budget

        # In PEPE, datasets typically pad all sequences to the same length (max_length)
        # Handle empty dataset case
        if len(dataset) == 0:
            self.batch_size = 1
            self.batches = []
            return

        # dataset[idx] -> (label, seq_str, toks, ...)
        sample_seq_len = len(dataset[0][2])
        
        if sample_seq_len > token_budget:
            logger.warning(
                f"A sequence has length {sample_seq_len}, which exceeds the specified token budget (batch_size) of {token_budget}. "
                "The budget will be effectively increased for batches containing such sequences."
            )
            self.batch_size = 1
        else:
            self.batch_size = max(1, token_budget // sample_seq_len)

        self.batches = self._create_batches()

    def _create_batches(self) -> List[List[int]]:
        indices = list(range(len(self.dataset)))
        return [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

    def __iter__(self) -> Any:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)




class SequenceDictDataset(Dataset):
    substring_dict: Optional[Dict[str, str]]
    encoded_data: List[Any]
    encoded_substring_data: List[Any]
    pad_token_id: int
    substring_masks: List[Any]

    def __init__(
        self,
        sequences: Dict[str, str],
        substring_dict: Optional[Dict[str, str]],
        context: Any,
        bracket_type: str = "square",
    ) -> None:
        self.data = list(sequences.items())  # (label, seq)
        if substring_dict:
            self.substring_dict = substring_dict
            self.context = context
            self.filtered_substring_data = self._filter_substrings()
        else:
            self.substring_dict = None
            self.context = None
            self.filtered_substring_data = None
        self.bracket_type = bracket_type

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _filter_substrings(self) -> Any:
        """Filter the substring_dict to only include sequences that are in the dataset."""
        assert self.substring_dict is not None
        labels, _ = zip(*self.data)
        filtered_substring_dict = {
            label: self.substring_dict[label]
            for label in labels
            if label in self.substring_dict
        }
        if len(filtered_substring_dict) != len(self.data):
            missing = [label for label in labels if label not in self.substring_dict]
            raise ValueError(
                "Not all sequences have matching substrings. "
                f"Missing substring entries for: {', '.join(missing)}"
            )
        return filtered_substring_dict.items()

    def _get_substring_masks(self):
        """Mask tokens from sequence that are not in the provided substring."""
        # Get the full sequences and substring
        full_sequence_tokens = [entry[2] for entry in self.encoded_data]
        substring_tokens = [entry[2] for entry in self.encoded_substring_data]

        # Create masks for each sequence
        masks = []
        for (label, _, _), full_seq, substring in zip(
            self.encoded_data, full_sequence_tokens, substring_tokens
        ):
            mask = self._find_subsequence(full_seq, substring, self.pad_token_id)
            if mask.sum() == 0:
                raise ValueError(
                    f"Substring for sequence '{label}' not found in the tokenized full sequence."
                )
            masks.append(mask)

        return masks

    def _find_subsequence(self, full_tensor, subtensor, pad_token_id=0):
        subsequence_mask = torch.zeros_like(full_tensor)

        # Remove padding from b
        trimmed_subtensor = subtensor[subtensor != pad_token_id]
        trimmed_subtensor_length = trimmed_subtensor.size(0)
        if trimmed_subtensor_length == 0:
            return subsequence_mask

        full_tensor_length = full_tensor.size(0)

        for start in range(full_tensor_length):
            match_positions = []
            full_tensor_index = start
            subtensor_index = 0

            while (
                full_tensor_index < full_tensor_length
                and subtensor_index < trimmed_subtensor_length
            ):
                if full_tensor[full_tensor_index] == pad_token_id:
                    full_tensor_index += 1
                    continue
                if full_tensor[full_tensor_index] == trimmed_subtensor[subtensor_index]:
                    match_positions.append(full_tensor_index)
                    full_tensor_index += 1
                    subtensor_index += 1
                else:
                    break

            if subtensor_index == trimmed_subtensor_length:
                subsequence_mask[match_positions] = 1

        return subsequence_mask


class HuggingFaceDataset(SequenceDictDataset):
    def __init__(
        self,
        sequences,
        substring_dict,
        context,
        bracket_type,
        tokenizer,
        max_length,
        add_special_tokens=True,
        gapped_sequences=True,
    ):
        super().__init__(sequences, substring_dict, context, bracket_type)
        self.gapped_sequences = gapped_sequences

        self.encoded_data = self._encode_sequences(
            self.data, tokenizer, max_length, add_special_tokens, gapped_sequences
        )  # (label, seq, toks, attention_mask)
        self.pad_token_id = (
            getattr(tokenizer, "pad_token_type_id", None) or tokenizer.pad_token_id
        )
        if self.substring_dict:
            logger.info("Tokenizing substrings...")
            self.encoded_substring_data = self._encode_sequences(
                self.filtered_substring_data,
                tokenizer,
                "max_length",
                add_special_tokens=False,
                gapped_sequences=gapped_sequences,
            )  # (label, seq, toks, attention_mask)
            self.substring_masks = self._get_substring_masks()

    def _encode_sequences(self, data, tokenizer, max_length, add_special_tokens, gapped_sequences = True):
        labels, raw_strs = zip(*data)
        strs: List[str]
        if gapped_sequences:
            strs = list(gap_sequence(list(raw_strs), self.bracket_type))
            max_token_length = max(len(seq.split(" ")) for seq in strs)
        else:
            # For other tokenizers, use the default tokenization
            strs = list(raw_strs)
            max_token_length = max(len(seq) for seq in strs)

        # Account for special tokens (if any)
        special_tokens_count = len(tokenizer.encode("", add_special_tokens=add_special_tokens)) if hasattr(tokenizer, "encode") else 0
        max_token_length += special_tokens_count

        if max_length == "max_length":
            max_length = max_token_length
            logger.info(f"Setting max_length to {max_length} (including {special_tokens_count} special tokens).")
        else:
            max_length = int(max_length)
            if max_length < max_token_length:
                logger.warning(
                    f"max_length {max_length} is less than the length of the longest sequence + special tokens ({max_token_length}). Setting max_length to {max_token_length}."
                )
                max_length = max_token_length
        logger.info(f"Tokenizing {len(strs)} sequences...")
        out = tokenizer(
            list(strs),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        toks = out.input_ids
        attention_masks = out.attention_mask
        return list(zip(labels, strs, list(toks), list(attention_masks)))

    def get_max_encoded_length(self):
        return max(len(toks) for _, _, toks, _ in self.encoded_data)

    def __getitem__(self, idx):
        labels, seqs, toks, attention_masks = self.encoded_data[idx]
        if self.substring_dict:
            substring_masks = self.substring_masks[idx]
            return labels, seqs, toks, attention_masks, substring_masks
        else:
            return labels, seqs, toks, attention_masks, None

    def safe_collate(self, batch):
        if self.substring_dict:
            labels, seqs, toks, attention_matrices, substring_masks = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                torch.stack(attention_matrices),
                torch.stack(substring_masks),
            )
        else:
            labels, seqs, toks, attention_matrices, _ = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                torch.stack(attention_matrices),
                None,
            )

    def __len__(self):
        return len(self.data)


class ESMDataset(SequenceDictDataset):
    def __init__(
        self,
        sequences,
        substring_dict,
        context,
        alphabet,
        max_length,
        prepend_bos=True,
        append_eos=True,
    ):
        super().__init__(sequences, substring_dict, context)
        self.encoded_data = self._encode_sequences(
            self.data,
            alphabet,
            max_length,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
        )  # (label, seq, toks)
        self.pad_token_id = alphabet.padding_idx
        if self.substring_dict:
            logger.info("Tokenizing substrings...")
            self.encoded_substring_data = self._encode_sequences(
                self.filtered_substring_data,
                alphabet,
                "max_length",
                prepend_bos=False,
                append_eos=False,
            )  # (label, seq, toks)
            logger.info("Matching substrings to full sequences...")
            self.substring_masks = self._get_substring_masks()

    def _encode_sequences(
        self, data, alphabet, max_length, prepend_bos=True, append_eos=True
    ):
        labels, strs = zip(*data)
        logger.info(f"Tokenizing {len(strs)} sequences...")

        orig_prepend = alphabet.prepend_bos
        orig_append = alphabet.append_eos
        alphabet.prepend_bos = prepend_bos
        alphabet.append_eos = append_eos
        try:
            _, _, tokens = alphabet.get_batch_converter()(list(zip(labels, strs)))
        finally:
            alphabet.prepend_bos = orig_prepend
            alphabet.append_eos = orig_append

        max_encoded_length = tokens.shape[1] - int(prepend_bos) - int(append_eos)
        if max_length == "max_length":
            target_length = max_encoded_length
        else:
            max_length = int(max_length)
            if max_length < max_encoded_length:
                logger.warning(
                    f"max_length {max_length} is less than the length of the longest sequence: {max_encoded_length}. Setting max_length to {max_encoded_length}."
                )
                target_length = max_encoded_length
            else:
                target_length = max_length

        target_width = target_length + int(prepend_bos) + int(append_eos)
        if tokens.shape[1] < target_width:
            padded = tokens.new_full(
                (tokens.shape[0], target_width), alphabet.padding_idx
            )
            padded[:, : tokens.shape[1]] = tokens
            tokens = padded

        return list(zip(labels, strs, list(tokens)))

    def get_max_encoded_length(self):
        return max(len(toks) for _, _, toks in self.encoded_data)

    def safe_collate(self, batch):
        if self.substring_dict:
            labels, seqs, toks, _, substring_masks = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                None,
                torch.stack(substring_masks),
            )
        else:
            labels, seqs, toks, _, _ = zip(*batch)
            return list(labels), list(seqs), torch.stack(toks), None, None

    def __getitem__(self, idx):
        labels, seqs, toks = self.encoded_data[idx]
        if self.substring_dict:
            substring_masks = self.substring_masks[idx]
            return labels, seqs, toks, None, substring_masks
        else:
            return labels, seqs, toks, None, None

class CustomDataset(SequenceDictDataset):
    """
    Dataset class for custom embedder.
    """

    def __init__(
        self,
        sequences,
        substring_dict,
        context,
        tokenizer,
        max_length,
        add_special_tokens=True,
        gapped_sequences = True,
    ):
        super().__init__(sequences, substring_dict, context)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.gapped_sequences = gapped_sequences

        # Encode sequences
        self.encoded_data = self._encode_sequences(
            self.data, tokenizer, max_length, add_special_tokens, self.gapped_sequences
        )

        # Handle substring sequences if provided
        if self.substring_dict:
            logger.info("Tokenizing substrings...")
            self.encoded_substring_data = self._encode_sequences(
                self.filtered_substring_data,
                tokenizer,
                "max_length",
                add_special_tokens=False,
            )
            self.substring_masks = self._get_substring_masks()

    def _encode_sequences(self, data, tokenizer, max_length, add_special_tokens, gapped_sequences = True):
        """Encode sequences using the tokenizer."""
        labels, raw_strs = zip(*data)
        strs: List[str]

        if gapped_sequences:
            strs = list(gap_sequence(list(raw_strs)))
            max_token_length = max(len(seq.split(" ")) for seq in strs)
        else:
            # For other tokenizers, use the default tokenization
            strs = list(raw_strs)
            max_token_length = max(len(seq) for seq in strs)

        # Convert max_length to int if needed
        if max_length == "max_length":
            max_token_length = max(len(s) for s in strs) + (2 if add_special_tokens else 0)

        logger.info(f"Tokenizing {len(strs)} sequences...")
        out = tokenizer(
            list(strs),
            truncation=True,
            padding="max_length",
            max_length=max_token_length,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        toks = out["input_ids"]
        attention_masks = out["attention_mask"]
        return list(zip(labels, strs, list(toks), list(attention_masks)))

    def get_max_encoded_length(self):
        """Get maximum encoded sequence length."""
        return max(len(toks) for _, _, toks, _ in self.encoded_data)

    def safe_collate(self, batch):
        """Collate function for data loader."""
        if self.substring_dict:
            labels, seqs, toks, attn_masks, substring_masks = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                torch.stack(attn_masks),
                torch.stack(substring_masks),
            )
        else:
            labels, seqs, toks, attn_masks = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                torch.stack(attn_masks),
                None,
            )

    def __getitem__(self, idx):
        """Get item from dataset."""
        labels, seqs, toks, attn_mask = self.encoded_data[idx]
        if self.substring_dict:
            substring_masks = self.substring_masks[idx]
            return labels, seqs, toks, attn_mask, substring_masks
        else:
            return labels, seqs, toks, attn_mask

def check_input_tokens(
    valid_tokens: Any,
    sequences: Dict[str, str],
    model_name: str,
    bracket_type: str = "square",
    split_long_sequences: bool = False,
) -> None:
    def __str_to_list(sequence, bracket_type="square"):
        if bracket_type == "square":
            return re.findall(r"\[.*?\]|.", sequence)
        elif bracket_type == "angle":
            return re.findall(r"<.*?>|.", sequence)
        else:
            raise ValueError(f"Invalid bracket type: {bracket_type}")

    with alive_bar(
        len(sequences), title="Checking input sequences for invalid tokens..."
    ) as bar:
        for label, sequence in sequences.items():
            sequence = __str_to_list(sequence, bracket_type)
            if "antiberta" in model_name and not split_long_sequences:  # check for longest sequence
                assert (
                    len(sequence) <= 256
                ), f"Antiberta2 does not support sequences longer than 256 tokens. Found {len(sequence)} tokens in sequence {label}. Use --split_long_sequences to process longer sequences."

            if not set(sequence).issubset(valid_tokens):
                raise ValueError(
                    f"Invalid tokens found in sequence {label}: {set(sequence) - set(valid_tokens)}"
                )
            bar()
    logger.info("No invalid tokens in input sequences.")


_CHARACTER_TOKENIZER_PROBE = "ACDEFGHIKLMNPQRSTVWY"


def is_character_tokenizer(tokenizer) -> bool:
    """Return True when one amino acid maps to one token (no subword merging)."""
    encoded = tokenizer.encode(
        _CHARACTER_TOKENIZER_PROBE, add_special_tokens=False
    )
    return len(encoded) == len(_CHARACTER_TOKENIZER_PROBE)


def warn_if_non_character_tokenizer(tokenizer: Any, model_name: Optional[str] = None) -> None:
    """Log a prominent warning when the tokenizer is not character-level."""
    if is_character_tokenizer(tokenizer):
        return
    name = model_name or "this model"
    logger.warning(
        "Warning: %s uses a subword tokenizer (not one amino acid per token). "
        "per_token and substring_pooled outputs will NOT align with residues; "
        "use mean_pooled for reliable sequence-level embeddings.",
        name,
    )


def fasta_to_dict(fasta_path: str) -> Dict[str, str]:
    """Convert FASTA file into a dictionary: {id: raw_sequence}."""
    seq_dict: Dict[str, str] = {}
    sequence_id: Optional[str] = None
    sequence_aa: List[str] = []

    def flush():
        nonlocal sequence_id, sequence_aa
        if sequence_id:
            seq_dict[sequence_id] = "".join(sequence_aa)
        sequence_id, sequence_aa = None, []

    with open(fasta_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line.startswith(">"):
                flush()
                sequence_id = line[1:] or f"seqnum{line_idx:09d}"
            else:
                sequence_aa.append(line)
    flush()

    return seq_dict
def get_bracket_type(tokenizer: Any) -> str:
    """
    Attempt to determine if special tokens use [ ] or < > brackets.
    Defaults to 'square' with a warning if unable to determine.
    """
    try:
        special_tokens = []
        if hasattr(tokenizer, "special_tokens_map") and "additional_special_tokens" in tokenizer.special_tokens_map:
            tokens = tokenizer.special_tokens_map["additional_special_tokens"]
            if tokens and len(tokens) > 0:
                special_tokens.append(tokens[0])
        
        if not special_tokens and hasattr(tokenizer, "all_special_tokens"):
            if tokenizer.all_special_tokens:
                special_tokens.append(tokenizer.all_special_tokens[0])
        
        if not special_tokens:
            logger.warning("Could not find any special tokens. Defaulting to 'square' brackets.")
            return "square"

        first_char = special_tokens[0][0]
        if first_char == "<":
            return "angle"
        elif first_char == "[":
            return "square"
        else:
            logger.warning(f"Unrecognized special token format (starts with '{first_char}'). Defaulting to 'square' brackets.")
            return "square"
    except Exception as e:
        logger.warning(f"Error determining bracket type: {e}. Defaulting to 'square' brackets.")
        return "square"
def gap_sequence(sequences: Sequence[str], bracket_type: str = "square") -> List[str]:
    if bracket_type == "square":
        seqs = [" ".join(re.findall(r"\[.*?\]|.", sequence)) for sequence in sequences]
    elif bracket_type == "angle":
        seqs = [" ".join(re.findall(r"<.*?>|.", sequence)) for sequence in sequences]
    else:
        raise ValueError(f"Invalid bracket type: {bracket_type}")
    return seqs

def flush_memmaps(obj: Any) -> None:
    """Recursively flush memory maps."""
    if hasattr(obj, "flush") and callable(obj.flush):
        obj.flush()
        gc.collect()
        logger.debug("Flushed output")
    elif isinstance(obj, dict):
        for value in obj.values():
            flush_memmaps(value)


def check_disk_free_space(path: str, min_free_bytes: int) -> None:
    _, _, free = shutil.disk_usage(path)
    if free < min_free_bytes:
        raise ValueError(
            f"Not enough disk space. Required: {min_free_bytes} bytes, Available: {free} bytes"
        )
    logger.info(f"Disk space check passed. Available: {free} bytes")


class IOFlushWorker(threading.Thread):
    def __init__(
        self,
        memmap_registry,
        flush_bytes_limit=64 * 1024 * 1024,
        global_dispatcher=None,
    ):  # e.g. 64 MB
        super().__init__()
        self.memmap_registry = (
            memmap_registry  # Dict: (output_type, layer, head) → memmap
        )
        self.flush_limit = flush_bytes_limit
        self.write_q: queue.Queue[Any] = queue.Queue(maxsize=128)
        self.buffer = {}  # (output_type, layer, head) → list of (offset, array)
        self.buffered_bytes = {}  # (output_type, layer, head) → total bytes
        self.total_buffered = 0
        self.lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        self.outstanding_enqueues = 0
        self.done_enqueuing = threading.Event()
        # Initially set done_enqueuing since there are no outstanding enqueues
        self.done_enqueuing.set()

        # Global checkpoint system (worker-agnostic)
        self.global_dispatcher = global_dispatcher
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 30  # seconds

    def is_range_completed(self, output_type, layer, head, offset, length):
        """Check if a range is completed using the global checkpoint system"""
        if self.global_dispatcher:
            return self.global_dispatcher.is_range_completed_global(
                output_type, layer, head, offset, length
            )
        return False

    def mark_range_completed(self, output_type, layer, head, offset, length):
        """Mark a range as completed using the global checkpoint system"""
        if self.global_dispatcher:
            self.global_dispatcher.mark_range_completed_global(
                output_type, layer, head, offset, length
            )

            # Save global checkpoint periodically
            now = time.time()
            if now - self.last_checkpoint_time > self.checkpoint_interval:
                self.global_dispatcher._save_global_checkpoint()
                self.last_checkpoint_time = now

    # Deprecated: _merge_ranges is no longer used (global checkpoint system in place)

    def queue_fullness(self):
        return self.write_q.qsize() / self.write_q.maxsize

    def run(self):
        while not self.shutdown_flag.is_set():
            try:
                item = self.write_q.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                # Mark the sentinel as done and break
                self.write_q.task_done()
                break
            try:
                key, offset, array = item
                arr_bytes = array.nbytes

                with self.lock:
                    self.buffer.setdefault(key, []).append((offset, array))
                    self.buffered_bytes[key] = (
                        self.buffered_bytes.get(key, 0) + arr_bytes
                    )
                    self.total_buffered += arr_bytes

                    if self.total_buffered >= self.flush_limit:
                        self.flush_all()
            except Exception as e:
                logger.error(f"[IOFlushWorker] Exception during processing: {e}")
            finally:
                self.write_q.task_done()

        # Process any remaining items in the queue after shutdown signal
        while True:
            try:
                item = self.write_q.get_nowait()
                if item is None:
                    self.write_q.task_done()
                    break
                try:
                    key, offset, array = item
                    arr_bytes = array.nbytes
                    with self.lock:
                        self.buffer.setdefault(key, []).append((offset, array))
                        self.buffered_bytes[key] = (
                            self.buffered_bytes.get(key, 0) + arr_bytes
                        )
                        self.total_buffered += arr_bytes
                except Exception as e:
                    logger.error(
                        f"[IOFlushWorker] Exception during shutdown processing: {e}"
                    )
                finally:
                    self.write_q.task_done()
            except queue.Empty:
                break

        # Final flush
        try:
            self.flush_all()
        except Exception as e:
            logger.error(f"[IOFlushWorker] Exception during final flush: {e}")

    def flush_all(self):
        for key in list(self.buffer.keys()):
            self.flush_key(key)
        self.total_buffered = 0

    def flush_key(self, key):
        try:
            t0 = time.time()
            mmap_handle = self.memmap_registry[key]
            batch_count = len(self.buffer[key])
            total_elements = sum(len(arr) for _, arr in self.buffer[key])
            for offset, arr in self.buffer[key]:
                mmap_handle[offset : offset + len(arr)] = arr
                # Mark this range as completed for crash recovery
                output_type, layer, head = key
                self.mark_range_completed(output_type, layer, head, offset, len(arr))
            mmap_handle.flush()
            duration = time.time() - t0
            if duration > 1.0: # Only log slow flushes
                logger.info(f"[IOFlushWorker] Slow flush for {key}: {duration:.2f}s for {batch_count} batches ({total_elements} elements)")
        except Exception as e:
            logger.error(f"[IOFlushWorker] Exception during flush: {e}")
            raise e
        finally:
            self.total_buffered -= self.buffered_bytes.get(key, 0)
            self.buffered_bytes[key] = 0
            self.buffer[key].clear()

    def enqueue(self, output_type, layer, head, offset, array):
        # Check if this range was already completed (crash recovery)
        if self.is_range_completed(output_type, layer, head, offset, len(array)):
            logger.debug(
                f"[IOFlushWorker] Skipping already completed range: {output_type}, {layer}, {head}, {offset}-{offset+len(array)}"
            )
            return

        with self.lock:
            if self.outstanding_enqueues == 0:
                # Clear the event since we're about to have outstanding enqueues
                self.done_enqueuing.clear()
            self.outstanding_enqueues += 1

        key = (output_type, layer, head)

        try:
            while True:
                try:
                    self.write_q.put((key, offset, array), timeout=1)
                    break
                except queue.Full:
                    logger.warning(
                        "[IOFlushWorker] Write queue full, waiting to enqueue..."
                    )
                    time.sleep(0.1)
        finally:
            with self.lock:
                self.outstanding_enqueues -= 1
                if self.outstanding_enqueues == 0:
                    self.done_enqueuing.set()

    def stop(self, max_wait_time=60, force_shutdown=True):
        """
        Stop the worker with practical timeouts for terabyte operations.

        Args:
            max_wait_time: Maximum time to wait for pending operations (seconds)
            force_shutdown: If True, force shutdown after max_wait_time even if work remains
        """
        logger.info(f"[IOFlushWorker] Initiating shutdown...")

        # Signal that we're shutting down
        self.shutdown_flag.set()

        # For terabyte operations, we can't wait indefinitely
        # Instead, we save our progress and allow graceful shutdown
        if max_wait_time > 0:
            logger.info(
                f"[IOFlushWorker] Waiting up to {max_wait_time}s for pending operations..."
            )
            completed = self.done_enqueuing.wait(timeout=max_wait_time)

            if not completed:
                with self.lock:
                    remaining = self.outstanding_enqueues
                    buffered_mb = self.total_buffered / (1024 * 1024)
                logger.warning(
                    f"[IOFlushWorker] Timeout with {remaining} enqueues and {buffered_mb:.1f}MB buffered"
                )

                if not force_shutdown:
                    logger.info(
                        "[IOFlushWorker] Continuing to wait since force_shutdown=False..."
                    )
                    self.done_enqueuing.wait()  # Wait indefinitely
                else:
                    logger.warning("[IOFlushWorker] Proceeding with forced shutdown")
        else:
            logger.info("[IOFlushWorker] Skipping wait for pending operations")

        # Save final checkpoint before shutdown
        if self.global_dispatcher:
            self.global_dispatcher._save_global_checkpoint()

        # Send sentinel value to stop the worker thread
        while True:
            try:
                self.write_q.put(None, timeout=1)
                break
            except queue.Full:
                logger.warning("Write queue full during shutdown; retrying...")
                time.sleep(0.1)

        # Wait for the worker thread to finish
        self.join(timeout=30)

        # Force one final flush and checkpoint
        with self.lock:
            try:
                self.flush_all()
                if self.global_dispatcher:
                    self.global_dispatcher._save_global_checkpoint()
            except Exception as e:
                logger.error(
                    f"[IOFlushWorker] Exception during final flush in stop(): {e}"
                )

            # Report final status
            remaining_in_queue = self.write_q.qsize()
            pending_buffered = sum(len(buf) for buf in self.buffer.values())

            if remaining_in_queue > 1 or pending_buffered > 0:
                logger.warning(
                    f"[IOFlushWorker] Final status: {remaining_in_queue} in queue, {pending_buffered} buffered"
                )
                if self.global_dispatcher:
                    total_ranges = sum(
                        len(ranges)
                        for ranges in self.global_dispatcher.global_completed_ranges.values()
                    )
                    logger.info(
                        f"[IOFlushWorker] Progress saved to global checkpoint: {len(self.global_dispatcher.global_completed_ranges)} keys, {total_ranges} ranges"
                    )
            else:
                logger.info("[IOFlushWorker] Clean shutdown - all data written")

        logger.info("[IOFlushWorker] Shutdown complete")


class MultiIODispatcher:
    def __init__(
        self,
        memmap_registry,
        num_workers=4,
        flush_bytes_limit=64 * 1024 * 1024,
        heavy_output_type="per_token",
        heavy_proportion=0.75,
        checkpoint_dir=None,
    ):
        self.num_workers = num_workers
        self.heavy_output_type = heavy_output_type
        self.checkpoint_dir = checkpoint_dir

        # Centralized checkpoint system (worker-agnostic)
        self.global_checkpoint_file = None
        self.global_completed_ranges = {}  # Shared across all workers
        self.checkpoint_lock = threading.Lock()

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.global_checkpoint_file = os.path.join(
                checkpoint_dir, "global_checkpoint.json"
            )
            self._load_global_checkpoint()

        # Check if heavy keys exist
        num_heavy_keys = sum(
            1 for key in memmap_registry if key[0] == self.heavy_output_type
        )

        if num_heavy_keys == 0:
            logger.warning(
                f"[MultiIODispatcher] WARNING: No keys found for heavy_output_type '{self.heavy_output_type}'. Reassigning all workers to light workload."
            )
            self.num_heavy_workers = 0
            self.num_light_workers = num_workers
        elif num_workers == 1:
            logger.info("[MultiIODispatcher] Only 1 worker available. Assigning all keys to this worker.")
            self.num_heavy_workers = 1
            self.num_light_workers = 1 # Both point to the same worker
        else:
            self.num_heavy_workers = max(1, int(num_workers * heavy_proportion))
            self.num_light_workers = num_workers - self.num_heavy_workers
            assert self.num_light_workers >= 1, "You need at least one light worker"
        self.workers = []
        self.heavy_workers = []
        self.light_workers = []

        # Distribute files across workers by hashing the key
        sharded_registries: List[Dict[Any, Any]] = [{} for _ in range(num_workers)]

        for key, mmap in memmap_registry.items():
            if num_workers == 1:
                shard_id = 0
            else:
                output_type = key[0]  # assuming key = (output_type, layer, head)
                if output_type == self.heavy_output_type and self.num_heavy_workers > 0:
                    # Only assign heavy keys to heavy workers
                    shard_id = hash(key) % self.num_heavy_workers
                else:
                    # Assign light keys to light workers
                    shard_id = self.num_heavy_workers + (hash(key) % self.num_light_workers)
            sharded_registries[shard_id][key] = mmap

        for i, reg in enumerate(sharded_registries):
            logger.info(f"[MultiIODispatcher] Worker {i} assigned {len(reg)} keys")

        for i in range(num_workers):
            worker = IOFlushWorker(
                memmap_registry=sharded_registries[i],
                flush_bytes_limit=flush_bytes_limit,
                global_dispatcher=self,  # Pass self as the global dispatcher
            )
            worker.start()
            self.workers.append(worker)
        # Assign worker groups
        if num_workers == 1:
            self.heavy_workers = self.workers
            self.light_workers = self.workers
        else:
            self.heavy_workers = (
                self.workers[: self.num_heavy_workers] if self.num_heavy_workers > 0 else []
            )
            self.light_workers = self.workers[self.num_heavy_workers :]

    def queue_fullness(self):
        # Return max fullness across all workers (pessimistic view)
        return max(worker.queue_fullness() for worker in self.workers)

    def enqueue(self, output_type, layer, head, offset, array):
        key = (output_type, layer, head)
        if self.num_workers == 1:
            self.workers[0].enqueue(output_type, layer, head, offset, array)
        elif output_type == self.heavy_output_type:
            # Route heavy output_type to heavy_workers (hashed for key affinity)
            worker_id = hash(key) % self.num_heavy_workers
            self.heavy_workers[worker_id].enqueue(
                output_type, layer, head, offset, array
            )
        else:
            # Route other output_types to light_workers
            worker_id = hash(key) % self.num_light_workers
            self.light_workers[worker_id].enqueue(
                output_type, layer, head, offset, array
            )

    def stop(self, max_wait_time=60, force_shutdown=True):
        """
        Stop all workers with practical timeouts for terabyte operations.

        Args:
            max_wait_time: Maximum time to wait for each worker
            force_shutdown: If True, force shutdown after timeout
        """
        logger.info(f"[MultiIODispatcher] Stopping {len(self.workers)} workers...")
        for i, worker in enumerate(self.workers):
            logger.info(f"[MultiIODispatcher] Stopping worker {i}...")
            worker.stop(max_wait_time=max_wait_time, force_shutdown=force_shutdown)
        logger.info("[MultiIODispatcher] All workers stopped")

    def get_resume_info(self):
        """Get information about what can be resumed after a crash"""
        total_completed_ranges = sum(
            len(ranges) for ranges in self.global_completed_ranges.values()
        )
        total_completed_bytes = 0

        resume_info: Dict[str, Any] = {
            "global_checkpoint_file": self.global_checkpoint_file,
            "total_completed_ranges": total_completed_ranges,
            "num_workers": len(self.workers),
            "keys_with_progress": {},
        }

        # Calculate total bytes and provide per-key breakdown
        for key, ranges in self.global_completed_ranges.items():
            key_bytes = sum(end - start for start, end in ranges)
            total_completed_bytes += key_bytes

            resume_info["keys_with_progress"][f"{key[0]}|{key[1]}|{key[2]}"] = {
                "ranges": len(ranges),
                "bytes": key_bytes,
                "mb": key_bytes / (1024 * 1024),
            }

        resume_info["total_completed_bytes"] = total_completed_bytes
        resume_info["total_completed_gb"] = total_completed_bytes / (1024 * 1024 * 1024)

        return resume_info

    def _load_global_checkpoint(self):
        """Load the global checkpoint that works regardless of worker count"""
        if self.global_checkpoint_file and os.path.exists(self.global_checkpoint_file):
            try:
                with open(self.global_checkpoint_file, "r") as f:
                    data = json.load(f)
                    # Convert string keys back to tuples and ranges to sets
                    for key_str, ranges in data.get("completed_ranges", {}).items():
                        # Parse key string back to tuple
                        parts = key_str.split("|")
                        if len(parts) == 3:
                            output_type, layer, head = parts
                            # Convert layer and head back to appropriate types
                            layer = int(layer) if layer != "None" else None
                            head = int(head) if head != "None" else None
                            key = (output_type, layer, head)
                            self.global_completed_ranges[key] = set(
                                tuple(r) for r in ranges
                            )

                total_ranges = sum(
                    len(ranges) for ranges in self.global_completed_ranges.values()
                )
                logger.info(
                    f"[MultiIODispatcher] Loaded global checkpoint: {len(self.global_completed_ranges)} keys, {total_ranges} ranges"
                )

                # Print summary of what was already completed
                if total_ranges > 0:
                    logger.info("[MultiIODispatcher] Resume info:")
                    for key, ranges in self.global_completed_ranges.items():
                        total_bytes = sum(end - start for start, end in ranges)
                        logger.info(
                            f"  {key}: {len(ranges)} ranges, {total_bytes / (1024*1024):.1f}MB"
                        )

            except Exception as e:
                logger.error(
                    f"[MultiIODispatcher] Failed to load global checkpoint: {e}"
                )
                # Reset to empty state on error
                self.global_completed_ranges = {}

    def _save_global_checkpoint(self):
        """Save the global checkpoint"""
        if not self.global_checkpoint_file:
            return

        with self.checkpoint_lock:
            try:
                # Convert tuples to strings for JSON serialization
                data = {
                    "completed_ranges": {
                        f"{key[0]}|{key[1]}|{key[2]}": list(ranges)
                        for key, ranges in self.global_completed_ranges.items()
                    },
                    "timestamp": time.time(),
                    "num_workers_used": self.num_workers,
                }

                # Write to temporary file first, then rename (atomic operation)
                temp_file = self.global_checkpoint_file + ".tmp"
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=2)

                os.rename(temp_file, self.global_checkpoint_file)

                total_ranges = sum(
                    len(ranges) for ranges in self.global_completed_ranges.values()
                )
                logger.info(
                    f"[MultiIODispatcher] Saved global checkpoint: {len(self.global_completed_ranges)} keys, {total_ranges} ranges"
                )

            except Exception as e:
                logger.error(
                    f"[MultiIODispatcher] Failed to save global checkpoint: {e}"
                )

    def is_range_completed_global(self, output_type, layer, head, offset, length):
        """Check if a range is completed in the global checkpoint (worker-agnostic)"""
        key = (output_type, layer, head)
        if key not in self.global_completed_ranges:
            return False

        end_offset = offset + length
        completed_set = self.global_completed_ranges[key]

        # Check if this range is fully covered by any completed range
        for start, end in completed_set:
            if start <= offset and end_offset <= end:
                return True
        return False

    def mark_range_completed_global(self, output_type, layer, head, offset, length):
        """Mark a range as completed in the global checkpoint (worker-agnostic)"""
        key = (output_type, layer, head)

        with self.checkpoint_lock:
            if key not in self.global_completed_ranges:
                self.global_completed_ranges[key] = set()

            self.global_completed_ranges[key].add((offset, offset + length))

            # Merge overlapping ranges to keep the set manageable
            self._merge_ranges_global(key)

    def _merge_ranges_global(self, key):
        """Merge overlapping ranges in the global checkpoint"""
        if key not in self.global_completed_ranges:
            return

        ranges = sorted(self.global_completed_ranges[key])
        merged: List[Tuple[int, int]] = []

        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                # Overlapping ranges - merge them
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        self.global_completed_ranges[key] = set(merged)
