# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PEPE (Pipeline for Easy Protein Embedding) is a CLI + Python library that extracts embeddings and attention matrices from protein sequences using pre-trained protein language models (ESM-1/2, ESMC, ProtT5, AntiBERTa2, and arbitrary HuggingFace / custom PyTorch models). Published to PyPI and Conda as `pepe-cli`; the import module is `pepe`.

## Source layout

The real package lives in `src/pepe/`.

## Commands

```sh
# Editable install (needed before running CLI, library, or tests)
pip install -e .

# For ESM-1 models only (fair-esm, not covered by the base deps)
pip install fair-esm

# For ESMC models only (biohub/ESMC-*): Biohub's transformers fork
pip install git+https://github.com/Biohub/transformers.git@main

# Run the CLI
pepe --model_name esm2_t6_8M_UR50D --fasta_path <file> --output_path <dir> --extract_embeddings mean_pooled

# Run the test suite (unittest-style, but pytest-compatible). Run from repo ROOT —
# tests reference relative paths like src/tests/test_files/.
python -m pytest src/tests/
python -m pytest src/tests/test_api_unittest.py            # single file
python -m pytest src/tests/test_api_unittest.py::TestPepeAPI::test_embed_to_disk   # single test
```

## Version / release flow

`src/pepe/__init__.py` `__version__` is the **single source of truth** — `pyproject.toml` and `setup.py` both read metadata from it (`__version__`, `__package_name__`, etc.). Bumping that string is what drives publishing:

- Push to `main` → `.github/workflows/publish-main-branch-trusted.yml` builds and publishes to PyPI. The version must **not** contain `-dev`/`-test`.
- Push to `test` → publishes to TestPyPI.

## Architecture

Two entry points converge on the same engine:

- **CLI**: `pepe.__main__:main` → `parse_arguments()` → `select_model(model_name)` → `Embedder(args).run()`.
- **Library**: `pepe.embed(...)` in `api.py` packs kwargs into a `SimpleNamespace` (mimicking the argparse `args`) and calls the same `select_model` + `.run()` path. When no `output_path` is given it writes to a temp dir, disables streaming, and returns in-memory results.

Both paths pass a single `args` object to every embedder constructor — so **CLI flags in `parse_arguments.py`, the `embed()` signature/`args_dict` in `api.py`, and attribute reads in `base_embedder.py` must stay in sync.**

### Model dispatch — `model_selecter.py`

`select_model(model_name)` is a factory that returns the embedder **class** by inspecting the name:
- contains `esm2` → `ESM2Embedder`; `esm1` → `ESMEmbedder` (fair-esm)
- looks like a local path / `.pt` / `.pth` / `custom:` prefix → `CustomEmbedder`
- contains `/` (HuggingFace id) → loads `AutoConfig` and dispatches on `config.model_type` (t5→`T5Embedder`, roformer→`Antiberta2Embedder`, esm→`ESM2Embedder`, esmc→`ESMCEmbedder`).

Heavy deps (torch, transformers, esm) are **lazily imported** inside functions/methods throughout, so `--help` and the CLI stay fast and optional backends (fair-esm, Biohub fork) only load when actually used. Preserve this pattern when adding models.

### Embedder hierarchy

`BaseEmbedder` (`embedders/base_embedder.py`) is the engine; subclasses only supply model/tokenizer loading and `_compute_outputs`:
- `ESMEmbedder` (`esm_embedder.py`) — ESM-1 via fair-esm.
- `HuggingfaceEmbedder` and subclasses `ESM2Embedder`, `ESMCEmbedder`, `T5Embedder`, `Antiberta2Embedder`, `GenericHuggingFaceEmbedder` (`huggingface_embedder.py`).
- `CustomEmbedder` (`custom_embedder.py`) — local `.pt` models.

`BaseEmbedder.run()` → `embed()` iterates the DataLoader, calls `_safe_compute` (which recursively **halves the batch and retries on CUDA OOM**), then routes each batch through the extraction methods selected by `--extract_embeddings`.

### Output types

`_set_output_objects()` defines one entry per output type — `per_token`, `mean_pooled`, `substring_pooled`, `attention_head`, `attention_layer`, `attention_model`, `logits` — each a dict of `{output_data, method (_extract_*), output_dir, shape}`. `extract_embeddings` picks which run. Adding an output type means adding both the dict entry and the matching `_extract_*` method. Files are saved as `.npy` under `<output_path>/<model_name>/<output_type>/`, plus a `*_idx.csv` mapping row index → sequence id. Note `logits` is only produced by ESM-2 (via `AutoModelForMaskedLM`); the generic HuggingFace path drops it.

### Streaming I/O (default) vs in-memory

`streaming_output=True` (default) preallocates `numpy.memmap` files (`preallocate_disk_space()` → `memmap_registry`) and streams batches to disk via `MultiIODispatcher` / `IOFlushWorker` threads (`utils.py`), with backpressure and a `global_checkpoint.json` for crash recovery/resume. This keeps memory flat for arbitrarily large datasets. With `streaming_output=False`, results accumulate in RAM and are written by `export_to_disk()`. `--discard_padding` is incompatible with streaming and silently disables it.

### Batching & datasets

`utils.py` holds the `Dataset` classes (`HuggingFaceDataset`, `ESMDataset`, `CustomDataset`, all extending `SequenceDictDataset`) and `TokenBudgetBatchSampler`. **`--batch_size` is a token budget (total tokens per batch), not a sequence count** — batch size is derived as `token_budget // padded_seq_len`.

### Long-sequence splitting

Models with hard length limits (ESM-2 ~1024, AntiBERTa2 256) are handled in `base_embedder.py`: `_check_max_input_length` / `_get_model_max_allowed` detect the limit, `_handle_sequence_splitting` chunks over-long sequences into overlapping pieces (`--split_long_sequences`, `--split_overlap`, `--force_split_length`), and `_reconstruct_chunks` stitches per-token/logits/mean-pooled outputs back together (in-memory mode only; under streaming, chunks are exported individually).
