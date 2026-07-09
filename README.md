# PEPE

**Parallel Extraction for Protein Embeddings**

[![PyPI](https://img.shields.io/pypi/v/pepe-cli)](https://pypi.org/project/pepe-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/csi-greifflab/pepe-cli/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtag375-blue)](https://doi.org/10.1093/bioinformatics/btag375)
[![Zenodo](https://img.shields.io/badge/Zenodo-15912054-blue)](https://zenodo.org/records/15912054)

PEPE is a **command-line tool and Python library** for high-throughput, multi-modal extraction of representations from protein language models (PLMs). It extracts embeddings and attention matrices from protein sequences in a single parallelized, streaming pass — letting you build embedding datasets that are limited only by the write speed and capacity of your storage, not by available RAM.

PEPE supports ESM-1, ESM-2, ProtT5, ProstT5, AntiBERTa2, ESMC (requires Biohub's transformers fork), and METL 1D models (requires metl-pretrained), along with any compatible model from the Hugging Face Hub and custom local PyTorch models.

---

## Table of contents

- [Why PEPE?](#why-pepe)
- [Installation](#installation)
  - [Docker](#docker)
- [Quick start (CLI)](#quick-start-cli)
- [Quick start (Python library)](#quick-start-python-library)
  - [Advanced usage](#advanced-usage)
  - [Memory management & large-scale processing](#memory-management--large-scale-processing)
  - [Handling long sequences](#handling-long-sequences-splitting--reconstruction)
  - [Performance optimization](#performance-optimization)
- [Supported models](#supported-models)
- [Arguments](#arguments)
- [Citation](#citation)
- [License](#license)

---

## Why PEPE?

Extracting embeddings from PLMs at scale runs into two bottlenecks: accumulating all outputs in memory before writing them to disk causes memory failures, and re-embedding the same sequences to extract different modes wastes computation. PEPE addresses both:

- **Multi-modal, single-pass extraction** — extract per-token, mean-pooled, and substring-pooled embeddings, plus per-head, per-layer, and per-model attention weights, from one forward pass instead of re-running the model per mode.
- **Streaming output** — memory-mapped, concurrently written batches keep peak memory low and constant, so total output size can exceed available RAM. In practice, scaling is limited only by the write speed and capacity of your storage.
- **Long-sequence handling** — automatic chunking and reconstruction for sequences that exceed a model's maximum input length.
- **Broad model support** — supported PLMs, any compatible Hugging Face Hub model, and custom local PyTorch models through one interface.
- **CLI and library** — run from the terminal or call `pepe.embed()` directly inside your own Python pipelines and notebooks.

---

## Installation

From PyPI:

```bash
pip install pepe-cli
```

From Conda:

```bash
conda install -c jahn_zhong pepe-cli
```

From source:

```bash
git clone https://github.com/csi-greifflab/pepe-cli
cd pepe-cli
pip install .
```

### Optional backends

Some models require additional packages that are not installed by default:

```bash
# ESMC models (e.g. EvolutionaryScale/esmc-300m-2024-12) — requires Biohub's transformers fork
pip install git+https://github.com/Biohub/transformers.git@main

# METL models (e.g. metl-g-20m-1d) — requires metl-pretrained
pip install git+https://github.com/gitter-lab/metl-pretrained.git
```

### Docker

A pre-built image with CUDA/GPU support is published on the GitHub Container Registry:

```bash
docker pull ghcr.io/csi-greifflab/pepe-cli:latest
```

Run PEPE from the container, passing through your GPU(s) and mounting a directory for input and output:

```bash
docker run --gpus all \
  -v "$(pwd)":/data \
  ghcr.io/csi-greifflab/pepe-cli:latest \
  --model_name esm2_t33_650M_UR50D \
  --fasta_path /data/sequences.fasta \
  --output_path /data/embeddings
```

The `--gpus all` flag requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Omit it to run on CPU.

---

## Quick start (CLI)

Extract mean-pooled embeddings from sequences in a FASTA file:

```bash
pepe \
  --model_name <model_name> \
  --fasta_path <file_path> \
  --output_path <directory> \
  --experiment_name <optional_string>
```

A minimum of three options is required: `--model_name`, `--fasta_path`, and `--output_path`. Outputs are saved as NumPy arrays in a subdirectory per output type. See [Arguments](#arguments) for the full set of options.

---

## Quick start (Python library)

PEPE can be used programmatically, giving you access to embeddings without the command-line interface.

```python
import pepe

# Embed sequences from a dictionary
sequences = {
    "prot1": "MADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "prot2": "MERIKELRDLMSQSRTREILTKLAEAGIDVPRLFK",
}

results = pepe.embed(
    model_name="facebook/esm2_t6_8M_UR50D",
    sequences=sequences,
    output_path="my_embeddings",
    extract_embeddings=["mean_pooled"],
    device="cpu",  # Use "cuda", "cuda:0", "cuda:1", etc. if available
)

# ...or from a FASTA file
pepe.embed(
    model_name="facebook/esm2_t6_8M_UR50D",
    fasta_path="path/to/sequences.fasta",
    output_path="my_embeddings",
    extract_embeddings=["mean_pooled"],
    device="cpu",
)
```

### Advanced usage

For finer control, use the embedder classes directly:

```python
from pepe.model_selecter import select_model

# Factory returning the correct subclass (ESMEmbedder, HuggingfaceEmbedder, etc.)
ModelClass = select_model("esm2_t6_8M_UR50D")

embedder = ModelClass(
    model_name="facebook/esm2_t6_8M_UR50D",
    fasta_path="path/to/sequences.fasta",
    output_path="output_directory",
    extract_embeddings=["mean_pooled", "attention_head"],
    layers=[[-1], [6]],  # Layers are expected as a list of lists of ints
)

embedder.run()
```

### Memory management & large-scale processing

PEPE handles datasets of any size using **streamed outputs** and **memory mapping**, enabled by default (`streaming_output=True`).

- **CLI**: no action needed — batches are streamed to disk automatically to avoid out-of-memory errors.
- **Library**: the returned object does **not** hold full embeddings in RAM. It provides `numpy.memmap` handles to the data on disk.

```python
import pepe

results = pepe.embed(
    model_name="facebook/esm2_t6_8M_UR50D",
    sequences=sequences,
    output_path="large_dataset_output",
    # streaming_output=True  <-- Default
)

# Embeddings are NOT loaded into RAM here.
# 'data' is a numpy.memmap pointing to the file on disk.
data = results.mean_pooled["output_data"][-1]

# Slice it like a normal array; only those rows are loaded into RAM.
first_100_embeddings = data[:100]

# Free GPU/CPU memory while keeping the memmaps by deleting the embedder object.
del results
```

### Handling long sequences (splitting & reconstruction)

Some models have strict input-length limits (e.g. 1024 for ESM-2, 256 for AntiBERTa2). PEPE can automatically detect sequences that exceed these limits and handle them through chunking and reconstruction.

- **Automatic detection** — with `--split_long_sequences` enabled, PEPE identifies sequences exceeding the model's capacity.
- **Overlapping chunks** — use `--split_overlap` to maintain context across chunk boundaries.
- **Reconstruction** — in library mode, sequences are reconstructed in memory automatically after `embed()`. In CLI mode, they are reconstructed if `streaming_output=False`; if `streaming_output=True`, chunks are exported individually to maximize efficiency and minimize RAM usage.

```python
# Process a 2000 aa protein with ESM-2 (1024 limit)
results = pepe.embed(
    model_name="facebook/esm2_t33_650M_UR50D",
    sequences={"long_prot": "M" * 2000},
    split_long_sequences=True,
    split_overlap=50,
)

# 'results.per_token' contains a single reconstructed tensor of length ~2002
# (including special tokens), despite the model's 1024 limit.
```

### Performance optimization

If you have enough RAM to hold the entire dataset in memory, disable streaming output for faster execution by avoiding disk writes during embedding:

```python
results = pepe.embed(
    model_name="facebook/esm2_t6_8M_UR50D",
    sequences=sequences,
    output_path="output",
    streaming_output=False,  # Keep everything in RAM for speed
)

# 'results.mean_pooled["output_data"]' is a standard NumPy object, in memory.
```

---

## Supported models

**ESM-family models**

- ESM-1: `esm1_t34_670M_UR50S`, `esm1_t34_670M_UR50D`, `esm1_t34_670M_UR100`, `esm1_t12_85M_UR50S`, `esm1_t6_43M_UR50S`, `esm1b_t33_650M_UR50S`, `esm1v_t33_650M_UR90S_1`, `esm1v_t33_650M_UR90S_2`, `esm1v_t33_650M_UR90S_3`, `esm1v_t33_650M_UR90S_4`, `esm1v_t33_650M_UR90S_5`
- ESM-2: `esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`, `esm2_t36_3B_UR50D`, `esm2_t48_15B_UR50D`

**Hugging Face Transformer models**

- T5: `Rostlab/prot_t5_xl_half_uniref50-enc`, `Rostlab/ProstT5`
- RoFormer: `alchemab/antiberta2-cssp`, `alchemab/antiberta2`
- ESMC: `biohub/ESMC-300M`, `biohub/ESMC-600M`, `biohub/ESMC-6B` (requires Biohub's transformers fork; see Installation)
- Any compatible Hugging Face Hub model (`username/model-name`), including private models with authentication and local Hugging Face directories

**METL models**

- METL 1D: `metl-*-1d` identifiers from `metl-pretrained` (requires `metl-pretrained`; logits and attention outputs are not supported)

**Custom models**

- Load your own PyTorch models with custom tokenizers
- Generate a template: `python examples/custom_model/create_example_custom_model.py`
---

## Arguments

### Required

- **`--model_name`** (str): Name of the model or a path/link to one. Choose from [Supported models](#supported-models), or:
  - ESM models: `esm2_t33_650M_UR50D`
  - Hugging Face models: `username/model-name`
  - Custom PyTorch models: `/path/to/model.pt` or `/path/to/model_directory/`
  - Local HF models: `/path/to/local_hf_directory/`
- **`--fasta_path`** (str): Path to the input FASTA file. If no experiment name is provided, output files are named after the input file.
- **`--output_path`** (str): Directory for output files. A subdirectory is created per output type.

### Model configuration

- **`--tokenizer_from`** (str, optional): Hugging Face address of the tokenizer. If omitted, PEPE searches for a tokenizer packaged with the model. For custom models, provide the path to the tokenizer directory.
- **`--disable_special_tokens`** (bool, optional): When `True`, PEPE does not pre-/append BOS/CLS and EOS/SEP tokens before embedding. Default `False`.
- **`--device`** (str, optional): Device to run the model on: `cuda`, `cpu`, or a specific index like `cuda:0`, `cuda:1`. Default `cuda`.

### Embedding configuration

- **`--layers`** (str, optional): Representation layers to extract. Default is the last layer. Example: `--layers -1 6`.
- **`--extract_embeddings`** (str, optional): One or more output modes:
  - `per_token` — embeddings for each token. Shape: `(num_sequences, max_length, embedding_size)`.
  - `mean_pooled` — average embedding across all tokens, excluding special tokens. Shape: `(num_sequences, embedding_size)`.
  - `substring_pooled` — average embedding for a specific substring per sequence (e.g. a CDR3 region). Requires `--substring_path`. Shape: `(num_sequences, embedding_size)`.
  - `attention_head` — raw attention weights for every head in the specified layers. Shape: `(num_sequences, max_length, max_length)` per head.
  - `attention_layer` — average attention across all heads within each specified layer. Shape: `(num_sequences, max_length, max_length)` per layer.
  - `attention_model` — average attention across all heads and specified layers. Shape: `(num_sequences, max_length, max_length)`.
  - `logits` — raw language model output (experimental).

  Default `mean_pooled`.
- **`--substring_path`** (str, optional): Path to a CSV with columns `sequence_id` and `substring`. Required for `substring_pooled`.
- **`--context`** (int, optional): Number of amino acids to include before and after the substring. Only used with `substring_pooled`. Default `0`.

### Processing configuration

- **`--batch_size`** (int, optional): Batch size for loading sequences. Default `1024`. Decrease if you hit out-of-memory errors.
- **`--max_length`** (int, optional): Length to pad sequences to. Defaults to the longest sequence in the input + special token(s). Values shorter than the longest sequence are forced up to that length + special token(s).
- **`--split_long_sequences`** (bool, optional): Automatically detect and chunk sequences exceeding the model's maximum length. Default `False`.
- **`--split_overlap`** (int, optional): Tokens to overlap when splitting long sequences, to maintain context across chunk boundaries. Default `0`.
- **`--force_split_length`** (int, optional): Force splitting at a specific length, overriding auto-detected limits. Default `None`.
- **`--discard_padding`** (bool, optional): Discard padding tokens from `per_token` output. **Note:** setting this to `True` automatically disables `--streaming_output`. Default `False`.

### Output configuration

- **`--experiment_name`** (str, optional): Prefix for output file names. Defaults to the input file name.
- **`--streaming_output`** (bool, optional): Preallocate disk space and write each batch concurrently. May cause issues on file systems that do not support memory mapping (e.g. some distributed file systems). When `False`, all outputs are held in RAM and written at once after computation. **Note:** automatically disabled if `--discard_padding` is `True`. Default `True`.
- **`--precision`** (str, optional): Output precision: `float16`/`16`/`half` or `float32`/`32`/`full`. Does not affect inference. Default `float32`.
- **`--flatten`** (bool, optional): Flatten 2D output arrays (`per_token` embeddings or attention weights) to 1D per sequence. Default `False`.

### Performance configuration

- **`--num_workers`** (int, optional): Workers for asynchronous data writing. Only relevant with `--streaming_output`. Default `8`.
- **`--flush_batches_after`** (int, optional): Output size (MB) accumulated in RAM per worker before flushing to disk. Default `128`.

---

## Citation

If you use PEPE in your work, please cite:

> **PEPE: Scalable extraction of multi-modal protein language model representations**
> Jahn Zhong, Niccolò Cardente, Geir Kjetil Sandve, Habib Bashour, Maria Francesca Abbate, Victor Greiff
> *Bioinformatics*, 2026, btag375. [https://doi.org/10.1093/bioinformatics/btag375](https://doi.org/10.1093/bioinformatics/btag375)

```bibtex
@article{zhong2026pepe,
  title   = {PEPE: Scalable extraction of multi-modal protein language model representations},
  author  = {Zhong, Jahn and Cardente, Niccol{\`o} and Sandve, Geir Kjetil and Bashour, Habib and Abbate, Maria Francesca and Greiff, Victor},
  journal = {Bioinformatics},
  year    = {2026},
  volume  = {42},
  number  = {6},
  pages   = {btag375},
  doi     = {10.1093/bioinformatics/btag375}
}
```

The archived source is also deposited on Zenodo: [https://zenodo.org/records/15912054](https://zenodo.org/records/15912054).

---

## License

PEPE is released under the [MIT License](https://github.com/csi-greifflab/pepe-cli/blob/main/LICENSE).
