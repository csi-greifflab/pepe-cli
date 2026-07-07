# PEPE

PEPE (Pipeline for Easy Protein Embedding) is a tool for extracting embeddings and attention matrices from protein sequences using pre-trained models. This tool supports various configurations for extracting embeddings and attention matrices, including options for handling CDR3 sequences. Currently implemented models are ESM2 from the 2023 paper ["Evolutionary-scale prediction of atomic-level protein structure with a language model"](https://science.org/doi/10.1126/science.ade2574) and AntiBERTa2-CSSP from the 2023 conference paper ["Enhancing Antibody Language Models with Structural Information"](https://www.mlsb.io/papers_2023/Enhancing_Antibody_Language_Models_with_Structural_Information.pdf). PEPE also supports custom PLMs from local files or from Huggingface Hub addresses. 

### Citation
> **PEPE: Scalable extraction of multi-modal protein language model representations**
> Jahn Zhong, Niccolò Cardente, Geir Kjetil Sandve, Habib Bashour, Maria Francesca Abbate, Victor Greiff
> *bioRxiv* (2026) 
> [DOI: 10.1101/2025.10.13.680902](https://doi.org/10.1101/2025.10.13.680902)

## Quick start

1. Install PEPE

    From PyPI:    
    ```sh
    pip install pepe-cli
    ```
    From Conda:
    ```sh
    conda install -c jahn_zhong pepe-cli
    ```
    Or install from the GitHub repository:    
    ```sh
    git clone https://github.com/csi-greifflab/pepe-cli
    cd pepe-cli
    pip install .
    ```

2. *(Optional)* For ESMC models (e.g. `biohub/ESMC-300M`), install Biohub's transformers fork:
    ```sh
    pip install git+https://github.com/Biohub/transformers.git@main
    ```

3. Extract embeddings:\
    Extract mean pooled embeddings from protein amino acid sequences in FASTA file:
    ```sh
    pepe --experiment_name <optional_string> --fasta_path <file_path> --output_path <directory> --model_name <model_name>
    ```

## Quick start with Library

PEPE can also be used as a Python library. This allows for programmatic access to protein embeddings without using the command-line interface.

1. Install PEPE (see above for details).
2. Use the `pepe.embed()` function in your script:

```python
import pepe

# Example: Embed sequences from a dictionary
sequences = {
    "prot1": "MADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "prot2": "MERIKELRDLMSQSRTREILTKLAEAGIDVPRLFK"
}

results = pepe.embed(
    model_name="facebook/esm2_t6_8M_UR50D",
    sequences=sequences,
    output_path="my_embeddings",
    extract_embeddings=["mean_pooled"],
    device="cpu"  # Use "cuda", "cuda:0", "cuda:1", etc. if available
)

# Or from a FASTA file
pepe.embed(
    model_name="facebook/esm2_t6_8M_UR50D",
    fasta_path="path/to/fasta",
    output_path="my_embeddings",
    extract_embeddings=["mean_pooled"],
    device="cpu"  # Use "cuda" if available
)
```

### Advanced Usage

For more control, you can use the embedder classes directly:

```python
from pepe.model_selecter import select_model

# Select the appropriate model class
# This acts as a factory returning the correct subclass (ESMEmbedder, HuggingfaceEmbedder, etc.)
ModelClass = select_model("esm2_t6_8M_UR50D")

# Initialize the embedder
embedder = ModelClass(
    model_name="facebook/esm2_t6_8M_UR50D",
    fasta_path="path/to/sequences.fasta",
    output_path="output_directory",
    extract_embeddings=["mean_pooled", "attention_head"],
    layers=[[-1], [6]]  # Layers are expected as a list of lists of ints
)

# Run the embedding pipeline
embedder.run()
```

### Memory Management & Large Scale Processing

PEPE is designed to handle protein datasets of any size by utilizing **streamed outputs** and **memory mapping**. This feature is enabled by default (`streaming_output=True`).

- **CLI Usage**: No action needed. PEPE automatically streams batches to disk to avoid OOM errors.
- **Library Usage**: When using `pepe.embed()` or the embedder classes, the returned object does **not** hold the full embeddings in RAM. Instead, it provides `numpy.memmap` handles to the data on disk.

```python
import pepe

# Returns an embedder object
results = pepe.embed(
    model_name="facebook/esm2_t6_8M_UR50D",
    sequences=sequences,
    output_path="large_dataset_output"
    # streaming_output=True  <-- Default
)

# The embeddings are NOT loaded into RAM here.
# 'data' is a numpy.memmap object pointing to the file on disk.
data = results.mean_pooled["output_data"][-1] 

# You can slice it like a normal array, which only loads those specific rows into RAM
first_100_embeddings = data[:100]

# Optimizing RAM usage:
# If you are done with the model but want to keep working with the data,
# you can delete the embedder object to free up GPU/CPU memory while keeping the memmaps.
del results 
```

### Handling Long Sequences (Splitting & Reconstruction)

Some models have strict architectural limits on input length (e.g., 1024 for ESM-2, 256 for AntiBERTa2). PEPE can automatically detect sequences that exceed these limits and handle them through chunking and reconstruction.

- **Automatic Detection**: When `--split_long_sequences` is enabled, PEPE automatically identifies sequences exceeding the model's capacity.
- **Overlapping Chunks**: Use `--split_overlap` to maintain context between chunks. 
- **Reconstruction**: 
    - In **Library mode**, sequences are reconstructed in memory automatically after `embed()`.
    - In **CLI mode**, sequences are reconstructed if `streaming_output=False`. If `streaming_output=True`, chunks are exported individually to maximize efficiency and minimize RAM usage.

```python
# Library Example: Process a 2000 AA protein with ESM-2 (1024 limit)
results = pepe.embed(
    model_name="facebook/esm2_t33_650M_UR50D",
    sequences={"long_prot": "M" * 2000},
    split_long_sequences=True,
    split_overlap=50
)

# 'results.per_token' will contain a single reconstructed tensor of length ~2002
# (including special tokens) despite the model's 1024 limit.
```


#### Performance Optimization

If you have sufficient RAM to hold the entire dataset in memory, you can disable streaming output for faster execution. This avoids the overhead of writing to disk during the embedding process.

```python
results = pepe.embed(
    model_name="facebook/esm2_t6_8M_UR50D",
    sequences=sequences,
    output_path="output",
    streaming_output=False  # Store everything in valid RAM for speed
)

# Now 'results.mean_pooled["output_data"]' is a standard Numpy object
# accessible immediately in memory.
```

## List of supported models:
- ESM-family models
    - ESM1:
        - esm1_t34_670M_UR50S
        - esm1_t34_670M_UR50D
        - esm1_t34_670M_UR100
        - esm1_t12_85M_UR50S
        - esm1_t6_43M_UR50S
        - esm1b_t33_650M_UR50S
        - esm1v_t33_650M_UR90S_1
        - esm1v_t33_650M_UR90S_2
        - esm1v_t33_650M_UR90S_3
        - esm1v_t33_650M_UR90S_4
        - esm1v_t33_650M_UR90S_5
    - ESM2:
        - esm2_t6_8M_UR50D
        - esm2_t12_35M_UR50D
        - esm2_t30_150M_UR50D
        - esm2_t33_650M_UR50D
        - esm2_t36_3B_UR50D
        - esm2_t48_15B_UR50D
- Huggingface Transformer models
    - T5 transformer models
        - Rostlab/prot_t5_xl_half_uniref50-enc
        - Rostlab/ProstT5
    - RoFormer models
        - alchemab/antiberta2-cssp
        - alchemab/antiberta2
    - ESMC models (requires Biohub transformers fork; see install instructions above)
        - biohub/ESMC-300M
        - biohub/ESMC-600M
        - biohub/ESMC-6B
    - Custom Hugging Face models
        - Any compatible model from Hugging Face Hub: `username/model-name`
        - Private models with authentication
        - Local Hugging Face models
- Custom Models
    - Load your own PyTorch models with custom tokenizers
    - Create example with: `python examples/custom_model/create_example_custom_model.py`


## Arguments

### Required Arguments
- **`--model_name`** (str): Name of model or link to model. Choose from [List of supported models](../README.md#list-of-supported-models) or use custom models:
  - ESM models: `esm2_t33_650M_UR50D`
  - ESMC models: `biohub/ESMC-300M` (requires Biohub transformers fork; see Quick start)
  - Hugging Face models: `username/model-name`
  - Custom PyTorch models: `/path/to/model.pt` or `/path/to/model_directory/`
  - Local HF models: `/path/to/local_hf_directory/`
- **`--fasta_path`** (str): Path to the input FASTA file. If no experiment name is provided, the output files will be named after the input file.
- **`--output_path`** (str): Directory for output files. Will generate a subdirectory for outputs of each output type.

### Model Configuration
- **`--tokenizer_from`** (str, optional): Huggingface address of the tokenizer to use. If not provided, will attempt to search for tokenizer packaged with model. If using a custom model, provide the path to the tokenizer directory.
- **`--disable_special_tokens`** (bool, optional): When True, PEPE disables pre- and appending BOS/CLS and EOS/SEP tokens before embedding. Default is `False`.
- **`--device`** (str, optional): Device to run the model on. Choose from `cuda`, `cpu`, or specific GPU indices like `cuda:0`, `cuda:1`. Default is `cuda`.

### Embedding Configuration
- **`--layers`** (str, optional): Representation layers to extract from the model. Default is the last layer. Example: `--layers -1 6`.
- **`--extract_embeddings`** (str, optional): Set the embedding return types. Choose one or more from:
  - `per_token`: Extracts embeddings for each token (amino acid) in the sequence. Output shape: `(num_sequences, max_length, embedding_size)`.
  - `mean_pooled`: Computes the average embedding across all tokens in a sequence, excluding special tokens (BOS, EOS, padding). Output shape: `(num_sequences, embedding_size)`.
  - `substring_pooled`: Computes the average embedding for a specific substring within each sequence (e.g., a CDR3 region). Requires `--substring_path`. Output shape: `(num_sequences, embedding_size)`.
  - `attention_head`: Extracts raw attention weights for every individual head in the specified layers. Output shape: `(num_sequences, max_length, max_length)` per head.
  - `attention_layer`: Extracts the average attention weights across all heads within each specified layer. Output shape: `(num_sequences, max_length, max_length)` per layer.
  - `attention_model`: Extracts the average attention weights across all heads and all specified layers. Output shape: `(num_sequences, max_length, max_length)`.
  - `logits`: Extracts the raw language model output (logits). (Experimental)
  Default is `mean_pooled`.
- **`--substring_path`** (str, optional): Path to a CSV file with columns "sequence_id" and "substring". Only required when selecting "substring_pooled" option.
- **`--context`** (int, optional): Only specify when including "substring_pooled" in `--extract_embeddings` option. Number of amino acids to include before and after the substring sequence. Default is `0`.

### Processing Configuration
- **`--batch_size`** (int, optional): Batch size for loading sequences. Default is `1024`. Decrease if encountering out-of-memory errors.
- **`--max_length`** (int, optional): Length to which sequences will be padded. Default is length of longest sequence in input file + special token(s). If shorter than longest sequence, will forcefully default to length of longest sequence + special token(s).
- **`--split_long_sequences`** (bool, optional): When True, automatically detect sequences exceeding the model's maximum allowed length and split them into chunks for processing. Default is `False`.
- **`--split_overlap`** (int, optional): Number of tokens to overlap when splitting long sequences. This helps maintain context across chunk boundaries. Default is `0`.
- **`--force_split_length`** (int, optional): Explicitly force sequence splitting at a specific length, overriding the model's auto-detected limits. Default is `None`.
- **`--discard_padding`** (bool, optional): Discard padding tokens from per_token embeddings output. **Note**: Setting this to `True` will automatically disable `--streaming_output`. Default is `False`.


### Output Configuration
- **`--experiment_name`** (str, optional): Prefix for names of output files. If not provided, name of input file will be used for prefix.
- **`--streaming_output`** (bool, optional): PEPE preallocates the required disk space and writes each batch of outputs concurrently. Can pose issues with file systems that do not support memory mapping (such as some distributed file systems.)
When False, all outputs are stored in RAM and written to disk at once after computation has finished. **Note**: Automatically disabled if `--discard_padding` is `True`. Default is `True`.
- **`--precision`** (str, optional): Precision of the output data. Choose from `float16`, `16`, `half`, `float32`, `32`, `full`. Inference during embedding is not affected. Default is `float32`.
- **`--flatten`** (bool, optional): Flatten 2D output arrays (per_token embeddings or attention weights) to 1D arrays per input sequence. Default is `False`.

### Performance Configuration
- **`--num_workers`** (int, optional): Number of workers for asynchronous data writing. Only relevant when `--streaming_output` is enabled. Default is `8`.
- **`--flush_batches_after`** (int, optional): Size (in MB) of outputs to accumulate in RAM per worker before flushing to disk. Default is `128`.
