# PEPE

PEPE (Pipeline for Easy Protein Embedding) is a tool for extracting embeddings and attention matrices from protein sequences using pre-trained models. This tool supports various configurations for extracting embeddings and attention matrices, including options for handling CDR3 sequences. Currently implemented models are ESM2 from the 2023 paper ["Evolutionary-scale prediction of atomic-level protein structure with a language model"](https://science.org/doi/10.1126/science.ade2574) and AntiBERTa2-CSSP from the 2023 conference paper ["Enhancing Antibody Language Models with Structural Information"](https://www.mlsb.io/papers_2023/Enhancing_Antibody_Language_Models_with_Structural_Information.pdf). PEPE also supports custom PLMs from local files or from Huggingface Hub addresses. 

## Quick start

1. Install PEPE \
    From PyPI:    
    ```sh
    pip install pepe-cli
    ```
    Or install from the GitHub repository:    
    ```sh
    git clone https://github.com/csi-greifflab/pepe-cli
    cd pepe-cli
    pip install .
    ```
2. Run the embedding script:\
    Extract mean pooled embeddings from protein amino acid sequences in FASTA file:
    ```sh
    pepe --experiment_name <optional_string> --fasta_path <file_path> --output_path <directory> --model_name <model_name>
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
  - Hugging Face models: `username/model-name`
  - Custom PyTorch models: `/path/to/model.pt` or `/path/to/model_directory/`
  - Local HF models: `/path/to/local_hf_directory/`
- **`--fasta_path`** (str): Path to the input FASTA file. If no experiment name is provided, the output files will be named after the input file.
- **`--output_path`** (str): Directory for output files. Will generate a subdirectory for outputs of each output type.

### Model Configuration
- **`--tokenizer_from`** (str, optional): Huggingface address of the tokenizer to use. If not provided, will attempt to search for tokenizer packaged with model. If using a custom model, provide the path to the tokenizer directory.
- **`--disable_special_tokens`** (bool, optional): When True, PEPE disables pre- and appending BOS/CLS and EOS/SEP tokens before embedding. Default is `False`.
- **`--device`** (str, optional): Device to run the model on. Choose from `cuda` or `cpu`. Default is `cuda`.

### Embedding Configuration
- **`--layers`** (str, optional): Representation layers to extract from the model. Default is the last layer. Example: `--layers -1 6`.
- **`--extract_embeddings`** (str, optional): Set the embedding return types. Choose one or more from: `per_token`, `mean_pooled`, `substring_pooled`, `attention_head`, `attention_layer`, `attention_model` and `logits` (experimental). Default is `mean_pooled`.
- **`--substring_path`** (str, optional): Path to a CSV file with columns "sequence_id" and "substring". Only required when selecting "substring_pooled" option.
- **`--context`** (int, optional): Only specify when including "substring_pooled" in `--extract_embeddings` option. Number of amino acids to include before and after the substring sequence. Default is `0`.

### Processing Configuration
- **`--batch_size`** (int, optional): Batch size for loading sequences. Default is `1024`. Decrease if encountering out-of-memory errors.
- **`--max_length`** (int, optional): Length to which sequences will be padded. Default is length of longest sequence in input file. If shorter than longest sequence, will forcefully default to length of longest sequence.
- **`--discard_padding`** (bool, optional): Discard padding tokens from per_token embeddings output. Default is `False`.

### Output Configuration
- **`--experiment_name`** (str, optional): Prefix for names of output files. If not provided, name of input file will be used for prefix.
- **`--streaming_output`** (bool, optional): PEPE preallocates the required disk space and writes each batch of outputs concurrently. Can pose issues with file systems that do not support memory mapping (such as some distributed file systems.)
When False, all outputs are stored in RAM and written to disk at once after computation has finished. Default is `True`.
- **`--precision`** (str, optional): Precision of the output data. Choose from `float16`, `16`, `half`, `float32`, `32`, `full`. Inference during embedding is not affected. Default is `float32`.
- **`--flatten`** (bool, optional): Flatten 2D output arrays (per_token embeddings or attention weights) to 1D arrays per input sequence. Default is `False`.

### Performance Configuration
- **`--num_workers`** (int, optional): Number of workers for asynchronous data writing. Only relevant when `--streaming_output` is enabled. Default is `8`.
- **`--flush_batches_after`** (int, optional): Size (in MB) of outputs to accumulate in RAM per worker before flushing to disk. Default is `128`.