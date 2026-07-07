# Embedding options
PEPE can extract numerous different representations from the input sequences while embedding each only once.
## Layer selection
Protein representations can be extracted from any of a PLMs hidden layers using the ```--layers``` argument and passing a list of integers. Use negative integers to index layers from the last element. E.g. ```"1"``` is first layer, ```"-1"``` is the last layer, ```"-2"``` is second to last layer, etc. Use ```"all"``` to select all layers. Default option: ```"-1"```.
```sh
pepe \
    --experiment_name "test" \
    --model_name "examples/custom_model/example_protein_model" \
    --fasta_path "src/tests/test_files/test.fasta" \
    --output_path "src/tests/test_files/test_output" \
    --layers 1 -1 # Select the first and last embedding layer
```

## Embedding modes
Multiple embedding modes can be selected at once using the ```--extract_embeddings``` argument. Default option is ```"mean_pooled"```:
- Embeddings:
    - ```"per_token"```: Uncompressed output of a transformer block. Contains high-dimensional representation of each amino acid in the protein sequence.
    - ```"mean_pooled"``` (default option): Average of ```"per_token"``` embedding over all amino acid tokens of the protein sequence.
    - ```"substring_pooled"```: Average of ```"per_token"``` embedding over a specified substring of the protein sequence. Additional arguments when selected:
        - ```--substring_path``` (required for ```"substring_pooled"```): Path to a CSV file with two columns. The first column contains the ```sequence_id``` and the second column must contain a substring of the sequence provided in the FASTA input file.
        - ```--context``` (optional): Specify the number of residues before and after the substring to include during pooling.
- Attention weights:
    - ```"attention_head"```: Asymmetrical pairwise attention weight matrices of input tokens from each self-attention head of the specified layer(s)
    - ```"attention_layer"```: Average of ```"attention_head"``` per specified layer.
    - ```"attention_model"```: Average of ```"attention_layer"``` per model.


```sh
pepe \
    --experiment_name "test" \
    --model_name "examples/custom_model/example_protein_model" \
    --fasta_path "src/tests/test_files/test.fasta" \
    --output_path "src/tests/test_files/test_output" \
    --extract_embeddings per_token mean_pooled attention_head substring_pooled \ # choose embedding modes that should be saved to disk
    --substring_path"src/tests/test_files/test_substring.csv"  # since "substring_pooled" option was passed to --extract_embeddings
```
## Output configuration
The way PEPE outputs representations can be configured with the following arguments:
- ```--streaming_output```: ```True``` (default) or ```False```.  PEPE preallocates the required disk space and writes each batch of outputs concurrently. Disable if encountering file system issues.
- ```--precision```: ```full``` (default) or ```half```. Specifies whether to save representations as ```float32``` (full precision) or ```float16``` (half precision) numerical values for smaller file sizes.
- ```--flatten```: ```True``` or ```False```(default). When enabled, two-dimensional embedding modes (e.g. ```"per_token"``` or ```"attention_layer"```) will be flattened to one-dimensional vector along the first axis.
