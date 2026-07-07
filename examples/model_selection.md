# Model selection
PEPE can extract embeddings from a wide range of PLMs through the ```--model_name``` argument.
## Supported Models ([list of currently supported models](../README.md#list-of-supported-models))
To select any of the supported models, pass the name of a model from the [list of currently supported models](../README.md#list-of-supported-models) (e.g. ```"esm2_t33_650M_UR50D"``` or ```"alchemab/antiberta2-cssp"```):
```sh
pepe \
--experiment_name "test" \
--model_name "esm2_t33_650M_UR50D" \ # pass the name of a supported model
--fasta_path "src/tests/test_files/test.fasta" \
--output_path "src/tests/test_files/test_output"
```
## Models hosted on Huggingface
The user can also choose from PLMs hosted on Huggingface Hub by passing the repository path as `--model_name`:
```sh
pepe \
    --experiment_name "test" \
    --model_name "alchemab/antiberta2-cssp" \ # pass the remote path to a Huggingface model
    --fasta_path "src/tests/test_files/test.fasta" \
    --output_path "src/tests/test_files/test_output"
```

### ESMC models
ESMC (ESM Cambrian) is a distinct architecture from ESM2. It requires Biohub's transformers fork, which coexists with ESM1's `fair-esm` package:
```sh
pip install git+https://github.com/Biohub/transformers.git@main
pepe \
    --experiment_name "esmc_test" \
    --model_name "biohub/ESMC-300M" \
    --fasta_path "src/tests/test_files/test.fasta" \
    --output_path "src/tests/test_files/test_output" \
    --extract_embeddings mean_pooled \
    --device cpu \
    --layers -1
```

## Custom models
To specify your own custom model, pass the path to a directory containing:
- .pt or .pth file (model weights)
- config.json file (model metadata)
- Tokenizer files:
    - tokenizer_config.json
    - vocab.json
    - vocab.txt
    - special_tokens_map.json

Instead of providing a custom tokenizer, it is also possible to select the tokenizer from a model hosten on Huggingface Hub by passing the repository path to the ```--tokenizer_from```.
```sh
pepe \
    --experiment_name "test" \
    --model_name "examples/custom_model/example_protein_model" \ # pass the directory path containing custom PyTorch model
    --tokenizer_from "alchemab/antiberta2-cssp" \ # Uses the same tokenizer as AntiBERTa2-CSSP
    --fasta_path "src/tests/test_files/test.fasta" \
    --output_path "src/tests/test_files/test_output" 
```
For details, see the [example_protein_model folder](examples/custom_model/example_protein_model) and the [python script](examples/custom_model/create_example_custom_model.py) for generating the example_protein_model files.
