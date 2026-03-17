#!/usr/bin/env bash
pepe \
    --experiment_name "test" \
    --model_name "examples/custom_model/example_protein_model" \
    --fasta_path "src/tests/test_files/test.fasta" \
    --output_path "src/tests/test_files/test_output" \
    --substring_path "src/tests/test_files/test_substring.csv" \
    --extract_embeddings "per_token" mean_pooled substring_pooled attention_head \
    --streaming_output true \
    --device cpu \
    --layers -2 -1

# Example: Long sequence splitting (using AntiBERTa2)
# pepe \
#     --model_name "alchemab/antiberta2-cssp" \
#     --fasta_path "src/tests/long_protein.fasta" \
#     --output_path "src/tests/cli_test_out" \
#     --split_long_sequences \
#     --split_overlap 50 \
#     --streaming_output false \
#     --device cpu