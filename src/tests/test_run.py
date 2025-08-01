import sys
import os
from pepe.__main__ import parse_arguments
from pepe.model_selecter import select_model


# os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = [
    "pepe",
    "--experiment_name",
    "test",
    "--model_name",
    "esm2_t33_650M_UR50D",
    "--fasta_path",
    "src/tests/test_files/test.fasta",
    "--output_path",
    "src/tests/test_files/test_output",
    "--substring_path",
    "src/tests/test_files/test_substring.csv",
    "--extract_embeddings",
    "mean_pooled",
    "per_token",
    "substring_pooled",
    "attention_head",
    "--streaming_output",
    "true",
]

args = parse_arguments()

# Check if output directory exists and creates it if it's missing
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

embedder = select_model(args.model_name)

embedder = embedder(args)
print("Embedder initialized")

embedder.run()
print("All outputs saved.")
