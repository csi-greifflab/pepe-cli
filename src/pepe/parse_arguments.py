import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2ints(value):
    """Convert a string to a list of integers."""
    if isinstance(value, str):
        if value.lower() == "all":
            return None
        elif value.lower() == "last":
            return [-1]
        else:
            try:
                return [int(x) for x in value.split(" ")]
            except ValueError:
                raise argparse.ArgumentTypeError(
                    "Invalid input. Expected integer(s) or spaced list of integers or 'all' or 'last'."
                )
    elif isinstance(value, int):
        return [value]


# Parsing command-line arguments for input and output file paths
def parse_arguments():
    """Parse command-line arguments for input and output file paths."""
    parser = argparse.ArgumentParser(description="Input path")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment. Will be used to name the output files. If not provided, the output files will be named after the input file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path to custom model. For custom models, use path to .pt/.pth file or directory containing model files. For predefined models, use model name from supported list.",
    )
    parser.add_argument(
        "--tokenizer_from",
        type=str,
        default=None,
        help="Huggingface address of the tokenizer to use. If not provided, will use the tokenizer from the model. If using a custom model, provide the path to the tokenizer directory.",
    )
    parser.add_argument(
        "--fasta_path",
        type=str,
        required=True,
        help="Path to the input FASTA file. Required. If no experiment name is provided, the output files will be named after the input file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory for output files \n Will generate a subdirectory for outputs of each output_type.\n Will output multiple files if multiple layers are specified with '--layers'. Output file is a single tensor or a list of tensors when --pooling is False.",
    )
    parser.add_argument(
        "--substring_path",
        default=None,
        type=str,
        help=" Path to a CSV file with columns 'sequence_id' and 'substring'. Only required when selecting 'substring' embedding option.",
    )
    parser.add_argument(
        "--context",
        default=0,
        type=int,
        help="Only specify when including 'substring_pooling' in '--extract_embeddings' option. Number of amino acids to include before and after the substring sequence. Default is 0.",
    )
    parser.add_argument(
        "--layers",
        type=str2ints,
        nargs="*",
        default=[[-1]],
        help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.",
    )
    parser.add_argument(
        "--extract_embeddings",
        choices=[
            "per_token",
            "mean_pooled",
            "substring_pooled",
            "attention_head",
            "attention_layer",
            "attention_model",
            "logits",
        ],
        default=["mean_pooled"],
        nargs="+",
        help="Set the embedding return types. Choose one or more from: 'per_token', 'mean_pooled', 'substring_pooled', 'attention_head', 'attention_layer', 'attention_model' and 'logits' (experimental). Default is 'pooled'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Number of tokens (not sequences!) per batch. Default is 1024.",
    )
    parser.add_argument(
        "--discard_padding",
        action="store_true",
        help="Discard padding tokens from per_token embeddings output. Not compatible with --streaming_output Default is False.",
    )
    parser.add_argument(
        "--max_input_length",
        type=str,
        default="max_input_length",
        help="Length to which sequences will be padded. Default is length of longest sequence in input file. If shorter than longest sequence, will forcefully default to length of longest sequence.",
    )
    parser.add_argument(
        "--split_long_sequences",
        action="store_true",
        help="If a sequence exceeds the maximum permitted length for the chosen model, split it into smaller overlapping chunks and concatenate the output representations.",
    )
    parser.add_argument(
        "--split_overlap",
        type=int,
        default=0,
        help="Number of tokens to overlap when splitting long sequences. Default is 0.",
    )
    parser.add_argument(
        "--streaming_output",
        type=str2bool,
        choices=[True, False],
        default=True,
        help="Preallocate output files and concurrently write embeddings to disk in batches while computing embeddings. Default is True.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for asynchronous data writing. Only relevant when --batch_writing is enabled. Default is 8.",
    )
    parser.add_argument(
        "--disable_special_tokens",
        type=str2bool,
        choices=[True, False],
        default=False,
        help="Disable special tokens in the model. Default is False.",
    )
    parser.add_argument(
        "--flatten",
        type=str2bool,
        choices=[True, False],
        default=False,
        help="Flatten the output tensors. Default is False.",
    )
    parser.add_argument(
        "--flush_batches_after",
        type=int,
        default=128,
        help="Size (in MB) of outputs to accumulate in RAM per worker (--num_workers) before flushing to disk. Default is 128.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["float16", "16", "half", "float32", "32", "full"],
        help="Precision of the output data. Inference during embedding is not affected. Default is 'float32'.",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on. Default is 'cuda'.",
    )

    args = parser.parse_args()
    return args
