import os
import tempfile
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Union, Any

from pepe.model_selecter import select_model
import pepe.utils

logger = logging.getLogger("pepe.api")

def embed(
    model_name: str,
    sequences: Optional[Union[Dict[str, str], List[str]]] = None,
    fasta_path: Optional[str] = None,
    output_path: Optional[str] = None,
    extract_embeddings: List[str] = ["mean_pooled"],
    layers: Union[List[int], List[List[int]]] = [[-1]],
    batch_size: int = 1024,
    device: str = "cuda",
    precision: str = "32",
    streaming_output: bool = True,
    discard_padding: bool = False,
    max_input_length: str = "max_length",
    experiment_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    High-level API for generating protein embeddings.

    Args:
        model_name: Model name or path to custom model.
        sequences: Dictionary mapping labels to protein sequences, or a list of sequences.
        fasta_path: Path to a FASTA file. Used if 'sequences' is not provided.
        output_path: Directory for output files. If not provided, a temporary directory will be used.
        extract_embeddings: List of embedding types to extract.
        layers: Representation layers to extract. Default is the last layer ([-1]).
        batch_size: Number of tokens per batch. Default is 1024.
        device: Device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.
        precision: Output precision (e.g., '32', '16'). Default is '32'.
        streaming_output: Whether to stream outputs to disk. Default is True.
        discard_padding: Whether to discard padding tokens. Default is False.
        max_input_length: Length to which sequences will be padded. Default is "max_length".
        experiment_name: Optional prefix for output files.
        **kwargs: Additional arguments supported by the embedders.

    Returns:
        A dictionary containing the results and/or the output path.
    """
    # Create a temporary fasta file if sequences are provided
    temp_fasta = None
    if sequences is not None:
        if isinstance(sequences, list):
            sequences = {f"seq_{i}": seq for i, seq in enumerate(sequences)}
            
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        for label, seq in sequences.items():
            temp_fasta.write(f">{label}\n{seq}\n")
        temp_fasta.close()
        fasta_path = temp_fasta.name

    if fasta_path is None:
        raise ValueError("Either 'sequences' or 'fasta_path' must be provided.")

    return_results = False
    if output_path is None:
        if streaming_output:
            logger.warning("No output_path provided. Disabling streaming_output and returning in-memory results.")
        output_path = tempfile.mkdtemp()
        streaming_output = False
        return_results = True
    
    # Create args object
    args_dict = {
        "model_name": model_name,
        "fasta_path": fasta_path,
        "output_path": output_path,
        "extract_embeddings": extract_embeddings,
        "layers": layers,
        "batch_size": batch_size,
        "device": device,
        "precision": precision,
        "streaming_output": streaming_output,
        "discard_padding": discard_padding,
        "max_input_length": max_input_length,
        "experiment_name": experiment_name,
        "tokenizer_from": kwargs.get("tokenizer_from"),
        "substring_path": kwargs.get("substring_path"),
        "context": kwargs.get("context", 0),
        "split_long_sequences": kwargs.get("split_long_sequences", False),
        "split_overlap": kwargs.get("split_overlap", 0),
        "force_split_length": kwargs.get("force_split_length"),
        "num_workers": kwargs.get("num_workers", 8),
        "disable_special_tokens": kwargs.get("disable_special_tokens", False),
        "flatten": kwargs.get("flatten", False),
        "flush_batches_after": kwargs.get("flush_batches_after", 128),
    }
    
    args = SimpleNamespace(**args_dict)
    
    selected_model_class = select_model(model_name)
    embedder = selected_model_class(args)
    embedder.run()
    
    results = {"output_path": output_path}
    
    if return_results:
        # Pick up in-memory results before they are lost
        for output_type in extract_embeddings:
            obj = getattr(embedder, output_type, None)
            if obj and "output_data" in obj:
                results[output_type] = obj["output_data"]
    
    # Cleanup temp file
    if temp_fasta:
        os.unlink(temp_fasta.name)
        
    return results
