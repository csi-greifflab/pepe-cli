import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))
import pepe
from pepe.model_selecter import select_model

def debug_pepe():
    base_dir = "/doctorai/userdata/pepe-cli"
    fasta_path = os.path.join(base_dir, "src/tests/data/verify_seqs.fasta")
    
    EmbedderClass = select_model("facebook/esm2_t6_8M_UR50D")
    embedder = EmbedderClass(
        model_name="facebook/esm2_t6_8M_UR50D",
        fasta_path=fasta_path,
        device="cpu"
    )
    
    print(f"PEPE Embedder Type: {type(embedder)}")
    print(f"Max Length: {embedder.max_length}")
    print(f"Tokenizer type: {type(embedder.tokenizer)}")
    
    # Check tokenizer special tokens
    print(f"All special ids: {embedder.tokenizer.all_special_ids}")
    print(f"All special tokens: {embedder.tokenizer.all_special_tokens}")
    
    # Run one batch
    for labels, strs, toks, attention_mask, substring_mask in embedder.data_loader:
        print(f"Toks shape: {toks.shape}")
        logits, representations, attention_matrices = embedder._compute_outputs(
            embedder.model, toks, attention_mask, True, False, False
        )
        print(f"Representations layer 6 shape: {representations[6].shape}")
        break

if __name__ == "__main__":
    debug_pepe()
