import os
import sys
import shutil
# Add src to path
sys.path.append(os.path.abspath("src"))

import pepe
from pepe.model_selecter import select_model

def test_programmatic_example():
    print("Testing Programmatic Embedding example...")
    sequences = {
        "prot1": "MADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "prot2": "MERIKELRDLMSQSRTREILTKLAEAGIDVPRLFK"
    }

    output_path = "readme_example_results"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    results = pepe.embed(
        model_name="facebook/esm2_t6_8M_UR50D",
        sequences=sequences,
        output_path=output_path,
        extract_embeddings=["mean_pooled"],
        device="cpu"
    )
    print("Programmatic example PASSED\n")

def test_advanced_example():
    print("Testing Advanced Usage example...")
    output_path = "readme_advanced_results"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # Create a dummy fasta file for the example
    fasta_path = "src/tests/test_files/test.fasta"

    # Select the appropriate model class
    ModelClass = select_model("esm2_t6_8M_UR50D")

    # Initialize the embedder
    embedder = ModelClass(
        model_name="facebook/esm2_t6_8M_UR50D",
        fasta_path=fasta_path,
        output_path=output_path,
        extract_embeddings=["mean_pooled", "attention_head"],
        layers=[[-1], [6]],
        device="cpu"
    )

    # Run the embedding pipeline
    embedder.run()
    print("Advanced example PASSED\n")

if __name__ == "__main__":
    try:
        test_programmatic_example()
        test_advanced_example()
        print("ALL README LIBRARY EXAMPLES PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
