import os
import sys
import subprocess
import torch
import numpy as np
import shutil

# Add src to path
sys.path.append(os.path.abspath("src"))
import pepe

def run_pepe(fasta_path, output_dir):
    print("Running PEPE...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    pepe.embed(
        model_name="facebook/esm2_t6_8M_UR50D",
        fasta_path=fasta_path,
        output_path=output_dir,
        experiment_name="pepe_output",
        extract_embeddings=["mean_pooled", "per_token"],
        streaming_output=False,
        device="cpu"
    )
    
    model_name = "esm2_t6_8M_UR50D"
    per_token_file = os.path.join(output_dir, model_name, "per_token", f"pepe_output_{model_name}_per_token_layer_6.npy")
    mean_pooled_file = os.path.join(output_dir, model_name, "mean_pooled", f"pepe_output_{model_name}_mean_pooled_layer_6.npy")
    
    return np.load(per_token_file), np.load(mean_pooled_file)

def run_plmfit(fasta_path, plmfit_repo_path, venv_path, output_dir):
    print("Running PLMFit...")
    # PLMFit needs data in plmfit_repo/data/verify/verify_data_full.csv
    # We already prepared it.
    
    # Run via subprocess
    cmd = [
        os.path.join(venv_path, "bin", "python3"),
        "-m", "plmfit",
        "--function", "extract_embeddings",
        "--data_type", "verify",
        "--plm", "esm2_t6_8M_UR50D",
        "--output_dir", output_dir,
        "--experiment_dir", "verify_exp",
        "--experiment_name", "verify",
        "--layer", "6",
        "--reduction", "mean"
    ]
    
    # We need to set DATA_DIR and CONFIG_DIR for PLMFit, and force CPU
    env = os.environ.copy()
    env["DATA_DIR"] = os.path.join(plmfit_repo_path, "data")
    env["CONFIG_DIR"] = os.path.join(plmfit_repo_path, "config")
    env["CUDA_VISIBLE_DEVICES"] = ""
    
    subprocess.run(cmd, check=True, cwd=plmfit_repo_path, env=env)
    
    # PLMFit output is a .pt file
    # Based on logs, it saves to experiment_dir/experiment_name.pt
    output_pt = os.path.join(plmfit_repo_path, "verify_exp", "verify.pt")
    
    # If not found, search for it
    if not os.path.exists(output_pt):
        print(f"Searching for PLMFit output in {os.path.join(plmfit_repo_path, 'verify_exp')}...")
        for root, dirs, files in os.walk(os.path.join(plmfit_repo_path, 'verify_exp')):
            for f in files:
                if f.endswith("verify.pt"):
                    output_pt = os.path.join(root, f)
                    break
    
    data = torch.load(output_pt, map_location="cpu")
    return data

def run_official_esm(fasta_path, venv_path, output_path):
    print("Running official ESM...")
    cmd = [
        os.path.join(venv_path, "bin", "python3"),
        "src/tests/run_official_esm.py",
        fasta_path,
        output_path
    ]
    subprocess.run(cmd, check=True)
    
    data = torch.load(output_path, map_location="cpu")
    return data

def main():
    base_dir = "/doctorai/userdata/pepe-cli"
    fasta_path = os.path.join(base_dir, "src/tests/data/verify_seqs.fasta")
    
    pepe_out_dir = os.path.join(base_dir, "test_verify_pepe")
    per_token_pepe, mean_pooled_pepe = run_pepe(fasta_path, pepe_out_dir)
    
    # Run official ESM
    esm_out_path = os.path.join(base_dir, "test_verify_esm_official.pt")
    esm_results = run_official_esm(fasta_path, os.path.join(base_dir, "venv_esm_official"), esm_out_path)
    mean_pooled_esm_residues = esm_results["mean_pooled_residues"]
    mean_pooled_esm_all = esm_results["mean_pooled_all"]
    
    # Run PLMFit
    plmfit_repo_path = os.path.join(base_dir, "plmfit_repo")
    plmfit_out_dir = "output_verify"
    per_token_plmfit = run_plmfit(fasta_path, plmfit_repo_path, os.path.join(base_dir, "venv_plmfit"), plmfit_out_dir)
    
    print("\n--- Comparison Results (Mean Pooled Only) ---")
    
    mean_pooled_pepe_torch = torch.from_numpy(mean_pooled_pepe)

    # PEPE vs Official ESM (Residues only)
    diff_mp_res = torch.allclose(mean_pooled_pepe_torch, mean_pooled_esm_residues, atol=1e-5)
    print(f"PEPE vs Official ESM (Mean Pooled Residues): {'MATCH' if diff_mp_res else 'FAIL'}")
    if not diff_mp_res:
        print(f"  Max absolute difference: {torch.max(torch.abs(mean_pooled_pepe_torch - mean_pooled_esm_residues)).item()}")
    
    # PLMFit vs Official ESM (All tokens)
    # PLMFit output is already mean pooled
    diff_plmfit = torch.allclose(per_token_plmfit, mean_pooled_esm_all, atol=1e-5)
    print(f"PLMFit vs Official ESM (Mean Pooled All Tokens): {'MATCH' if diff_plmfit else 'FAIL'}")
    if not diff_plmfit:
        print(f"  Max absolute difference: {torch.max(torch.abs(per_token_plmfit - mean_pooled_esm_all)).item()}")
    
    # Summary
    if diff_mp_res and diff_plmfit:
        print("\nSUCCESS: Mean pooled embeddings match across tools.")
    else:
        print("\nFAILURE: Mean pooled embeddings do not match.")
        sys.exit(1)

if __name__ == "__main__":
    main()
