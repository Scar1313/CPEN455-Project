import os
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths

def main():
    generated_dir = "samples"
    reference_dir = "data/test"
    batch_size = 32
    dims = 192
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert os.path.exists(generated_dir), f"Generated image folder not found: {generated_dir}"
    assert os.path.exists(reference_dir), f"Reference image folder not found: {reference_dir}"

    print(f"Calculating FID between:\n - Generated: {generated_dir}\n - Reference: {reference_dir}")
    fid_score = calculate_fid_given_paths([generated_dir, reference_dir], batch_size, device, dims=dims)
    print(f"✅ FID score (dims={dims}): {fid_score:.4f}")

if __name__ == "__main__":
    main()

