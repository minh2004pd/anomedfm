import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
import shutil

def pad_and_normalize_slice(slices):
    """
    slices: list of 4 (240, 240) numpy arrays
    Returns: (4, 256, 256) normalized and padded numpy array
    """
    processed_slices = []
    for slice_data in slices:
        # 1. Pad to 256x256
        h, w = slice_data.shape
        pad_h = (256 - h) // 2
        pad_w = (256 - w) // 2
        padded = np.zeros((256, 256), dtype=np.float32)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = slice_data
        
        # 2. Normalize 0-1
        mn = padded.min()
        mx = padded.max()
        if mx - mn > 1e-8:
            padded = (padded - mn) / (mx - mn)
        else:
            padded = np.zeros_like(padded)
        
        processed_slices.append(padded)
        
    return np.stack(processed_slices, axis=0) # (4, 256, 256)

def process_subject(subject_path, output_root, subject_id):
    modalities = ['t1', 't1ce', 't2', 'flair'] # Standard order
    
    # Load headers and dataobj (proxy for data)
    data_proxies = {}
    for mod in modalities:
        file_path = os.path.join(subject_path, f"{subject_id}_{mod}.nii.gz")
        if not os.path.exists(file_path):
            return None
        img = nib.load(file_path)
        data_proxies[mod] = img.dataobj

    seg_path = os.path.join(subject_path, f"{subject_id}_seg.nii.gz")
    if not os.path.exists(seg_path):
        return None
    seg_proxy = nib.load(seg_path).dataobj

    # Range: [80, 155 - 26] = [80, 129]
    start_slice = 80
    end_slice = 155 - 26 
    
    healthy_files = []
    diseased_files = []
    
    for i in range(start_slice, end_slice):
        # Load slice for each modality
        slices = [np.asarray(data_proxies[mod][..., i]).astype(np.float32) for mod in modalities]
        
        # Process (Pad & Normalize)
        combined_slice = pad_and_normalize_slice(slices)
        
        # Check label
        mask_slice = np.asarray(seg_proxy[..., i])
        is_diseased = np.any(mask_slice > 0)
        
        # Folder management
        status_folder = "unhealthy" if is_diseased else "healthy"
        subject_status_dir = os.path.join(output_root, status_folder, subject_id)
        os.makedirs(subject_status_dir, exist_ok=True)
        
        # Save slice
        output_filename = f"slice_{i:03d}.npy"
        save_path = os.path.join(subject_status_dir, output_filename)
        np.save(save_path, combined_slice.astype(np.float32))
        
        # Store relative file paths for statistics/lists
        rel_path = os.path.join(status_folder, subject_id, output_filename)
        if is_diseased:
            diseased_files.append(rel_path)
        else:
            healthy_files.append(rel_path)
            
    return healthy_files, diseased_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/k66/minhdd/flow-matching-main/data/brats2021/extracted_data")
    parser.add_argument("--output_dir", type=str, default="/home/k66/minhdd/flow-matching-main/data/brats2021")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--clean", action="store_true", help="Delete existing output folders before processing")
    args = parser.parse_args()

    if args.clean:
        for folder in ["healthy", "unhealthy"]:
            target_path = os.path.join(args.output_dir, folder)
            if os.path.exists(target_path):
                print(f"Cleaning existing folder: {target_path}")
                shutil.rmtree(target_path)

    subjects = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    subjects.sort()
    
    if args.subset:
        subjects = subjects[:args.subset]
    
    all_healthy = []
    all_diseased = []
    
    for subject_id in tqdm(subjects, desc="Processing BraTS2021"):
        subject_path = os.path.join(args.input_dir, subject_id)
        res = process_subject(subject_path, args.output_dir, subject_id)
        if res:
            h, d = res
            all_healthy.extend(h)
            all_diseased.extend(d)
            
    # Save lists
    with open(os.path.join(args.output_dir, "healthy_slices.txt"), "w") as f:
        f.write("\n".join(all_healthy))
    with open(os.path.join(args.output_dir, "diseased_slices.txt"), "w") as f:
        f.write("\n".join(all_diseased))
        
    print(f"\nTotal healthy slices: {len(all_healthy)}")
    print(f"Total diseased (unhealthy) slices: {len(all_diseased)}")

if __name__ == "__main__":
    main()
