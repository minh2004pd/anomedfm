
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from torchvision import transforms

import json

class BraTSDataset(Dataset):
    def __init__(self, root, image_size=64, transform=None, mode='train', split_file='./data/brats_split.json', return_mask=False, healthy_only=False):
        """
        BraTS2021 Dataset for 2D Flow Matching.
        
        Args:
            root (str): Path to the BraTS2021 dataset root directory.
            image_size (int): Output image size (will resize slices to this).
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train' or 'valid'.
            split_file (str): Path to the JSON file containing train/val splits.
            return_mask (bool): Whether to return the segmentation mask.
            healthy_only (bool): If True, inpaint tumor regions to create "healthy" 
                                 versions of all slices. Label is always 0.
        """
        self.root = root
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.return_mask = return_mask
        self.healthy_only = healthy_only
        
        # Load split config
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found at {split_file}. Run create_split_config.py first.")
            
        with open(split_file, 'r') as f:
            split_data = json.load(f)
            
        # Map 'valid' mode to 'val' key in JSON if needed, or stick to convention
        # JSON has 'train' and 'val'.
        json_mode = 'val' if mode == 'valid' else mode
        
        if json_mode not in split_data:
             raise ValueError(f"Mode {json_mode} not found in split file {split_file}. Keys: {list(split_data.keys())}")
             
        self.case_dirs = split_data[json_mode]
        
        if not self.case_dirs:
            raise ValueError(f"No cases found for mode {mode} in {split_file}")
            
        mode_str = f"{mode} (healthy_only={healthy_only})" if healthy_only else mode
        print(f"Loaded {len(self.case_dirs)} BraTS cases for {mode_str} mode from {split_file}.")

        # Pre-define modalities
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        
        # Slice configurations from user requirements
        # "exclude the lowest 80 slices and the uppermost 26 slices"
        # Total slices usually 155.
        # Range: [80, 155 - 26] = [80, 129] (exclusive of 129)
        self.start_slice = 80
        self.end_slice = 155 - 26 # 129
        
    def __len__(self):
        # We define length as number of cases, but in __getitem__ we pick a random slice.
        # This keeps the epoch size related to number of patients.
        return len(self.case_dirs)

    def _load_mask(self, case_dir):
        # Mask is usually named *seg.nii.gz
        pattern = os.path.join(case_dir, f"*seg.nii.gz")
        files = glob.glob(pattern)
        if not files:
             return None
        img = nib.load(files[0])
        return img.get_fdata()

    def _inpaint_tumor(self, image, mask_slice):
        """
        Inpaint tumor regions by replacing tumor pixels with the median intensity
        of surrounding non-tumor brain tissue (per channel).
        
        Args:
            image: (H, W, C) numpy array, raw intensity values
            mask_slice: (H, W) numpy array, BraTS segmentation (0=bg, 1/2/4=tumor)
        
        Returns:
            inpainted_image: (H, W, C) numpy array with tumor regions filled
        """
        tumor_mask = mask_slice > 0  # Binary: True where tumor exists
        
        if not np.any(tumor_mask):
            return image  # No tumor, return as-is
        
        inpainted = image.copy()
        
        for ch in range(image.shape[-1]):
            ch_data = image[..., ch]
            # Non-tumor AND non-background pixels = healthy brain tissue
            # Background is typically 0 intensity
            brain_mask = (~tumor_mask) & (ch_data > 1e-6)
            
            if np.any(brain_mask):
                # Use median of healthy brain tissue for smooth fill
                fill_value = np.median(ch_data[brain_mask])
            else:
                # Fallback: use overall median (rare edge case)
                fill_value = np.median(ch_data[ch_data > 1e-6]) if np.any(ch_data > 1e-6) else 0.0
            
            inpainted[..., ch][tumor_mask] = fill_value
        
        return inpainted

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        
        try:
            # 1. Pick a random slice within valid range
            slice_idx = np.random.randint(self.start_slice, self.end_slice)
            
            # 2. Load the 4 modalities for this slice
            slices = []
            for mod in self.modalities:
                case_name = os.path.basename(case_dir)
                pattern = os.path.join(case_dir, f"*{mod}.nii.gz")
                files = glob.glob(pattern)
                if not files:
                    raise FileNotFoundError(f"Missing {mod} in {case_dir}")
                    
                img_obj = nib.load(files[0])
                # Assuming shape (240, 240, 155) -> (H, W, D)
                slice_data = np.asarray(img_obj.dataobj[..., slice_idx]).astype(np.float32)
                slices.append(slice_data)
                
            # Stack channels: (240, 240, 4)
            image = np.stack(slices, axis=-1)
            
            # 3. Load Mask for label
            mask_pattern = os.path.join(case_dir, f"*seg.nii.gz")
            mask_files = glob.glob(mask_pattern)
            label = 0 # Healthy by default
            mask_slice = None
            if mask_files:
                mask_obj = nib.load(mask_files[0])
                mask_slice = np.asarray(mask_obj.dataobj[..., slice_idx])
                # Check if tumor present in this slice
                # BraTS labels: 1, 2, 4 are tumor classes. 0 is background.
                if np.any(mask_slice > 0):
                    label = 1 # Diseased
            
            # 4. Healthy-only mode: inpaint tumor regions
            if self.healthy_only and label == 1 and mask_slice is not None:
                image = self._inpaint_tumor(image, mask_slice)
                # After inpainting, the image is "healthy" — label becomes 0
                label = 0
            
            # 5. Preprocessing from user specs
            # "padded to a size of 256 x 256" (From 240x240)
            # "normalized to values between 0 and 1"
            
            # Pad
            h, w, c = image.shape
            pad_h = (256 - h) // 2
            pad_w = (256 - w) // 2
            
            # Simple padding with zeros
            padded_image = np.zeros((256, 256, 4), dtype=np.float32)
            # Center crop placement
            padded_image[pad_h:pad_h+h, pad_w:pad_w+w, :] = image
            
            # Normalize 0-1
            for ch in range(4):
                ch_data = padded_image[..., ch]
                mn = ch_data.min()
                mx = ch_data.max()
                if mx - mn > 1e-8:
                    padded_image[..., ch] = (ch_data - mn) / (mx - mn)
                else:
                    padded_image[..., ch] = 0.0 # flat constant, usually background
                    
            # Resize to target image_size for training (e.g. 64)
            # Torchvision transforms expect (C, H, W)
            image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1) # (4, 256, 256)
            
            if self.image_size != 256:
                resize = transforms.Resize((self.image_size, self.image_size), antialias=True)
                image_tensor = resize(image_tensor)
                
            if self.return_mask:
                # Prepare mask tensor (always return original mask for evaluation)
                if mask_slice is not None:
                    mask_binary = (mask_slice > 0).astype(np.float32)
                else:
                    mask_binary = np.zeros((240, 240), dtype=np.float32)
                padded_mask = np.zeros((256, 256), dtype=np.float32)
                padded_mask[pad_h:pad_h+h, pad_w:pad_w+w] = mask_binary
                mask_tensor = torch.from_numpy(padded_mask).unsqueeze(0) # (1, 256, 256)
                if self.image_size != 256:
                    resize_mask = transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)
                    mask_tensor = resize_mask(mask_tensor)
                return image_tensor, torch.tensor(label, dtype=torch.float32), mask_tensor

            return image_tensor, torch.tensor(label, dtype=torch.float32)
            
        except (EOFError, OSError, Exception) as e:
            print(f"Error loading case {case_dir}: {e}. Skipping...")
            # Pick a new random index
            new_idx = np.random.randint(0, len(self.case_dirs))
            return self.__getitem__(new_idx)


class BraTSPreprocessedDataset(Dataset):
    """
    Dataset for training/evaluation on preprocessed BraTS2021 .npy slices.

    Requires a pre-generated split JSON produced by ``create_brats_split.py``.
    The JSON stores a case-level 80-20 split so that every slice of a patient
    belongs exclusively to train or val.  The train list is pre-shuffled with
    seed=42; the DataLoader re-shuffles per epoch on top.

    Args:
        root (str): Root directory containing healthy/ and unhealthy/ folders.
        mode (str): 'train' or 'val'.
        healthy_only (bool): If True, only label-0 (healthy) slices are loaded.
        image_size (int): Resize slices if != 256 (the stored size).
        split_file (str): Path to the JSON generated by create_brats_split.py.
                          Defaults to <root>/preprocessed_split.json.
        transform: Optional torchvision transform applied after loading.
        return_label (bool): Return (image, label) instead of just image.
    """

    def __init__(
        self,
        root: str,
        mode: str = 'train',
        healthy_only: bool = False,
        image_size: int = 256,
        split_file: str = None,
        transform=None,
        return_label: bool = False,
    ):
        assert mode in ('train', 'val', 'test'), f"mode must be 'train', 'val', or 'test', got '{mode}'"
        self.root         = root
        self.mode         = mode
        self.healthy_only = healthy_only
        self.image_size   = image_size
        self.return_label = return_label

        if transform is None and image_size != 256:
            from torchvision import transforms
            self.transform = transforms.Resize((image_size, image_size))
        else:
            self.transform = transform

        split_file = split_file or os.path.join(root, "preprocessed_split.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Run:  python create_brats_split.py --data_path {root}"
            )

        with open(split_file) as f:
            split = json.load(f)

        entries = split[mode]           # list of {"path": ..., "label": ...}

        if healthy_only:
            entries = [e for e in entries if e["label"] == 0]

        self.slice_paths = [e["path"]  for e in entries]
        self.labels      = [e["label"] for e in entries]

        meta = split.get("meta", {}).get(mode, {})
        print(
            f"BraTSPreprocessedDataset [{mode}|healthy_only={healthy_only}]: "
            f"{meta.get('n_cases', '?')} cases — "
            f"{len(self.slice_paths)} slices loaded  "
            f"(split seed={split.get('seed', '?')})"
        )

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        rel_path = self.slice_paths[idx]
        abs_path = os.path.join(self.root, rel_path)
        image = np.load(abs_path).astype(np.float32)
        image_tensor = torch.from_numpy(image)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.return_label:
            return image_tensor, torch.tensor(self.labels[idx], dtype=torch.float32)
        return image_tensor
