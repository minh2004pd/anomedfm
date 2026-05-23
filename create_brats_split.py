#!/usr/bin/env python3
"""
Generate case-level train/val splits for BraTS2021 and save to two JSON files:

  preprocessed_split.json  — slice-level entries for BraTSPreprocessedDataset (training)
  inference_split.json     — case directory paths for BraTSDataset (inference / evaluation)

A patient case is assigned entirely to train OR val across both files (no leakage).
The train slice list is pre-shuffled with seed=42; DataLoader re-shuffles per epoch.

Usage:
    python create_brats_split.py \
        --data_path /mnt/apple/k66/minhdd/data/brats2021 \
        --train_ratio 0.8 --seed 42
"""

import argparse
import json
import os
import random


def read_txt(path: str) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def case_id(rel_path: str) -> str:
    return rel_path.replace("\\", "/").split("/")[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    default="/mnt/apple/k66/minhdd/data/brats2021")
    p.add_argument("--extracted_dir", default=None,
                   help="Folder with NIfTI case dirs for inference split. "
                        "Defaults to <data_path>/extracted_data")
    p.add_argument("--train_ratio",  type=float, default=0.8)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    preprocessed_out = os.path.join(args.data_path, "preprocessed_split.json")
    inference_out    = os.path.join(args.data_path, "inference_split.json")
    extracted_dir    = args.extracted_dir or os.path.join(args.data_path, "extracted_data")

    healthy_txt  = os.path.join(args.data_path, "healthy_slices.txt")
    diseased_txt = os.path.join(args.data_path, "diseased_slices.txt")

    healthy_paths  = read_txt(healthy_txt)
    diseased_paths = read_txt(diseased_txt)

    # ── Global case-level split ───────────────────────────────────────────────
    all_cases = sorted(set(case_id(p) for p in healthy_paths + diseased_paths))
    rng = random.Random(args.seed)
    rng.shuffle(all_cases)
    n_train = int(len(all_cases) * args.train_ratio)
    train_cases = set(all_cases[:n_train])
    val_cases   = set(all_cases[n_train:])

    assert not (train_cases & val_cases), "BUG: case overlap between splits"

    # ── Assign slices ─────────────────────────────────────────────────────────
    def assign(paths: list[str], label: int, allowed: set) -> list[dict]:
        return [{"path": p, "label": label}
                for p in paths if case_id(p) in allowed]

    train_slices = assign(healthy_paths,  0, train_cases) + \
                   assign(diseased_paths, 1, train_cases)
    val_slices   = assign(healthy_paths,  0, val_cases)   + \
                   assign(diseased_paths, 1, val_cases)

    # Shuffle train with fixed seed so order is reproducible.
    # DataLoader will re-shuffle per epoch on top of this.
    random.Random(args.seed).shuffle(train_slices)

    # ── Summary ───────────────────────────────────────────────────────────────
    def summarise(slices, cases):
        h = sum(1 for s in slices if s["label"] == 0)
        d = sum(1 for s in slices if s["label"] == 1)
        return {"n_cases": len(cases), "n_healthy_slices": h,
                "n_diseased_slices": d, "n_slices": len(slices)}

    train_info = summarise(train_slices, train_cases)
    val_info   = summarise(val_slices,   val_cases)

    print(f"Total cases : {len(all_cases)}")
    print(f"Train → {train_info['n_cases']} cases | "
          f"{train_info['n_healthy_slices']} healthy + "
          f"{train_info['n_diseased_slices']} diseased = "
          f"{train_info['n_slices']} slices")
    print(f"Val   → {val_info['n_cases']} cases | "
          f"{val_info['n_healthy_slices']} healthy + "
          f"{val_info['n_diseased_slices']} diseased = "
          f"{val_info['n_slices']} slices")

    # ── Save preprocessed_split.json (for BraTSPreprocessedDataset / training) ─
    with open(preprocessed_out, "w") as f:
        json.dump({
            "seed":        args.seed,
            "train_ratio": args.train_ratio,
            "train":       train_slices,
            "val":         val_slices,
            "meta":        {"train": train_info, "val": val_info},
        }, f, indent=2)
    print(f"Preprocessed split → {preprocessed_out}")

    # ── Save inference_split.json (for BraTSDataset / inference with masks) ──
    # BraTSDataset expects {"train": [dir_path, ...], "val": [dir_path, ...]}
    available_cases = set(os.listdir(extracted_dir)) if os.path.isdir(extracted_dir) else set()

    def case_dirs_for(case_set):
        dirs = [os.path.join(extracted_dir, c) for c in sorted(case_set) if c in available_cases]
        missing = case_set - available_cases
        if missing:
            print(f"  WARNING: {len(missing)} cases not found in {extracted_dir}")
        return dirs

    inf_train_dirs = case_dirs_for(train_cases)
    inf_val_dirs   = case_dirs_for(val_cases)

    with open(inference_out, "w") as f:
        json.dump({
            "train": inf_train_dirs,
            "val":   inf_val_dirs,
        }, f, indent=2)
    print(f"Inference split    → {inference_out}")
    print(f"  train: {len(inf_train_dirs)} case dirs  |  val: {len(inf_val_dirs)} case dirs")


if __name__ == "__main__":
    main()
