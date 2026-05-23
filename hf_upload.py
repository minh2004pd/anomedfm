#!/usr/bin/env python3
"""
Upload checkpoint to HuggingFace Hub after training.

Usage:
    uv run python hf_upload.py --repo vipghn2003/brats-flow-matching
    uv run python hf_upload.py --repo vipghn2003/brats-flow-matching --checkpoint ./output_brats/checkpoint.pth
    uv run python hf_upload.py --repo vipghn2003/brats-flow-matching --all  # upload all checkpoints
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, login


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. username/repo-name")
    parser.add_argument("--checkpoint", default="./output_brats/checkpoint.pth", help="Path to checkpoint file")
    parser.add_argument("--output_dir", default="./output_brats", help="Output dir to upload args.json from")
    parser.add_argument("--all", action="store_true", help="Upload all checkpoint-*.pth files too")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        login()  # interactive prompt

    api = HfApi()

    # Create repo if not exists
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)

    files_to_upload = []

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        files_to_upload.append(checkpoint)
    else:
        print(f"Warning: {checkpoint} not found, skipping.")

    if args.all:
        for p in sorted(Path(args.output_dir).glob("checkpoint-*.pth")):
            files_to_upload.append(p)

    args_json = Path(args.output_dir) / "args.json"
    if args_json.exists():
        files_to_upload.append(args_json)

    for f in files_to_upload:
        print(f"Uploading {f} ...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  -> done: https://huggingface.co/{args.repo}/blob/main/{f.name}")

    print(f"\nAll uploads complete. Repo: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
