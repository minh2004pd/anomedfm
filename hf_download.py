#!/usr/bin/env python3
"""
Download checkpoints from HuggingFace Hub.

Modes:
  - Single file (default):  --filename checkpoint.pth
  - Sync all checkpoints:   --sync_all  (downloads files missing locally;
                                         always overwrites checkpoint.pth)

Usage:
    uv run python hf_download.py --repo minh2k4/brats-flow-matching --sync_all
    uv run python hf_download.py --repo minh2k4/brats-flow-matching --filename checkpoint_epoch0020.pth
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def download_one(repo: str, filename: str, output_dir: Path, token: str | None, force: bool = False):
    target = output_dir / filename
    if target.exists() and not force:
        print(f"  skip (exists): {filename}")
        return
    print(f"  downloading: {filename}")
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(output_dir),
        token=token,
        force_download=force,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. username/repo-name")
    parser.add_argument("--output_dir", default="./output_brats", help="Where to save downloaded files")
    parser.add_argument("--filename", default="checkpoint.pth", help="Single filename to download (ignored with --sync_all)")
    parser.add_argument("--token", default=None, help="HF token for private repos (or set HF_TOKEN env var)")
    parser.add_argument("--sync_all", action="store_true", help="Download all .pth files missing locally; always overwrite checkpoint.pth")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sync_all:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=args.repo)
        ckpts = sorted([f for f in files if f.endswith(".pth")])
        print(f"Found {len(ckpts)} .pth files on {args.repo}")
        for f in ckpts:
            # always overwrite checkpoint.pth (it tracks the latest), skip others if present
            force = (f == "checkpoint.pth")
            download_one(args.repo, f, output_dir, token, force=force)
        # also grab args.json
        try:
            download_one(args.repo, "args.json", output_dir, token, force=True)
        except Exception as e:
            print(f"  args.json: skipped ({e})")
    else:
        for filename in [args.filename, "args.json"]:
            try:
                download_one(args.repo, filename, output_dir, token, force=True)
            except Exception as e:
                print(f"  {filename}: skipped ({e})")

    print(f"\nDone. Files at: {output_dir}")


if __name__ == "__main__":
    main()
