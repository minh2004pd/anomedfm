#!/bin/bash

# Load .env so HF_REPO / HF_TOKEN are picked up regardless of which shell
# starts the script (avoids surprises like an old exported HF_REPO in the
# parent shell silently overriding the .env).
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# --v2 : train the 5-level UNet (bratsv2 arch) instead of the legacy 4-level
#        brats arch. Output dir defaults to ./output_brats_v2/ so the two
#        runs don't collide. Pass --v2 anywhere on the CLI.
USE_V2=0
for arg in "$@"; do
    case "$arg" in
        --v2) USE_V2=1 ;;
    esac
done

# On Vast.ai server data is at /workspace/brats2021; locally at <repo>/../data/brats2021
if [ -d "/workspace/brats2021" ]; then
    DATA_PATH="${DATA_PATH:-/workspace/brats2021}"
else
    DATA_PATH="${DATA_PATH:-$(cd "$(dirname "$0")/.." && pwd)/data/brats2021}"
fi

if [ "$USE_V2" -eq 1 ]; then
    DATASET="bratsv2"
    # On the server the v2 run lives in output_brats_perbatch/; prefer it
    # when present so training auto-resumes from the existing checkpoint.
    if [ -z "${OUTPUT_DIR:-}" ]; then
        if [ -d "./output_brats_perbatch" ]; then
            OUTPUT_DIR="./output_brats_perbatch"
        else
            OUTPUT_DIR="./output_brats_v2"
        fi
    fi
    LOG_NAME="train_brats_v2.log"
else
    DATASET="brats"
    OUTPUT_DIR="${OUTPUT_DIR:-./output_brats_perbatch}"
    LOG_NAME="train_brats_perbatch.log"
fi
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Auto-resume if checkpoint exists
RESUME_ARG=""
if [ -f "$OUTPUT_DIR/checkpoint.pth" ]; then
    RESUME_ARG="--resume=$OUTPUT_DIR/checkpoint.pth"
    echo "Resuming from $OUTPUT_DIR/checkpoint.pth"
fi

# Train BraTS (healthy + unhealthy, 80-20 split built into BraTSPreprocessedDataset)
# use_preprocessed loads from .npy slices with auto 80-20 case-level split (seed=42)
# Gradient checkpointing is enabled via model_configs.py ("brats" config has use_checkpoint=True)
set -o pipefail
echo "Training arch=$DATASET into $OUTPUT_DIR"
uv run python train.py \
  --dataset="$DATASET" \
  --data_path="$DATA_PATH" \
  --use_preprocessed \
  --image_size=256 \
  --batch_size=4 \
  --accum_iter=8 \
  --epochs=50 \
  --lr=1e-4 \
  --lr_scheduler=cosine \
  --min_lr=1e-6 \
  --precision=bf16 \
  --class_drop_prob=0.15 \
  --cfg_scale=3.0 \
  --use_ema \
  --eval_frequency=10 \
  --num_workers=6 \
  --output_dir="$OUTPUT_DIR" \
  $RESUME_ARG \
  # --wandb \
  # --wandb_project="flow-matching-brats" \
  2>&1 | tee "$LOG_DIR/$LOG_NAME"

echo "Training completed!"

# Upload checkpoint to HuggingFace if HF_REPO is set
# Usage: HF_REPO=minh2k4/brats-flow-matching HF_TOKEN=hf_xxx bash train_brats.sh
if [ -n "$HF_REPO" ]; then
    echo "Uploading checkpoint to HuggingFace: $HF_REPO ..."
    uv run python hf_upload.py \
        --repo "$HF_REPO" \
        --checkpoint "$OUTPUT_DIR/checkpoint.pth" \
        --output_dir "$OUTPUT_DIR" \
        --all \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
else
    echo "Tip: set HF_REPO=username/repo-name to auto-upload checkpoint after training."
fi