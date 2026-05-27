#!/bin/bash
# Run once on a new server to set up the environment.
# Usage: DATA_PATH=./data/brats2021 bash setup.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# On Vast.ai server data is at /workspace/brats2021; locally at <repo>/../data/brats2021
if [ -d "/workspace/brats2021" ]; then
    DATA_PATH="${DATA_PATH:-/workspace/brats2021}"
else
    DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/../data/brats2021}"
fi
INPUT_DIR="${INPUT_DIR:-$DATA_PATH/extracted_data}"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$(dirname "$0")"
uv sync

# Step 1: Preprocess raw .nii.gz -> .npy slices (skip if already done)
if [ -d "$DATA_PATH/healthy" ] && [ -d "$DATA_PATH/unhealthy" ]; then
    echo "Preprocessed slices found at $DATA_PATH, skipping process_brats.py."
elif [ -d "$INPUT_DIR" ]; then
    echo "Preprocessing BraTS .nii.gz -> .npy slices..."
    uv run python process_brats.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$DATA_PATH"
else
    echo "Warning: No raw data at $INPUT_DIR and no preprocessed data at $DATA_PATH/healthy."
    echo "Skipping preprocessing. Make sure DATA_PATH points to preprocessed data."
fi

# Step 2: Generate train/val split JSON
if [ ! -f "$DATA_PATH/preprocessed_split.json" ]; then
    echo "Generating BraTS split files..."
    uv run python create_brats_split.py --data_path "$DATA_PATH"
else
    echo "Split files already exist, skipping."
fi

echo "Setup complete. Run: DATA_PATH=$DATA_PATH bash train_brats.sh"
