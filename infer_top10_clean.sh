#!/bin/bash
# Infer top-10 dice>0.90 samples, save clean images to clean_samples/fm/<idx>/
set -e
cd "$(dirname "$0")"

INDICES="4547 4358 3853 2609 2606 5874 4356 4995 4357 4548"
# dice:  0.965 0.962 0.957 0.956 0.956 0.953 0.953 0.952 0.952 0.951

for IDX in $INDICES; do
    echo "============================================"
    echo "Inferring idx=$IDX ..."
    echo "============================================"
    SINGLE_IDX="$IDX" \
    CLEAN_BASE_DIR="/mnt/apple/k66/minhdd/clean_samples/fm" \
    SPLIT_FILE="/mnt/apple/k66/minhdd/data/brats2021/preprocessed_split.json" \
    CHECKPOINT_OVERRIDE="/mnt/apple/k66/minhdd/flow-matching-main/brats-perbatch-e11/checkpoint_epoch0011.pth" \
    NUM_UNHEALTHY=1 NUM_HEALTHY=0 \
    INFER_SPLIT="test" \
    OUTPUT_DIR="/tmp/infer_top10_tmp" \
    bash execute_anomaly_detection.sh --v2 --best
done

echo ""
echo "=== Done! Clean images saved to /mnt/apple/k66/minhdd/clean_samples/fm/ ==="
ls /mnt/apple/k66/minhdd/clean_samples/fm/
