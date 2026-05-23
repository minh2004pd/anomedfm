#!/bin/bash
# Evaluate anomaly detection on the held-out TEST split.
# Mirrors execute_anomaly_detection.sh but targets split["test"] and writes
# results under test_results_* so val and test outputs stay separate.
#
# Usage:
#   ./evaluate_test.sh [options]
#
# Options (same as execute_anomaly_detection.sh):
#   --update       Enable CFG-Zero* optimized-scale guidance
#   --best         Oracle per-sample best-candidate selection (default: on)
#   --no-best      Disable oracle selection
#   --hysteresis   Use hysteresis threshold instead of Otsu
#   --encode_y1    Conditional reverse-encode with label=0 (healthy)
#   --encode_cfg   CFG-style reverse encode (implies --encode_y1)
#   --v2           Use bratsv2 (5-level UNet) checkpoint
#
# Env overrides (all optional):
#   DATA_PATH            Path to brats2021 root
#   CHECKPOINT_OVERRIDE  Explicit .pth path
#   EPOCH                Pin to checkpoint-N.pth
#   V2_CKPT_DIR          Override v2 checkpoint directory
#   T_VAL                Encoding depth  (default: 0.2)
#   STEP_SIZE            ODE step size   (default: 0.02)
#   CFG_SCALE            CFG scale       (default: 0.5)
#   NUM_UNHEALTHY        Cap on unhealthy samples; -1 = all (default: -1)
#   NUM_HEALTHY          Cap on healthy samples;   -1 = all (default: -1)
#   MIN_COMPONENT_SIZE   Blob filter     (default: 100)
#   BORDER_EROSION       Rim noise erosion pixels (default: 3)
#   BORDER_OVERLAP_THR   Rim noise overlap threshold (default: 0.6)
#   COMBINED_MODALITIES  Modalities for union mask (default: T2,FLAIR)
#   HYST_HIGH_PCT        Hysteresis high percentile (default: 60.0)
#   HYST_LOW_PCT         Hysteresis low percentile  (default: 30.0)
#   ENCODE_CFG_SCALE     Reverse CFG weight (default: 3.0)
#   OUTPUT_DIR           Override output directory name

set -euo pipefail
set -a && source .env 2>/dev/null; set +a

# ── Flags ──────────────────────────────────────────────────────────────────────
USE_CFG_ZERO_STAR=0
USE_BEST=1
USE_HYSTERESIS=0
USE_ENCODE_Y1=0
USE_ENCODE_CFG=0
USE_V2=0

for arg in "$@"; do
    case "$arg" in
        --update)     USE_CFG_ZERO_STAR=1 ;;
        --best)       USE_BEST=1 ;;
        --no-best)    USE_BEST=0 ;;
        --hysteresis) USE_HYSTERESIS=1 ;;
        --encode_y1)  USE_ENCODE_Y1=1 ;;
        --encode_cfg) USE_ENCODE_CFG=1; USE_ENCODE_Y1=1 ;;
        --v2)         USE_V2=1 ;;
    esac
done

# ── Env defaults ───────────────────────────────────────────────────────────────
ZERO_INIT_STEPS="${ZERO_INIT_STEPS:-0}"
MIN_COMPONENT_SIZE="${MIN_COMPONENT_SIZE:-100}"
HYST_HIGH_PCT="${HYST_HIGH_PCT:-60.0}"
HYST_LOW_PCT="${HYST_LOW_PCT:-30.0}"
ENCODE_CFG_SCALE="${ENCODE_CFG_SCALE:-3.0}"
BORDER_EROSION="${BORDER_EROSION:-3}"
BORDER_OVERLAP_THR="${BORDER_OVERLAP_THR:-0.6}"
COMBINED_MODALITIES="${COMBINED_MODALITIES:-T2,FLAIR}"
# -1 = evaluate the full test split (all samples)
NUM_UNHEALTHY="${NUM_UNHEALTHY:--1}"
NUM_HEALTHY="${NUM_HEALTHY:--1}"
NO_IMAGES="${NO_IMAGES:-0}"

# ── Data path ──────────────────────────────────────────────────────────────────
if [ -d "/workspace/brats2021" ]; then
    DATA_PATH="${DATA_PATH:-/workspace/brats2021}"
else
    DATA_PATH="${DATA_PATH:-$(cd "$(dirname "$0")/.." && pwd)/data/brats2021}"
fi

# ── Checkpoint selection ───────────────────────────────────────────────────────
if [ "$USE_V2" -eq 1 ]; then
    if [ -n "${V2_CKPT_DIR:-}" ]; then
        CKPT_DIR="$V2_CKPT_DIR"
    elif [ -d "./output_brats_perbatch" ]; then
        CKPT_DIR="./output_brats_perbatch"
    else
        CKPT_DIR="./output_brats_v2"
    fi
    ARCH="bratsv2"
    OUTPUT_SUFFIX="_v2"
else
    CKPT_DIR="./output_brats"
    ARCH="brats"
    OUTPUT_SUFFIX=""
fi

CHECKPOINT="${CHECKPOINT_OVERRIDE:-}"
CKPT_TAG=""
if [ -n "${CHECKPOINT_OVERRIDE:-}" ]; then
    if [ ! -f "$CHECKPOINT_OVERRIDE" ]; then
        echo "Error: CHECKPOINT_OVERRIDE=$CHECKPOINT_OVERRIDE not found" >&2
        exit 1
    fi
elif [ -n "${EPOCH:-}" ]; then
    CHECKPOINT="${CKPT_DIR}/checkpoint-${EPOCH}.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: EPOCH=$EPOCH but $CHECKPOINT not found" >&2
        exit 1
    fi
    CKPT_TAG="_ckpt_e${EPOCH}"
elif [ -f "${CKPT_DIR}/checkpoint.pth" ]; then
    CHECKPOINT="${CKPT_DIR}/checkpoint.pth"
else
    CHECKPOINT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pth 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found in ${CKPT_DIR}/" >&2
    exit 1
fi

# ── Inference hyperparams ─────────────────────────────────────────────────────
T_VAL="${T_VAL:-0.2}"
STEP_SIZE="${STEP_SIZE:-0.02}"
CFG_SCALE="${CFG_SCALE:-0.5}"

# ── Output directory ───────────────────────────────────────────────────────────
if [ "${USE_CFG_ZERO_STAR}" -eq 1 ]; then
    _prefix="test_results_update"
else
    _prefix="test_results"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${_prefix}_t_${T_VAL}_${STEP_SIZE}_cfg_${CFG_SCALE}${OUTPUT_SUFFIX}${CKPT_TAG}}"

echo "============================================================"
echo "  BraTS2021 TEST-SET evaluation"
echo "  Checkpoint : $CHECKPOINT  (arch=$ARCH)"
echo "  Data       : $DATA_PATH"
echo "  Output     : $OUTPUT_DIR"
echo "  t=$T_VAL  step=$STEP_SIZE  cfg=$CFG_SCALE"
echo "  Samples    : unhealthy=${NUM_UNHEALTHY}  healthy=${NUM_HEALTHY}"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# ── Extra args ─────────────────────────────────────────────────────────────────
EXTRA_ARGS=()
[ "$NO_IMAGES" -eq 1 ] && EXTRA_ARGS+=(--no_images)
if [ "$USE_CFG_ZERO_STAR" -eq 1 ]; then
    EXTRA_ARGS+=(--cfg_zero_star --zero_init_steps "$ZERO_INIT_STEPS")
    echo "CFG-Zero* enabled (zero_init_steps=$ZERO_INIT_STEPS)"
fi
if [ "$USE_BEST" -eq 1 ]; then
    EXTRA_ARGS+=(--best)
    echo "Best-of oracle selection enabled"
fi
if [ "$USE_HYSTERESIS" -eq 1 ]; then
    EXTRA_ARGS+=(--hysteresis --hyst_high_pct "$HYST_HIGH_PCT" --hyst_low_pct "$HYST_LOW_PCT")
    echo "Hysteresis threshold enabled (high=${HYST_HIGH_PCT}%  low=${HYST_LOW_PCT}%)"
fi
if [ "$USE_ENCODE_Y1" -eq 1 ]; then
    EXTRA_ARGS+=(--encode_label 0)
    echo "Encode-with-label=0 enabled"
fi
if [ "$USE_ENCODE_CFG" -eq 1 ]; then
    EXTRA_ARGS+=(--encode_cfg_scale "$ENCODE_CFG_SCALE")
    echo "Reverse CFG encoding enabled (encode_cfg_scale=$ENCODE_CFG_SCALE)"
fi

# ── Run ────────────────────────────────────────────────────────────────────────
uv run python infer_anomaly.py \
  --checkpoint          "$CHECKPOINT" \
  --arch                "$ARCH" \
  --data_path           "$DATA_PATH" \
  --split               test \
  --t                   "$T_VAL" \
  --step_size           "$STEP_SIZE" \
  --cfg_scale           "$CFG_SCALE" \
  --num_unhealthy       "$NUM_UNHEALTHY" \
  --num_healthy         "$NUM_HEALTHY" \
  --output_dir          "$OUTPUT_DIR" \
  --device              cuda \
  --min_component_size  "$MIN_COMPONENT_SIZE" \
  --border_erosion      "$BORDER_EROSION" \
  --border_overlap_thr  "$BORDER_OVERLAP_THR" \
  --combined_modalities "$COMBINED_MODALITIES" \
  "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "  Test evaluation complete → $OUTPUT_DIR"
echo "  metrics.csv   : ${OUTPUT_DIR}/metrics.csv"
echo "  summary JSON  : ${OUTPUT_DIR}/summary.json  (if generated)"
echo "============================================================"
