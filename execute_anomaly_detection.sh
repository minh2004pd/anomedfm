#!/bin/bash

set -a && source .env 2>/dev/null; set +a

# --update     : enable CFG-Zero* optimized-scale guidance at inference.
#                Zero-init step skipping is OFF (ZERO_INIT_STEPS=0) — only the
#                scalar projection s* is applied. Plug-in only, no retraining.
# --hysteresis : use hysteresis (high+low percentile) threshold instead of Otsu.
#                Tuneable via env vars HYST_HIGH_PCT (default 99.5) and
#                HYST_LOW_PCT (default 95.0). Pixel kept iff its CC contains
#                at least one pixel above HIGH percentile.
# Mask post-processing (always on; tuneable via env vars):
#   MIN_COMPONENT_SIZE — drop blobs smaller than this many pixels.
# --best       : oracle selection — for each unhealthy sample picks the candidate
#                mask (per-modality or combined) with the highest DICE vs GT and
#                reports it as 'best'. Requires GT, not deployable.
# --encode_y1  : conditional reverse-encode with label=1 (unhealthy) for ALL
#                samples. Class-agnostic. Decode with label=0 + CFG subtracts
#                the unhealthy signal.
# --encode_cfg : CFG-style reverse encode with label=1 (variant A). Class-agnostic.
#                Encode uses v = (1+w)*v(y=1) - w*v_uncond, w from
#                ENCODE_CFG_SCALE env var (default 3.0). Pushes latent deeper
#                into the unhealthy manifold so the forward decode toward
#                healthy subtracts tumor more strongly. Implies --encode_y1.
USE_CFG_ZERO_STAR=0
# --best is on by default: per-sample, per-metric oracle selection picks the
# candidate (T1/T1ce/T2/FLAIR/combined) with the highest score for each of
# DICE / IoU / AUROC independently, then the run-level "best" row is the
# mean of those per-sample maxima. Pass --no-best to disable.
USE_BEST=1
USE_HYSTERESIS=0
USE_ENCODE_Y1=0
USE_ENCODE_CFG=0
# --v2 : use the 5-level UNet checkpoint (bratsv2 arch). Looks for the
#        checkpoint in ./output_brats_perbatch/ first (server layout),
#        then falls back to ./output_brats_v2/ (local layout). Override
#        with V2_CKPT_DIR=... env var if needed. Default is the legacy
#        4-level brats checkpoint in ./output_brats/. The two checkpoints
#        are NOT interchangeable — their state_dicts have different shapes.
USE_V2=0
ZERO_INIT_STEPS="${ZERO_INIT_STEPS:-0}"
MIN_COMPONENT_SIZE="${MIN_COMPONENT_SIZE:-100}"
HYST_HIGH_PCT="${HYST_HIGH_PCT:-60.0}"
HYST_LOW_PCT="${HYST_LOW_PCT:-30.0}"
ENCODE_CFG_SCALE="${ENCODE_CFG_SCALE:-3.0}"
# Border-noise suppression (always on; set BORDER_EROSION=0 to disable).
# BORDER_EROSION       — thickness (pixels) of the brain rim used to detect
#                        edge-hugging artefacts. Larger = more aggressive.
# BORDER_OVERLAP_THR   — a CC is dropped if more than this fraction of its
#                        pixels lie inside the rim. Tumours touching the
#                        cortex still extend inward, so they survive at 0.6.
#                        Increase (e.g. 0.8) if peripheral tumours get cut;
#                        decrease (e.g. 0.4) if rim noise still leaks through.
BORDER_EROSION="${BORDER_EROSION:-3}"
BORDER_OVERLAP_THR="${BORDER_OVERLAP_THR:-0.6}"
# Modalities used to build the 'combined' (union) mask. Per-modality masks
# (T1/T1ce/T2/FLAIR) are still computed and reported individually — this only
# controls which subset is unioned into the 'combined' row. T1 and T1ce are
# excluded by default because they typically yield weaker anomaly maps on
# this checkpoint.
COMBINED_MODALITIES="${COMBINED_MODALITIES:-T2,FLAIR}"
# Sample caps. Pass -1 to evaluate the entire val pool for that class.
# Default 100 unhealthy + 50 healthy keeps the standard sweep cost.
NUM_UNHEALTHY="${NUM_UNHEALTHY:-100}"
NUM_HEALTHY="${NUM_HEALTHY:-50}"
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

if [ -d "/workspace/brats2021" ]; then
    DATA_PATH="${DATA_PATH:-/workspace/brats2021}"
else
    DATA_PATH="${DATA_PATH:-$(cd "$(dirname "$0")/.." && pwd)/data/brats2021}"
fi

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

# Inference hyperparams. Defaults are the winners from the epoch-11 grid
# sweep (see grid_sweep_results_ckpt_e11.md): t=0.2, cfg=0.5, step=0.02.
T_VAL="${T_VAL:-0.2}"
STEP_SIZE="${STEP_SIZE:-0.02}"
CFG_SCALE="${CFG_SCALE:-0.5}"

HF_REPO="${HF_REPO:-}"
HF_TOKEN="${HF_TOKEN:-}"

# Checkpoint selection. EPOCH=N pins to ${CKPT_DIR}/checkpoint-N.pth and
# tags the output dir with _ckpt_eN; otherwise prefer checkpoint.pth, then
# fall back to the most recently modified checkpoint-*.pth.
CHECKPOINT="${CHECKPOINT_OVERRIDE:-}"
CKPT_TAG=""
if [ -n "$CHECKPOINT_OVERRIDE" ]; then
    if [ ! -f "$CHECKPOINT_OVERRIDE" ]; then
        echo "Error: CHECKPOINT_OVERRIDE=$CHECKPOINT_OVERRIDE not found"
        exit 1
    fi
elif [ -n "${EPOCH:-}" ]; then
    CHECKPOINT="${CKPT_DIR}/checkpoint-${EPOCH}.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: EPOCH=$EPOCH but $CHECKPOINT not found"
        exit 1
    fi
    CKPT_TAG="_ckpt_e${EPOCH}"
elif [ -f "${CKPT_DIR}/checkpoint.pth" ]; then
    CHECKPOINT="${CKPT_DIR}/checkpoint.pth"
else
    CHECKPOINT=$(ls -t ${CKPT_DIR}/checkpoint-*.pth 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found in ${CKPT_DIR}/"
    exit 1
fi

if [ "$USE_CFG_ZERO_STAR" -eq 1 ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-./anomaly_results_update_t_${T_VAL}_${STEP_SIZE}_cfg_scale_${CFG_SCALE}_new_ckpt_otsu_border_combined_T2_FLAIR${OUTPUT_SUFFIX}${CKPT_TAG}}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-./anomaly_results_t_${T_VAL}_${STEP_SIZE}_cfg_scale_${CFG_SCALE}_new_ckpt_otsu_border_combined_T2_FLAIR${OUTPUT_SUFFIX}${CKPT_TAG}}"
fi

echo "Using checkpoint: $CHECKPOINT (arch=$ARCH)"
mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [ "$USE_CFG_ZERO_STAR" -eq 1 ]; then
    EXTRA_ARGS+=(--cfg_zero_star --zero_init_steps "$ZERO_INIT_STEPS")
    echo "CFG-Zero* enabled (zero_init_steps=$ZERO_INIT_STEPS)"
fi
if [ "$USE_BEST" -eq 1 ]; then
    EXTRA_ARGS+=(--best)
    echo "Best-of oracle selection enabled (per-sample picks max-DICE candidate)"
fi
if [ "$USE_HYSTERESIS" -eq 1 ]; then
    EXTRA_ARGS+=(--hysteresis --hyst_high_pct "$HYST_HIGH_PCT" --hyst_low_pct "$HYST_LOW_PCT")
    echo "Hysteresis threshold enabled (high=${HYST_HIGH_PCT}%  low=${HYST_LOW_PCT}%)"
fi
if [ "$USE_ENCODE_Y1" -eq 1 ]; then
    EXTRA_ARGS+=(--encode_label 0)
    echo "Encode-with-label=0 enabled (all samples encoded as healthy)"
fi
if [ "$USE_ENCODE_CFG" -eq 1 ]; then
    EXTRA_ARGS+=(--encode_cfg_scale "$ENCODE_CFG_SCALE")
    echo "Reverse CFG encoding enabled (encode_cfg_scale=$ENCODE_CFG_SCALE, label=0)"
fi

SINGLE_IDX="${SINGLE_IDX:-}"
SPLIT_FILE="${SPLIT_FILE:-}"
NO_IMAGES="${NO_IMAGES:-0}"
[ "$NO_IMAGES" -eq 1 ] && EXTRA_ARGS+=(--no_images)

uv run python infer_anomaly.py \
  --checkpoint          "$CHECKPOINT" \
  --arch                "$ARCH" \
  --data_path           "$DATA_PATH" \
  ${SPLIT_FILE:+--split_file "$SPLIT_FILE"} \
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
  ${SINGLE_IDX:+--target_idx "$SINGLE_IDX" --save_clean_idx "$SINGLE_IDX" --clean_base_dir "${CLEAN_BASE_DIR:-/mnt/apple/k66/minhdd/clean_samples/fm}"} \
  ${INFER_SPLIT:+--split "$INFER_SPLIT"} \
  "${EXTRA_ARGS[@]}"

echo "Anomaly Detection completed!"

# # Upload results to HuggingFace so k66 can download
# if [ -n "$HF_REPO" ] && [ -n "$HF_TOKEN" ]; then
#     echo "Uploading anomaly results to HuggingFace: $HF_REPO ..."
#     uv run python -c "
# from huggingface_hub import HfApi, login
# login(token='${HF_TOKEN}')
# api = HfApi()
# api.create_repo('${HF_REPO}', repo_type='dataset', exist_ok=True, private=True)
# api.upload_folder(
#     folder_path='${OUTPUT_DIR}',
#     path_in_repo='anomaly_results',
#     repo_id='${HF_REPO}',
#     repo_type='dataset',
# )
# print('Done: https://huggingface.co/datasets/${HF_REPO}')
# "
# else
#     echo "Tip: set HF_REPO and HF_TOKEN in .env to auto-upload results."
# fi
