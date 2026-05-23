#!/bin/bash
# run_grid.sh — coordinate-descent sweep over (cfg, t, step) for anomaly
# detection on BraTS. Each phase fixes the trajectory found in the previous
# phase. All runs share --seed 42 so the same 100 unhealthy + 50 healthy
# slices are evaluated across configs (enables paired comparison).
#
# Locked-in choices (no sweep):
#   threshold  = otsu
#   border     = on  (BORDER_EROSION=3, BORDER_OVERLAP_THR=0.6)
#   combined   = T2,FLAIR
#   checkpoint = ./output_brats/checkpoint.pth (or ./output_brats_v2/ with --v2)
#   N          = 100 unhealthy + 50 healthy
#
# Pass --v2 anywhere on the CLI to switch to the 5-level UNet checkpoint
# (bratsv2 arch) in ./output_brats_v2/. Output dirs are tagged with _v2 so
# v1 and v2 sweeps don't collide on disk.
#
# Usage:
#   bash run_grid.sh 1                # phase 1: cfg sweep at t=0.2, step=0.02
#   bash run_grid.sh 2 <best_cfg>     # phase 2: t sweep at given cfg, step=0.02
#   bash run_grid.sh 3 <best_t> <best_cfg>   # phase 3: step sweep at given (t, cfg)
#   bash run_grid.sh 1 --v2           # same as above but using bratsv2 ckpt
#
# After each phase: read summary.txt of all p<N>_* runs, pick the winner,
# pass it to the next phase. Skip-if-exists logic prevents accidental
# re-runs.

set -euo pipefail

ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Strip --v2 from positional args so PHASE/cfg/t ordering still works
USE_V2=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --v2) USE_V2=1 ;;
        *)    POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]:-}"

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
    V2_TAG="_v2"
else
    CKPT_DIR="./output_brats"
    ARCH="brats"
    V2_TAG=""
fi

CHECKPOINT=""
# CHECKPOINT_OVERRIDE lets you point directly at any .pth file regardless of
# naming convention (e.g. checkpoint_epoch0011.pth). Takes precedence over
# all other checkpoint-discovery logic. RUN_TAG is auto-derived from the
# filename stem if not already set.
if [ -n "${CHECKPOINT_OVERRIDE:-}" ]; then
    CHECKPOINT="$CHECKPOINT_OVERRIDE"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: CHECKPOINT_OVERRIDE=$CHECKPOINT not found" >&2
        exit 1
    fi
    _stem=$(basename "$CHECKPOINT" .pth)
    RUN_TAG="${RUN_TAG:-${_stem}}"
# EPOCH=N pins to ${CKPT_DIR}/checkpoint-N.pth and auto-suffixes output
# dirs with _ckpt_eN so different epochs don't clobber each other.
elif [ -n "${EPOCH:-}" ]; then
    CHECKPOINT="${CKPT_DIR}/checkpoint-${EPOCH}.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: EPOCH=$EPOCH but $CHECKPOINT not found" >&2
        exit 1
    fi
    RUN_TAG="${RUN_TAG:-ckpt_e${EPOCH}}"
elif [ -f "${CKPT_DIR}/checkpoint.pth" ]; then
    CHECKPOINT="${CKPT_DIR}/checkpoint.pth"
else
    CHECKPOINT=$(ls -t ${CKPT_DIR}/checkpoint-*.pth 2>/dev/null | head -1)
fi
if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found in ${CKPT_DIR}/" >&2
    exit 1
fi

# Sample caps. Pass -1 to evaluate the entire val pool for that class.
# Defaults match the standard sweep size (paired comparison via fixed seed=42).
NUM_UNHEALTHY="${NUM_UNHEALTHY:-100}"
NUM_HEALTHY="${NUM_HEALTHY:-50}"

# HuggingFace upload of grid results. After each completed run, the whole
# output dir is uploaded as a *single* commit via upload_folder — this
# matters because per-file uploads on a 5-modality x 5-column visualisation
# sweep blow past the HF rate limit. Set HF_RESULTS_REPO in .env to enable;
# leave empty to skip uploads entirely.
HF_RESULTS_REPO="${HF_RESULTS_REPO:-}"

PHASE="${1:?Usage: bash run_grid.sh <phase 1|2|3|auto> [best_cfg|best_t best_cfg]}"

upload_run() {
    local outdir=$1 tag=$2
    if [ -z "$HF_RESULTS_REPO" ] || [ -z "${HF_TOKEN:-}" ]; then
        echo "[UPL ] skipped (HF_RESULTS_REPO or HF_TOKEN not set)"
        return 0
    fi
    if [ ! -f "$outdir/summary.txt" ]; then
        echo "[UPL ] skipped — $outdir/summary.txt missing (run failed?)"
        return 0
    fi
    echo "[UPL ] $tag  →  hf://$HF_RESULTS_REPO/$tag"
    HF_TOKEN="$HF_TOKEN" HF_RESULTS_REPO="$HF_RESULTS_REPO" \
    OUTDIR="$outdir" TAG="$tag" \
    uv run python -c '
import os
from huggingface_hub import HfApi, login
login(token=os.environ["HF_TOKEN"])
api = HfApi()
repo = os.environ["HF_RESULTS_REPO"]
tag = os.environ["TAG"]
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
api.upload_folder(
    folder_path=os.environ["OUTDIR"],
    path_in_repo=tag,
    repo_id=repo,
    repo_type="dataset",
    commit_message="grid run " + tag,
)
print("  -> https://huggingface.co/datasets/" + repo + "/tree/main/" + tag)
' || echo "[UPL ] upload failed for $tag (continuing)"
}

run_one() {
    local t=$1 step=$2 cfg=$3 phase=$4
    # Tag includes N so runs at different sample sizes don't collide on disk.
    # RUN_TAG (env var) lets you append a free-form suffix — use it when
    # rerunning the same hyperparams against a new checkpoint, e.g.
    #   RUN_TAG=ckpt_e25 bash run_grid.sh 1 --v2
    local n_tag="n${NUM_UNHEALTHY}_${NUM_HEALTHY}"
    local run_suffix="${RUN_TAG:+_${RUN_TAG}}"
    local tag="p${phase}_t${t}_s${step}_cfg${cfg}_${n_tag}${V2_TAG}${run_suffix}"
    local outdir="./anomaly_results_grid_${tag}"

    if [ -f "$outdir/summary.txt" ]; then
        echo "[SKIP] $tag — summary.txt already exists"
        upload_run "$outdir" "$tag"
        return 0
    fi

    echo "[RUN ] $tag  →  $outdir"
    mkdir -p "$outdir"
    uv run python infer_anomaly.py \
        --checkpoint          "$CHECKPOINT" \
        --arch                "$ARCH" \
        --data_path           "$DATA_PATH" \
        --split_file          "${SPLIT_FILE:-${DATA_PATH}/preprocessed_split.json}" \
        --split               "${SPLIT:-val}" \
        --t                   "$t" \
        --step_size           "$step" \
        --cfg_scale           "$cfg" \
        --num_unhealthy       "$NUM_UNHEALTHY" \
        --num_healthy         "$NUM_HEALTHY" \
        --output_dir          "$outdir" \
        --device              cuda \
        --seed                42 \
        --min_component_size  100 \
        --border_erosion      3 \
        --border_overlap_thr  0.6 \
        --combined_modalities T2,FLAIR \
        --no_images \
        --best

    upload_run "$outdir" "$tag"
}

echo "Sample caps: NUM_UNHEALTHY=$NUM_UNHEALTHY  NUM_HEALTHY=$NUM_HEALTHY  (-1 = full val)"
echo "Checkpoint : $CHECKPOINT (arch=$ARCH)"
[ -n "${RUN_TAG:-}" ] && echo "Run tag    : $RUN_TAG (output dirs suffixed with _$RUN_TAG)"
echo "Data path  : $DATA_PATH"
if [ -n "$HF_RESULTS_REPO" ]; then
    echo "HF upload  : $HF_RESULTS_REPO (one upload_folder commit per run)"
else
    echo "HF upload  : disabled (set HF_RESULTS_REPO + HF_TOKEN in .env to enable)"
fi
echo ""

case "$PHASE" in
    1)
        echo "=== PHASE 1: CFG sweep (t=0.2, step=0.02) ==="
        # cfg=1.0 already covered by Run B (T2+FLAIR). Sweep four points
        # densely around the suspected peak at cfg=1.0:
        #   0.5 and 0.7 — search below 1
        #   1.5 and 2.0 — search above 1 (cfg=2.0 redone with T2+FLAIR for
        #                                  fair comparison; existing Run C
        #                                  used 4-mod combined).
        # Combined with Run B (cfg=1.0) and old Run D (cfg=3.0, 4-mod), we
        # get six cfg points spanning [0.5, 3.0] for the ablation chart.
        for cfg in ${CFG_LIST:-0.5 0.7 1.5 2.0}; do
            run_one 0.2 0.02 "$cfg" 1
        done
        echo ""
        echo "Phase 1 done. Compare DICE in:"
        echo "  ./anomaly_results_grid_p1_*"
        echo "  + existing ./anomaly_results_t_0.2_0.02_cfg_scale_1_new_ckpt_otsu_border_combined_T2_FLAIR (cfg=1.0)"
        echo "Then run:  bash run_grid.sh 2 <best_cfg>"
        ;;
    2)
        BEST_CFG="${2:?Phase 2 needs the winning cfg from phase 1: bash run_grid.sh 2 <best_cfg>}"
        echo "=== PHASE 2: t sweep (cfg=${BEST_CFG}, step=0.02) ==="
        # If BEST_CFG=1.0, t=0.2 and t=0.3 are already covered (runs B and G).
        # Otherwise t=0.2 and t=0.3 also need to be re-run at the new cfg.
        for t in 0.10 0.15 0.25; do
            run_one "$t" 0.02 "$BEST_CFG" 2
        done
        if [ "$BEST_CFG" != "1.0" ]; then
            run_one 0.20 0.02 "$BEST_CFG" 2
            run_one 0.30 0.02 "$BEST_CFG" 2
        fi
        echo ""
        echo "Phase 2 done. Find best t, then run:  bash run_grid.sh 3 <best_t> ${BEST_CFG}"
        ;;
    3)
        BEST_T="${2:?Phase 3 needs best t and cfg: bash run_grid.sh 3 <best_t> <best_cfg>}"
        BEST_CFG="${3:?Phase 3 needs best t and cfg: bash run_grid.sh 3 <best_t> <best_cfg>}"
        echo "=== PHASE 3: step sweep (t=${BEST_T}, cfg=${BEST_CFG}) ==="
        # step=0.02 already covered by phase 2 winner. Sweep faster steps.
        for step in 0.04 0.07; do
            run_one "$BEST_T" "$step" "$BEST_CFG" 3
        done
        echo ""
        echo "Phase 3 done. All sweeps complete."
        echo "Build the speed-vs-quality table from ./anomaly_results_grid_p3_*"
        ;;
    auto)
        # ── Chạy cả 3 phase liên tiếp, tự động chọn winner sau mỗi phase ──
        # Metric dùng để chọn winner: DICE của modality 'combined'.
        # Đọc từ summary.txt dòng "  combined    DICE=X.XXXX"
        pick_best_cfg() {
            local best_cfg="" best_dice=-1
            local n_tag="n${NUM_UNHEALTHY}_${NUM_HEALTHY}"
            for cfg in 0.5 0.7 1.5 2.0; do
                local run_suffix="${RUN_TAG:+_${RUN_TAG}}"
                local tag="p1_t0.2_s0.02_cfg${cfg}_${n_tag}${V2_TAG}${run_suffix}"
                local summary="./anomaly_results_grid_${tag}/summary.txt"
                [ -f "$summary" ] || continue
                local dice
                dice=$(grep "^  combined " "$summary" | grep -oP 'DICE=\K[0-9.]+' | head -1)
                [ -z "$dice" ] && continue
                if awk "BEGIN{exit !($dice > $best_dice)}"; then
                    best_dice=$dice; best_cfg=$cfg
                fi
            done
            echo "$best_cfg"
        }

        pick_best_t() {
            local best_cfg=$1 best_t="" best_dice=-1
            local n_tag="n${NUM_UNHEALTHY}_${NUM_HEALTHY}"
            for t in 0.10 0.15 0.20 0.25 0.30; do
                local run_suffix="${RUN_TAG:+_${RUN_TAG}}"
                local tag="p2_t${t}_s0.02_cfg${best_cfg}_${n_tag}${V2_TAG}${run_suffix}"
                local summary="./anomaly_results_grid_${tag}/summary.txt"
                [ -f "$summary" ] || continue
                local dice
                dice=$(grep "^  combined " "$summary" | grep -oP 'DICE=\K[0-9.]+' | head -1)
                [ -z "$dice" ] && continue
                if awk "BEGIN{exit !($dice > $best_dice)}"; then
                    best_dice=$dice; best_t=$t
                fi
            done
            echo "$best_t"
        }

        echo "=== AUTO: PHASE 1 — CFG sweep (t=0.2, step=0.02) ==="
        for cfg in 0.5 0.7 1.5 2.0; do
            run_one 0.2 0.02 "$cfg" 1
        done

        BEST_CFG=$(pick_best_cfg)
        if [ -z "$BEST_CFG" ]; then
            echo "Error: không tìm được best_cfg từ phase 1 — kiểm tra summary.txt" >&2
            exit 1
        fi
        echo ""
        echo ">>> Phase 1 winner: cfg=${BEST_CFG}"

        echo ""
        echo "=== AUTO: PHASE 2 — t sweep (cfg=${BEST_CFG}, step=0.02) ==="
        for t in 0.10 0.15 0.25; do
            run_one "$t" 0.02 "$BEST_CFG" 2
        done
        if [ "$BEST_CFG" != "1.0" ]; then
            run_one 0.20 0.02 "$BEST_CFG" 2
            run_one 0.30 0.02 "$BEST_CFG" 2
        fi

        BEST_T=$(pick_best_t "$BEST_CFG")
        if [ -z "$BEST_T" ]; then
            echo "Error: không tìm được best_t từ phase 2 — kiểm tra summary.txt" >&2
            exit 1
        fi
        echo ""
        echo ">>> Phase 2 winner: t=${BEST_T}"

        echo ""
        echo "=== AUTO: PHASE 3 — step sweep (t=${BEST_T}, cfg=${BEST_CFG}) ==="
        for step in 0.04 0.07; do
            run_one "$BEST_T" "$step" "$BEST_CFG" 3
        done
        echo ""
        echo "=== AUTO DONE: best cfg=${BEST_CFG}  t=${BEST_T}  (step=0.02 vs 0.04 vs 0.07 — xem p3_*) ==="
        ;;
    *)
        echo "Unknown phase: $PHASE  (expected 1, 2, 3, or auto)" >&2
        exit 1
        ;;
esac
