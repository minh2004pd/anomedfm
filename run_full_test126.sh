#!/bin/bash
set -e

WORKDIR="/mnt/apple/k66/minhdd/flow-matching-main"
CHECKPOINT="./brats-perbatch-e11/checkpoint_epoch0011.pth"
DATA_PATH="/mnt/apple/k66/minhdd/data/brats2021"
SPLIT_FILE="$DATA_PATH/preprocessed_split.json"
OUTPUT_MISSING="./anomaly_results_epoch11_missing"
OUTPUT_STILL="./anomaly_results_epoch11_still_missing"
OUTPUT_FINAL="./anomaly_results_final_full_test126"
VAL251="./anomaly_results_final_full_val251"

cd "$WORKDIR"

log() { echo "[$(date '+%H:%M:%S %d/%m')] $*"; }

# ── Bước 1: chờ job missing (2554 slices) xong ──────────────────────────────
log "Chờ job missing đang chạy xong..."
while pgrep -f "anomaly_results_epoch11_missing" > /dev/null; do
    LINES=$(wc -l < "$OUTPUT_MISSING/metrics.csv" 2>/dev/null || echo 0)
    log "  missing: $LINES rows / ~14557 dự kiến"
    sleep 60
done
log "Job missing xong! $(wc -l < $OUTPUT_MISSING/metrics.csv) rows"

# ── Bước 2: tính still_missing ───────────────────────────────────────────────
log "Tính still_missing..."
python3 << 'PYEOF'
import csv, json

with open('/mnt/apple/k66/minhdd/data/brats2021/preprocessed_split.json') as f:
    cur = json.load(f)

covered = set()

# val251 test portion (global idx >= 6125 -> local)
with open('./anomaly_results_final_full_val251/metrics.csv') as f:
    for r in csv.DictReader(f):
        if int(r['idx']) >= 6125:
            covered.add(int(r['idx']) - 6125)

# job missing vừa xong
with open('./anomaly_results_epoch11_missing/metrics.csv') as f:
    for r in csv.DictReader(f):
        covered.add(int(r['idx']))

all_test = set(range(len(cur['test'])))
still = sorted(all_test - covered)
print(f'Covered: {len(covered)}, Still missing: {len(still)}')
with open('/tmp/still_missing_test_idxs.txt', 'w') as f:
    f.write(','.join(str(i) for i in still))
PYEOF

STILL_COUNT=$(python3 -c "
t = open('/tmp/still_missing_test_idxs.txt').read().strip()
print(len(t.split(',')) if t else 0)
")
log "Still missing: $STILL_COUNT slices"

# ── Bước 3: chạy still_missing ───────────────────────────────────────────────
if [ "$STILL_COUNT" -gt 0 ]; then
    STILL_IDXS=$(cat /tmp/still_missing_test_idxs.txt)
    log "Chạy tiếp $STILL_COUNT slices -> $OUTPUT_STILL"
    uv run python infer_anomaly.py \
      --checkpoint "$CHECKPOINT" \
      --arch bratsv2 \
      --data_path "$DATA_PATH" \
      --split test \
      --split_file "$SPLIT_FILE" \
      --target_idx "$STILL_IDXS" \
      --t 0.2 --step_size 0.02 --cfg_scale 0.5 \
      --num_unhealthy -1 --num_healthy -1 \
      --output_dir "$OUTPUT_STILL" \
      --device cuda \
      --min_component_size 100 --border_erosion 3 --border_overlap_thr 0.6 \
      --combined_modalities T2,FLAIR \
      --best --no_images
    log "Still missing xong! $(wc -l < $OUTPUT_STILL/metrics.csv) rows"
else
    log "Không có still_missing, bỏ qua bước 3"
fi

# ── Bước 4: merge tất cả -> OUTPUT_FINAL ─────────────────────────────────────
log "Merge val251 + missing + still_missing -> $OUTPUT_FINAL..."
python3 << 'PYEOF'
import csv, json, os

with open('/mnt/apple/k66/minhdd/data/brats2021/preprocessed_split.json') as f:
    cur = json.load(f)

all_rows = []
fieldnames = None

# Nguồn 1: val251 test portion (remap global -> local)
with open('./anomaly_results_final_full_val251/metrics.csv') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        if int(r['idx']) >= 6125:
            r = dict(r)
            r['idx'] = str(int(r['idx']) - 6125)
            all_rows.append(r)
print(f'val251 test rows: {len(all_rows)}')

# Nguồn 2: missing
n_before = len(all_rows)
with open('./anomaly_results_epoch11_missing/metrics.csv') as f:
    for r in csv.DictReader(f):
        all_rows.append(dict(r))
print(f'missing rows: {len(all_rows) - n_before}')

# Nguồn 3: still_missing
n_before = len(all_rows)
try:
    with open('./anomaly_results_epoch11_still_missing/metrics.csv') as f:
        for r in csv.DictReader(f):
            all_rows.append(dict(r))
    print(f'still_missing rows: {len(all_rows) - n_before}')
except FileNotFoundError:
    print('still_missing: không có file (bỏ qua)')

# Dedup theo (idx, label, modality)
seen = set()
deduped = []
for r in all_rows:
    key = (r['idx'], r['label'], r['modality'])
    if key not in seen:
        seen.add(key)
        deduped.append(r)

deduped.sort(key=lambda r: (int(r['idx']), r['label'], r['modality']))

os.makedirs('./anomaly_results_final_full_test126', exist_ok=True)
with open('./anomaly_results_final_full_test126/metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deduped)

unique_slices = len(set(r['idx'] for r in deduped))
total = len(cur['test'])
print(f'Merged: {len(deduped)} rows, {unique_slices}/{total} unique slices')
if unique_slices < total:
    print(f'WARNING: còn thiếu {total - unique_slices} slices!')
else:
    print('OK: đủ toàn bộ test set!')
PYEOF

log "DONE! Kết quả tại $OUTPUT_FINAL/metrics.csv"
