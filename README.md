# anomedfm

Weakly supervised brain tumor anomaly detection using Flow Matching generative models on BraTS 2021.

Only **image-level labels** (healthy / unhealthy) are used during training — no pixel-wise tumor annotations. At inference, a test slice is partially encoded along the flow trajectory then decoded back toward the healthy distribution using Classifier-Free Guidance (CFG); the pixel-wise reconstruction residual reveals tumor regions.

## Method

- **Architecture**: UNet (bratsv2, 5-level) with timestep and class conditioning, trained with Conditional Flow Matching
- **Supervision**: image-level labels only — healthy (y=0) / unhealthy (y=1); no segmentation masks used during training
- **Input**: 4-modality MRI slices (T1, T1ce, T2, FLAIR) — shape `(4, 256, 256)`, normalized to `[0, 1]`
- **Inference**: reverse-ODE encode to `t_start=0.2` → forward-ODE decode with CFG (w=0.5) toward healthy label → pixel-wise residual → Otsu threshold + morphological post-processing
- **Best result**: combined T2+FLAIR mask on full test set (126 cases)

## Dataset

BraTS 2021 preprocessed slices (.npy format). Download from Kaggle:

```bash
pip install kaggle
# place kaggle.json at ~/.kaggle/kaggle.json
kaggle datasets download minhdon/brats2021-preprocessed -p ./data/ --unzip
```

Expected layout after extraction:
```
data/brats2021/
  healthy/BraTS2021_XXXXX/slice_YYY.npy        # (4, 256, 256)
  unhealthy/BraTS2021_XXXXX/slice_YYY.npy      # (4, 256, 256)
  unhealthy/BraTS2021_XXXXX/slice_YYY_seg.npy  # (1, 256, 256)
```

Generate the train/val/test split:

```bash
python create_brats_split.py \
  --data_root ./data/brats2021 \
  --output ./data/brats2021/preprocessed_split.json
```

## Installation

```bash
pip install -r requirements.txt
# or with uv:
uv sync
```

## Training

Train on BraTS healthy slices:

```bash
python train.py \
  --dataset brats \
  --data_path ./data/brats2021 \
  --split_file ./data/brats2021/preprocessed_split.json \
  --arch bratsv2 \
  --healthy_only \
  --cfg_scale 0.5 \
  --output_dir ./output_brats \
  --epochs 50 \
  --batch_size 32
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--arch` | `brats` | Model architecture (`brats` / `bratsv2`) |
| `--healthy_only` | off | Train on healthy slices only |
| `--cfg_scale` | `3.0` | Classifier-free guidance scale |
| `--epochs` | `921` | Number of training epochs |
| `--batch_size` | — | Batch size per GPU |
| `--resume` | — | Resume from checkpoint path |

## Inference

Via the shell wrapper (recommended):

```bash
# Evaluate on val set (100 unhealthy + 50 healthy)
bash execute_anomaly_detection.sh --v2 --best

# Evaluate on full test set
NUM_UNHEALTHY=-1 NUM_HEALTHY=-1 \
SPLIT_FILE=./data/brats2021/preprocessed_split.json \
bash execute_anomaly_detection.sh --v2 --best --split test

# Single sample — also saves clean individual images
SINGLE_IDX=4547 \
CLEAN_BASE_DIR=./clean_samples \
SPLIT_FILE=./data/brats2021/preprocessed_split.json \
INFER_SPLIT=test \
bash execute_anomaly_detection.sh --v2 --best
```

Or directly:

```bash
python infer_anomaly.py \
  --checkpoint ./output_brats/checkpoint_epoch0011.pth \
  --arch bratsv2 \
  --data_path ./data/brats2021 \
  --split_file ./data/brats2021/preprocessed_split.json \
  --split test \
  --t 0.2 \
  --step_size 0.02 \
  --cfg_scale 0.5 \
  --num_unhealthy -1 \
  --num_healthy -1 \
  --output_dir ./anomaly_results \
  --best
```

Key inference arguments:

| Argument | Default | Description |
|---|---|---|
| `--t` | `0.6` | Encode end time (0→1); lower = lighter edit |
| `--step_size` | `0.02` | ODE step size |
| `--cfg_scale` | `3.0` | CFG guidance scale at decode |
| `--split` | `val` | Split to evaluate (`val` / `test`) |
| `--best` | off | Oracle: pick best-DICE modality per sample |
| `--combined_modalities` | all | Modalities to union into combined mask (e.g. `T2,FLAIR`) |
| `--min_component_size` | `50` | Drop connected components smaller than N pixels |
| `--border_erosion` | `3` | Brain rim thickness for edge-noise suppression |

## Infer top samples by dice

```bash
# Infer top-10 dice>0.90 samples and save clean images
bash infer_top10_clean.sh
```

Outputs per sample (under `clean_samples/fm/<idx>/`):
- `fm_original_{T1,T1ce,T2,FLAIR,combined}.png`
- `fm_recon_{T1,T1ce,T2,FLAIR,combined}.png`
- `fm_anomaly_map_{T1,T1ce,T2,FLAIR,combined}.png`
- `fm_binary_mask_{T1,T1ce,T2,FLAIR,combined}.png`
- `fm_gt_mask.png`

## Pretrained checkpoint

Download from HuggingFace:

```bash
python hf_download.py
# or manually:
# huggingface-cli download minh2k4/brats-flow-matching checkpoint_epoch0011.pth
```

## Repository structure

```
anomedfm/
  train.py                      # training entry point
  train_arg_parser.py           # training arguments
  infer_anomaly.py              # inference + evaluation
  execute_anomaly_detection.sh  # shell wrapper for inference
  infer_top10_clean.sh          # infer top-10 dice samples
  process_brats.py              # preprocess raw BraTS NIfTI → .npy slices
  create_brats_split.py         # generate train/val/test split JSON
  hf_download.py                # download checkpoint from HuggingFace
  hf_upload.py                  # upload results to HuggingFace
  datasets/                     # dataset loaders (BraTS, simple shapes)
  models/                       # UNet architectures + EMA
  flow_matching/                # flow matching path, solver, loss
  training/                     # train loop, eval loop, distributed utils
```
