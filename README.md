# MobileNetV2 + FPN/SEAM/CSAF/AMFF — Steel Surface Defect Classification with CAM-Based Localization

A reproducible research pipeline for steel-surface-defect **classification with CAM-based localization**, built on a MobileNetV2 backbone with a lightweight feature pyramid (FPN), a spatial-enhancement attention module (SEAM), a cross-scale attention fusion block (CSAF), and a proposed attention-based multi-scale fusion block (AMFF). The same pipeline is run end-to-end on two public defect datasets — **NEU-DET** (6 classes) and **GC10-DET** (10 classes) — and compared against a wide set of baseline backbones, a custom FCOS-style detector, and standard YOLO models under an identical evaluation protocol.

Everything — architecture, hyperparameters, splits, metrics, and every table/figure in the eventual paper — is produced by a single notebook: [`new_model_code.ipynb`](new_model_code.ipynb).

## What this is (and isn't)

- **Task**: image-level classification (categorical cross-entropy), *not* object detection. There is no bounding-box supervision during training.
- **Localization**: regions are recovered post hoc from Grad-CAM heatmaps and scored as **CAM-based localization** — reported separately from, and never conflated with, supervised detection.
- **Supervised detection is also included**, for comparison: a custom FCOS-style anchor-free detector (with optional PKI/AIFI/GFPN neck additions inspired by APG-YOLO), standard YOLOv8/11/12 baselines, and a custom SDF-YOLO variant — all trained on the same verified ground-truth boxes and scored with the identical AP/mAP code as the CAM-based numbers, so every localization number in this repo is directly comparable.

## Datasets

| | NEU-DET | GC10-DET |
|---|---|---|
| Classes | 6 (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) | 10 (punching_hole, welding_line, crescent_gap, water_spot, oil_spot, silk_spot, inclusion, rolled_pit, crease, waist_folding) |
| Native image size | 200×200, grayscale | 2048×1000, grayscale |
| Working resolution | 200×200 (native, no resampling) | 224×224 (resized; see below) |
| Annotation format | PASCAL-VOC XML | PASCAL-VOC XML |
| Images | 1,800 | 2,304 (after cleaning, see below) |

Both datasets are consumed through the identical directory contract — `IMAGES/<class_name>/*.jpg` plus a flat `ANNOTATIONS/<stem>.xml` — so every data-loading, training, evaluation, ablation, and YOLO-export function in the notebook works unmodified across both.

**GC10-DET preparation** (`new_model_code.ipynb`, Section 0b): the raw release ships as 10 numbered class folders plus a flat label folder, at native 2048×1000. The prep cell resizes every image to 224×224 (rescaling boxes to match) and fixes three real data-quality issues found by inspection, never silently:
- A class-name typo (`10_yaozhed` → `10_yaozhe`, 131 files).
- One file with a single garbage annotation object literally named `d` (the file's other, valid object is kept; only the bad one is dropped).
- **12 filenames reused across different class folders** — fixed by prefixing every prepared filename with its class name, since the flat annotations folder and the stem-keyed image lookup would otherwise silently collide and corrupt ground truth.

Raw `GC10-DET/1..10` and `GC10-DET/lable` are left untouched by this step; the cleaned, resized copy is written to `GC10-DET_PREPARED/`.

> Datasets themselves are not included in this repository. Place NEU-DET under `NEU-DET/` and raw GC10-DET under `GC10-DET/` (both in the repo root) before running.

## Architecture

- **Backbone**: MobileNetV2 (ImageNet-pretrained), tapped at 4 strides (C2–C5).
- **FPN**: lightweight top-down pyramid, configurable channel width.
- **Fusion attention** at the lateral-fusion point — pluggable, compared head-to-head: `none | se | cbam | eca | coordatt | ema | amff` (**AMFF** is the proposed method).
- **SEAM**: multi-dilation depthwise-conv spatial gate applied to each pyramid level.
- **CSAF**: cross-scale attention fusion, competing weights across P2–P5 via softmax.
- **Head**: `GlobalAveragePooling2D → Dropout → Dense(num_classes, softmax)`.
- **Detection variant** (Section 14): the same backbone/FPN/attention stack feeds a shared anchor-free FCOS-style head, optionally extended with **PKI** (multi-kernel context), **AIFI** (transformer-style context at the coarsest level), and a **GFPN**-style denser neck — inspired by APG-YOLO.
- **SDF-YOLO** (Section 17, n-scale prototype): a custom Ultralytics YOLO11n-based architecture with a Context Anchor Attention (CAA) module inserted, sized to keep pretrained-weight transfer intact. The full GPU version (s-scale, SPPF, COCO-pretrained head) lives in [`colab_sdf_yolo_v2/`](colab_sdf_yolo_v2/) — see [SDF-YOLO v2](#sdf-yolo-v2-gpu-detection-pipeline) below.

## Baselines compared

Pretrained (ImageNet): ResNet50, VGG16, DenseNet121, MobileNet, MobileNetV2, MobileNetV3Large, EfficientNetB0, EfficientNetV2B0.
From-scratch: ResNet18, ResNet34, ShuffleNetV2.
Detection: YOLOv8n, YOLO11n, YOLO12n, FCOS (ours), APG-YOLO-inspired (ours), SDF-YOLO (ours, optional).
Detection, SDF-YOLO v2: YOLOv5s, YOLOv8s, YOLOv9s, YOLOv10s, YOLO11n, YOLO11s, YOLO12s.

All baselines are trained on the *identical* split, augmentation, schedule, and seed as the proposed model, and scored with the same metric code — see `table_classification_comparison.csv` and `table_final_comparison.csv`.

## Repository structure

```
new_model_code.ipynb          # the entire pipeline — config, data, model, training, eval, ablations, baselines
NEU-DET/                      # raw NEU-DET (not included — place here)
GC10-DET/                     # raw GC10-DET (not included — place here)
GC10-DET_PREPARED/            # generated: cleaned + resized GC10-DET (Section 0b)
paper_results/                # generated: NEU-DET outputs (checkpoints, logs, figures, tables)
paper_results_gc10det/        # generated: GC10-DET outputs, same structure
paper_results_comparison/     # generated: cross-dataset comparison tables/figures (Section 23)
colab_sdf_yolo_v2/            # SDF-YOLO v2: GPU training script, run scripts, runs/ (see below)
paper_results_v2/             # generated: SDF-YOLO v2 paper tables (md/tex/csv) + figures
```

Each `paper_results*/` folder follows the same layout:

```
checkpoints/   *.weights.h5 per trained variant (gitignored — large, regenerated by training)
logs/          per-run CSV training logs + history JSON
figures/       confusion matrices, qualitative CAM/detection grids, architecture diagram
tables/        every CSV/JSON table referenced above, including summary.json
yolo/          exported YOLO-format dataset + Ultralytics run artifacts (gitignored)
splits/        the frozen train/val/test split (with a sha256 sidecar) — this *is* versioned
```

## Setup

```bash
pip install tensorflow==2.15.0 ultralytics==8.3.174 torch opencv-python numpy pandas scikit-learn matplotlib
```

Also requires PyTorch (for the SDF-YOLO section) and a Jupyter/VS Code notebook environment. Everything runs on CPU; no GPU is required (though training is considerably slower without one — see Notes below).

## How to run

The notebook is organized top-to-bottom as a single linear script; running all cells in order reproduces everything.

| Sections | What they do |
|---|---|
| 0 – 0b | Config (`CFG`, `GC10_CFG`), dataset prep for both datasets |
| 1 – 12 | Ground truth, frozen split, input pipeline, architecture, training, classification eval, baseline comparison, CAM localization, IoU/AP/mAP metrics, ablation studies, efficiency measurement, improved CAM tuning |
| 13 | **NEU-DET**: run everything — multi-seed training, per-class metrics, CAM localization, efficiency, classification comparison, ablation, YOLO baselines, `summary.json` |
| 14 – 18 | Anchor-free detector (FCOS/PKI/AIFI/GFPN), NEU-DET detector comparison, SDF-YOLO (flag-gated, off by default) |
| 19 – 21 | **GC10-DET**: the same "run everything" (13), detector comparison (16), and SDF-YOLO trigger (18), mirrored |
| 22 | Multi-seed ablation driver (fills a gap: single-seed-only ablation, run here across additional seeds) |
| 23 | **Cross-dataset comparison** — reads every table above from both datasets, writes unified tables + grouped bar charts to `paper_results_comparison/` |

A `QUICK` flag at the top of each "run everything" cell (13, 16, 19, 20) toggles between a fast smoke test (1 seed, 2+2 epochs) and the full reported run (all seeds, full epoch budget). Set `QUICK = False` for numbers worth quoting.

## Results

Every number here comes from `paper_results*/tables/summary.json` and the CSV tables — nothing is hand-typed. Current numbers below are from a `QUICK=True` smoke-test run (single seed, 2 frozen + 2 finetune epochs) and are **not** the reported numbers; re-run with `QUICK=False` before quoting anything from this repo.

| Dataset | Test accuracy | Macro F1 | CAM localization AP50 |
|---|---|---|---|
| NEU-DET | 93.33% | 92.95% | 9.71 |
| GC10-DET | *(training — see `paper_results_gc10det/tables/summary.json` once complete)* | | |

See `paper_results/tables/table_final_comparison.csv` (and its GC10-DET / cross-dataset equivalents) for every model's classification/localization/detection numbers side by side under the identical protocol.

## SDF-YOLO v2 (GPU detection pipeline)

[`colab_sdf_yolo_v2/`](colab_sdf_yolo_v2/) trains SDF-YOLO and seven Ultralytics baselines on NEU-DET and GC10-DET with supervised boxes, all under one protocol. Everything there is driven by [`train.py`](colab_sdf_yolo_v2/train.py); run details and history are in [`IMPROVE_PLAN.md`](colab_sdf_yolo_v2/IMPROVE_PLAN.md).

**Model (current default).** YOLO11s-scale backbone with yolo11's **SPPF** at the P5 level (the original AIFI variant is `--p5 aifi`). The neck adds a multi-kernel DWConv block (3/5/7/9) at P3 and **CAA** (Context Anchor Attention) at P3/P4/P5. The backbone, SPPF and Detect head are initialised from `yolo11s.pt` (`--coco-head`, 115 of 121 head tensors; only the class-output layers start random), so SDF-YOLO starts from the same COCO pretraining as the baselines. Size: 11.6M params, 27.4 GFLOPs at 640.

**Protocol.** Every model uses 640×640 on both datasets, cosine LR, mosaic off for the last 20 epochs, repeat-factor sampling (threshold 0.1, train split only) and seed 42. Baselines train for 200 epochs and SDF-YOLO for 250. `best.pt` is chosen on the **val** split; the test split is evaluated once, after training. FPS is measured at batch 1 on an RTX 3060 (torch 2.11, ultralytics 8.3.174).

```powershell
cd colab_sdf_yolo_v2
python train.py --data neu  --model sdf            # SDF-YOLO -> runs/neu_sdf_s_ch_sppf_e250
python train.py --data gc10 --model yolo12s.pt     # one baseline
python train.py --data all  --model all --skip-done
python make_comparison.py                          # -> ../paper_results_v2/
```

Use `--imgsz 1024` for GC10 at high resolution (the batch drops to 8 automatically; runs get an `_i1024` suffix). Pass `--seeds 42,1,2` to report mean ± std.

### Results — test split, 640×640, seed 42 (single run)

Generated by `make_comparison.py` → [`paper_results_v2/comparison.md`](paper_results_v2/comparison.md) (LaTeX: `comparison.tex`). The best value in each column is in bold.

**NEU-DET**

| Model | mAP50 | mAP75 | mAP50-95 | F1 | P | R | Params (M) | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|---|---|
| **SDF-YOLO (ours)** | 75.23 | 41.59 | 42.46 | 72.61 | 72.99 | 72.54 | 11.60 | 27.4 | 50 |
| YOLO12s | **77.51** | **44.51** | **44.82** | 71.70 | 69.55 | **74.53** | 9.23 | 21.2 | 46 |
| YOLOv9s | 77.27 | 42.10 | 44.10 | **72.96** | 75.57 | 70.77 | 7.17 | 26.7 | 28 |
| YOLO11n | 76.88 | 43.54 | 44.18 | 72.31 | 74.91 | 70.67 | **2.58** | **6.3** | 59 |
| YOLO11s | 76.20 | 42.32 | 43.85 | 71.01 | 77.07 | 67.22 | 9.42 | 21.3 | 54 |
| YOLOv8s | 75.84 | 40.82 | 43.02 | 71.82 | **77.63** | 67.72 | 11.13 | 28.4 | **71** |
| YOLOv5s | 75.70 | 43.25 | 43.72 | 71.29 | 74.49 | 69.24 | 9.11 | 23.8 | 54 |
| YOLOv10s | 75.32 | 42.49 | 42.91 | 72.64 | 75.38 | 70.58 | 7.22 | 21.4 | 68 |

**GC10-DET**

| Model | mAP50 | mAP75 | mAP50-95 | F1 | P | R | Params (M) | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|---|---|
| **SDF-YOLO (ours)** | 64.85 | 28.54 | 31.97 | **65.41** | **74.60** | 61.10 | 11.60 | 27.4 | 54 |
| YOLO12s | **67.21** | 27.37 | 32.42 | 64.71 | 65.97 | **66.96** | 9.23 | 21.2 | 41 |
| YOLO11n | 65.05 | **29.53** | **32.57** | 63.65 | 63.59 | 65.27 | **2.58** | **6.3** | 65 |
| YOLOv10s | 63.99 | 28.71 | 31.86 | 63.80 | 64.78 | 64.23 | 7.22 | 21.4 | 73 |
| YOLOv9s | 63.95 | 26.02 | 30.95 | 63.51 | 65.55 | 63.67 | 7.17 | 26.7 | 32 |
| YOLO11s | 63.86 | 26.63 | 30.33 | 61.70 | 72.79 | 58.05 | 9.42 | 21.3 | 63 |
| YOLOv5s | 63.74 | 28.87 | 32.08 | 61.10 | 60.57 | 62.98 | 9.12 | 23.8 | 74 |
| YOLOv8s | 63.48 | 28.63 | 31.80 | 62.35 | 69.18 | 59.74 | 11.13 | 28.5 | **84** |

### SDF-YOLO variants (selected on val mAP50)

| Variant | NEU val / test mAP50 | GC10 val / test mAP50 |
|---|---|---|
| AIFI, random-init head (`sdf_s`) | 74.18 / 76.04 | 63.78 / 63.46 |
| + COCO head (`sdf_s_ch`) | 73.94 / 76.59 | 62.59 / 62.63 |
| + COCO head, 25-epoch warm freeze (`sdf_s_ch_wf25`) | 73.69 / 74.94 | 64.21 / 64.81 |
| **SPPF + COCO head, 250 epochs (`sdf_s_ch_sppf_e250`)** | **74.67** / 75.23 | **65.87** / 64.85 |
| same, GC10 at 1024 (`sdf_s_ch_sppf_i1024_e250`) | — | 65.12 / 66.40 |

The SPPF variant has the highest val mAP50 on both datasets, so it is the reported model. The first three variants used 13.05M params / 28.6 GFLOPs. Only the summaries of these older runs are kept, in `colab_sdf_yolo_v2/old_runs_summaries.zip`.

**Caveats.** These are single-seed results. Most gaps between models are 1–2 mAP50 points, which is within typical seed-to-seed variation, so the 3-seed runs (`run_final.ps1`) are still needed before quoting any ranking. SDF-YOLO trains for 250 epochs against 200 for the baselines; state this in the paper or give the baselines 250 as well.

## Reproducibility notes

- The train/val/test split is frozen (`splits/split_v1.csv` + a sha256 sidecar) and stratified per class — identical across every seed and every model compared.
- `load_boxes()` **raises** rather than fabricating ground truth when an annotation is missing or malformed; images without a verified box are excluded from localization evaluation, never given a synthetic one.
- Every ablation/baseline/attention-variant comparison trains on the exact same split, augmentation, and schedule, varying only the one factor under study.
- Multi-seed results are always reported as mean ± sd (`model_seeds = (42, 1337, 2026, 7, 99)`); a gap smaller than the sd is not evidence of anything.
- Checkpoints (`*.h5`, `*.weights.h5`), YOLO export/run directories, and MLflow run logs are gitignored — they're large (tens to 200+ MB per file) and fully regenerated by re-running the notebook. Only the small, citable artifacts (tables, figures, logs, the frozen split) are version-controlled.

## Known limitations

- All notebook training runs on **CPU only** (the SDF-YOLO v2 detection runs in `colab_sdf_yolo_v2/` use an RTX 3060) (no CUDA/ROCm-enabled TensorFlow build) — a full `QUICK=False` run across both datasets, all ablations, and all baselines takes on the order of many hours.
- The results table above reflects a quick single-seed smoke test, not the reported multi-seed numbers.
