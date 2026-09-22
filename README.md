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
- **SDF-YOLO** (Section 17): a custom Ultralytics YOLO11n-based architecture with a Context Anchor Attention (CAA) module inserted, sized to keep pretrained-weight transfer intact.

## Baselines compared

Pretrained (ImageNet): ResNet50, VGG16, DenseNet121, MobileNet, MobileNetV2, MobileNetV3Large, EfficientNetB0, EfficientNetV2B0.
From-scratch: ResNet18, ResNet34, ShuffleNetV2.
Detection: YOLOv8n, YOLO11n, YOLO12n, FCOS (ours), APG-YOLO-inspired (ours), SDF-YOLO (ours, optional).

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

## Reproducibility notes

- The train/val/test split is frozen (`splits/split_v1.csv` + a sha256 sidecar) and stratified per class — identical across every seed and every model compared.
- `load_boxes()` **raises** rather than fabricating ground truth when an annotation is missing or malformed; images without a verified box are excluded from localization evaluation, never given a synthetic one.
- Every ablation/baseline/attention-variant comparison trains on the exact same split, augmentation, and schedule, varying only the one factor under study.
- Multi-seed results are always reported as mean ± sd (`model_seeds = (42, 1337, 2026, 7, 99)`); a gap smaller than the sd is not evidence of anything.
- Checkpoints (`*.h5`, `*.weights.h5`), YOLO export/run directories, and MLflow run logs are gitignored — they're large (tens to 200+ MB per file) and fully regenerated by re-running the notebook. Only the small, citable artifacts (tables, figures, logs, the frozen split) are version-controlled.

## Known limitations

- All training in this repo runs on **CPU only** (no CUDA/ROCm-enabled TensorFlow build) — a full `QUICK=False` run across both datasets, all ablations, and all baselines takes on the order of many hours.
- The results table above reflects a quick single-seed smoke test, not the reported multi-seed numbers.
