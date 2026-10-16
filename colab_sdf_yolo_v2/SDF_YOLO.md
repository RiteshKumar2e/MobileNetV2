# SDF-YOLO on NEU-DET and GC10-DET

This document covers everything in `colab_sdf_yolo_v2/`: the SDF-YOLO architecture, how both datasets are prepared, the training protocol, the full results on NEU-DET and GC10-DET, how the model got to its current form, and how to reproduce or extend the runs.

All numbers below come from `runs/*/summary.json` (aggregated in `runs/results_table.csv` and `../paper_results_v2/comparison.md`). Nothing is hand-typed. Every result is **single-seed (seed 42)** unless stated otherwise.

---

## 1. What is in this folder

| File / folder | Purpose |
|---|---|
| `train.py` | The single entry point: builds SDF-YOLO, trains SDF-YOLO and all baselines, evaluates on the test split, writes `summary.json` and `results_table.csv` |
| `make_comparison.py` | Builds paper tables (`comparison.md/.tex/.csv`) and plots into `../paper_results_v2/`; groups seeds into mean ± std |
| `prepare_gc10_native.py` | Builds `datasets/gc10/` from raw GC10-DET at native resolution, keeping the frozen split |
| `prepare_gc10_native_voc.py` | Same prep, writing a VOC-style (`IMAGES/` + `ANNOTATIONS/`) copy |
| `rename_gc10_like_neu.py` | Renames GC10 files to NEU-style names (`<class>_<n>`), writing a name map so raw names stay traceable |
| `splits/gc10_split_v1.csv` | Frozen GC10 train/val/test split (renamed stems) |
| `splits/gc10_split_v1_rawnames.csv` | Same split with the original raw stems |
| `splits/gc10_name_map.csv` | old stem → new stem → raw file → split |
| `datasets/neu/`, `datasets/gc10/` | YOLO-format data. `images/` and `labels/` are gitignored; the `*_data.yaml` files and `train_rfs_0.1.txt` are regenerated on every run |
| `run_improve.ps1` | Phase 1 / Phase 2 setting-search runs (history, see §7) |
| `run_final.ps1` | Final paper runs: TTA re-eval, extra seeds, m-scale, GC10 at 1024 |
| `IMPROVE_PLAN.md` | Running log of decisions (Hinglish) |
| `gc10_det.ipynb`, `neu_det.ipynb` | Notebook front-ends that call `train.py` per dataset |
| `results_table.xlsx` | Spreadsheet copy of an earlier results table |
| `old_runs_summaries.zip` | `summary.json` + table of the earlier SDF variants (their weights were deleted) |

Note: the committed `datasets/*/*_data.yaml` files contain absolute paths from the machine that produced them. `train.py` rewrites them for each run, so they do not need editing.

---

## 2. Datasets

| | NEU-DET | GC10-DET |
|---|---|---|
| Classes | 6 | 10 |
| Class names | crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches | punching_hole, welding_line, crescent_gap, water_spot, oil_spot, silk_spot, inclusion, rolled_pit, crease, waist_folding |
| Native image size | 200×200 grayscale | 2048×1000 grayscale |
| Train / val / test images | 1260 / 180 / 360 | 1602 / 229 / 461 |
| Train images after repeat-factor sampling (t = 0.1) | 1260 (unchanged; balanced) | 1723 |
| Training input size | 640 (letterboxed) | 640 (letterboxed); 1024 optional |

### GC10-DET preparation (`prepare_gc10_native.py`)

The notebook's earlier GC10 copy (`GC10-DET_PREPARED/`) was resized to 224×224 and labelled every box with its image's folder class. The detection pipeline here does not reuse it. Instead:

1. **Native resolution.** Images are copied byte-for-byte at 2048×1000. The network input size is chosen at train time with `--imgsz`, and ultralytics letterboxes while keeping the aspect ratio.
2. **Per-box class from the XML.** Each box keeps the class in its own `<object><name>`. GC10 images can contain more than one defect type, so labelling by folder was wrong for those images.
3. **Same frozen split.** The same stems go into the same train/val/test partitions as the notebook (`splits/gc10_split_v1.csv`).
4. **NEU-style names.** `rename_gc10_like_neu.py` renames files to `<class>_<n>` and writes `gc10_name_map.csv`.

### Class imbalance and repeat-factor sampling

GC10 is heavily imbalanced: crease has 51 training boxes and rolled_pit has 44. `train.py` applies LVIS repeat-factor sampling (Gupta et al., 2019) to the **train split only**:

- r(c) = max(1, √(t / f(c))), where f(c) is the fraction of train images that contain class c and t = 0.1 by default (`--rfs`).
- Each image is repeated by the max r over its classes; the fractional part is resolved with a seeded coin flip.

Repeat factors this produces on GC10:

| Class | inclusion | rolled_pit | crease | waist_folding | all others |
|---|---|---|---|---|---|
| Repeat factor | 1.08 | **2.31** | **2.03** | 1.28 | 1.00 |

NEU-DET is balanced, so every factor is 1.0 and its training list is unchanged. Val and test are never resampled.

---

## 3. SDF-YOLO architecture

SDF-YOLO is a YOLO11-based detector with a redesigned neck. The default, reported model is **SDF-YOLO-s with SPPF at P5** (run suffix `_ch_sppf`).

### Layer layout (built by `build_sdf_yaml()` in `train.py`)

```
Backbone (layers 0–10) — identical to yolo11s, COCO-pretrained
  0  Conv 64, 3, s2
  1  Conv 128, 3, s2
  2  C3k2 256
  3  Conv 256, 3, s2
  4  C3k2 512                         ── P3 feature
  5  Conv 512, 3, s2
  6  C3k2 512                         ── P4 feature
  7  Conv 1024, 3, s2
  8  C3k2 1024                        ── P5 feature
  9  SPPF 1024, k5   (or AIFI with --p5 aifi)
 10  C2PSA 1024

Top-down FPN (11–16) — same as yolo11s
 11–13  Upsample → Concat(P4) → C3k2 512
 14–16  Upsample → Concat(P3) → C3k2 256

SDF neck (17–31)
 17     Conv 256, 3
 18–21  DWConv k=3 / 5 / 7 / 9   (parallel, multi-scale depthwise)
 22     Concat(17, 18, 19, 20, 21)
 23     Conv 256, 1×1              (fuse)
 24     CAA, k=11                  ── P3 out
 25–27  Conv s2 → Concat(25, 13, 6) → C3k2 512
 28     CAA, k=9                   ── P4 out
 29–31  Conv s2 → Concat(29, 10, 8) → C3k2 1024
 32     CAA, k=7                   ── P5 out

 33  Detect(P3, P4, P5)            — COCO-pretrained except class outputs
```

### What SDF adds over YOLO11

| Component | Where | What it does |
|---|---|---|
| **Multi-scale DWConv block** | P3 (layers 17–23) | Four parallel depthwise convs with kernels 3/5/7/9 plus the identity, concatenated and fused by a 1×1 conv. Gives the high-resolution level several receptive fields at low cost, aimed at small and thin defects (crazing, scratches, crease). |
| **CAA (Context Anchor Attention)** | P3, P4, P5 outputs | AvgPool 3×3 → 1×1 conv → depthwise 1×k then k×1 strip convs → 1×1 conv → sigmoid gate; output `x + x·gate`. Strip kernels capture long, elongated context (k = 11/9/7 at P3/P4/P5), which suits line-shaped defects such as welding_line, scratches and crease. |
| **Dense PAN** | P4, P5 bottom-up | Each bottom-up level concatenates three inputs: the downsampled lower level, the top-down feature, **and the raw backbone feature** (layers 6 and 8). YOLO11's PAN uses only the first two. |
| **P5 block** | layer 9 | Default `sppf`: yolo11's SPPF at the same index and shape, so it loads COCO weights. Alternative `aifi`: transformer-style intra-scale attention (from RT-DETR), which starts from random init. |

### Pretrained-weight transfer

- `model.load("yolo11s.pt")` loads by layer name, which covers the backbone (0–10) and the top-down FPN. With SPPF at P5, **295 of 550 tensors** start pretrained.
- Ultralytics matches weights by layer *name*, and the SDF Detect head is `model.33` while yolo11s's is `model.23`, so the head was never loaded. `load_coco_head()` (`--coco-head`, on by default) copies every head tensor whose shape matches: **115 of 121 tensors**. Only the class-output layers start random, because the class count differs from COCO. This gives SDF the same starting point as the baselines, which fine-tune their full COCO head.

### Size

| | Params | GFLOPs @640 | GFLOPs @1024 |
|---|---|---|---|
| SDF-YOLO-s (SPPF) | 11.60 M | 27.4 | 70.2 |
| SDF-YOLO-s (AIFI, older) | 13.05 M | 28.6 | — |

---

## 4. Training and evaluation protocol

The same protocol applies to SDF-YOLO and all seven baselines.

| Setting | Value |
|---|---|
| Input size | 640×640 on both datasets (GC10 at 1024 optional) |
| Batch | 32 at 640, 8 at 1024 (set per imgsz/scale so it fits a 12 GB RTX 3060; same for every model) |
| Epochs | **250 for SDF-YOLO, 200 for baselines** |
| Schedule | Cosine LR, ultralytics default optimizer and warmup |
| Augmentation | Ultralytics defaults; mosaic off for the last 20 epochs |
| Sampling | Repeat-factor sampling, t = 0.1, train split only |
| Seed | 42 (`deterministic=True`); `--seeds 42,1,2` for mean ± std |
| Model selection | `best.pt` by **val** mAP50 |
| Test | Evaluated once after training; TTA (flips + 3 scales) reported in separate `*_TTA` columns, never in place of the plain numbers |
| Speed | FPS at batch 1 (pre-process + inference + post-process), RTX 3060, torch 2.11, ultralytics 8.3.174 |

Baselines: YOLOv5s (anchor-free `su` build), YOLOv8s, YOLOv9s, YOLOv10s, YOLO11n, YOLO11s, YOLO12s, all starting from COCO checkpoints.

Safeguards built into `train.py`:

- **GPU queue.** A lock file makes concurrent `train.py` calls wait, because two runs on a 12 GB card make Windows page VRAM to system RAM.
- **VRAM guard.** If the first epoch uses over 97% of VRAM, the run stops and suggests a smaller batch instead of silently slowing down about 5×.
- **Eval-only safety.** `--eval-only` refuses to re-evaluate if `best.pt` is newer than `summary.json`, which would mean the weights were overwritten by an unfinished retrain.

---

## 5. Results — NEU-DET

Test split, 640×640, seed 42. Best value per column in bold.

| Model | mAP50 | mAP75 | mAP50-95 | F1 | P | R | Params (M) | GFLOPs | FPS | val mAP50 |
|---|---|---|---|---|---|---|---|---|---|---|
| **SDF-YOLO (ours)** | 75.23 | 41.59 | 42.46 | 72.61 | 72.99 | 72.54 | 11.60 | 27.4 | 50 | 74.67 |
| YOLO12s | **77.51** | **44.51** | **44.82** | 71.70 | 69.55 | **74.53** | 9.23 | 21.2 | 46 | 75.28 |
| YOLOv9s | 77.27 | 42.10 | 44.10 | **72.96** | 75.57 | 70.77 | 7.17 | 26.7 | 28 | **77.29** |
| YOLO11n | 76.88 | 43.54 | 44.18 | 72.31 | 74.91 | 70.67 | **2.58** | **6.3** | 59 | 75.87 |
| YOLO11s | 76.20 | 42.32 | 43.85 | 71.01 | 77.07 | 67.22 | 9.42 | 21.3 | 54 | 75.28 |
| YOLOv8s | 75.84 | 40.82 | 43.02 | 71.82 | **77.63** | 67.72 | 11.13 | 28.4 | **71** | 74.66 |
| YOLOv5s | 75.70 | 43.25 | 43.72 | 71.29 | 74.49 | 69.24 | 9.11 | 23.8 | 54 | 73.75 |
| YOLOv10s | 75.32 | 42.49 | 42.91 | 72.64 | 75.38 | 70.58 | 7.22 | 21.4 | 68 | 74.68 |

With TTA, SDF-YOLO reaches 75.65 mAP50 / 43.55 mAP50-95 on NEU (YOLOv9s: 77.35 / 45.44).

**SDF-YOLO per-class AP on NEU-DET (test):**

| Class | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|
| AP50 | 40.5 | 81.6 | 89.9 | 85.4 | 60.8 | 93.2 |
| AP50-95 | 13.2 | 44.2 | 61.3 | 53.6 | 25.4 | 57.1 |

**Reading:** SDF-YOLO is 8th of 8 on test mAP50 but its F1 (72.61) is within 0.35 of the best and its recall (72.54) is second-best. Crazing and rolled-in_scale are its weak classes; both are diffuse textures with loosely defined boxes. Training took 110 min. The best val epoch was 127 of 250.

---

## 6. Results — GC10-DET

### At 640 (the main table)

Test split, 640×640, seed 42.

| Model | mAP50 | mAP75 | mAP50-95 | F1 | P | R | Params (M) | GFLOPs | FPS | val mAP50 |
|---|---|---|---|---|---|---|---|---|---|---|
| **SDF-YOLO (ours)** | 64.85 | 28.54 | 31.97 | **65.41** | **74.60** | 61.10 | 11.60 | 27.4 | 54 | 65.87 |
| YOLO12s | **67.21** | 27.37 | 32.42 | 64.71 | 65.97 | **66.96** | 9.23 | 21.2 | 41 | 65.70 |
| YOLO11n | 65.05 | **29.53** | **32.57** | 63.65 | 63.59 | 65.27 | **2.58** | **6.3** | 65 | 68.15 |
| YOLOv10s | 63.99 | 28.71 | 31.86 | 63.80 | 64.78 | 64.23 | 7.22 | 21.4 | 73 | 63.51 |
| YOLOv9s | 63.95 | 26.02 | 30.95 | 63.51 | 65.55 | 63.67 | 7.17 | 26.7 | 32 | 67.62 |
| YOLO11s | 63.86 | 26.63 | 30.33 | 61.70 | 72.79 | 58.05 | 9.42 | 21.3 | 63 | **68.25** |
| YOLOv5s | 63.74 | 28.87 | 32.08 | 61.10 | 60.57 | 62.98 | 9.12 | 23.8 | 74 | 63.45 |
| YOLOv8s | 63.48 | 28.63 | 31.80 | 62.35 | 69.18 | 59.74 | 11.13 | 28.5 | **84** | 64.64 |

**Reading:** SDF-YOLO is 3rd of 8 on test mAP50 and has the **best F1 and best precision** on GC10. It beats five of the seven baselines on mAP50; YOLO12s (+2.4) and YOLO11n (+0.2) are ahead.

### At 1024 (optional higher-resolution setting)

Test split, 1024, batch 8 (YOLO12s batch 4), seed 42.

| Model | mAP50 | mAP75 | mAP50-95 | F1 | mAP50 (TTA) | GFLOPs | val mAP50 |
|---|---|---|---|---|---|---|---|
| YOLO11s | **67.52** | 28.72 | **33.83** | 65.74 | 65.88 | 54.6 | 68.03 |
| YOLO12s | 66.97 | 31.07 | 33.53 | 65.59 | **67.14** | 54.3 | **69.89** |
| **SDF-YOLO (ours)** | 66.40 | 28.12 | 33.35 | **65.96** | 67.01 | 70.2 | 65.12 |
| YOLO11n | 65.83 | 27.82 | 33.14 | 65.83 | 64.48 | 16.2 | 66.64 |
| YOLOv5s | 65.03 | 26.61 | 31.42 | 62.51 | 65.55 | 61.0 | 67.67 |
| YOLOv9s | 64.74 | 31.37 | 33.16 | 64.10 | 65.29 | 68.5 | 66.88 |
| YOLOv8s | 64.45 | **31.46** | 32.84 | 64.58 | 63.37 | 72.8 | 66.11 |
| YOLOv10s | 64.05 | 28.54 | 32.54 | 62.41 | 64.05 | 54.9 | 67.42 |

At 1024 SDF-YOLO gains +1.55 test mAP50 and has the best F1, but its **val** mAP50 drops (65.87 → 65.12). Settings are chosen on val, so the paper table uses 640 for GC10.

### SDF-YOLO per-class AP on GC10-DET (test)

| Class | 640 AP50 | 640 AP50-95 | 1024 AP50 | 1024 AP50-95 |
|---|---|---|---|---|
| punching_hole | 96.6 | 56.4 | 98.5 | 56.4 |
| welding_line | 86.5 | 28.8 | 94.2 | 44.3 |
| crescent_gap | 95.8 | 63.7 | 94.3 | 59.1 |
| water_spot | 71.1 | 39.4 | 77.0 | 40.7 |
| oil_spot | 68.6 | 28.3 | 71.1 | 31.3 |
| silk_spot | 59.7 | 25.8 | 52.7 | 20.6 |
| inclusion | 37.3 | 14.0 | 35.3 | 13.5 |
| rolled_pit | 46.9 | 20.1 | 41.7 | 17.6 |
| crease | **17.2** | 7.8 | **33.2** | 17.8 |
| waist_folding | 68.9 | 35.4 | 66.1 | 32.2 |

**Reading:** the rare classes (crease, rolled_pit) and inclusion dominate the error. The first random-head SDF scored crease 14.8 / rolled_pit 21.2. The COCO head, repeat-factor sampling and SPPF together lifted rolled_pit to 46.9. Crease nearly doubles at 1024 (17.2 → 33.2), and welding_line gains strongly too, consistent with thin, elongated defects benefiting from resolution. At 640 the best val epoch was 69 of 250 (at 1024: epoch 30), so GC10 overfits early. Training took 133 min at 640 and 340 min at 1024.

---

## 7. How SDF-YOLO got here (variant history)

All variants were selected on **val** mAP50; test is shown only for reference.

| Variant | Change | NEU val / test mAP50 | GC10 val / test mAP50 |
|---|---|---|---|
| `sdf_s` | AIFI at P5, random-init Detect head | 74.18 / 76.04 | 63.78 / 63.46 |
| `sdf_s_ch` | + COCO head (`--coco-head`) | 73.94 / 76.59 | 62.59 / 62.63 |
| `sdf_s_ch_wf25` | + 25-epoch warm-freeze (2 stages) | 73.69 / 74.94 | 64.21 / 64.81 |
| **`sdf_s_ch_sppf_e250`** | SPPF at P5 (COCO-pretrained), 250 epochs, no warm-freeze | **74.67** / 75.23 | **65.87** / 64.85 |
| `sdf_s_ch_sppf_i1024_e250` | same, GC10 at 1024 | — | 65.12 / 66.40 |

Key findings along the way:

1. **Random Detect head (root cause).** Initially 58% of SDF's weights, including the entire Detect head, started random while baselines fine-tuned a full COCO model. `--coco-head` fixes this, and it is now the default.
2. **Random new modules disturb the pretrained backbone.** AIFI, the DWConv block, CAA and the dense PAN start random. Warm-freeze (`--warm-freeze 25`) trains them first with the pretrained layers frozen. It helped GC10 but hurt NEU, so it is off by default. (Unfreezing inside a single run caused NaN loss at epoch 26; the two-stage version fixes that.)
3. **SPPF instead of AIFI.** This is the biggest single gain. SPPF at layer 9 loads yolo11s's COCO weights, cuts parameters from 13.05 M to 11.60 M, and gives the best val mAP50 on both datasets.
4. **640 vs 1024 on GC10.** 1024 improves test mAP50 but lowers val, so 640 is the reported setting.

Earlier runs' weights were deleted; their `summary.json` files are in `old_runs_summaries.zip`.

---

## 8. Reproducing the runs

All commands run from `colab_sdf_yolo_v2/` on a CUDA GPU.

```powershell
# 0) GC10 data (once) — NEU-DET is expected already in datasets/neu/
python prepare_gc10_native.py --raw <path to raw GC10-DET>

# 1) SDF-YOLO, reported setting
python train.py --data neu  --model sdf      # -> runs/neu_sdf_s_ch_sppf_e250
python train.py --data gc10 --model sdf      # -> runs/gc10_sdf_s_ch_sppf_e250

# 2) One baseline, or everything (skips finished runs)
python train.py --data gc10 --model yolo12s.pt
python train.py --data all  --model all --skip-done

# 3) Paper tables and figures -> ../paper_results_v2/
python make_comparison.py
```

### Useful flags

| Flag | Effect | Run-name suffix |
|---|---|---|
| `--imgsz 1024` | Higher resolution (batch auto → 8) | `_i1024` |
| `--seeds 42,1,2` | One run per seed; table shows mean ± std | `_seed1`, `_seed2` |
| `--scale m` | SDF-YOLO-m; baselines become their m versions (`yolo11s.pt` → `yolo11m.pt`) | `sdf_m` |
| `--rfs 0.3` / `--rfs 0` | Stronger / no oversampling | `_rfs0.3` / `_rfs0` |
| `--p5 aifi` | Original AIFI P5 block | no `_sppf` |
| `--no-coco-head` | Random-init Detect head (original) | no `_ch` |
| `--warm-freeze 25` | Two-stage training | `_wf25` |
| `--epochs N` | Override 250 (SDF) / 200 (baselines) | `_eN` |
| `--no-tta` | Skip TTA evaluation | — |
| `--eval-only` | Re-evaluate an existing `best.pt` on test | — |
| `--skip-done` / `--redo` | Skip / force-retrain runs that already have `summary.json` | — |

### Outputs per run (`runs/<name>/`)

- `weights/best.pt` (selected on val), `results.csv` (per-epoch curves)
- `summary.json`: test metrics, per-class AP50 / AP75 / AP50-95, TTA numbers, val best epoch, config (including RFS stats and COCO-head transfer counts), environment, training time
- `<name>_test/`: confusion matrix and PR curves on test
- `runs/results_table.csv`: one row per run, rebuilt from every `summary.json`

### Remaining paper runs (`run_final.ps1`)

```powershell
powershell -ExecutionPolicy Bypass -File run_final.ps1          # all steps
powershell -ExecutionPolicy Bypass -File run_final.ps1 -Step 2  # one step
```

| Step | What | Time (RTX 3060) |
|---|---|---|
| 1 | TTA re-evaluation of the existing 640 runs (no training) | ~30 min |
| 2 | NEU, seeds 1 and 2: SDF + YOLOv9s / YOLO11n / YOLO11s / YOLO12s | ~16 h |
| 3 | NEU, m-scale: SDF-YOLO-m vs YOLO11m / YOLOv9m / YOLO12m | ~12 h |
| 4 | GC10 at 1024: SDF + YOLO11s / YOLO11n / YOLOv9s | ~65 h |

Do not use **Run All** in the GC10 notebook. It retrains all eight models.

---

## 9. Honest status and caveats

- **Single seed.** Most gaps between models are 1–2 mAP50 points, which is within typical seed-to-seed variation. No ranking should be quoted until the 3-seed runs (`run_final.ps1` step 2) are done and reported as mean ± std.
- **Where SDF-YOLO stands now:**
  - NEU-DET: lowest test mAP50 of the eight models (75.23 vs 77.51 for YOLO12s), but 2nd-best recall and an F1 within 0.35 of the best.
  - GC10-DET: 3rd on test mAP50 (64.85 vs 67.21 for YOLO12s), with the **best F1 (65.41) and precision (74.60)**. At 1024 it also has the best F1.
- **Epoch budget.** SDF trains for 250 epochs and baselines for 200. The paper must state this, or the baselines should be given 250 too.
- **Efficiency.** At 11.6 M params / 27.4 GFLOPs, SDF-YOLO is heavier than YOLO11n (2.58 M / 6.3 G) and will not beat it on size or speed. The fair claim is accuracy within the s-size class.
- **Rare classes.** GC10 crease (17.2 AP50 at 640) and inclusion (37.3) remain the main weaknesses.
- **No test-set tuning.** Every setting above was chosen on val. Picking settings by test, giving baselines weaker settings, or reporting only the best seed would invalidate the comparison.
