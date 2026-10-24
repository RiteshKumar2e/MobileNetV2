# NEW_SDF_GPU — SDF-YOLO v2

Self-contained copy of `colab_sdf_yolo_v2/` with a new default model (SDF-YOLO v2) and a
fair, multi-seed training protocol. The original folder is untouched. No metric is
hard-coded anywhere: every table number comes from `runs/*/summary.json`, written after
real training and a single test-split evaluation.

## Quick start (GPU machine)

```bash
pip install -r requirements.txt
# NEU-DET/ and GC10-DET/ (repo root) must sit next to this folder, or pass NEU_VOC / GC10_VOC
bash run_all.sh                                        # Linux / Colab
powershell -ExecutionPolicy Bypass -File run_all.ps1   # Windows
```

| Step | What | Runs |
|---|---|---|
| 0 | `prepare_data.py`: YOLO datasets from the frozen splits (NEU 1260/180/360, GC10 1602/229/461) | — |
| 1 | v2 screening, s @640, seed 42: default / PKI 3,3,3,5,5 / P2 head / no AIFI / v1 + YOLO11s. Pick each dataset's winner by **val** mAP50 (`../paper_results_v3/comparison.md`, ablation table) | 12 |
| 2 | v2 ablation, seeds 42,1,2: full / no PKI / no AIFI / no MSDW + YOLO11s | 30 |
| 3 | High-accuracy setting: **L scale**, 300 epochs, strong aug, GC10 @1024 + rfs 0.3. SDF-YOLO-v2-L (step-1 winner flags via `NEU_FLAGS` / `GC10_FLAGS`, or `-NeuFlags` / `-Gc10Flags`) vs YOLO11l / YOLO12l / YOLOv8l, 3 seeds | 24 |
| 4 | Tables + figures → `../paper_results_v3/` (s @640) and `../paper_results_v3_large/` (step 3), mean ± std over seeds | — |

One step: `STEP=1 bash run_all.sh` or `run_all.ps1 -Step 1`. Finished runs are skipped, so
rerun the same command after a crash. Step 3 is slow (L @1024 on a 12 GB card: ~1 day per
GC10 run); screen with `SEEDS=42` first.

## SDF-YOLO v2 (default `--arch v2`, code in `sdf_v2.py`)

SDF-YOLO combined with APG-YOLO's components (PKI block, AIFI, GFPN), arranged so the
YOLO11 COCO weights still transfer:

| Part | SDF-YOLO v2 |
|---|---|
| Backbone | YOLO11 (COCO) + a **PKI block** after the C3k2 stages at P3/P4/P5 (`--pki-levels 4,6,8`). PKI = Conv3x3 → parallel depthwise kernels (`--pki-kernels 3,5,7,9,11`; APG's best NEU set is `3,3,3,5,5`) → 1×1, gated by CAA. Added as a zero-init residual `x + γ·PKI(x)`, so the pretrained C3k2 is kept, not replaced (APG replaces it and loses the pretraining). |
| P5 | Pretrained SPPF **+ AIFI** (`--no-aifi` drops it) → C2PSA |
| Neck | Pretrained YOLO11 top-down FPN, then a **GFPN-style bottom-up path**: each level concatenates the downsampled lower level, the top-down feature, the raw backbone feature (dense skip) and the upsampled higher level (queen fusion). P3 also gets the stride-4 feature. SDF's MSDW block + CAA (identity-init) on the outputs. |
| Head | YOLO11 Detect, COCO-initialised (97/103 tensors; only the class outputs start random). `--p2-head` adds a stride-4 level for tiny defects (crease, rolled_pit). |
| Size | s: 15.9 M params (16.3 M with P2); L + P2: 31.8 M |
| Weight transfer | Explicit layer-index map (`transfer_mapped`), checked tensor-by-tensor by value; result stored in `summary.json → config.transfer_check` |

`--arch v1` trains the published SDF-YOLO.

## Other changes vs `colab_sdf_yolo_v2/train.py`

| Area | Change |
|---|---|
| Fair protocol | `--epochs` default is **200 for every model** (old: SDF 250, baselines 200). |
| Model size | `--scale l` / `--scale x`; baselines scale with it (yolo11s.pt → yolo11l.pt). |
| Augmentation | `--extra "mixup=0.1 flipud=0.5 ..."` passes ultralytics train args to every model (run name gets `_aug`). Don't pass `close_mosaic` there. |
| v1 ablation flags | `--msdw-kernels`, `--caa-levels`, `--caa-kernels`, `--no-dense-pan`, `--caa-init identity` |
| Error analysis | `summary.json` also has per-class Precision / Recall, per-class TP / FP / FN (conf 0.25, confusion matrix) and the bs-1 latency breakdown. |
| Data | `prepare_data.py` builds both datasets from the repo's VOC copies + `splits/*.csv`. |

## Reporting rules

- Pick settings by **val** mAP50, never by test.
- Report mean ± std over 3 seeds for SDF **and** baselines; compare L with L-size baselines.
- APG-YOLO's 82.4 (NEU) / 67.6 (GC10) used a different split (NEU 90/5/5, imgsz 224, 100 epochs, SGD). Cite them as "reported", not as a same-protocol row.
