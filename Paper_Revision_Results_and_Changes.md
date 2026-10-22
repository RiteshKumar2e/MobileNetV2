# Paper revision: final results and manuscript changes

**Source of every number below:** `MobileNetV2_NEU_GPU_results.zip`, the full run of 22 Oct 2026 (stages `protocol`, `section13`, `detection`, `ablation_multiseed`; `QUICK = False`).
`MobileNetV2_NEU_GPU_quick_smoke.zip` is a 2-epoch smoke test. **Do not use any number from it.**

**Independent checks performed on the results** (not just copied from the tables):
- All 30 runs of the main protocol were re-scored from their saved raw predictions (`predictions/*.npz`).
- Each run predicts exactly the 360 frozen test images. Labels match `split_v1.csv`, and recomputed accuracy equals the reported accuracy.
- The split hash (`389fea71fa3c85b4`) matches the protocol manifest.
- The test set was never used for checkpoint selection (`test_used_for_model_selection: false`).

---

## 0. What the results mean (read first)

1. **Image-level classification on NEU-DET is saturated.** Under one controlled protocol, every MobileNetV2 variant reaches **100.00 ± 0.00 %** test accuracy over 5 seeds. That includes the plain backbone, FPN, AMFF, the full model, and the SE and CBAM controls. 7 of the 8 ImageNet-pretrained backbones in the backbone comparison also reach 100 % at seed 42. Classification accuracy therefore **cannot** distinguish the proposed modules from the baseline. The paper must not claim that any module improves classification.
2. **The old 93.33 % (full model) vs 99.72 % (FPN+AMFF ablation) discrepancy is resolved.** Under the corrected protocol, the full model achieves **99.94 ± 0.12 %** (5 seeds, Section 13 run) and **100.00 ± 0.00 %** (5 seeds, protocol run). That is 1 error in 3,600 test predictions across 10 independent trainings. The 93.33 % value is **not reproducible** and must be removed everywhere. (The previous training code had two checkpoint-selection defects, fixed in this revision. Phase 2 overwrote the best phase-1 checkpoint, and selection on a saturated validation accuracy picked an arbitrary epoch. These are the most likely cause, but this was not separately proven.)
3. **The modules can only be judged on CAM-based (weak) localization.** That is where differences appear (paired over 5 seeds):
   - **FPN clearly helps:** AP50 +2.65 pp (p = 0.018), mAP50:95 +2.41 pp (p = 0.002) over the plain backbone.
   - **CSAF gives a small gain:** mAP50:95 +0.45 pp over FPN+AMFF (p = 0.018). The AP50 change is not significant (+0.22, p = 0.19).
   - **AMFF shows no measurable gain.** Versus plain FPN the change is AP50 −0.19 (p = 0.33). Versus SE, CBAM, ECA and no-attention, all differences are ≤ 0.6 pp with p ≥ 0.32. Coordinate Attention scores higher than AMFF (AP50 +2.18, p = 0.081, not significant).
   - **SEAM shows no measurable gain:** AP50 +0.67 (p = 0.34), mAP50:95 +0.08 (p = 0.71).

   These are 5-seed paired t-tests without multiple-comparison correction. Report them as such.
4. **CAM localization is weak in absolute terms:** AP50 11.04 ± 1.65 %. A supervised detector on the same split reaches 42 % (our FCOS head) and 71–75 % (YOLOv8n/11n/12n). The paper must present CAM localization as **post-hoc weak localization**, never as detection.
5. **Speed:** the full model runs at **7.67 ms/image (130.4 FPS) on an RTX 3060 GPU** and **170.63 ms/image (5.9 FPS) on CPU** (batch 1, forward pass only). "Real-time" is supportable **only** for the GPU setting, stated with the hardware.

**Recommended framing for the revision:** the contribution becomes (i) a rigorous, reproducible evaluation protocol on NEU-DET, (ii) evidence that classification is saturated, so architectural claims must rest on localization, (iii) FPN-based multiscale fusion measurably improves weak localization, with CSAF adding a small mAP50:95 gain, and (iv) an honest negative result: AMFF and SEAM give no measurable benefit over standard attention. Claims that AMFF or SEAM are improvements must be withdrawn.

---

## 1. Experimental setup (paste into Methods / Experimental settings)

> All classification experiments use the NEU-DET dataset (six classes, 300 images per class, 1,800 images, 200×200 single-channel). We use a fixed stratified split of 1,260 training, 180 validation and 360 test images (70/10/20, split seed 42). The split file is released with the code. Images are kept at native 200×200 resolution, and the grey channel is replicated to three channels for the ImageNet-pretrained MobileNetV2 stem. Pixel values are scaled to [−1, 1]. Training-time augmentation (training split only) consists of random horizontal and vertical flips and ±10 % brightness and contrast.
>
> Every classifier is trained with the same two-phase schedule. Phase 1 trains for up to 30 epochs with the backbone frozen (Adam, learning rate 10⁻³). Phase 2 trains for up to 30 epochs with the top 30 backbone layers unfrozen (Adam, 10⁻⁴), with BatchNorm statistics kept frozen. Both phases use batch size 32, categorical cross-entropy with label smoothing 0.1, ReduceLROnPlateau on validation loss (factor 0.5, patience 4, minimum 10⁻⁶) and early stopping with patience 10. The checkpoint is selected on the **validation** split by macro-F1, with ties broken by lower validation loss. The selection rule is fixed in advance. The test split is evaluated once per run, after selection, and is never used for model, checkpoint or hyper-parameter selection.
>
> Each configuration is trained with five seeds (42, 1337, 2026, 7, 99). The seed controls weight initialisation, data shuffling and augmentation; the split is identical across seeds. We report mean ± sample standard deviation (ddof = 1) over seeds and compare variants with paired t-tests over seeds.
>
> Experiments were run with TensorFlow 2.10.1 and Python 3.10.11 on Windows 11 (build 22631). The machine had an NVIDIA GeForce RTX 3060 (12 GB), an Intel 10-core/20-thread CPU and 31.7 GB RAM. GPU execution is not bit-wise deterministic, which is why results are reported over multiple seeds.

**Data-integrity note (add one sentence):**
> A cross-split duplicate check (byte hash plus pixel-confirmed perceptual hash) found one exact duplicate: `patches_101` in the training split and `patches_105` in the validation split. No test image is duplicated in the training or validation splits.

---

## 2. Main classification table (controlled protocol, 5 seeds)

Source: `paper_results/protocol_3925b60d23/table_protocol_main.csv`

| Model | Runs | Accuracy (%) | Macro-F1 (%) | Macro-P (%) | Macro-R (%) | Params (M) | MACs (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 5 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 2.42 | 269.4 |
| MobileNetV2 + FPN | 5 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.20 | 1362.1 |
| MobileNetV2 + FPN + AMFF | 5 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.36 | 1525.2 |
| Full model (FPN + AMFF + CSAF + SEAM) | 5 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.36 | 1373.1 |
| MobileNetV2 + FPN + SE | 5 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.31 | 1470.5 |
| MobileNetV2 + FPN + CBAM | 5 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.31 | 1471.3 |

All per-class F1 scores are 100.0 for every model. Paired differences against the full model are 0.00 for every variant.

**Text for Results:**
> Under the controlled protocol all six configurations classify the 360 held-out test images without error in all five seeds (100.00 ± 0.00 % accuracy and macro-F1). This includes the plain MobileNetV2 backbone and the SE and CBAM controls. Image-level classification on NEU-DET is therefore saturated for ImageNet-pretrained lightweight backbones, and accuracy cannot attribute any benefit to the proposed modules. We consequently evaluate the modules on weak localization (Section 4), where performance is far from ceiling.

---

## 3. Proposed full model: headline run (5 seeds, with localization)

Source: `paper_results/tables/table_multiseed.csv`, `table_per_class_f1_full_seeds.csv`, figure `confusion_full_pooled_5seeds.png`.

| Metric | Mean ± SD (5 seeds) | Per seed (42 / 1337 / 2026 / 7 / 99) |
|---|---:|---|
| Accuracy (%) | 99.94 ± 0.12 | 100 / 100 / 99.72 / 100 / 100 |
| Macro-precision (%) | 99.95 ± 0.12 | |
| Macro-recall (%) | 99.94 ± 0.12 | |
| Macro-F1 (%) | 99.94 ± 0.12 | |

Per-class F1 over 5 seeds: Crazing 100.00 ± 0.00, **Inclusion 99.83 ± 0.37**, Patches 100.00 ± 0.00, Pitted surface 100.00 ± 0.00, Rolled-in scale 100.00 ± 0.00, **Scratches 99.83 ± 0.38**.

**Text:**
> Across five seeds the full model reaches 99.94 ± 0.12 % accuracy. The only error in 1,800 test predictions (seed 2026) is one scratches image classified as inclusion. Both classes contain thin, elongated dark structures, which plausibly explains the confusion. Together with the five protocol runs in Table 2 (100 % each), the full model makes one error in 3,600 test predictions over ten independent trainings.

*(Note for the authors: the protocol run and this run are separate trainings of the same configuration under the same protocol. The single-error difference reflects GPU non-determinism. Quote both, or only the combined statement above.)*

---

## 4. CAM-based weak localization

Source: `tables/table_multiseed.csv`, `tables/summary.json`, `tables/table_per_class_localization.csv`.

Ground truth: the 867 PASCAL-VOC boxes of the 360 test images. Regions are derived post hoc from Grad-CAM maps. No box is used for training or for selecting the classifier.

| Metric (test, 867 GT boxes) | Full model, mean ± SD over 5 seeds |
|---|---:|
| AP50 (%) | 11.04 ± 1.65 |
| AP75 (%) | 3.41 ± 0.24 |
| mAP50:95 (%) | 4.74 ± 0.43 |

The representative seed (2026, chosen on validation) gives AP50 11.26, AP75 3.64 and mAP50:95 5.08, with 612 proposed regions. Precision@0.5 is 11.76 % and recall@0.5 is 8.47 %.
Per-class AP50 for that seed: Pitted surface 56.32, Scratches 10.22, Patches 0.99, Crazing 0.03, Inclusion 0.00, Rolled-in scale 0.00.

**Text:**
> CAM-derived regions localise compact, high-contrast defects reasonably well (pitted surface, AP50 56.3 %). They fail on diffuse or texture-like defects (crazing, rolled-in scale) and on small inclusions (AP50 ≈ 0). Overall AP50 is 11.04 ± 1.65 %. We therefore present these regions as post-hoc weak localization that supports interpretation of the classifier, not as a substitute for a supervised detector (Section 6).

---

## 5. Ablation on weak localization (5 seeds) — Table for "Ablation study"

Source: `tables/table_ablation_multiseed.csv` (studies A and B have 5 seeds; studies C to E have only seed 42). Classification accuracy is 99.89–100 % for every row, so localization is the discriminating metric.

**5a. Module ladder (study A)**

| Variant | FPN | AMFF | CSAF | SEAM | AP50 (%) | AP75 (%) | mAP50:95 (%) |
|---|:-:|:-:|:-:|:-:|---:|---:|---:|
| Baseline (MobileNetV2) | – | – | – | – | 7.02 ± 1.77 | 0.50 ± 0.94 | 1.74 ± 0.87 |
| + FPN | ✓ | – | – | – | 9.66 ± 0.35 | 2.98 ± 1.07 | 4.15 ± 0.37 |
| + FPN + AMFF | ✓ | ✓ | – | – | 9.47 ± 0.31 | 3.18 ± 0.41 | 4.11 ± 0.18 |
| + FPN + CSAF | ✓ | – | ✓ | – | 9.15 ± 0.15 | 3.38 ± 0.56 | 4.38 ± 0.33 |
| + FPN + AMFF + CSAF | ✓ | ✓ | ✓ | – | 9.70 ± 0.28 | 3.59 ± 0.27 | 4.56 ± 0.14 |
| Full model (+ SEAM) | ✓ | ✓ | ✓ | ✓ | 10.36 ± 1.47 | 3.46 ± 0.53 | 4.64 ± 0.43 |

**5b. Fusion attention at the FPN lateral-fusion point (study B; everything else identical)**

| Attention | Accuracy (%) | AP50 (%) | AP75 (%) | mAP50:95 (%) |
|---|---:|---:|---:|---:|
| None (concat + 1×1) | 99.94 ± 0.12 | 10.35 ± 1.39 | 3.74 ± 0.25 | 4.78 ± 0.43 |
| SE | 99.94 ± 0.12 | 10.67 ± 1.54 | 3.69 ± 0.25 | 4.93 ± 0.28 |
| ECA | 100.00 ± 0.00 | 10.24 ± 1.24 | 4.01 ± 0.40 | 5.02 ± 0.55 |
| CBAM | 99.89 ± 0.15 | 9.71 ± 0.25 | 3.60 ± 0.35 | 4.68 ± 0.29 |
| Coordinate Attention | 99.89 ± 0.15 | **12.48 ± 1.89** | **4.23 ± 0.24** | **5.52 ± 0.70** |
| EMA | 99.89 ± 0.15 | 10.43 ± 1.55 | 3.62 ± 0.44 | 4.89 ± 0.60 |
| AMFF (ours) | 100.00 ± 0.00 | 10.31 ± 1.24 | 3.79 ± 0.24 | 4.97 ± 0.36 |

**5c. Paired comparisons (5 seeds, paired t-test, uncorrected)**

| Comparison | ΔAP50 (pp) | p | ΔmAP50:95 (pp) | p |
|---|---:|---:|---:|---:|
| + FPN vs baseline | +2.65 ± 1.52 | 0.018 | +2.41 ± 0.74 | 0.002 |
| Full vs baseline | +3.34 ± 3.06 | 0.071 | +2.90 ± 1.23 | 0.006 |
| + AMFF vs + FPN | −0.19 ± 0.39 | 0.331 | −0.04 ± 0.27 | 0.754 |
| + CSAF vs + FPN + AMFF | +0.22 ± 0.32 | 0.194 | +0.45 ± 0.26 | 0.018 |
| + SEAM (full) vs + FPN + AMFF + CSAF | +0.67 ± 1.38 | 0.341 | +0.08 ± 0.44 | 0.706 |
| AMFF vs no attention | −0.05 ± 1.05 | 0.925 | +0.20 ± 0.34 | 0.273 |
| AMFF vs SE | −0.36 ± 1.61 | 0.639 | +0.04 ± 0.27 | 0.730 |
| AMFF vs CBAM | +0.60 ± 1.35 | 0.377 | +0.29 ± 0.58 | 0.322 |
| AMFF vs ECA | +0.07 ± 0.61 | 0.814 | −0.05 ± 0.37 | 0.790 |
| AMFF vs Coordinate Att. | −2.18 ± 2.10 | 0.081 | −0.54 ± 0.84 | 0.222 |

**Text for the ablation section:**
> Because classification is saturated, the ablation is judged on weak localization. Multiscale fusion is the component with a clear effect. Adding the FPN raises AP50 by 2.65 pp (p = 0.018) and mAP50:95 by 2.41 pp (p = 0.002) over the plain backbone. CSAF yields a small further gain in mAP50:95 (+0.45 pp, p = 0.018) but not in AP50. AMFF does not measurably improve localization over plain FPN fusion (−0.19 pp AP50). It is statistically indistinguishable from SE, ECA, CBAM and attention-free fusion when placed at the same position. Coordinate Attention obtains the highest mean AP50 (12.48 %), although its advantage over AMFF is not significant with five seeds (p = 0.081). SEAM does not produce a significant change. We therefore do not claim AMFF or SEAM as performance-improving components. All tests are paired over five seeds and not corrected for multiple comparisons.

**CSAF weights (descriptive, `tables/table_csaf_weights.csv`, test set, seed-99 model):**

| Class | P2 (50×50) | P3 (25×25) | P4 (13×13) | P5 (7×7) |
|---|---:|---:|---:|---:|
| Crazing | 0.275 | 0.279 | 0.276 | 0.169 |
| Inclusion | 0.315 | 0.262 | 0.262 | 0.160 |
| Patches | 0.286 | 0.248 | 0.372 | 0.093 |
| Pitted surface | 0.259 | 0.286 | 0.370 | 0.086 |
| Rolled-in scale | 0.278 | 0.291 | 0.312 | 0.119 |
| Scratches | 0.303 | 0.271 | 0.308 | 0.117 |

Across the five protocol runs, the maximum deviation of the mean CSAF weight from uniform (0.25) is 0.09–0.14 (`csaf_max_dev_from_uniform`).

> CSAF learns non-uniform level weights. It consistently down-weights the coarsest level P5 (0.09–0.17 against 0.25 for uniform) and gives the largest weight to P4 for patches and pitted surface (≈0.37), the classes with large, blob-like defects. The weighting is therefore class-dependent. However, it translates only into the small mAP50:95 gain reported above.

**Studies C–E (single seed 42 only):** dilation control, AMFF channel/spatial composition, and CSAF fusion strategy. Report these only in an appendix, labelled "single seed, not statistically tested". The seed-to-seed SD of AP50 is 1.2–1.9 pp, which is larger than most of their differences, so no conclusion should be drawn from them.

---

## 6. Detection / localization comparison (test split, same AP code)

Source: `tables/table_final_comparison.csv`, `tables/table_yolo_baselines.csv`.

YOLO protocol (fixes the P0 issue): **trained on the 1,260-image training split; `best.pt` selected on the 180-image validation split; evaluated once on the 360-image test split (867 boxes)**. Settings: 100 epochs, 224×224 input (Ultralytics requires multiples of 32), batch 32, seed 42, Ultralytics 8.3.174. Scoring uses the same AP implementation as every other row.

| Model | Supervision | AP50 (%) | AP75 (%) | mAP50:95 (%) | Params (M) |
|---|---|---:|---:|---:|---:|
| Proposed (CAM-based, full model) | image labels only | 11.26 | 3.64 | 5.08 | 3.36 |
| FCOS head on our backbone (AMFF+CSAF+SEAM) | boxes | 42.03 | 6.17 | 13.29 | 3.97 |
| + PKI + AIFI + GFPN (APG-YOLO-inspired) | boxes | 42.23 | 7.68 | 14.68 | 5.49 |
| YOLOv8n | boxes | 72.30 | 36.67 | 38.53 | 3.01 |
| YOLO11n | boxes | 71.18 | 33.10 | 37.70 | 2.58 |
| YOLO12n | boxes | **74.98** | **39.51** | **41.69** | 2.56 |

The CAM row is the representative seed (2026); its 5-seed mean is AP50 11.04 ± 1.65. The FCOS rows and YOLO rows are single runs (seed 42).

**Text:**
> For context we trained supervised detectors on the same split and scored them with the same AP code. These numbers are not a head-to-head comparison with the weakly supervised CAM regions, which use no box labels. Supervised YOLO models reach 71–75 % AP50, compared with 11 % for CAM-derived regions, which quantifies the cost of not using box supervision. A supervised FCOS head on our backbone reaches 42 % AP50. Adding APG-YOLO-inspired PKI/AIFI/GFPN blocks raises mAP50:95 by 1.4 pp in a single run, a difference we do not treat as conclusive.

**Response to Reviewer 3 (YOLO-11 / YOLO-26):**
> We now report YOLOv8n, YOLO11n and YOLO12n, trained on the training split, selected on the validation split and evaluated on the test split for 100 epochs. YOLO26 was not available in the toolchain used (Ultralytics 8.3.174).

*(Delete any sentence that says YOLO-11 was not evaluated. YOLO11n is now in the table.)*

---

## 7. Efficiency (batch 1, forward pass only, preprocessing excluded)

Source: `tables/table_efficiency.csv`. Settings: 20 warm-up runs and 200 timed runs, median reported. Hardware as in Section 1.

| Model | Params (M) | FLOPs (G) | MACs (G) | Size (MiB) | CPU ms (median) | CPU FPS | GPU ms (median) | GPU FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 2.42 | 0.54 | 0.27 | 9.24 | 84.66 | 11.8 | 4.18 | 239.4 |
| MobileNetV2 + FPN | 3.20 | 2.72 | 1.36 | 12.19 | 144.61 | 6.9 | 5.28 | 189.2 |
| **Full model** | 3.36 | 2.75 | 1.37 | 12.83 | 170.63 | 5.9 | 7.67 | 130.4 |

Pyramid width trade-off (`table_pyramid_cost.csv`, CPU): FPN width 32 → 2.38 M params, 349 M MACs, 9.6 FPS; width 64 → 2.61 M, 563 M MACs, 7.6 FPS; width 128 (used) → 3.36 M, 1,373 M MACs, 5.4 FPS.

**Text:**
> The full model has 3.36 M parameters and 1.37 GMACs at 200×200. On an RTX 3060 it processes one image in 7.67 ms (130 FPS). On the CPU of the same machine it needs 170.6 ms (5.9 FPS). The model is therefore real-time capable on a desktop GPU but not on CPU. Most of the cost comes from running the 128-channel pyramid at 50×50 resolution. Reducing the pyramid width to 32 channels cuts MACs by 75 %.

*(Note: the protocol stage re-measured CPU latency on the same machine and got 190.74 ms. CPU timing varied by about 12 % between sessions. Quote the table above and state the hardware. YOLO latencies in `table_yolo_baselines.csv` are end-to-end Ultralytics predict calls, so they are not comparable to the forward-only numbers above. Do not put them in the same column.)*

---

## 8. Backbone comparison (single seed 42, same protocol)

Source: `tables/table_classification_comparison.csv`. Label it explicitly as **single-run, seed 42**.

| Backbone | Init | Accuracy (%) | Macro-F1 (%) | Params (M) | MACs (M) | CPU ms |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 | ImageNet | 100.00 | 100.00 | 23.60 | 3317.9 | 421.26 |
| VGG16 | ImageNet | 100.00 | 100.00 | 14.72 | 12155.5 | 671.71 |
| DenseNet121 | ImageNet | 99.72 | 99.72 | 7.04 | 2212.5 | 468.92 |
| MobileNet | ImageNet | 100.00 | 100.00 | 3.24 | 430.0 | 71.25 |
| MobileNetV2 | ImageNet | 100.00 | 100.00 | 2.27 | 261.6 | 91.80 |
| MobileNetV3-Large | ImageNet | 100.00 | 100.00 | 3.00 | 188.0 | 75.06 |
| EfficientNetB0 | ImageNet | 100.00 | 100.00 | 4.06 | 348.5 | 134.94 |
| EfficientNetV2-B0 | ImageNet | 100.00 | 100.00 | 5.93 | 624.8 | 161.51 |
| ResNet18 | scratch | 99.44 | 99.44 | 11.19 | 1556.7 | 190.87 |
| ResNet34 | scratch | 97.78 | 97.77 | 21.30 | 3138.3 | 344.36 |
| ShuffleNetV2 | scratch | 97.78 | 97.78 | 1.28 | 127.8 | 40.20 |
| Proposed (full) | ImageNet | 99.72 | 99.72 | 3.36 | 1373.1 | 170.63 |

> Seven of eight ImageNet-pretrained backbones classify the test set without error. Only scratch-trained networks fall below 99.5 %. This supports the conclusion that NEU-DET classification is saturated.

*(The "Proposed (full)" row is the validation-selected seed of the Section 3 run (seed 2026). It is not a separate seed-42 training.)*

---

## 9. Old value → new value (search the manuscript for each)

| Old text / value | Replace with | Where |
|---|---|---|
| Full model **93.33 %** accuracy | **99.94 ± 0.12 %** (5 seeds). Optionally add: 100.00 ± 0.00 % in the controlled protocol | abstract, results, tables, conclusion |
| Ablation FPN+AMFF **99.72 %** | **100.00 ± 0.00 %** (5 seeds, Table 2) | ablation table and text |
| **94.68**, **92.95** (old P/R/F1 values) | macro-P 99.95 ± 0.12, macro-R 99.94 ± 0.12, macro-F1 99.94 ± 0.12 | results table |
| AP50 **9.71 %**, mAP50:95 **4.30 %** | AP50 **11.04 ± 1.65 %**, AP75 **3.41 ± 0.24 %**, mAP50:95 **4.74 ± 0.43 %** | abstract, localization section |
| **867** GT boxes | unchanged (867 boxes on 360 test images) | — |
| **570** proposals / **58** hits (old CAM counts) | 612 proposals, P@0.5 11.76 %, R@0.5 8.47 % (seed 2026), or report only AP | localization text |
| CPU **105.12 ms (~9.5 FPS)** full model | CPU **170.63 ms (5.9 FPS)** and GPU **7.67 ms (130.4 FPS)**, RTX 3060, batch 1 | abstract, efficiency |
| **47.58 ms** (MobileNetV2), **92.79 ms** (+FPN) | 84.66 ms / 4.18 ms GPU; 144.61 ms / 5.28 ms GPU | efficiency table |
| YOLO trained on **180 val images / 423 boxes**, **224×224**, **3 epochs** | trained on 1,260 training images, selected on val, tested on 360 test images / 867 boxes, 224×224, **100 epochs** | YOLO section |
| "YOLO-11 not compared" | YOLO11n: AP50 71.18, mAP50:95 37.70 (plus v8n, v12n) | YOLO prose |
| "real-time" (unqualified) | "real-time on a desktop GPU (RTX 3060, 130 FPS); 5.9 FPS on CPU" | abstract, intro, conclusion |
| "AMFF / SEAM improve performance" | "AMFF and SEAM did not measurably improve localization (paired 5-seed tests)" | contributions, discussion, conclusion |
| "the full model is the best classifier" | "all configurations reach ≥ 99.9 % accuracy; classification does not discriminate between them" | everywhere |
| "detection" / "object detector" for CAM regions | "CAM-derived weak localization" / "post-hoc localization" | title, abstract, method |

After replacing values, search the source again for `93.33`, `99.72`, `94.68`, `92.95`, `9.71`, `4.30`, `105.12`, `9.5 FPS`, `47.58`, `92.79`, `570`, `58`, `3 epochs` and `180 images`. None of them may remain except `99.72` where it is a genuine new value (DenseNet121; full model seed 2026).

---

## 10. Abstract and conclusion: suggested replacement sentences

**Abstract (results part):**
> On NEU-DET we evaluate under a fixed 1,260/180/360 split with validation-only model selection and five seeds. The proposed network (3.36 M parameters, 1.37 GMACs) reaches 99.94 ± 0.12 % test accuracy. Image-level classification is, however, saturated: the plain MobileNetV2 backbone and the SE and CBAM variants also reach 100 %. Evaluated on post-hoc CAM-based weak localization, multiscale pyramid fusion improves AP50 from 7.02 % to 9.66 % (p = 0.018), and cross-scale adaptive fusion adds a small mAP50:95 gain. The attention-modulated fusion and spatial enhancement modules give no measurable benefit over standard attention. The model runs at 130 FPS on an RTX 3060 GPU.

**Conclusion:**
> Classification of the six NEU-DET defect classes is effectively solved by ImageNet-pretrained lightweight backbones, so architectural contributions on this dataset must be judged on localization. Multiscale fusion (FPN) is the component with a clear, statistically supported benefit for weak localization, and CSAF contributes a small additional gain together with interpretable, class-dependent scale weights. AMFF and SEAM did not improve results relative to standard attention under a controlled multi-seed protocol. CAM-derived localization remains far below supervised detectors (AP50 11 % vs 71–75 % for YOLO), so applications that need defect boxes should use box supervision.

---

## 11. Limitations (replace the current section)

- **Single dataset (NEU-DET).** Classification is saturated on it, which limits what accuracy can show about architecture.
- **Weak localization.** CAM-based localization is weak (AP50 11.04 ± 1.65 %) and fails on diffuse classes (crazing, rolled-in scale) and small inclusions.
- **Five seeds only.** Paired t-tests over five seeds have low power and are not corrected for multiple comparisons. Studies C–E (dilation, AMFF composition, fusion strategy), the supervised detectors and the YOLO baselines are single-seed.
- **Hardware-specific timings.** Latency was measured on one machine, with forward pass only. CPU timings varied by about 12 % between sessions.
- **Different YOLO input size.** YOLO baselines use 224×224 input, while our models use the native 200×200.
- **No deterministic GPU execution.** Runs are not bit-reproducible, so variability is reported through seeds.

---

## 12. Reference [1]: corrected entry

The current entry ends in `Springer, ??? (2018)`; the `???` is the missing publisher location. Verified book record: *Fifty Materials That Make the World*, Ian Baker, Springer, Cham, 2018, DOI [10.1007/978-3-319-78766-4](https://doi.org/10.1007/978-3-319-78766-4). "Steel" is a chapter of this book (listed as chapter 41 in library tables of contents).

```text
Baker, I.: Steel. In: Fifty Materials That Make the World. Springer, Cham (2018).
https://doi.org/10.1007/978-3-319-78766-4
```

**Still to check on SpringerLink:** the Steel chapter's own DOI and page range. The pages 215–222 currently cited could not be confirmed online. Keep the page numbers only after checking them on the book's table of contents.

Sources: [Springer book page](https://www.springer.com/gp/book/9783319787640), [library TOC record](https://psnz.umt.edu.my/seal/Record/978-3-319-78766-4/TOC)

---

## 13. Not covered by this run (decide before submission)

- **SDF-YOLO (Sections 17–18)** was not part of this run. If the paper keeps SDF-YOLO numbers, they come from an earlier run. Either re-run the `sdf_yolo` stage under the same split, or remove them.
- **Literature comparison table** (`table_literature_neu_det.csv`) is still empty. Fill it only from the cited papers, with their split protocol, and mark which rows are comparable.
- **Section 8b tuned CAM localization** (threshold tuned on validation) was not run. The CAM numbers above use the fixed default threshold of 0.35.
- **Declarations:** check that the data-availability statement matches the journal portal. The dataset is public NEU-DET, and the split file and code can be released.
