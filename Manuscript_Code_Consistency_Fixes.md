# Manuscript ↔ code consistency fixes (resolves every `% TODO-VERIFY`)

Checked against `new_model_code.ipynb`, the exact code that produced the 22 Oct 2026 results, and against the run logs.
The manuscript was only partly available: the pasted text stops at the dataset-split table. Sections after that point still need the same check.

**Summary.** Every parameter count in the layer-wise table matches the code exactly:

| Component | Parameters |
|---|---:|
| Lateral C5 | 163,968 |
| Lateral C4 | 73,856 |
| Lateral C3 | 24,704 |
| Lateral C2 | 18,560 |
| AMFF, each | 54,003 |
| Smooth, each | 147,584 |
| CSAF | 214,080 |
| SEAM | 5,377 |
| Classifier | 774 |
| Backbone | 2,257,984 |
| **Total** | **3,364,064** |

But the **equations and figures for AMFF, CSAF and SEAM describe modules that are not in the code**. The FPN/AMFF ordering in Algorithm 1 is also wrong. The preprocessing description and three claims in the text do not match the code or the results either. These must be fixed before submission: a reviewer who re-derives 54,003 or 214,080 from the current equations will not get those numbers.

---

## 1. AMFF: equations, text and Fig. `amff.png` are wrong

**What the code does** (`class AMFF`, `mode="both"`). AMFF *is* the top-down merge: it replaces the FPN addition. It takes the coarser level and the lateral feature as two inputs.
- The coarser level is upsampled (nearest neighbour) to the lateral resolution.
- Two attention gates are computed from that upsampled map, in parallel:
  - a channel gate: GAP, then FC 128→16 with ReLU, then FC 16→128 with sigmoid;
  - a spatial gate: channel-wise max and mean, then a 7×7 convolution with sigmoid.
- Two separately gated copies of the upsampled map are concatenated with the lateral feature and fused by a 1×1 convolution, BN and ReLU.

There is **no depth-wise convolution and no dilation in AMFF**.
Parameter check: 2,064 + 2,176 + 99 + 49,152 + 512 = **54,003**. This matches the table.

Replace the AMFF equations, from `Each backbone feature is first projected...` through `\mathbf{F}'=\mathbf{F}\odot\mathbf{m}`, with:

```latex
Each backbone feature is first projected to 128 channels,
$\mathbf{L}_l=\operatorname{Conv}_{1\times1}(\mathbf{C}_l)$, $l=2,\ldots,5$, and the
top-down pathway starts from $\tilde{\mathbf{P}}_5=\mathbf{L}_5$. At levels $l=4,3,2$ the plain
FPN addition is replaced by AMFF, which merges the coarser level into the lateral
feature. The coarser map is first upsampled to the lateral resolution,
\begin{equation}
\mathbf{H}_l=\operatorname{Up}_{\mathrm{nearest}}\!\big(\tilde{\mathbf{P}}_{l+1}\big),
\end{equation}
and two attention gates are computed from it in parallel: a channel gate and a spatial gate,
\begin{equation}
\mathbf{a}^{c}_l=\sigma\!\Big(\mathbf{W}_2\,\delta\big(\mathbf{W}_1\operatorname{GAP}(\mathbf{H}_l)\big)\Big),
\qquad
\mathbf{a}^{s}_l=\sigma\!\Big(\operatorname{Conv}_{7\times7}\big([\operatorname{Max}_c(\mathbf{H}_l);\operatorname{Mean}_c(\mathbf{H}_l)]\big)\Big),
\end{equation}
with $\mathbf{W}_1\in\mathbb{R}^{16\times128}$ and $\mathbf{W}_2\in\mathbb{R}^{128\times16}$.
The two gated copies of $\mathbf{H}_l$ are concatenated with the lateral feature and fused:
\begin{equation}
\tilde{\mathbf{P}}_l=\delta\!\Big(\operatorname{BN}\big(\operatorname{Conv}_{1\times1}
[\,\mathbf{H}_l\odot\mathbf{a}^{c}_l\,;\ \mathbf{H}_l\odot\mathbf{a}^{s}_l\,;\ \mathbf{L}_l\,]\big)\Big),
\qquad l=4,3,2 .
\end{equation}
After the top-down pass, each merged level is smoothed,
$\mathbf{P}_l=\delta\big(\operatorname{Conv}_{3\times3}(\tilde{\mathbf{P}}_l)\big)$ for $l=2,3,4$, and
$\mathbf{P}_5=\mathbf{L}_5$.
```

Replace the AMFF novelty paragraph (`AMFF should not be interpreted...`) with:

```latex
AMFF should not be interpreted as a new attention mechanism. Its two gates are the
channel gate of squeeze-and-excitation~\cite{hu2018squeeze} and the max/mean spatial gate
of CBAM~\cite{woo2018cbam}. The difference lies in composition and placement. CBAM applies
the two gates sequentially to one tensor. AMFF applies them in parallel to the upsampled
coarser level, keeps the two gated copies separate, and lets a $1\times1$ convolution fuse
them with the lateral feature inside the FPN merge. AMFF is therefore a task-specific
fusion design. Its value is judged by the controlled multi-seed ablation in
Section~\ref{sec:res_abl}, which finds no measurable benefit over SE, ECA, CBAM or
attention-free fusion.
```

**Figure `amff.png` must be redrawn** to show this structure: two inputs (P_{l+1}, L_l), nearest-neighbour upsampling, a channel gate (GAP→FC→FC→σ) and a spatial gate (max/mean→7×7→σ) in parallel, concat [gated, gated, L_l], then 1×1 conv, BN and ReLU. The current figure shows depth-wise local/dilated branches, which do not exist in the code.

---

## 2. Algorithm 1, line 5, and the architecture paragraph

The current line is `P_l ← Smooth(AMFF(L_l + Up(P_{l+1})))`. In the code, AMFF *is* the merge (there is no addition before it), and smoothing is applied **after** the full top-down pass, to P4, P3 and P2. Replace with:

```latex
\State $\tilde{\mathbf{P}}_5 \gets \mathbf{L}_5$;\quad
$\tilde{\mathbf{P}}_l \gets \mathrm{AMFF}\big(\tilde{\mathbf{P}}_{l+1},\mathbf{L}_l\big)$, $l=4,3,2$
\State $\mathbf{P}_l \gets \mathrm{ReLU}\big(\mathrm{Conv}_{3\times3}(\tilde{\mathbf{P}}_l)\big)$, $l=4,3,2$;\quad $\mathbf{P}_5 \gets \tilde{\mathbf{P}}_5$
```

In the architecture paragraph, replace `AMFF and a $3\times3$ smoothing convolution are applied at levels 4, 3 and 2, and CSAF fuses the levels top-down to a $50\times50\times128$ representation.` with:

```latex
AMFF performs the top-down merge at levels 4, 3 and 2, a $3\times3$ convolution then smooths
each merged level, and CSAF resamples all four levels to $50\times50$ and fuses them into one
$50\times50\times128$ representation.
```

Delete `% TODO-VERIFY: order AMFF/Smooth/CSAF against the code`.

---

## 3. CSAF: equations, text, table row and Fig. `csaf.png` are wrong

**What the code does** (`class CSAF`, `mode="csaf"`). CSAF is **not** top-down and has **no** depth-wise spatial gate or GAP→1×1 global gate.
- All four levels are resized (bilinear) to the P2 resolution.
- A joint descriptor concatenates the GAP of every level (4 × 128 = 512).
- A two-layer MLP (512→64 with ReLU, then 64→512) produces one logit per level and channel.
- A **softmax across the four levels**, separately for each channel, makes the levels compete.
- The levels are summed with these weights, followed by a 3×3 convolution (no bias), BN and ReLU.

Parameter check: 32,832 + 33,280 + 147,456 + 512 = **214,080**. This matches the table. The per-level weights in the CSAF-weights table are exactly these softmax weights (mean over channels and over the images of each class). So the table *is* produced by this module, and that TODO is resolved once the equations below replace the current ones.

Replace the CSAF paragraph and equations with:

```latex
CSAF (Fig.~\ref{fig:csaf}) fuses the four pyramid levels into a single map at the finest
resolution, with input-dependent weights that compete across scales. Each level is resampled
to $50\times50$ by bilinear interpolation, $\mathbf{R}_l=\operatorname{Resize}(\mathbf{P}_l)$,
and a joint cross-scale descriptor is formed from all levels,
\begin{equation}
\mathbf{d}=\big[\operatorname{GAP}(\mathbf{R}_2);\operatorname{GAP}(\mathbf{R}_3);
\operatorname{GAP}(\mathbf{R}_4);\operatorname{GAP}(\mathbf{R}_5)\big]\in\mathbb{R}^{512}.
\end{equation}
A two-layer perceptron maps $\mathbf{d}$ to one logit per level and channel,
$\mathbf{Z}=\mathbf{W}_b\,\delta(\mathbf{W}_a\mathbf{d}+\mathbf{b}_a)+\mathbf{b}_b$, reshaped to
$\mathbb{R}^{4\times128}$ ($\mathbf{W}_a\in\mathbb{R}^{64\times512}$,
$\mathbf{W}_b\in\mathbb{R}^{512\times64}$). The logits are normalized with a softmax across the
four levels, separately for every channel $c$:
\begin{equation}
w_{l,c}=\frac{\exp(Z_{l,c})}{\sum_{l'=2}^{5}\exp(Z_{l',c})},
\qquad
\mathbf{F}_{\mathrm{CSAF}}=\delta\!\Big(\operatorname{BN}\big(\operatorname{Conv}_{3\times3}
\textstyle\sum_{l=2}^{5}\mathbf{w}_l\odot\mathbf{R}_l\big)\Big).
\end{equation}
Because every gate is computed from all levels and the levels compete through the softmax, the
mixing proportion adapts to each image. This is the difference from fixed addition or
concatenation. Its measured effect is a small gain in mAP$_{50:95}$
(Section~\ref{sec:res_abl}), and the learned weights are class-dependent
(Table~\ref{tab:csaf_weights}).
```

Delete both `% TODO-VERIFY` comments in the CSAF paragraph.
In the layer table, change the CSAF row's operation from `Cross-scale spatial attention fusion` to `Cross-scale adaptive fusion (softmax over levels)`.
**Figure `csaf.png` must be redrawn**: four inputs P2–P5 resized to 50×50, then GAP of each, concat (512), MLP 512→64→512, reshape 4×128, softmax over levels, weighted sum, 3×3 conv, BN and ReLU.

---

## 4. SEAM: one-sentence description, and the dilation rates in Table 3

**What the code does** (`class SEAM`). Three parallel 3×3 depth-wise convolutions with **dilation rates 1, 3 and 5**, each followed by BN and ReLU. Their outputs are concatenated (384 channels), and a 1×1 convolution to one channel with a sigmoid gives a spatial map *a*. The output is applied residually, **F ⊙ (1 + a)**.
Parameter check: 3 × 1,152 + 3 × 512 + 385 = **5,377**. This matches.

Replace `The SEAM stage (5{,}377 parameters) enhances the spatial response ... % TODO-VERIFY ...` with:

```latex
SEAM (5{,}377 parameters) widens the receptive field without further downsampling. Three
parallel $3\times3$ depth-wise convolutions with dilation rates 1, 3 and 5, each followed by
BN and ReLU, are concatenated and projected by a $1\times1$ convolution and a sigmoid to a
single-channel, class-agnostic spatial map $\mathbf{a}$, which is applied residually:
\begin{equation}
\mathbf{F}_{\mathrm{SEAM}}=\mathbf{F}_{\mathrm{CSAF}}\odot(1+\mathbf{a}).
\end{equation}
Its measured effect is not significant (Section~\ref{sec:res_abl}).
```

In the hyperparameter table, replace `AMFF dilation rates & $\{1,2\}$` with these three rows:

```latex
AMFF gates & Channel (reduction 8) and $7\times7$ spatial, in parallel \\
CSAF & Joint descriptor, MLP $512\to64\to512$, softmax over 4 levels \\
SEAM dilation rates & $\{1,3,5\}$ \\
```

---

## 5. Preprocessing: equation (1) and the preprocessing table are wrong

**What the code does.** It decodes the 8-bit image as one channel and replicates it to 3. It **divides by 255** to [0, 1], applies augmentation (training only) on [0, 1], clips, and then maps to [−1, 1] with 2x − 1. The output range equals Keras MobileNetV2 `preprocess_input`, but the manuscript's "0–255, no division by 255" statement is false for this code. The augmentation details also need correcting:
- brightness adds a random offset δ ∈ [−0.1, 0.1] on the [0, 1] scale, i.e. up to ±10 % of the full intensity range;
- contrast multiplies by a factor in [0.9, 1.1].

Replace equation (1) and the sentence before it with:

```latex
The single channel $\mathbf{I}_g\in\{0,\ldots,255\}^{200\times200\times1}$ is replicated three
times to match the ImageNet-pretrained MobileNetV2 stem and scaled to the range expected by
that stem:
\begin{equation}
\mathbf{I}_{3c}=\operatorname{Concat}(\mathbf{I}_{g},\mathbf{I}_{g},\mathbf{I}_{g})/255,
\qquad
\mathbf{I}=2\,\mathbf{I}_{3c}-1\in[-1,1]^{200\times200\times3},
\end{equation}
which is the same mapping as the Keras MobileNetV2 \texttt{preprocess\_input}.
```

In the augmentation sentence, replace `independent brightness and contrast variation of up to $\pm10\%$` with `a random brightness offset of up to $\pm10\%$ of the intensity range and a random contrast factor in $[0.9,1.1]$, applied on the $[0,1]$ scale before the mapping to $[-1,1]$`.

Preprocessing table, rows to change:

```latex
Input intensity range & $[0,255]\rightarrow[0,1]$ (division by 255) $\rightarrow[-1,1]$ \\
Backbone preprocessing & $2x-1$ (equivalent to MobileNetV2 \texttt{preprocess\_input}) \\
Brightness variation & additive offset, $\pm0.1$ on the $[0,1]$ scale \\
Contrast variation & factor in $[0.9,1.1]$ \\
```

In Algorithm 1, line 2, `PreprocessMV2` can stay if the caption or a footnote says it denotes "/255 then 2x − 1".

---

## 6. Grad-CAM / localization: details to correct or add

**What the code does** (`_cam_batch`, `cam_to_boxes`):
- **Grad-CAM source:** the gradient of the **softmax probability** of the predicted class, taken with respect to the **SEAM output feature map** (50×50×128).
- **CAM computation:** channel weights are the spatial mean of the gradients, followed by ReLU of the weighted sum.
- **Resize and normalisation:** bilinear resize to 200×200, then division by the map's maximum. There is no min-max step, so "normalized to [0, 1]" holds only because of the ReLU.
- **Threshold and morphology:** threshold τ = 0.35 · max. Morphological **closing then opening with a 5×5 kernel**.
- **Region filtering:** 8-connected components; components with **< 64 pixels are discarded**; at most **10 regions** per image.
- **Boxes:** each box is the **axis-aligned bounding box** of its component and carries the **predicted class** as its label.
- **Score:** s_r = p_cls · mean(A over the region). This matches the manuscript's equation.

Replace the first sentence of the localization paragraph with:

```latex
After classification, Grad-CAM~\cite{selvaraju2017grad} is computed for the predicted class
from the gradient of its softmax probability with respect to the SEAM output feature map
($50\times50\times128$). The map is rectified, resized to $200\times200$ by bilinear
interpolation and divided by its maximum to give $A(x,y)\in[0,1]$.
```

Replace `Morphological closing and opening suppress small isolated responses, connected components are extracted, and the minimum enclosing rectangles of the components are taken as the weakly localized regions` with:

```latex
Morphological closing followed by opening with a $5\times5$ kernel suppresses small isolated
responses. 8-connected components smaller than 64 pixels are discarded, and the axis-aligned
bounding box of each remaining component (at most ten per image) is taken as a weakly
localized region labelled with the predicted class
```

Add to the metrics description (the part after the paste cut-off, wherever AP is defined):

```latex
Following the PASCAL VOC convention, boxes marked \emph{difficult} are excluded from matching
(17 of the 867 test boxes, leaving 850). Predictions of each class are ranked by $s_r$ over the
whole test set and greedily matched one-to-one to unmatched ground-truth boxes of the same class
at the given IoU threshold. AP is computed with 101-point interpolation and averaged over the six
classes; mAP$_{50:95}$ averages AP over IoU thresholds $0.50{:}0.05{:}0.95$. The threshold
$\tau=0.35$ was fixed in advance and not tuned on the test split.
```

In Algorithm 2: change `MinEnclosingRect` to `AxisAlignedBBox`, and add the step `discard components with area < 64 px; keep at most 10`.

---

## 7. Dataset section: verified, plus two corrections

Verified from the XML files and the split file:
- 4,189 boxes in total, of which 81 are difficult.
- Training split: 2,899 boxes (60 difficult).
- Validation split: 423 boxes (4 difficult).
- Test split: 867 boxes (17 difficult).
- 210 / 30 / 60 images per class in train / validation / test.

Corrections:
1. `The validation set was used for model selection and hyperparameter tuning` should become `The validation set was used only for checkpoint selection, early stopping and the learning-rate schedule; all hyperparameters, including the CAM threshold, were fixed in advance`. No hyperparameter search was run.
2. Add the per-split box counts above to Table~\ref{tab:dataset_split} (an extra "Boxes" column), or to the text.

---

## 8. Claims in the Introduction and Related Work that contradict the results

| Location | Current text | Problem | Replace with |
|---|---|---|---|
| Intro, 2nd paragraph, last sentence | "the similarity between inclusion and pitted surface, in particular, is a persistent source of confusion for automated systems, as our experiments confirm" | Not what the experiments show: the only test error in 3,600 predictions is **scratches → inclusion**. Inclusion and pitted surface were never confused. | "Thin, elongated dark structures make scratches and inclusions visually similar; the single test error observed across ten trainings of the proposed model is a scratches image classified as inclusion." |
| Intro, paragraph starting "We propose a lightweight..." | "AMFF gates each pyramid feature with a modulation map computed from a local and a dilated depth-wise receptive field; CSAF fuses the levels top-down with local and global sigmoid gates" | Describes modules that are not in the code (Sections 1 and 3) | "AMFF merges each coarser pyramid level into the lateral feature through parallel channel and spatial gates; CSAF fuses all four levels at the finest resolution with input-dependent weights normalized by a softmax across scales;" |
| Related work, last paragraph | "The contribution is the task-oriented integration and ablation-verified justification of these components" | The ablation does **not** verify AMFF or SEAM | "The contribution is a controlled, multi-seed evaluation of these components, which shows which of them measurably help (multiscale fusion, and to a small degree CSAF) and which do not (AMFF, SEAM)." |
| Related work, attention paragraph, last sentence | "...an issue we address by restricting attention to fused pyramid levels only" | Acceptable, but SEAM operates on the fused 50×50 map, so "pyramid levels and the fused map" is accurate | "...an issue we address by restricting attention to the pyramid merges and the fused map only" |
| Related work, last paragraph | "(AMFF)" listed as the only gated fusion | CSAF is the component with a measured gain | "...and deliberately scoped gated fusion (AMFF, CSAF)" |

---

## 9. Small corrections in the layer table and setup

- **Footnote to the layer table.** "of which 1,104,288 are trainable" is the **Phase I** count. Phase II has **2,615,008** trainable parameters (log: `unfroze top 30 backbone layers (52 BatchNorm layers held frozen)`). Write: `...1{,}104{,}288 trainable in Phase~I and 2{,}615{,}008 in Phase~II; 2{,}259{,}776 non-trainable in Phase~I.`
- **52 BatchNorm layers frozen:** verified from the log. Keep.
- **Smooth convolutions:** they include a ReLU and a bias (147,584 = 3·3·128·128 + 128). Write `$3\times3$ convolution + ReLU` in the table.
- **Classifier head:** GAP → Dropout (rate **0.5**) → Dense(6) + softmax. Add "rate 0.5" to the Dropout row.
- **Backbone taps:** state the layer names once for reproducibility: C2 = `block_3_expand_relu`, C3 = `block_6_expand_relu`, C4 = `block_13_expand_relu`, C5 = `out_relu`.

---

## 10. Left open (not in the pasted text, or not verifiable here)

- Everything after the dataset-split table (results, tables, discussion, conclusion, declarations, references) was cut off in the paste. Check it against `Paper_Revision_Results_and_Changes.md`.
- `\tbd{repository URL}` still needs the actual URL.
- The figures `amff.png` and `csaf.png` have to be redrawn by you (Sections 1 and 3). `Architecture.png` should be checked against Section 2: AMFF is the merge, smoothing comes after, and CSAF takes all four levels.

---

# Part B: Status of the 11-item fix list (checked 22 Oct 2026)

Every statement below was checked against the run logs, the saved predictions (`predictions/*.npz`), the checkpoints in `MobileNetV2_NEU_GPU_results.zip` and the notebook code. No result was changed to make the paper consistent.

| # | Item | Status | Evidence / action |
|---|---|---|---|
| 1 | `Table ??` in Sections 5.1/5.2 | **Open: needs the full `.tex`** | The pasted manuscript is cut off inside Section 3. Every `\ref{tab:...}` needs a matching `\label{tab:...}` *after* `\caption`. Compile twice and search the PDF for `??`. |
| 2 | Train/val duplicate (`patches_101` ↔ `patches_105`) | **Disclosed; optional rerun prepared** | Only validation is affected, and the test split has no duplicate. In all 30 protocol runs the selected checkpoint already had **100.0 % validation macro-F1** (`results_per_run.csv`, column `val_macro_f1`). The duplicate could therefore only change the val-loss tie-break between epochs that were all at 100 %, never which test predictions were made. Keeping the current results with the disclosure is defensible. If a duplicate-free rerun is wanted: `python run.py --stage protocol section13 detection ablation_multiseed --drop-val-duplicate`. That gives 179 validation images, a new protocol folder `protocol_4892144be2`, and about 9 h on the RTX 3060. The default mode was verified to still map to the reported `protocol_3925b60d23`. |
| 3 | Seed mismatch in the backbone table | **Fixed in code; correct row available now** | The proposed row came from the validation-chosen seed 2026, while every baseline row uses seed 42. The code now uses the seed-42 run (`include_proposed=per_seed[0]`). The corrected row from the existing logs (Section 13, seed 42) is **Accuracy 100.00, Precision 100.00, Recall 100.00, F1 100.00**, with 3,364,064 params, 1,373.1 M MACs and 170.63 ms CPU latency (`table_efficiency.csv`). The protocol run's seed-42 full model is also 100.00. Replace the 99.72 row, and note "seed 42, same protocol as all rows". |
| 4 | Verify 100.00 ± 0.00 and 99.94 ± 0.12; are they separate experiments? | **Verified: two separate groups** | *Group A, protocol (Section 13b):* checkpoints `proto3925b60d23_full_model_csaf_seam_s{42,1337,2026,7,99}`. All 30 runs (6 models × 5 seeds) were re-scored from raw predictions; each covers exactly the 360 frozen test images, labels match `split_v1.csv`, and the recomputed accuracy is 100.0 %. *Group B, headline run (Section 13):* checkpoints `full_seed{42,1337,2026,7,99}`, accuracies 100 / 100 / 99.72 / 100 / 100, giving 99.94 ± 0.12. Same configuration and protocol, but independent trainings. State this in the paper: "two independent sets of five trainings; one error in 3,600 test predictions". |
| 5 | Classifier vs detector wording | **Done in the pasted part** | Search the rest of the `.tex` for `detect`, `detector`, `detection head`. Allowed only for the YOLO/FCOS baselines. |
| 6 | Ground truth and AP/mAP documentation | **Text ready (Part A §6–7)** | Verified from the XML files: test split 867 boxes, 17 difficult and excluded, so 850 are matched. Greedy one-to-one matching per class over the whole test set, ranked by $s_r$; 101-point AP; mAP over IoU 0.50:0.05:0.95; evaluated on all 360 test images. |
| 7 | 200×200, 3-channel, no 128/224 resize | **Verified, but one sentence is wrong** | The classifier, CAM and FCOS pipelines run at native 200×200: `tf.image.resize`/`cv2.resize` go to `CFG.img_size = 200`, an identity. **But the YOLO baselines train and infer at 224×224** (`yolo_imgsz = 224`, because Ultralytics needs multiples of 32). Change `No spatial resizing to $128\times128$ or $224\times224$ is performed anywhere in this study.` to: `The classification network and the CAM localization operate at the native $200\times200$ resolution throughout; no resizing to $128\times128$ or $224\times224$ is applied. Only the YOLO baselines use $224\times224$, because Ultralytics requires input sizes divisible by 32.` |
| 8 | AMFF/CSAF/SEAM aligned with code | **Mostly done; leftovers below** | The new equations match the code. See the leftover items listed below. Figures `amff.png`, `csaf.png` and `Architecture.png` must be redrawn to match. |
| 9 | Novelty, ablation, null results | **Done in the pasted part** | The paired p-values and the "5 seeds, uncorrected" caveat must also appear in the results section (Part A of `Paper_Revision_Results_and_Changes.md`, §5c). |
| 10 | Speed and YOLO reporting | **Text ready** | Hardware: RTX 3060 12 GB, Intel 10C/20T, TF 2.10.1, batch 1, 20 warm-up + 200 timed runs, median, forward pass only. YOLO: Ultralytics 8.3.174; YOLOv8n / YOLO11n / YOLO12n; train split → `best.pt` on validation → scored once on test; 100 epochs; 224×224; batch 32; seed 42. YOLO26 was not tested; do not list it. SDF-YOLO was **not** part of this run: remove it, or rerun the `sdf_yolo` stage. |
| 11 | Six main sections, less repetition | **Open: needs the full `.tex`** | The pasted part already has Sections 1–3 plus Setup (4). Results/Discussion (5) and Conclusion (6) make six, so Discussion and Limitations must sit inside Section 5. The Introduction repeats the "classification vs post-hoc localization" explanation three times (paragraphs 4, 5, 6 and the Grad-CAM paragraph); merge them into one. |

**Leftovers found in the newly pasted manuscript (Section 3):**

1. The architecture paragraph still contains the old sentence `AMFF and a $3\times3$ smoothing convolution are applied at levels 4, 3 and 2, and CSAF fuses the levels top-down to a $50\times50\times128$ representation.` and the comment `% TODO-VERIFY: order AMFF/Smooth/CSAF against the code`. It contradicts the corrected paragraph below it. Replace it with `AMFF performs the top-down merge at levels 4, 3 and 2, a $3\times3$ convolution then smooths each merged level, and CSAF resamples all four levels to $50\times50$ and fuses them into one $50\times50\times128$ representation.` and delete the comment.
2. `AMFF addresses inter-class similarity by gating ...` states a benefit that the ablation does not support. Write `AMFF is intended to address inter-class similarity by gating ...; Section~\ref{sec:res_abl} tests whether it does.`
3. In the nomenclature, $r$ (dilation rate) is no longer used in any equation. Remove it, or write the SEAM rates as $r\in\{1,3,5\}$ in the SEAM sentence.
4. Table~\ref{tab:hyperparams}: `AMFF gates & Channel (reduction 8) ...`. The hidden width is 16 = 128/8. That is correct, keep it.
