# SDF-YOLO ko NEU aur GC10 dono pe upar lana — kya chalana hai

Sab commands **isi folder (`colab_sdf_yolo_v2/`) ke andar** se chalani hain, usi GPU machine pe jahan pichle 16 runs chale the (RTX 3060).

## Abhi ki haalat (seed 42, test split)

| | SDF-YOLO-s | Sabse achha baseline |
|---|---|---|
| NEU mAP50 | 76.04 (8 mein se 6th) | 77.76 (YOLOv8s) |
| GC10 mAP50 | 63.46 (8 mein se 8th) | 66.87 (YOLO12s) |

GC10 pe sabse badi kamzori: **crease 14.8** aur **rolled_pit 21.2** (baaki models ~26–40).

## Sabse bada kaaran jo mila: SDF ka head random se shuru ho raha tha

Baselines poore COCO-pretrained model se training shuru karte hain. SDF mein sirf backbone pretrained tha — **58% weights, poora Detect head bhi, random se shuru ho rahe the**. Wajah: ultralytics weights ko layer ke *naam* se milata hai, aur SDF mein layers zyada hain to head `model.33` hai, jabki yolo11s mein `model.23`.

Naya flag **`--coco-head`** yolo11s ka pretrained head SDF mein copy karta hai (121 mein se 115 tensors; sirf class-output wali layers nahi, kyunki classes alag hain). Yeh baselines ke barabar ki shuruaat hai, koi unfair fayda nahi. Kam images wali classes (crease, rolled_pit) ko isse sabse zyada fayda hona chahiye.

Isliye **aage ke saare SDF runs `--coco-head` ke saath** hain. Ab yeh `train.py` ka **default** hai, to notebook ki seedhi cell `train(f"--data {DATASET} --model sdf")` bhi dono datasets pe COCO head ke saath train karti hai (run ka naam `neu_sdf_s_ch` / `gc10_sdf_s_ch`; purane `neu_sdf_s` / `gc10_sdf_s` jaise ke taise rehte hain). Purana tareeka chahiye to `--no-coco-head`.

**Ab ka default SDF (1 Oct):** `--p5 sppf` (layer 9 pe AIFI ki jagah yolo11s ka COCO-pretrained SPPF; 295/550 tensors pretrained, 11.62M params), coco-head on, warm-freeze **off** (ek hi stage), aur SDF ke liye **250 epochs** (baselines 200). Run ke naam `neu_sdf_s_ch_sppf_e250` / `gc10_sdf_s_ch_sppf_e250`. Purana AIFI wala model: `--p5 aifi`. Saare purane runs hata diye gaye; unke `summary.json` + `results_table.csv` `old_runs_summaries.zip` mein hain.

**Warm-freeze (ab default nahi, `--warm-freeze 25` se chalta hai):** coco-head ke baad bhi SDF baselines se neeche tha (NEU 76.59, GC10 62.63), kyunki SDF ke naye hisse (AIFI, multi-scale DWConv, CAA, PAN) random se shuru hote hain aur training ki shuruaat mein pretrained backbone ko bigaadte hain. Isliye SDF ab 2 stage mein train hota hai: **stage 1** = 25 epochs, pretrained layers (0–8, 10, 13, 16) freeze, sirf naye hisse seekhte hain (`runs/<naam>_stage1`); **stage 2** = unhi weights se saare layers, 175 epochs, normal warmup ke saath. Kul 200 epochs, baselines jitne. Run ka naam `neu_sdf_s_ch_wf25` / `gc10_sdf_s_ch_wf25`. Band karna ho to `--warm-freeze 0`. (Pehla version ek hi run mein unfreeze karta tha, usme epoch 26 se loss NaN ho gaya tha; woh runs hata diye gaye.)

---

## Phase 1 — Dono datasets pe SDF ki settings dhoondo (~16 ghante)

```powershell
powershell -ExecutionPolicy Bypass -File run_improve.ps1 -Phase 1
```

Yeh 7 runs ek ke baad ek chalata hai (sab SDF-YOLO-s, seed 42):

| # | Dataset | Flags | Kya test ho raha hai | Time (andaaza) |
|---|---|---|---|---|
| 1 | NEU | `--coco-head` | Sirf head fix | ~1.5 ghante |
| 2 | GC10 | `--coco-head` | Sirf head fix | ~2 ghante |
| 3 | GC10 | `--coco-head --rfs 0.3` | Rare classes zyada baar dikhana | ~2 ghante |
| 4 | NEU | `--coco-head --epochs 150` | Kam epochs | ~1 ghanta |
| 5 | GC10 | `--coco-head --epochs 120` | Kam epochs (200 pe overfit ho raha tha) | ~1 ghanta |
| 6 | GC10 | `--coco-head --imgsz 1024 --batch 8` | Badi image, patle/chhote defects | ~4–5 ghante |
| 7 | GC10 | `--coco-head --imgsz 1024 --batch 8 --rfs 0.3` | 3 + 6 saath | ~4–5 ghante |

Pehle 3 runs (~5 ghante) sabse zaroori hain. Unke baad `runs/` Claude ko dikha doge to aage ka faisla jaldi ho jayega.

- Beech mein band ho jaye (crash, laptop band) → **wahi command dobara chalao**. Jo runs ho chuke hain wo skip ho jayenge.
- `--imgsz 1024` pe batch **8** hi rakho: batch 16 ko 14 GB chahiye (GPU 12 GB), Windows use chupchaap system RAM mein bhej deta hai aur training ~5x slow ho jaati hai. Batch 8 = 7.1 GB, ~1.15 s/iteration (test kiya).

### Winner kaise chune

Script ke end mein `make_comparison.py` chalta hai. `../paper_results_v2/comparison.md` ki aakhri table ("SDF-YOLO settings") kholo aur **har dataset ke liye** wo run chuno jiska **val mAP50** sabse zyada hai. Test mAP50 dekh ke mat chunna — warna paper ka test number bharosemand nahi rahega.

## Phase 2 — Winner setting pe fair comparison, 3 seeds (har dataset alag)

Winner ke flags `-Winner` mein do. Baselines ko bhi **wahi setting** milti hai (`--coco-head` baselines pe kuch nahi karta, unka head pehle se pretrained hai). Udaharan:

```powershell
# GC10: agar run 7 jeeta
powershell -ExecutionPolicy Bypass -File run_improve.ps1 -Phase 2 -Data gc10 -Winner "--coco-head --imgsz 1024 --batch 8 --rfs 0.3"
# NEU: agar run 1 jeeta
powershell -ExecutionPolicy Bypass -File run_improve.ps1 -Phase 2 -Data neu -Winner "--coco-head"
```

Har dataset pe: SDF-YOLO-s + us dataset ke 3 sabse strong baselines (GC10: YOLO12s, YOLOv10s, YOLO11s; NEU: YOLOv8s, YOLO11n, YOLOv10s), seeds 42, 1, 2 = 12 runs. Jo pehle ho chuke hain wo skip.

Time: 640 pe ~15–20 ghante per dataset; GC10 1024 pe ~2 din. Time kam ho to Ctrl+C karke beech mein rok sakte ho; baad mein wahi command chalao, aage se shuru hoga.

## Phase 2 ke baad: paper table

Main table winner setting pe banao:

```powershell
# udaharan: GC10 winner = 1024 + rfs 0.3, NEU winner = default setting
python make_comparison.py --gc10 "imgsz=1024 rfs=0.3" --neu ""
```

Output `../paper_results_v2/` mein: `comparison.md`, `comparison.tex` (paper mein paste), `comparison.csv`, aur 4 PNG plots. 3 seeds ho to numbers **mean ± std** mein aate hain.

---

## Seedhi baat — "sab aspects mein SDF aage"

- **Accuracy (mAP50, mAP50-95, F1)** mein dono datasets pe aage aana hi target hai, aur upar ke runs isi ke liye hain. Guarantee nahi hai; runs ke baad hi pata chalega.
- **Params / GFLOPs / FPS** mein SDF-YOLO-s (13.05M, 28.6G) YOLO11n (2.58M, 6.3G) se kabhi nahi jeet sakta. Paper mein aam taur pe claim hota hai: *"same size class ke models mein sabse zyada accuracy"*. Agar SDF halka bhi chahiye, to architecture badalni padegi (jaise AIFI aur P5 wala hissa, jo ~5M params hain) — Phase 1 ke baad us pe baat karenge.
- Jo **nahi** karenge, kyunki reviewer pakad lete hain aur paper reject ho sakta hai: test set dekh ke settings chunna, baselines ko kamzor settings dena, ya sirf sabse achha seed dikhana.

Runs ke baad Claude ko bolo **"improve results dekho"**.

## Folder mein kya badla

- `train.py` — naye flags: `--coco-head`, `--seed`, `--skip-done`. Pehle wale 16 runs ke naam aur numbers wahi hain. `results_table.csv` mein ab `val_mAP50` column bhi hai.
- `run_improve.ps1` — upar ke Phase 1 / Phase 2 runs.
- `make_comparison.py` — paper tables + plots; seeds ko mean ± std mein jodta hai.
- Paper mein FPS ke saath GPU **RTX 3060** likhna.

---

## Final runs (7 Oct) — `run_final.ps1`

640 / seed 42 pe saare 8 models NEU pe 75.3–77.5 aur GC10 pe 63.5–67.2 mAP50 ke beech hain; SDF dono pe beech mein hai. Training settings se 85 nahi aata. Ab jo 4 cheezein add hui hain (har ek SDF **aur** baselines dono pe, fair comparison ke liye):

| # | Kya | Default / flag |
|---|---|---|
| 1 | GC10 1024 pe (patle defects) | GC10 ka default imgsz ab **1024**, batch 8; NEU 640 hi |
| 2 | Bada model | `--scale m` → SDF-m, aur `yolo11s.pt` apne aap `yolo11m.pt` banta hai |
| 3 | TTA | default on; table mein alag `mAP50_TTA` / `mAP75_TTA` / `mAP50_95_TTA` columns |
| 4 | 3 seeds | `--seeds 42,1,2` → mean ± std |

Batch apne aap (imgsz, scale) ke hisaab se: 640 → 32 (m: 16), 1024 → 8 (m: 4). Agar batch GPU mein fit nahi hota to run pehle hi epoch mein ruk jaata hai aur chhota batch batata hai (pehle Windows chupchaap RAM mein bhejta tha aur run ghanton atakta tha).

```powershell
powershell -ExecutionPolicy Bypass -File run_final.ps1          # sab steps
powershell -ExecutionPolicy Bypass -File run_final.ps1 -Step 1  # sirf ek step
```

1. TTA re-eval purane 16 runs ka (~30 min) 2. NEU seeds 1,2: SDF + YOLOv9s/11n/11s/12s (~16 h) 3. NEU `--scale m`: SDF-m + YOLO11m/v9m/12m (~12 h) 4. GC10 1024: SDF + YOLO11s/11n/v9s (~65 h).

**GC10 notebook pe Run All mat chalana**: ab woh saare 8 models 1024 pe chalayega (~5 din). Purane 640 wale GC10 runs ke naam alag hain (`_i1024` nahi), woh table mein bane rehte hain.

---

## 15 Oct — GC10 default wapas 640

`train.py` mein GC10 ka default imgsz ab **640** hai (NEU jaisa). Paper table dono datasets pe 640 pe hai, kyunki GC10 pe SDF ka val mAP50 640 pe zyada aaya (65.87 vs 1024 pe 65.12). 1024 chahiye to `--imgsz 1024` do (batch apne aap 8 ho jaata hai). `run_final.ps1` Step 4 mein `--imgsz 1024` ab alag se likha hai. Paper/tables mein model ka naam ab **SDF-YOLO** hai.
