"""SDF-YOLO (and same-protocol baselines) on NEU-DET / GC10-DET, GPU edition.

Differences from the notebook's n-scale run, all aimed at accuracy:
  * model scale selectable (default "s" -- the size class of APG-YOLO's
    YOLO11s baseline, 21.3 GFLOPs), backbone transferred from yolo11{scale}.pt
  * imgsz 640, 200 epochs, cosine LR, mosaic off for the last 20 epochs,
    images cached in RAM
  * repeat-factor sampling (Gupta et al., LVIS 2019) over the TRAIN split only:
    images containing rare classes are repeated, so GC10's crease (51 boxes)
    and rolled_pit (44 boxes) are seen more often. NEU-DET is balanced, so it
    is unaffected at the default threshold.

Model selection (best.pt) uses the val split only. The test split is touched
exactly once, after training, and those are the numbers to report.

Run as a script so CAA pickles as __main__.CAA (the same path the local
notebook uses), which keeps best.pt loadable there.
"""
import argparse
import csv
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path

import torch
import ultralytics
import ultralytics.nn.tasks as _tasks
import yaml
from ultralytics import YOLO
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parent

DATASETS = {
    "neu": ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"),
    "gc10": ("punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot",
             "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"),
}

# Same-protocol baselines: checkpoint -> name used in the results table / paper.
# `--model all` trains SDF-YOLO plus every one of these, in this order.
BASELINES = {
    "yolo11s.pt": "YOLO11s",
    "yolov8s.pt": "YOLOv8s",
    "yolo11n.pt": "YOLO11n",
    "yolov5su.pt": "YOLOv5s",    # ultralytics' anchor-free YOLOv5s ("u" build)
    "yolov9s.pt": "YOLOv9s",
    "yolov10s.pt": "YOLOv10s",
    "yolo12s.pt": "YOLO12s",
}
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{}"

# depth, width, max_channels -- identical to ultralytics' yolo11.yaml
SCALES = {"n": (0.50, 0.25, 1024), "s": (0.50, 0.50, 1024), "m": (0.50, 1.00, 512)}

# 640 on both datasets: the paper table is 640 (SDF val mAP50 on GC10: 640 = 65.87 >
# 1024 = 65.12). GC10 at 1024 is still available with --imgsz 1024 (runs get _i1024).
# Applied to EVERY model of that dataset, baselines included, so the comparison stays fair.
DEFAULT_IMGSZ = {"neu": 640, "gc10": 640}
# Batch that fits a 12 GB RTX 3060 without Windows silently paging VRAM to system RAM
# (measured: SDF-s 640/32 = 10.2 GB, 1024/16 = 14 GB -> paging, 1024/8 = 7.1 GB).
# Same batch for every model at a given (imgsz, scale), again for fairness.
DEFAULT_BATCH = {(640, "n"): 32, (640, "s"): 32, (640, "m"): 16,
                 (1024, "n"): 8, (1024, "s"): 8, (1024, "m"): 4}


def scaled_baseline(model, scale):
    """yolo11s.pt -> yolo11m.pt at --scale m (yolov5su.pt -> yolov5mu.pt), so a bigger
    SDF is always compared with baselines of the same size class. Nano stays nano."""
    if scale == "s" or not model.endswith(".pt"):
        return model
    stem = Path(model).stem
    if stem.endswith("su"):
        return f"{stem[:-2]}{scale}u.pt"
    if stem.endswith("s"):
        return f"{stem[:-1]}{scale}.pt"
    return model


MODEL_LABELS = {Path(scaled_baseline(k, sc)).stem: v[:-1] + sc if v.endswith("s") else v
                for k, v in BASELINES.items() for sc in ("s", "m")}
MODEL_LABELS["SDF-YOLO-s"] = "SDF-YOLO"   # paper name; the m scale stays SDF-YOLO-m


class CAA(torch.nn.Module):
    """Context Anchor Attention -- identical to new_model_code.ipynb code-17-1."""

    def __init__(self, c1, kernel_size=11):
        super().__init__()
        pad = kernel_size // 2
        self.pool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.reduce = torch.nn.Conv2d(c1, c1, 1, bias=False)
        self.h_conv = torch.nn.Conv2d(c1, c1, (1, kernel_size), padding=(0, pad),
                                      groups=c1, bias=False)
        self.v_conv = torch.nn.Conv2d(c1, c1, (kernel_size, 1), padding=(pad, 0),
                                      groups=c1, bias=False)
        self.gate = torch.nn.Conv2d(c1, c1, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        ctx = self.reduce(self.pool(x))
        ctx = self.v_conv(self.h_conv(ctx))
        gate = self.act(self.gate(ctx))
        return x + x * gate


_tasks.CAA = CAA


class _SdfYamlBuilder:
    def __init__(self):
        self.layers = []
        self.next_idx = 0

    def add(self, frm, module, args, repeats=1):
        self.layers.append([frm, repeats, module, list(args)])
        i = self.next_idx
        self.next_idx += 1
        return i


def build_sdf_yaml(nc, scale, out_path, p5="aifi"):
    """Same topology as the notebook's build_sdf_yolo_yaml(), at any scale.
    CAA is not a base ultralytics module, so its channel argument is resolved
    here with the exact width rule parse_model applies to the layers around it.
    p5="sppf" puts yolo11's SPPF back at layer 9 instead of AIFI: same index and
    shape as in yolo11{scale}.pt, so it starts COCO-pretrained (AIFI starts random)."""
    depth, width, max_ch = SCALES[scale]

    def sc(c):
        return make_divisible(min(c, max_ch) * width, 8)

    b = _SdfYamlBuilder()
    i0 = b.add(-1, "Conv", [64, 3, 2])
    i1 = b.add(i0, "Conv", [128, 3, 2])
    i2 = b.add(i1, "C3k2", [256, False, 0.25])
    i3 = b.add(i2, "Conv", [256, 3, 2])
    i4 = b.add(i3, "C3k2", [512, False, 0.25])
    i5 = b.add(i4, "Conv", [512, 3, 2])
    i6 = b.add(i5, "C3k2", [512, True])
    i7 = b.add(i6, "Conv", [1024, 3, 2])
    i8 = b.add(i7, "C3k2", [1024, True])
    i9 = b.add(i8, "SPPF", [1024, 5]) if p5 == "sppf" else b.add(i8, "AIFI", [1024, 8])
    i10 = b.add(i9, "C2PSA", [1024])

    i11 = b.add(i10, "nn.Upsample", [None, 2, "nearest"])
    i12 = b.add([i11, i6], "Concat", [1])
    i13 = b.add(i12, "C3k2", [512, False])
    i14 = b.add(i13, "nn.Upsample", [None, 2, "nearest"])
    i15 = b.add([i14, i4], "Concat", [1])
    i16 = b.add(i15, "C3k2", [256, False])

    i17 = b.add(i16, "Conv", [256, 3, 1])
    i18 = b.add(i17, "DWConv", [256, 3, 1])
    i19 = b.add(i17, "DWConv", [256, 5, 1])
    i20 = b.add(i17, "DWConv", [256, 7, 1])
    i21 = b.add(i17, "DWConv", [256, 9, 1])
    i22 = b.add([i17, i18, i19, i20, i21], "Concat", [1])
    i23 = b.add(i22, "Conv", [256, 1, 1])
    p3 = b.add(i23, "CAA", [sc(256), 11])

    i25 = b.add(p3, "Conv", [512, 3, 2])
    i26 = b.add([i25, i13, i6], "Concat", [1])
    i27 = b.add(i26, "C3k2", [512, False])
    p4 = b.add(i27, "CAA", [sc(512), 9])

    i29 = b.add(p4, "Conv", [1024, 3, 2])
    i30 = b.add([i29, i10, i8], "Concat", [1])
    i31 = b.add(i30, "C3k2", [1024, True])
    p5 = b.add(i31, "CAA", [sc(1024), 7])

    b.add([p3, p4, p5], "Detect", ["nc"])

    cfg = {"nc": nc, "scale": scale, "scales": {scale: [depth, width, max_ch]},
           "backbone": b.layers[:11], "head": b.layers[11:]}
    with open(out_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=None, width=100)
    return out_path


def load_coco_head(model, ckpt_path):
    """ultralytics' load() matches weights by layer *name*, so SDF's Detect head
    (model.33) never received yolo11s' pretrained head (model.23) and started from
    random init, while every baseline fine-tunes its full COCO head. The head's
    input widths (P3/P4/P5) are the same as yolo11's, so copy every tensor whose
    shape matches -- all but the nc-sized class outputs."""
    src_model = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"].float()
    src = src_model.state_dict()
    s_det, d_det = len(src_model.model) - 1, len(model.model.model) - 1
    sd = model.model.state_dict()
    prefix = f"model.{d_det}."
    upd = {k: src[f"model.{s_det}.{k[len(prefix):]}"] for k in sd
           if k.startswith(prefix) and src.get(f"model.{s_det}.{k[len(prefix):]}") is not None
           and src[f"model.{s_det}.{k[len(prefix):]}"].shape == sd[k].shape}
    sd.update(upd)
    model.model.load_state_dict(sd)
    n_head = sum(k.startswith(prefix) for k in sd)
    print(f"COCO Detect head: transferred {len(upd)}/{n_head} tensors "
          f"(model.{s_det} -> model.{d_det})", flush=True)
    return {"transferred": len(upd), "total": n_head}


def pretrained_layer_ids(model, ckpt_path, coco_head):
    """Indices of SDF layers whose every tensor came from the yolo11 checkpoint
    (same name and shape), i.e. the layers that start pretrained. The Detect head
    counts only with --coco-head, and is left trainable anyway: its class outputs
    are new."""
    src = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"].float().state_dict()
    sd = model.model.state_dict()
    ids = []
    # layers >= 17 are SDF's own neck (multi-scale DWConv, CAA, PAN): a shape match
    # there with yolo11's layer of the same index is a coincidence, not the same layer
    for i in range(17):
        keys = [k for k in sd if k.startswith(f"model.{i}.") and sd[k].is_floating_point()]
        if keys and all(k in src and src[k].shape == sd[k].shape for k in keys):
            ids.append(i)
    return ids


def repeat_factor_train_list(ds_dir, threshold, seed=42):
    """LVIS repeat-factor sampling: r(c) = max(1, sqrt(t / f(c))), f(c) = fraction
    of TRAIN images containing class c; each image repeats max over its classes
    (fractional part resolved with a seeded coin flip)."""
    img_dir, lbl_dir = ds_dir / "images" / "train", ds_dir / "labels" / "train"
    img_classes = {}
    for img in sorted(img_dir.glob("*.jpg")):
        lf = lbl_dir / f"{img.stem}.txt"
        rows = lf.read_text().splitlines() if lf.exists() else []
        img_classes[img] = {int(r.split()[0]) for r in rows if r.strip()}

    n = len(img_classes)
    freq = Counter(c for cs in img_classes.values() for c in cs)
    r_class = {c: max(1.0, math.sqrt(threshold / (k / n))) for c, k in freq.items()}

    rng = random.Random(seed)
    lines = []
    for img, cs in img_classes.items():
        r = max((r_class[c] for c in cs), default=1.0)
        k = int(r) + (1 if rng.random() < r - int(r) else 0)
        lines.extend([img.resolve().as_posix()] * k)

    out = ds_dir / f"train_rfs_{threshold:g}.txt"
    out.write_text("\n".join(lines) + "\n")
    return out, {"images": n, "after_rfs": len(lines),
                 "repeat_per_class": {c: round(r, 2) for c, r in sorted(r_class.items())}}


def write_data_yaml(ds_dir, names, train_entry, tag):
    # not "data_{tag}.yaml": ultralytics' check_yolov5u_filename rewrites any path with
    # "yolov5" and "u.yaml" in it, so data_neu_yolov5su.yaml became data_neu_yolov5s.yaml
    y = ds_dir / f"{tag}_data.yaml"
    y.write_text(yaml.safe_dump({
        "path": ds_dir.resolve().as_posix(), "train": train_entry,
        "val": "images/val", "test": "images/test",
        "nc": len(names), "names": list(names)}, sort_keys=False))
    return y


def evaluate_on_test(weights, data_yaml, imgsz, device, project, name, tta=False):
    m = YOLO(str(weights))
    tta_res = {}
    if tta:
        # test-time augmentation: flips + 3 scales, merged. Reported NEXT TO the plain
        # numbers, never instead of them. YOLOv10 (NMS-free) has no TTA in ultralytics
        # and silently falls back to single-scale.
        b = m.val(data=str(data_yaml), split="test", imgsz=imgsz, batch=16, device=device,
                  augment=True, project=project, name=f"{name}_test_tta", exist_ok=True,
                  plots=False, verbose=False).box
        tta_res = {"mAP50_TTA": round(float(b.map50) * 100, 2),
                   "mAP75_TTA": round(float(b.map75) * 100, 2),
                   "mAP50_95_TTA": round(float(b.map) * 100, 2)}
    met = m.val(data=str(data_yaml), split="test", imgsz=imgsz, batch=16, device=device,
                project=project, name=f"{name}_test", exist_ok=True, plots=True, verbose=True)
    speed = m.val(data=str(data_yaml), split="test", imgsz=imgsz, batch=1, device=device,
                  project=project, name=f"{name}_speed", exist_ok=True, plots=False,
                  verbose=False).speed
    ms = speed["preprocess"] + speed["inference"] + speed["postprocess"]
    box = met.box
    cls_names = [m.names[int(c)] for c in box.ap_class_index]

    def per_class(ap):
        return {n: round(float(a) * 100, 1) for n, a in zip(cls_names, ap)}

    return {
        "mAP50": round(float(box.map50) * 100, 2),
        "mAP75": round(float(box.map75) * 100, 2),
        "mAP50_95": round(float(box.map) * 100, 2),
        "F1": round(float(box.f1.mean()) * 100, 2),
        "Precision": round(float(box.mp) * 100, 2),
        "Recall": round(float(box.mr) * 100, 2),
        "GFLOPs": round(float(get_flops(m.model, imgsz)), 1),
        "Params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
        "FPS_bs1": round(1000.0 / ms, 1),
        "per_class_AP50": per_class(box.ap50),
        "per_class_AP75": per_class(box.all_ap[:, 5]),   # IoU thresholds 0.50:0.05:0.95
        "per_class_AP50_95": per_class(box.ap),
        **tta_res,
    }


TABLE_COLS = ["run", "dataset", "model", "mAP50", "mAP75", "mAP50_95", "F1", "Precision",
              "Recall", "GFLOPs", "Params_M", "FPS_bs1",
              "mAP50_TTA", "mAP75_TTA", "mAP50_95_TTA",
              "imgsz", "batch", "epochs", "rfs_threshold", "seed", "val_mAP50", "val_best_epoch"]


def rebuild_results_table(project):
    """results_table.csv = one row per runs/*/summary.json (rebuilt, not appended, so
    re-evaluated runs replace their old row and new columns never misalign)."""
    rows = []
    for s in sorted(Path(project).glob("*/summary.json")):
        d = json.loads(s.read_text())
        t, cfg = d["test_split"], d.get("config", {})
        vb = d.get("val_best") or val_best(s.parent, cfg.get("epochs", 200)) or {}
        rows.append({"run": d["run"], "dataset": d["dataset"],
                     "model": MODEL_LABELS.get(d["model"], d["model"]),
                     **{k: t.get(k, "") for k in TABLE_COLS[3:15]},
                     "imgsz": cfg.get("imgsz", ""), "batch": cfg.get("batch", ""),
                     "epochs": cfg.get("epochs", ""),
                     "rfs_threshold": cfg.get("rfs_threshold", ""),
                     "seed": cfg.get("seed", 42), "val_mAP50": vb.get("mAP50", ""),
                     "val_best_epoch": vb.get("epoch", "")})
    with open(Path(project) / "results_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_COLS)
        w.writeheader()
        w.writerows(rows)


def _try_lock(fh):
    fh.seek(0)
    if platform.system() == "Windows":
        import msvcrt
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def wait_for_gpu(device, max_used_gb, poll_s=60):
    """Queue behind any other training so runs go one at a time, like one Colab
    runtime per run. One run at batch 32 / 640 needs ~10 GB; two on a 12 GB card
    make Windows (WDDM) page VRAM to system RAM -- ~60 s per iteration and finally
    "illegal memory access". The OS drops the lock when the holding process exits
    (even on a crash), so the next queued run then starts on its own."""
    (ROOT / "runs").mkdir(exist_ok=True)
    fh = open(ROOT / "runs" / ".gpu.lock", "a+")
    waited = False
    while True:
        try:
            _try_lock(fh)
            break
        except OSError:
            if not waited:
                print("another train.py run is using the GPU -- waiting in queue, "
                      "this run starts automatically when it finishes...", flush=True)
                waited = True
            time.sleep(poll_s)
    # also wait out GPU jobs that don't take the lock (old runs, other programs such
    # as a DINO pretraining). The GPU must look free on 4 checks over 30 s: a single
    # check can land in another job's short gap (e.g. between its epochs).
    while True:
        samples = []
        for i in range(4):
            free, total = torch.cuda.mem_get_info(device)
            samples.append((total - free) / 2**30)
            if samples[-1] > max_used_gb:
                break
            if i < 3:
                time.sleep(10)
        used_gb = max(samples)
        if used_gb <= max_used_gb:
            break
        if not waited:
            print(f"GPU has {used_gb:.1f} GB in use by other processes (limit {max_used_gb} "
                  f"GB) -- waiting, see `nvidia-smi`...", flush=True)
            waited = True
        time.sleep(poll_s)
    if waited:
        print("GPU free, starting.", flush=True)
    return fh


def ensure_weights(model):
    """Fetch a baseline checkpoint into the project folder if it is missing.
    ultralytics' own downloader refuses when its online check fails (it probes
    1.1.1.1, which VPNs such as Cloudflare WARP can block) even though GitHub
    itself is reachable, so download directly."""
    path = ROOT / model
    if path.exists():
        return path
    import urllib.request
    url = WEIGHTS_URL.format(model)
    part = path.with_suffix(".part")
    for attempt in range(3):
        try:
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, part)
            part.replace(path)
            return path
        except OSError as e:
            print(f"  download failed ({e}), retry {attempt + 1}/3", flush=True)
            time.sleep(10)
    raise SystemExit(f"could not download {model} -- fetch {url} manually into {ROOT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=sorted(DATASETS) + ["all"], required=True,
                    help="'all' runs neu then gc10")
    ap.add_argument("--model", default="sdf",
                    help="'sdf' for SDF-YOLO, an ultralytics checkpoint for a baseline "
                         f"({', '.join(BASELINES)}), or 'all' for SDF-YOLO + every baseline")
    ap.add_argument("--scale", choices=sorted(SCALES), default="s",
                    help="model size for SDF and the baselines alike: with 'm', yolo11s.pt "
                         "trains as yolo11m.pt etc. (nano stays nano)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help=f"default per dataset: {DEFAULT_IMGSZ}")
    ap.add_argument("--epochs", type=int, default=None,
                    help="default: 250 for SDF-YOLO, 200 for the baselines")
    ap.add_argument("--batch", type=int, default=None,
                    help=f"default per (imgsz, scale) so it fits 12 GB: {DEFAULT_BATCH}")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rfs", type=float, default=0.1,
                    help="repeat-factor threshold; 0 disables oversampling")
    ap.add_argument("--cache", default="ram", help="'ram', 'disk' or 'false'")
    ap.add_argument("--device", default="0")
    ap.add_argument("--fraction", type=float, default=1.0, help="smoke tests only")
    ap.add_argument("--seed", type=int, default=42,
                    help="training + oversampling seed")
    ap.add_argument("--seeds", default=None,
                    help="comma list, e.g. 42,1,2: one run per seed (report mean +- std); "
                         "overrides --seed")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                    help="also evaluate the test split with test-time augmentation; reported "
                         "in extra *_TTA columns next to the plain numbers")
    ap.add_argument("--coco-head", action=argparse.BooleanOptionalAction, default=True,
                    help="SDF only: also start the Detect head from yolo11's COCO head, like the "
                         "baselines do (run name gets _ch). On by default; --no-coco-head gives "
                         "the original random-head setup (runs named gc10_sdf_s / neu_sdf_s)")
    ap.add_argument("--p5", choices=["sppf", "aifi"], default="sppf",
                    help="SDF only: block at P5 (layer 9). 'sppf' = yolo11's COCO-pretrained "
                         "SPPF (run name gets _sppf); 'aifi' = the original AIFI transformer, "
                         "which starts from random weights")
    ap.add_argument("--warm-freeze", type=int, default=0,
                    help="SDF only: first train N epochs with the layers that start from yolo11 "
                         "COCO weights frozen, so the randomly initialised SDF parts (AIFI, "
                         "multi-scale DWConv, CAA, PAN) learn first; then a second stage trains "
                         "everything for --epochs minus N. Run name gets _wfN; 0 disables")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-gpu-used", type=float, default=2.0,
                    help="wait while more than this many GB of VRAM are already in use")
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; re-evaluate the run's existing best.pt on the TEST split")
    ap.add_argument("--redo", action="store_true",
                    help="with 'all': also retrain runs that already have a summary.json")
    ap.add_argument("--skip-done", action="store_true",
                    help="single run: exit if it already has a summary.json (for re-runnable scripts)")
    args = ap.parse_args()

    if args.device != "cpu":
        assert torch.cuda.is_available(), (
            "No GPU visible. Colab: Runtime > Change runtime type > T4 GPU.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if not args.allow_shared_gpu:
            _gpu_lock = wait_for_gpu(int(args.device.split(",")[0]), args.max_gpu_used)  # noqa: F841
    print(f"torch {torch.__version__} | ultralytics {ultralytics.__version__}")

    datasets = ["neu", "gc10"] if args.data == "all" else [args.data]
    models = ["sdf", *BASELINES] if args.model == "all" else [args.model]
    models = [scaled_baseline(m, args.scale) for m in models]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    batch_mode = len(datasets) * len(models) * len(seeds) > 1
    assert not (batch_mode and args.name), "--name only works for a single run"

    failed = []
    for data, model, seed in [(d, m, s) for d in datasets for m in models for s in seeds]:
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed
        if run_args.epochs is None:
            run_args.epochs = 250 if model == "sdf" else 200
        if run_args.imgsz is None:
            run_args.imgsz = DEFAULT_IMGSZ[data]
        if run_args.batch is None:
            run_args.batch = DEFAULT_BATCH.get((run_args.imgsz, args.scale),
                                               8 if run_args.imgsz > 640 else 32)
        name = run_name(data, model, run_args)
        if ((batch_mode or args.skip_done) and not args.redo and not args.eval_only
                and (ROOT / "runs" / name / "summary.json").exists()):
            print(f"skip {name}: already done (--redo to retrain)", flush=True)
            continue
        print(f"\n{'#' * 70}\n# {name}\n{'#' * 70}", flush=True)
        try:
            run_one(run_args, data, model)
        except Exception:  # keep going with the rest of an 'all' batch
            if not batch_mode:
                raise
            import traceback
            traceback.print_exc()
            failed.append(name)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"failed runs: {failed}")


def run_name(data, model, args):
    """Default settings keep the original names (gc10_sdf_s, neu_yolo11s, ...);
    each non-default setting adds a suffix, e.g. gc10_sdf_s_i1024_rfs0.3_seed1."""
    base = f"{data}_sdf_{args.scale}" if model == "sdf" else f"{data}_{Path(model).stem}"
    if model == "sdf" and getattr(args, "coco_head", False):
        base += "_ch"
    if model == "sdf" and getattr(args, "p5", "aifi") == "sppf":
        base += "_sppf"
    if model == "sdf" and getattr(args, "warm_freeze", 0):
        base += f"_wf{args.warm_freeze}"
    if args.imgsz != 640:
        base += f"_i{args.imgsz}"
    if args.rfs != 0.1:
        base += f"_rfs{args.rfs:g}"
    if args.epochs != 200:
        base += f"_e{args.epochs}"
    if args.seed != 42:
        base += f"_seed{args.seed}"
    return base


def val_best(run_dir, epochs):
    """Best VAL mAP50 of the last `epochs` rows of results.csv (earlier rows of an
    interrupted-then-restarted run are dropped) -- use this, not test, to pick settings."""
    f = run_dir / "results.csv"
    if not f.exists():
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(f))][-epochs:]
    b = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    return {"mAP50": round(float(b["metrics/mAP50(B)"]) * 100, 2),
            "mAP50_95": round(float(b["metrics/mAP50-95(B)"]) * 100, 2),
            "epoch": int(float(b["epoch"]))}


def vram_guard(trainer):
    """Stop early with a clear message if the batch does not fit in VRAM. On Windows
    (WDDM) CUDA does not raise out-of-memory: it pages to system RAM and the run gets
    ~5x slower without any error, which is how earlier runs crawled for hours."""
    if trainer.epoch > 0 or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(trainer.device).total_memory
    used = torch.cuda.memory_reserved(trainer.device)
    if used > 0.97 * total:
        raise RuntimeError(f"batch {trainer.args.batch} needs {used / 2**30:.1f} GB of the GPU's "
                         f"{total / 2**30:.1f} GB -- Windows would page to system RAM and crawl. "
                         f"Rerun with --batch {max(1, trainer.args.batch // 2)}")


def run_one(args, data, model_arg):
    names = DATASETS[data]
    ds_dir = ROOT / "datasets" / data
    assert (ds_dir / "images" / "train").is_dir(), (
        f"{ds_dir} missing -- for gc10 run prepare_gc10_native.py --raw <raw GC10-DET folder> first")
    is_sdf = model_arg == "sdf"
    name = args.name or run_name(data, model_arg, args)
    label = ("SDF-YOLO" if args.scale == "s" else f"SDF-YOLO-{args.scale}") if is_sdf else Path(model_arg).stem
    project = str(ROOT / "runs")

    rfs_stats = None
    if args.rfs > 0:
        train_txt, rfs_stats = repeat_factor_train_list(ds_dir, args.rfs, seed=args.seed)
        train_entry = train_txt.resolve().as_posix()
        print(f"repeat-factor sampling: {rfs_stats}")
    else:
        train_entry = "images/train"
    data_yaml = write_data_yaml(ds_dir, names, train_entry, name)
    config = {"imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch,
              "rfs_threshold": args.rfs, "rfs": rfs_stats, "cos_lr": True,
              "close_mosaic": 20, "seed": args.seed, "fraction": args.fraction}

    if args.eval_only:
        run_dir = Path(project) / name
        weights = run_dir / "weights" / "best.pt"
        assert weights.exists(), f"{weights} missing -- train this run first"
        s_path = run_dir / "summary.json"
        if s_path.exists() and weights.stat().st_mtime > s_path.stat().st_mtime + 600:
            # e.g. a notebook "Run All" restarted a finished run and was stopped part-way:
            # best.pt is now that unfinished model, the summary still the real one
            raise RuntimeError(f"{weights} is newer than {s_path} -- the weights were overwritten "
                               f"after this run finished; re-evaluating would replace its real "
                               f"numbers. Retrain the run (--redo) to get TTA numbers for it.")
        summary = json.loads(s_path.read_text()) if s_path.exists() else {
            "run": name, "dataset": data, "model": label, "config": config,
            "train_minutes": None}
        summary["test_split"] = evaluate_on_test(weights, data_yaml, args.imgsz, args.device,
                                                 project, name, tta=args.tta)
        s_path.write_text(json.dumps(summary, indent=2))
        report(summary, run_dir, project)
        return

    if is_sdf:
        Path(project).mkdir(exist_ok=True)
        cfg = build_sdf_yaml(len(names), args.scale,
                             Path(project) / f"sdf_yolo_{args.scale}_{args.p5}_{data}.yaml",
                             p5=args.p5)
        config["p5"] = args.p5
        model = YOLO(str(cfg))
        ckpt = ensure_weights(f"yolo11{args.scale}.pt")
        model.load(str(ckpt))
        if args.coco_head:
            config["coco_head"] = load_coco_head(model, ckpt)
    else:
        model = YOLO(str(ensure_weights(model_arg)) if model_arg.endswith(".pt") else model_arg)
    model.add_callback("on_train_batch_end", vram_guard)

    common = dict(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch,
                  workers=args.workers, device=args.device, seed=args.seed, deterministic=True,
                  cos_lr=True, cache=(False if args.cache == "false" else args.cache),
                  fraction=args.fraction, project=project, exist_ok=True)
    epochs = args.epochs
    t0 = time.time()
    if is_sdf and args.warm_freeze:
        # Stage 1: pretrained layers frozen, SDF's new layers learn. Stage 2 is a
        # separate training from stage 1's weights, so it gets ultralytics' normal LR
        # warmup. (Unfreezing inside one run instead let AdamW move all backbone
        # weights at full LR in one step -> loss NaN from that epoch on.)
        freeze = pretrained_layer_ids(model, ckpt, args.coco_head)
        config["warm_freeze"] = {"epochs": args.warm_freeze, "layers": freeze}
        print(f"warm-freeze stage 1: layers {freeze} frozen for {args.warm_freeze} epochs",
              flush=True)
        model.train(freeze=freeze, epochs=args.warm_freeze, close_mosaic=0,
                    patience=args.warm_freeze, name=f"{name}_stage1", **common)
        stage1 = Path(project) / f"{name}_stage1" / "weights" / "last.pt"
        print(f"warm-freeze stage 2: all layers, {epochs - args.warm_freeze} epochs from {stage1}",
              flush=True)
        model = YOLO(str(stage1))
        epochs -= args.warm_freeze
    model.train(epochs=epochs, close_mosaic=20, patience=epochs, name=name, **common)
    train_min = round((time.time() - t0) / 60, 1)

    run_dir = Path(project) / name
    test = evaluate_on_test(run_dir / "weights" / "best.pt", data_yaml, args.imgsz,
                            args.device, project, name, tta=args.tta)

    summary = {
        "run": name, "dataset": data, "model": label,
        "test_split": test,
        "val_best": val_best(run_dir, args.epochs),
        "config": config,
        "env": {"gpu": torch.cuda.get_device_name(0) if args.device != "cpu" else "cpu",
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "python": platform.python_version()},
        "train_minutes": train_min,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    report(summary, run_dir, project)


def report(summary, run_dir, project):
    rebuild_results_table(project)
    test = summary["test_split"]
    print("\n" + "=" * 70)
    label = MODEL_LABELS.get(summary["model"], summary["model"])
    print(f"{label} on {summary['dataset']} -- TEST split")
    for k in ("mAP50", "mAP75", "mAP50_95", "F1", "GFLOPs", "Params_M", "FPS_bs1"):
        print(f"  {k:<10} {test[k]}")
    for k in ("per_class_AP50", "per_class_AP75", "per_class_AP50_95"):
        print(f"  {k}: {test[k]}")
    print(f"  train time: {summary.get('train_minutes')} min | summary -> {run_dir / 'summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
