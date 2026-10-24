"""Paper tables + figures from runs/*/summary.json.

Runs that differ only by seed (gc10_sdf_s, gc10_sdf_s_seed1, gc10_sdf_s_seed2)
are grouped and reported as mean +- std. Main table = one setting per dataset
(default imgsz 640, rfs 0.1, 200 epochs; change with --gc10/--neu) for every
model; every SDF config also goes to the ablation table with its VAL mAP50
(pick settings by val, not test).

    python make_comparison.py [--gc10 "imgsz=1024 rfs=0.3"] [--neu "epochs=150"]
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent
DS_TITLE = {"neu": "NEU-DET", "gc10": "GC10-DET"}
LABELS = {"yolo11n": "YOLO11n", "yolo11s": "YOLO11s", "yolo12s": "YOLO12s",
          "yolov10s": "YOLOv10s", "yolov5su": "YOLOv5s", "yolov8s": "YOLOv8s",
          "yolov9s": "YOLOv9s", "SDF-YOLO-s": "SDF-YOLO",
          "yolo11l": "YOLO11l", "yolo12l": "YOLO12l", "yolov8l": "YOLOv8l", "yolov10l": "YOLOv10l",
          "SDF-YOLO-l": "SDF-YOLO-L", "yolo11m": "YOLO11m", "SDF-YOLO-m": "SDF-YOLO-M"}
DEFAULT = {"imgsz": 640, "rfs_threshold": 0.1, "epochs": 200}
SDF_EPOCHS = 200   # new protocol: same epochs for every model
SETTING = {"neu": dict(DEFAULT), "gc10": dict(DEFAULT)}   # main-table setting, --neu/--gc10
METRICS = ["mAP50", "mAP75", "mAP50_95", "F1", "Precision", "Recall"]
COST = ["Params_M", "GFLOPs", "FPS_bs1"]

# reference palette (dataviz skill): blue sequential ramp, categorical slots 1-3
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
C_BASE, C_SDF, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(runs):
    groups = {}
    for s in sorted(Path(runs).glob("*/summary.json")):
        d = json.loads(s.read_text())
        key = re.sub(r"_seed\d+$", "", d["run"])
        d["_dir"] = s.parent
        groups.setdefault(key, []).append(d)
    return groups


def label(d):
    return LABELS.get(d["model"], d["model"])


def is_default(d):
    c = d.get("config", {})
    want = dict(SETTING[d["dataset"]])
    if d["run"].startswith(f"{d['dataset']}_sdf") and want["epochs"] == DEFAULT["epochs"]:
        want["epochs"] = SDF_EPOCHS   # train.py default for SDF-YOLO
    return all(c.get(k, v) == v for k, v in want.items())


def parse_setting(s):
    """'imgsz=1024 rfs=0.3 epochs=200' -> table setting (unnamed keys keep defaults)."""
    out = dict(DEFAULT)
    for tok in s.split():
        k, v = tok.split("=")
        k = {"rfs": "rfs_threshold"}.get(k, k)
        out[k] = float(v) if k == "rfs_threshold" else int(v)
    return out


def agg(ds, k):
    vals = [d["test_split"][k] for d in ds if d["test_split"].get(k) not in (None, "")]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def val_mAP50(d):
    vb = d.get("val_best")
    if vb:
        return vb["mAP50"]
    f = d["_dir"] / "results.csv"
    if not f.exists():
        return None
    n = d.get("config", {}).get("epochs", 200)
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(f))][-n:]
    return round(max(float(r["metrics/mAP50(B)"]) for r in rows) * 100, 2)


def fmt(m, s, bold=False, tex=False):
    if m is None:
        return "-"
    t = f"{m:.2f}" + (f" ± {s:.2f}" if s is not None else "")
    if tex:
        t = t.replace("±", r"$\pm$")
        return rf"\textbf{{{t}}}" if bold else t
    return f"**{t}**" if bold else t


def main_rows(groups, ds):
    """Every model at the dataset's table setting (SETTING[ds]). Several SDF variants
    can match it (e.g. with and without --coco-head): keep the one with the best
    VAL mAP50 -- chosen on val, never on test."""
    rows = []
    for key, runs in groups.items():
        if runs[0]["dataset"] != ds or not is_default(runs[0]):
            continue
        r = {"key": key, "model": label(runs[0]), "n": len(runs), "sdf": key.startswith(f"{ds}_sdf")}
        for k in METRICS + COST:
            r[k] = agg(runs, k)
        r["val"] = statistics.mean(v for v in map(val_mAP50, runs) if v is not None)
        rows.append(r)
    sdf = [r for r in rows if r["sdf"]]
    if len(sdf) > 1:
        keep = max(sdf, key=lambda r: r["val"])
        rows = [r for r in rows if not r["sdf"] or r is keep]
    rows.sort(key=lambda r: (not r["sdf"], -r["mAP50"][0]))
    return rows


def write_tables(groups, out):
    csv_rows, md, tex = [], [], []
    for ds in ("neu", "gc10"):
        rows = main_rows(groups, ds)
        if not rows:
            continue
        best = {k: max(r[k][0] for r in rows if r[k][0] is not None) for k in METRICS}
        seeds = max(r["n"] for r in rows)
        md += [f"## {DS_TITLE[ds]} — test split"
               + (f" (mean ± std over {seeds} seeds where available)" if seeds > 1 else " (seed 42, single run)"),
               "", "| Model | mAP50 | mAP75 | mAP50-95 | F1 | P | R | Params (M) | GFLOPs | FPS | val mAP50 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        tex += [r"\begin{table}[t]\centering\small",
                rf"\caption{{Detection results on the {DS_TITLE[ds]} test split.}}",
                r"\begin{tabular}{lcccccccc}\toprule",
                r"Model & mAP50 & mAP75 & mAP50-95 & F1 & P & R & Params (M) & GFLOPs \\ \midrule"]
        for r in rows:
            name = f"**{r['model']}** (ours)" if r["sdf"] else r["model"]
            cells = [fmt(*r[k], bold=abs(r[k][0] - best[k]) < 1e-9) for k in METRICS]
            md.append(f"| {name} | " + " | ".join(cells)
                      + f" | {r['Params_M'][0]:.2f} | {r['GFLOPs'][0]:.1f} | {r['FPS_bs1'][0]:.0f}"
                      + f" | {r['val']:.2f} |")
            tname = rf"{r['model']} (ours)" if r["sdf"] else r["model"]
            tex.append(f"{tname} & " + " & ".join(
                fmt(*r[k], bold=abs(r[k][0] - best[k]) < 1e-9, tex=True) for k in METRICS)
                + rf" & {r['Params_M'][0]:.2f} & {r['GFLOPs'][0]:.1f} \\")
            csv_rows.append({"dataset": ds, "model": r["model"], "seeds": r["n"],
                             **{k: round(r[k][0], 2) for k in METRICS + COST},
                             **{f"{k}_std": (round(r[k][1], 2) if r[k][1] is not None else "")
                                for k in ("mAP50", "mAP50_95")},
                             "val_mAP50": round(r["val"], 2)})
        tex += [r"\bottomrule\end{tabular}\end{table}", ""]
        md.append("")

    # ablation: every SDF config (incl. default), ranked by VAL mAP50
    abl = []
    for key, runs in groups.items():
        if "_sdf_" not in key:
            continue
        c = runs[0].get("config", {})
        abl.append((runs[0]["dataset"], key, c, len(runs), agg(runs, "mAP50"), agg(runs, "mAP50_95"),
                    statistics.mean(v for v in map(val_mAP50, runs) if v is not None),
                    {cl: statistics.mean(d["test_split"]["per_class_AP50"].get(cl, 0) for d in runs)
                     for cl in runs[0]["test_split"]["per_class_AP50"]}))
    if abl:
        md += ["## SDF-YOLO settings (choose by **val** mAP50; test shown for reference)", "",
               "| Dataset | Run | COCO head | imgsz | rfs | epochs | seeds | **val mAP50** | test mAP50 | test mAP50-95 | rarest classes AP50 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for ds, key, c, n, m50, m5095, v, pc in sorted(abl, key=lambda a: (a[0], -a[6])):
            rare = ", ".join(f"{k} {pc[k]:.1f}" for k in ("crease", "rolled_pit") if k in pc) or \
                ", ".join(f"{k} {v_:.1f}" for k, v_ in sorted(pc.items(), key=lambda x: x[1])[:2])
            md.append(f"| {DS_TITLE[ds]} | {key} | {'yes' if c.get('coco_head') else 'no'} | "
                      f"{c.get('imgsz')} | {c.get('rfs_threshold')} | "
                      f"{c.get('epochs')} | {n} | **{v:.2f}** | {fmt(*m50)} | {fmt(*m5095)} | {rare} |")
        md.append("")

    (out / "comparison.md").write_text("# SDF-YOLO v2 comparison\n\n" + "\n".join(md), encoding="utf-8")
    (out / "comparison.tex").write_text("% needs \\usepackage{booktabs}\n" + "\n".join(tex), encoding="utf-8")
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_per_class(groups, out):
    cmap = LinearSegmentedColormap.from_list("blue", SEQ)
    for ds in ("neu", "gc10"):
        rows = main_rows(groups, ds)
        if not rows:
            continue
        runs = {r["model"]: groups[r["key"]] for r in rows}
        classes = list(next(iter(runs.values()))[0]["test_split"]["per_class_AP50"])
        mat = [[statistics.mean(d["test_split"]["per_class_AP50"][c] for d in rs) for c in classes]
               for rs in runs.values()]
        fig, ax = plt.subplots(figsize=(1.0 + 0.78 * len(classes), 0.9 + 0.36 * len(rows)))
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(classes)), [c.replace("_", " ") for c in classes],
                      rotation=30, ha="right", fontsize=8, color=INK2)
        ax.set_yticks(range(len(rows)), [f"{m} (ours)" if r["sdf"] else m
                                         for m, r in zip(runs, rows)], fontsize=8, color=INK2)
        for j in range(len(classes)):
            col = [m[j] for m in mat]
            for i, v in enumerate(mat):
                ax.text(j, i, f"{v[j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v[j] > 55 else INK,
                        fontweight="bold" if v[j] == max(col) else "normal")
        ax.set_xticks([x - 0.5 for x in range(1, len(classes))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        cb.outline.set_visible(False)
        cb.set_label("test AP50 (%)", fontsize=8, color=INK2)
        ax.set_title(f"{DS_TITLE[ds]}: per-class test AP50 (bold = best per class)",
                     fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(out / f"per_class_ap50_{ds}.png", dpi=200, facecolor="white")
        plt.close(fig)


def fig_accuracy_vs_cost(groups, out):
    dss = [ds for ds in ("neu", "gc10") if main_rows(groups, ds)]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.6))
    for ax, ds in zip(axes if len(dss) > 1 else [axes], dss):
        rows = main_rows(groups, ds)
        for r in rows:
            x, y = r["GFLOPs"][0], r["mAP50"][0]
            c = C_SDF if r["sdf"] else C_BASE
            ax.scatter(x, y, s=46, color=c, edgecolor="white", linewidth=2, zorder=3)
            if r["mAP50"][1] is not None:
                ax.errorbar(x, y, yerr=r["mAP50"][1], color=c, linewidth=1.2, capsize=3, zorder=2)
            # a neighbour just to the right at a similar height -> put this label on the left
            yr = max(q["mAP50"][0] for q in rows) - min(q["mAP50"][0] for q in rows)
            crowded = any(0 < q["GFLOPs"][0] - x < 4 and abs(q["mAP50"][0] - y) < 0.08 * yr
                          for q in rows)
            ax.annotate(r["model"] + (" (ours)" if r["sdf"] else ""), (x, y),
                        xytext=(-5, 4) if crowded else (5, 4), ha="right" if crowded else "left",
                        textcoords="offset points", fontsize=7.5,
                        color=INK if r["sdf"] else INK2, fontweight="bold" if r["sdf"] else "normal")
        style(ax)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.18 * (hi - lo))   # room for the right-most labels
        ax.set_xlabel("GFLOPs (640×640)", fontsize=8, color=INK2)
        ax.set_ylabel("test mAP50 (%)", fontsize=8, color=INK2)
        ax.set_title(DS_TITLE[ds], fontsize=9, color=INK, loc="left")
    fig.suptitle("Accuracy vs compute (orange = SDF-YOLO, blue = baselines)", fontsize=9,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "map50_vs_gflops.png", dpi=200, facecolor="white")
    plt.close(fig)


def curve(d):
    n = d.get("config", {}).get("epochs", 200)
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(d["_dir"] / "results.csv"))][-n:]
    return [int(float(r["epoch"])) for r in rows], [float(r["metrics/mAP50(B)"]) * 100 for r in rows]


def fig_val_curves(groups, out):
    dss = [ds for ds in ("neu", "gc10") if main_rows(groups, ds)]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.4))
    for ax, ds in zip(axes if len(dss) > 1 else [axes], dss):
        rows = main_rows(groups, ds)
        sdf = next(r for r in rows if r["sdf"])
        base = [r for r in rows if not r["sdf"]]
        y11s = next(r for r in base if r["model"] == "YOLO11s")
        top = max((r for r in base if r is not y11s), key=lambda r: r["mAP50"][0])
        for r, c in ((sdf, C_SDF), (y11s, C_BASE), (top, C_3)):
            x, y = curve(groups[r["key"]][0])
            ax.plot(x, y, color=c, linewidth=1.6,
                    label=r["model"] + (" (ours)" if r["sdf"] else ""))
            b = max(range(len(y)), key=y.__getitem__)
            ax.scatter(x[b], y[b], s=40, color=c, edgecolor="white", linewidth=2, zorder=3)
        style(ax)
        ax.set_xlabel("epoch", fontsize=8, color=INK2)
        ax.set_ylabel("val mAP50 (%)", fontsize=8, color=INK2)
        ax.set_title(f"{DS_TITLE[ds]} (dot = best epoch → best.pt)", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "val_map50_curves.png", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT.parent / "paper_results_v2"))
    ap.add_argument("--gc10", default="", help="main-table setting, e.g. 'imgsz=1024 rfs=0.3'")
    ap.add_argument("--neu", default="", help="main-table setting, e.g. 'epochs=150'")
    a = ap.parse_args()
    SETTING["gc10"], SETTING["neu"] = parse_setting(a.gc10), parse_setting(a.neu)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = load(a.runs)
    write_tables(groups, out)
    fig_per_class(groups, out)
    fig_accuracy_vs_cost(groups, out)
    fig_val_curves(groups, out)
    print(f"{sum(len(v) for v in groups.values())} runs in {len(groups)} groups -> {out}")
    for p in sorted(out.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
