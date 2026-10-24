"""Build datasets/<neu|gc10> (YOLO format) from the repo's VOC copies, using the FROZEN
splits in splits/ so every model sees identical train/val/test.

    python prepare_data.py --data neu  --voc ../NEU-DET
    python prepare_data.py --data gc10 --voc ../GC10-DET

VOC layout: <voc>/IMAGES/<class>/<stem>.jpg + <voc>/ANNOTATIONS/<stem>.xml.
Each box keeps the class in its own <object><name>; images copied byte-for-byte.
"""
import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from train import DATASETS

ROOT = Path(__file__).resolve().parent
SPLITS = {"neu": ROOT / "splits" / "neu_split_v1.csv", "gc10": ROOT / "splits" / "gc10_split_v1.csv"}


def parse(xml_path, names):
    r = ET.parse(xml_path).getroot()
    w, h = int(r.find("size").findtext("width")), int(r.find("size").findtext("height"))
    boxes, dropped = [], 0
    for o in r.findall("object"):
        n = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        x1, y1 = max(0.0, float(bb.findtext("xmin"))), max(0.0, float(bb.findtext("ymin")))
        x2, y2 = min(float(w), float(bb.findtext("xmax"))), min(float(h), float(bb.findtext("ymax")))
        if n not in names or x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        boxes.append((names.index(n), x1, y1, x2, y2))
    return w, h, boxes, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=sorted(DATASETS), required=True)
    ap.add_argument("--voc", required=True, help="folder with IMAGES/ and ANNOTATIONS/")
    a = ap.parse_args()
    names = list(DATASETS[a.data])
    voc, out = Path(a.voc), ROOT / "datasets" / a.data
    rows = list(csv.DictReader(open(SPLITS[a.data], newline="")))

    for sub in ("images", "labels"):
        shutil.rmtree(out / sub, ignore_errors=True)
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    n_img, n_box, missing, dropped = Counter(), Counter(), [], 0
    for r in rows:
        stem, split = r["stem"], r["split"]
        img = voc / "IMAGES" / r["class"] / f"{stem}.jpg"
        xml = voc / "ANNOTATIONS" / f"{stem}.xml"
        if not img.exists() or not xml.exists():
            missing.append(stem)
            continue
        w, h, boxes, d = parse(xml, names)
        dropped += d
        if not boxes:
            missing.append(stem)
            continue
        shutil.copyfile(img, out / "images" / split / f"{stem}.jpg")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(
            f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            for c, x1, y1, x2, y2 in boxes) + "\n")
        n_img[split] += 1
        for b in boxes:
            n_box[(split, names[b[0]])] += 1

    print(f"images per split: {dict(n_img)}")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} boxes: " + ", ".join(f"{n} {n_box[(s, n)]}" for n in names))
    print(f"objects dropped: {dropped} | stems skipped: {len(missing)} {missing[:5]}")
    assert sum(n_img.values()) > 0.95 * len(rows), "too many images missing -- wrong --voc folder?"
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
