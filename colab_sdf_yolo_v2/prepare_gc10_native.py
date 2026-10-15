"""Build datasets/gc10 from RAW GC10-DET at native resolution (no resizing).

Keeps the notebook's frozen split (splits/gc10_split_v1.csv: same stems in the
same train/val/test partitions). Two differences from the notebook's 224x224
GC10-DET_PREPARED, both deliberate:

  1. Images are copied byte-for-byte at their original size (2048x1000); the
     network input size is chosen at train time with --imgsz, exactly like
     every YOLO paper (ultralytics letterboxes, keeping the aspect ratio).
  2. Each box keeps the class written in its own XML <object><name>. The
     notebook's prep labelled every box with the image's FOLDER class, which is
     wrong for GC10 images that contain more than one defect type.

Raw GC10-DET layout expected anywhere under --raw: ten numbered class folders
1..10 of .jpg images plus one folder of VOC .xml files (named 'lable' in the
original release).
"""
import argparse
import csv
import hashlib
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CLASS_MAP = {1: "punching_hole", 2: "welding_line", 3: "crescent_gap", 4: "water_spot",
             5: "oil_spot", 6: "silk_spot", 7: "inclusion", 8: "rolled_pit",
             9: "crease", 10: "waist_folding"}
NAMES = [CLASS_MAP[i] for i in range(1, 11)]
FOLDER_OF = {v: k for k, v in CLASS_MAP.items()}


def find_raw_dirs(raw):
    img_root = None
    for d in [raw, *[p for p in raw.rglob("*") if p.is_dir()]]:
        if all((d / str(i)).is_dir() and any((d / str(i)).glob("*.jpg")) for i in range(1, 11)):
            img_root = d
            break
    assert img_root, f"no folder with subfolders 1..10 of .jpg images under {raw}"
    xml_dirs = sorted((p for p in raw.rglob("*") if p.is_dir() and any(p.glob("*.xml"))),
                      key=lambda p: -sum(1 for _ in p.glob("*.xml")))
    assert xml_dirs, f"no folder of .xml annotations under {raw}"
    return img_root, xml_dirs[0]


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    w, h = int(size.findtext("width")), int(size.findtext("height"))
    boxes, dropped = [], 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        head = name.split("_", 1)[0]
        if not head.isdigit() or int(head) not in CLASS_MAP:
            dropped += 1                      # e.g. the stray object named "d"
            continue
        bb = obj.find("bndbox")
        x1 = max(0.0, min(float(bb.findtext("xmin")), w))
        y1 = max(0.0, min(float(bb.findtext("ymin")), h))
        x2 = max(0.0, min(float(bb.findtext("xmax")), w))
        y2 = max(0.0, min(float(bb.findtext("ymax")), h))
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        boxes.append((int(head) - 1, x1, y1, x2, y2))
    return w, h, boxes, dropped


def jpeg_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder where raw GC10-DET was extracted")
    ap.add_argument("--split", default=str(ROOT / "splits" / "gc10_split_v1_rawnames.csv"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "gc10"))
    args = ap.parse_args()

    img_root, xml_dir = find_raw_dirs(Path(args.raw))
    print(f"raw images: {img_root}\nraw xml:    {xml_dir}")

    with open(args.split, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"frozen split: {Counter(r['split'] for r in rows)}")

    # Same raw filename in two class folders shares ONE flat XML. Identical
    # images -> keep one copy (the one in val/test if the twins straddle
    # splits, so no train/test leakage). Different images -> the XML's own
    # <folder> tag says which image it annotates; the other copy has no
    # annotation of its own and is excluded. If <folder> matches neither,
    # both are excluded rather than guessed.
    by_raw = defaultdict(list)
    for r in rows:
        by_raw[r["stem"].split("__", 1)[1]].append(r)
    exclude, twin_log = set(), []
    for raw_stem, group in by_raw.items():
        if len(group) < 2:
            continue
        hashes = {r["stem"]: hashlib.md5(
            (img_root / str(FOLDER_OF[r["class"]]) / f"{raw_stem}.jpg").read_bytes()).hexdigest()
            for r in group}
        if len(set(hashes.values())) == 1:
            keep = sorted(group, key=lambda r: {"test": 0, "val": 1, "train": 2}[r["split"]])[0]
            exclude |= {r["stem"] for r in group if r is not keep}
            twin_log.append(f"{raw_stem}: identical, kept {keep['stem']} ({keep['split']})")
            continue
        xml_folder = (ET.parse(xml_dir / f"{raw_stem}.xml").getroot().findtext("folder") or "").strip()
        owners = [r for r in group if str(FOLDER_OF[r["class"]]) == xml_folder]
        if len(owners) == 1:
            exclude |= {r["stem"] for r in group if r is not owners[0]}
            twin_log.append(f"{raw_stem}: different images, XML <folder>={xml_folder} "
                            f"-> kept {owners[0]['stem']} ({owners[0]['split']})")
        else:
            exclude |= {r["stem"] for r in group}
            twin_log.append(f"{raw_stem}: different images, XML <folder>={xml_folder!r} "
                            f"matches neither -> both excluded")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    n_img = Counter()
    n_box = Counter()
    relabelled = dropped = missing = 0
    sizes = Counter()
    for r in rows:
        stem, split, folder_cls = r["stem"], r["split"], r["class"]
        if stem in exclude:
            continue
        raw_stem = stem.split("__", 1)[1]
        img = img_root / str(FOLDER_OF[folder_cls]) / f"{raw_stem}.jpg"
        xml = xml_dir / f"{raw_stem}.xml"
        if not img.exists() or not xml.exists():
            missing += 1
            continue
        w, h, boxes, d = parse_xml(xml)
        dropped += d
        iw, ih = jpeg_size(img)
        assert (iw, ih) == (w, h), f"{xml}: XML says {w}x{h}, image is {iw}x{ih}"
        sizes[(iw, ih)] += 1
        if not boxes:
            missing += 1
            continue
        relabelled += sum(1 for b in boxes if NAMES[b[0]] != folder_cls)
        shutil.copyfile(img, out / "images" / split / f"{stem}.jpg")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(
            f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            for c, x1, y1, x2, y2 in boxes) + "\n")
        n_img[split] += 1
        for b in boxes:
            n_box[NAMES[b[0]]] += 1

    print("\n".join(twin_log) or "no duplicated raw filenames")
    print(f"images per split : {dict(n_img)}")
    print(f"image sizes      : {dict(sizes)}")
    print(f"boxes per class  : {dict(n_box)}")
    print(f"boxes whose XML class != folder class (now labelled correctly): {relabelled}")
    print(f"objects dropped (unknown name / degenerate): {dropped}")
    print(f"split stems skipped (missing file or no valid box): {missing}")
    print(f"excluded as duplicated/ambiguous: {len(exclude)}")
    assert sum(n_img.values()) > 0.95 * len(rows), "too many split images missing -- wrong raw folder?"
    print(f"\ndone -> {out}")


if __name__ == "__main__":
    main()
