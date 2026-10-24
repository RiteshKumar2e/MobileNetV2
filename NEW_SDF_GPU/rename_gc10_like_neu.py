"""Rename GC10-DET (native, PREPARED layout) to NEU-DET's naming:
IMAGES/<class>/<class>_<n>.jpg + ANNOTATIONS/<class>_<n>.xml, n from 1 within
each class, in sorted order of the old stem. Writes GC10-DET/name_map.csv
(old_stem, new_stem, class, raw_file, split) so the frozen split and the raw
GC10-DET filenames stay traceable."""
import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="GC10-DET folder (IMAGES/ + ANNOTATIONS/)")
    ap.add_argument("--split", required=True, help="split csv with stem,class,split (old stems)")
    args = ap.parse_args()

    root = Path(args.root)
    split_of = {r["stem"]: r["split"] for r in csv.DictReader(open(args.split, newline=""))}
    assert not (root / "name_map.csv").exists(), "already renamed (name_map.csv exists)"

    rows = []
    for cls_dir in sorted(p for p in (root / "IMAGES").iterdir() if p.is_dir()):
        cls = cls_dir.name
        for n, img in enumerate(sorted(cls_dir.glob("*.jpg")), start=1):
            old = img.stem
            assert "__" in old, f"unexpected name {img}"
            new = f"{cls}_{n}"
            rows.append({"old_stem": old, "new_stem": new, "class": cls,
                         "raw_file": f"{old.split('__', 1)[1]}.jpg", "split": split_of[old]})

    news = [r["new_stem"] for r in rows]
    assert len(set(news)) == len(news)

    for r in rows:
        cls, old, new = r["class"], r["old_stem"], r["new_stem"]
        (root / "IMAGES" / cls / f"{old}.jpg").rename(root / "IMAGES" / cls / f"{new}.jpg")
        xml_old = root / "ANNOTATIONS" / f"{old}.xml"
        tree = ET.parse(xml_old)
        tree.getroot().find("filename").text = f"{new}.jpg"
        tree.write(root / "ANNOTATIONS" / f"{new}.xml", encoding="utf-8", xml_declaration=True)
        xml_old.unlink()

    with open(root / "name_map.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["old_stem", "new_stem", "class", "raw_file", "split"])
        w.writeheader()
        w.writerows(rows)

    imgs = sorted(p.stem for p in (root / "IMAGES").rglob("*.jpg"))
    xmls = sorted(p.stem for p in (root / "ANNOTATIONS").glob("*.xml"))
    assert imgs == xmls == sorted(news), "image/xml names do not line up after rename"
    for x in (root / "ANNOTATIONS").glob("*.xml"):
        assert ET.parse(x).getroot().findtext("filename") == f"{x.stem}.jpg"
    print(f"renamed {len(rows)} images + xmls; map -> {root / 'name_map.csv'}")


if __name__ == "__main__":
    main()
