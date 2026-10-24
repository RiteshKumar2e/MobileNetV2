"""Raw GC10-DET -> GC10-DET_NATIVE in GC10-DET_PREPARED's exact layout
(IMAGES/<class_name>/<class_name>__<raw_stem>.jpg + flat ANNOTATIONS/<stem>.xml),
but at native 2048x1000 (byte-for-byte copies) and with each box keeping the
class from its own XML <name> instead of the image's folder class.

Same cleaning as the notebook's prepare_gc10det(): unknown object names (the
stray "d") and degenerate boxes dropped, images left with no valid box
excluded. Filenames reused across two class folders share one flat XML; its
<folder> tag decides which image it annotates, the other copy is excluded.
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from prepare_gc10_native import CLASS_MAP, NAMES, find_raw_dirs, jpeg_size, parse_xml


def write_voc(path, folder, filename, w, h, boxes):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = folder
    ET.SubElement(ann, "filename").text = filename
    size = ET.SubElement(ann, "size")
    for k, v in (("width", w), ("height", h), ("depth", 1)):
        ET.SubElement(size, k).text = str(v)
    ET.SubElement(ann, "segmented").text = "0"
    for c, x1, y1, x2, y2 in boxes:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = NAMES[c]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        for k, v in (("xmin", x1), ("ymin", y1), ("xmax", x2), ("ymax", y2)):
            ET.SubElement(bb, k).text = str(round(v, 2))
    ET.ElementTree(ann).write(path, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_root, xml_dir = find_raw_dirs(Path(args.raw))
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "ANNOTATIONS").mkdir(parents=True)

    folders_of = defaultdict(list)
    for num in CLASS_MAP:
        for img in sorted((img_root / str(num)).glob("*.jpg")):
            folders_of[img.stem].append(num)

    per_cls_img, per_cls_box = Counter(), Counter()
    relabelled = dropped = missing_xml = empty = twin_skipped = 0
    for num, cls in CLASS_MAP.items():
        (out / "IMAGES" / cls).mkdir(parents=True)
        for img in sorted((img_root / str(num)).glob("*.jpg")):
            xml = xml_dir / f"{img.stem}.xml"
            if not xml.exists():
                missing_xml += 1
                continue
            if len(folders_of[img.stem]) > 1:
                owner = (ET.parse(xml).getroot().findtext("folder") or "").strip()
                if owner != str(num):
                    twin_skipped += 1
                    continue
            w, h, boxes, d = parse_xml(xml)
            dropped += d
            assert jpeg_size(img) == (w, h), f"{xml}: size mismatch"
            if not boxes:
                empty += 1
                continue
            stem = f"{cls}__{img.stem}"
            shutil.copyfile(img, out / "IMAGES" / cls / f"{stem}.jpg")
            write_voc(out / "ANNOTATIONS" / f"{stem}.xml", cls, f"{stem}.jpg", w, h, boxes)
            per_cls_img[cls] += 1
            for b in boxes:
                per_cls_box[NAMES[b[0]]] += 1
                relabelled += NAMES[b[0]] != cls

    print("images per folder class:", dict(per_cls_img))
    print("boxes per XML class    :", dict(per_cls_box))
    print(f"total: {sum(per_cls_img.values())} images, {sum(per_cls_box.values())} boxes")
    print(f"boxes whose XML class != folder class (kept as XML class): {relabelled}")
    print(f"objects dropped: {dropped} | images without XML: {missing_xml} | "
          f"images with no valid box: {empty} | duplicate-name copies not owned by their XML: {twin_skipped}")


if __name__ == "__main__":
    main()
