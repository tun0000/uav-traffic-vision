"""Evaluate the trained VisDrone detector.

Reports overall/per-class AP via ultralytics' val(), plus a COCO-style
area-bucketed breakdown for small-object performance: adds a custom "tiny"
(<16px side) bucket below COCO's standard "small" (<32px side), since the
EDA showed most VisDrone boxes fall below COCO's own small/medium split.
Also saves a few representative prediction overlays (dense scenes and
small-object-heavy scenes) to reports/figures/.

Usage:
    python scripts/evaluate.py [--weights weights/yolo26s_visdrone_640.pt] [--imgsz 640]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# custom buckets: COCO's "small" (<32px side) split into tiny/small at 16px,
# matching the EDA bucketing in scripts/dataset_stats.py
AREA_RNG = [
    [0 ** 2, 1e5 ** 2],
    [0 ** 2, 16 ** 2],
    [16 ** 2, 32 ** 2],
    [32 ** 2, 96 ** 2],
    [96 ** 2, 1e5 ** 2],
]
AREA_LBL = ["all", "tiny", "small", "medium", "large"]


def build_gt_coco(data_root: Path, split: str) -> tuple[dict, list[Path]]:
    """Convert YOLO-format val labels to a COCO-format GT dict."""
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    images = sorted(img_dir.glob("*.jpg"))

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)],
    }
    ann_id = 1
    for img_id, img_path in enumerate(images):
        with Image.open(img_path) as im:
            w, h = im.size
        coco["images"].append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            abs_w, abs_h = bw * w, bh * h
            x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cls + 1,
                "bbox": [x0, y0, abs_w, abs_h], "area": abs_w * abs_h, "iscrowd": 0,
            })
            ann_id += 1
    return coco, images


def predict_coco(model: YOLO, images: list[Path], imgsz: int) -> list[dict]:
    """Run predictions and format them as a COCO-format detections list."""
    dets = []
    results = model.predict(
        source=[str(p) for p in images], imgsz=imgsz, conf=0.001,
        verbose=False, stream=True,
    )
    for img_id, r in enumerate(results):
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for (x0, y0, x1, y1), score, cls in zip(boxes, scores, classes):
            dets.append({
                "image_id": img_id, "category_id": int(cls) + 1,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "score": float(score),
            })
    return dets


def summarize_bucket(coco_eval: COCOeval, ap: int, area_lbl: str, max_det: int = 100) -> float:
    """Re-implements pycocotools' private `_summarize` closure (not accessible
    outside `COCOeval.summarize()`) so custom area-range labels can be queried.
    """
    p = coco_eval.params
    aind = [i for i, lbl in enumerate(p.areaRngLbl) if lbl == area_lbl]
    mind = [i for i, m in enumerate(p.maxDets) if m == max_det]
    if ap == 1:
        s = coco_eval.eval["precision"][:, :, :, aind, mind]
    else:
        s = coco_eval.eval["recall"][:, :, aind, mind]
    return float(np.mean(s[s > -1])) if len(s[s > -1]) else -1.0


def run_coco_eval(gt: dict, dt: list[dict], out_dir: Path) -> dict[str, float]:
    gt_path = out_dir / "gt_coco.json"
    dt_path = out_dir / "dt_coco.json"
    gt_path.write_text(json.dumps(gt))
    dt_path.write_text(json.dumps(dt))

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(dt_path))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.areaRng = AREA_RNG
    ev.params.areaRngLbl = AREA_LBL

    ev.evaluate()
    ev.accumulate()
    ev.summarize()  # prints pycocotools' own all/small/medium/large/AR summary

    results = {}
    for lbl in AREA_LBL:
        results[f"AP_{lbl}"] = summarize_bucket(ev, 1, lbl, 100)
        results[f"AR100_{lbl}"] = summarize_bucket(ev, 0, lbl, 100)
    return results


def pick_representative_images(data_root: Path, split: str, images: list[Path], n: int = 2) -> dict[str, list[Path]]:
    """Pick dense-scene and small-object-heavy val images for overlay figures."""
    lbl_dir = data_root / "labels" / split
    counts, tiny_fracs = [], []
    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            counts.append(0)
            tiny_fracs.append(0.0)
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        n_obj, n_tiny = 0, 0
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            bw, bh = float(parts[3]) * w, float(parts[4]) * h
            n_obj += 1
            if bw * bh < 16 ** 2:
                n_tiny += 1
        counts.append(n_obj)
        tiny_fracs.append(n_tiny / n_obj if n_obj else 0.0)

    counts_arr, tiny_arr = np.array(counts), np.array(tiny_fracs)
    dense_idx = np.argsort(-counts_arr)[:n]
    # among images with a meaningful number of objects, pick the highest tiny fraction
    eligible = np.where(counts_arr >= 20)[0]
    tiny_idx = eligible[np.argsort(-tiny_arr[eligible])[:n]]
    return {
        "dense": [images[i] for i in dense_idx],
        "tiny": [images[i] for i in tiny_idx],
    }


def save_overlays(model: YOLO, picks: dict[str, list[Path]], imgsz: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, paths in picks.items():
        for img_path in paths:
            r = model.predict(str(img_path), imgsz=imgsz, conf=0.25, verbose=False)[0]
            dst = out_dir / f"{tag}_{img_path.stem}.jpg"
            r.save(str(dst))
            print(f"  saved {dst} ({len(r.boxes)} detections @ conf>=0.25)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=Path("weights/yolo26s_visdrone_640.pt"))
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets" / "VisDrone")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=Path("reports/evaluation.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--work-dir", type=Path, default=Path("reports/.eval_coco"))
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    model = YOLO(str(args.weights))

    print("=== ultralytics model.val() ===")
    metrics = model.val(data="VisDrone.yaml", imgsz=args.imgsz, split="val")

    per_class = []
    for idx, cls_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(idx)
        per_class.append((CLASS_NAMES[cls_id], p, r, ap50, ap))

    print("\n=== building COCO-format GT/DT for area-bucketed eval ===")
    gt, images = build_gt_coco(data_root, "val")
    dt = predict_coco(model, images, args.imgsz)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    bucket_results = run_coco_eval(gt, dt, args.work_dir)

    print("\n=== saving representative overlays ===")
    picks = pick_representative_images(data_root, "val", images)
    save_overlays(model, picks, args.imgsz, args.figures_dir)

    lines = ["# Evaluation: yolo26s @ imgsz=640 on VisDrone2019-DET val", ""]
    lines.append(f"Weights: `{args.weights}`. All numbers from actual script execution.")
    lines.append("")
    lines.append("## Overall (ultralytics `model.val()`)")
    lines.append("")
    lines.append(f"- mAP50: **{metrics.box.map50:.4f}**")
    lines.append(f"- mAP50-95: **{metrics.box.map:.4f}**")
    lines.append(f"- mean precision: {metrics.box.mp:.4f}, mean recall: {metrics.box.mr:.4f}")
    lines.append("")
    lines.append("## Per-class")
    lines.append("")
    lines.append("| class | P | R | AP50 | AP50-95 |")
    lines.append("|-------|---|---|------|---------|")
    for name, p, r, ap50, ap in per_class:
        lines.append(f"| {name} | {p:.3f} | {r:.3f} | {ap50:.3f} | {ap:.3f} |")
    lines.append("")

    lines.append("## Small-object analysis (COCO-style area buckets)")
    lines.append("")
    lines.append(
        "Independent COCO-format evaluation (pycocotools) over the same val set and "
        "predictions, with an extra **tiny** bucket (<16px side) split out from COCO's "
        "standard **small** (16-32px side), matching the EDA bucketing in "
        "`reports/dataset_stats.md`."
    )
    lines.append("")
    lines.append("| bucket | AP@[.5:.95] | AR@100 |")
    lines.append("|--------|-------------|--------|")
    for lbl in AREA_LBL:
        ap_v = bucket_results[f"AP_{lbl}"]
        ar_v = bucket_results[f"AR100_{lbl}"]
        lines.append(f"| {lbl} | {ap_v:.3f} | {ar_v:.3f} |")
    lines.append("")

    lines.append("## Representative predictions")
    lines.append("")
    for tag, paths in picks.items():
        label = "Dense scenes" if tag == "dense" else "Small-object-heavy scenes"
        lines.append(f"**{label}:**")
        for p in paths:
            lines.append(f"![{tag}](figures/{tag}_{p.stem}.jpg)")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
