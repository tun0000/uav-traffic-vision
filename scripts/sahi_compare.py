"""Direct inference vs SAHI sliced inference on VisDrone val.

The project's centerpiece experiment. Both modes are evaluated with the exact
same COCO protocol (pycocotools, custom tiny/small/medium/large buckets,
maxDets up to 500 — VisDrone scenes are dense enough that COCO's default 100
truncates recall), so every delta in the table is attributable to slicing.

Stages:
1. Optional parameter sweep (SAHI slice size / overlap) on a seeded random
   subset, picking the config with the best tiny-bucket AP.
2. Full-val evaluation: direct vs the chosen SAHI config, including per-image
   latency and a detection-count analysis of YOLO26's 300-detection e2e cap.
3. Side-by-side visualizations (same image, direct left / SAHI right) drawn
   with one renderer so the comparison is styling-fair.

Writes reports/sahi_comparison.md and reports/figures/sahi_vs_direct_*.jpg.

Usage:
    python scripts/sahi_compare.py [--weights weights/yolo26s_visdrone_640.pt]
    python scripts/sahi_compare.py --skip-sweep --slice 640 --overlap 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from evaluate import AREA_LBL, CLASS_NAMES, build_gt_coco, run_coco_eval, summarize_bucket

CONF = 0.01          # same low threshold for both modes (eval-oriented)
VIZ_CONF = 0.25      # threshold for drawn figures
MAX_DETS = [10, 100, 500]
E2E_CAP = 300        # fixed max detections of YOLO26's end-to-end head

# BGR palette, one color per class
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
]


def predict_direct(weights: Path, images: list[Path], imgsz: int) -> tuple[list[dict], list[int], float]:
    """Per-image direct inference. Returns (coco dets, per-image det counts, mean ms/img)."""
    model = YOLO(str(weights))
    for p in images[:3]:  # CUDA warmup, not timed
        model.predict(str(p), imgsz=imgsz, conf=CONF, verbose=False)

    dets, counts, elapsed = [], [], 0.0
    for img_id, img_path in enumerate(images):
        t0 = time.perf_counter()
        r = model.predict(str(img_path), imgsz=imgsz, conf=CONF, verbose=False)[0]
        elapsed += time.perf_counter() - t0
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        counts.append(len(boxes))
        for (x0, y0, x1, y1), score, cls in zip(boxes, scores, classes):
            dets.append({
                "image_id": img_id, "category_id": int(cls) + 1,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "score": float(score),
            })
    return dets, counts, 1000 * elapsed / len(images)


def predict_sahi(
    weights: Path, images: list[Path], imgsz: int, slice_size: int, overlap: float
) -> tuple[list[dict], list[int], float]:
    """Per-image SAHI sliced inference. Returns (coco dets, det counts, mean ms/img)."""
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights),
        confidence_threshold=CONF, device="cuda:0", image_size=imgsz,
    )
    get_sliced_prediction(  # warmup, not timed
        str(images[0]), model, slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0,
    )

    dets, counts, elapsed = [], [], 0.0
    for img_id, img_path in enumerate(images):
        t0 = time.perf_counter()
        result = get_sliced_prediction(
            str(img_path), model, slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0,
        )
        elapsed += time.perf_counter() - t0
        preds = result.object_prediction_list
        counts.append(len(preds))
        for op in preds:
            x, y, w, h = op.bbox.to_xywh()
            dets.append({
                "image_id": img_id, "category_id": int(op.category.id) + 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(op.score.value),
            })
    return dets, counts, 1000 * elapsed / len(images)


def eval_mode(gt: dict, dets: list[dict], work_dir: Path) -> dict[str, float]:
    """Unified COCO eval; returns AP/AR per area bucket at maxDets=500 (+AR@100)."""
    _, ev = run_coco_eval(gt, dets, work_dir, max_dets=MAX_DETS)
    out = {"AP50_all": _ap50(ev)}
    for lbl in AREA_LBL:
        out[f"AP_{lbl}"] = summarize_bucket(ev, 1, lbl, 500)
        out[f"AR500_{lbl}"] = summarize_bucket(ev, 0, lbl, 500)
        out[f"AR100_{lbl}"] = summarize_bucket(ev, 0, lbl, 100)
    return out


def _ap50(ev) -> float:
    p = ev.params
    t = np.where(np.isclose(p.iouThrs, 0.5))[0]
    aind = [i for i, lbl in enumerate(p.areaRngLbl) if lbl == "all"]
    mind = [i for i, m in enumerate(p.maxDets) if m == 500]
    s = ev.eval["precision"][t][:, :, :, aind, mind]
    return float(np.mean(s[s > -1])) if len(s[s > -1]) else -1.0


def cap_stats(counts: list[int]) -> dict[str, float]:
    arr = np.array(counts)
    return {
        "mean": float(arr.mean()),
        "max": int(arr.max()),
        "n_at_cap": int((arr >= E2E_CAP).sum()),
        "n_over_cap": int((arr > E2E_CAP).sum()),
    }


def draw_dets(img: np.ndarray, dets: list[dict], conf: float) -> tuple[np.ndarray, int]:
    out = img.copy()
    thickness = max(1, round(min(out.shape[:2]) / 600))
    n = 0
    for d in dets:
        if d["score"] < conf:
            continue
        x, y, w, h = (int(round(v)) for v in d["bbox"])
        color = PALETTE[(d["category_id"] - 1) % len(PALETTE)]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        n += 1
    return out, n


def side_by_side(
    img_path: Path, direct_dets: list[dict], sahi_dets: list[dict],
    sahi_tag: str, out_path: Path,
) -> None:
    img = cv2.imread(str(img_path))
    left, n_l = draw_dets(img, direct_dets, VIZ_CONF)
    right, n_r = draw_dets(img, sahi_dets, VIZ_CONF)

    header_h = max(40, img.shape[0] // 20)
    scale = header_h / 55
    font = cv2.FONT_HERSHEY_SIMPLEX

    def with_header(panel: np.ndarray, text: str) -> np.ndarray:
        bar = np.full((header_h, panel.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(bar, text, (10, int(header_h * 0.7)), font, scale, (20, 20, 20), 2, cv2.LINE_AA)
        return np.vstack([bar, panel])

    left = with_header(left, f"Direct 640 - {n_l} dets (conf>={VIZ_CONF})")
    right = with_header(right, f"SAHI {sahi_tag} - {n_r} dets (conf>={VIZ_CONF})")
    gap = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    combo = np.hstack([left, gap, right])

    if combo.shape[1] > 2600:  # keep README-friendly file sizes
        s = 2600 / combo.shape[1]
        combo = cv2.resize(combo, (2600, int(combo.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), combo, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  saved {out_path} (direct {n_l} vs SAHI {n_r} dets)")


def group_by_image(dets: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for d in dets:
        grouped.setdefault(d["image_id"], []).append(d)
    return grouped


def pick_viz_images(data_root: Path, images: list[Path], n: int = 3) -> list[int]:
    """Indices of small-object-heavy val images (>=20 objects, highest tiny fraction)."""
    from PIL import Image

    lbl_dir = data_root / "labels" / "val"
    scores = []
    for i, img_path in enumerate(images):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            scores.append((i, 0, 0.0))
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        n_obj, n_tiny = 0, 0
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            if float(parts[3]) * w * float(parts[4]) * h < 16 ** 2:
                n_tiny += 1
            n_obj += 1
        scores.append((i, n_obj, n_tiny / n_obj if n_obj else 0.0))
    eligible = [s for s in scores if s[1] >= 20]
    eligible.sort(key=lambda s: -s[2])
    return [s[0] for s in eligible[:n]]


def fmt_row(name: str, m: dict[str, float], ms: float) -> str:
    return (
        f"| {name} | {m['AP50_all']:.3f} | {m['AP_all']:.3f} | {m['AP_tiny']:.3f} "
        f"| {m['AP_small']:.3f} | {m['AP_medium']:.3f} | {m['AP_large']:.3f} "
        f"| {m['AR500_all']:.3f} | {m['AR500_tiny']:.3f} | {ms:.0f} |"
    )


TABLE_HEADER = (
    "| mode | AP50 | AP | AP tiny | AP small | AP medium | AP large | AR@500 | AR@500 tiny | ms/img |\n"
    "|------|------|----|---------|----------|-----------|----------|--------|-------------|--------|"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=Path("weights/yolo26s_visdrone_640.pt"))
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets" / "VisDrone")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-sweep", action="store_true", help="skip the sweep, use --slice/--overlap")
    parser.add_argument("--sweep-n", type=int, default=100, help="sweep subset size")
    parser.add_argument("--slice", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="use only first N val images (smoke test)")
    parser.add_argument("--out", type=Path, default=Path("reports/sahi_comparison.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--work-dir", type=Path, default=Path("reports/.eval_coco"))
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    img_dir = data_root / "images" / "val"
    images = sorted(img_dir.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    print(f"val images: {len(images)}")

    sweep_lines: list[str] = []
    slice_size, overlap = args.slice, args.overlap
    if not args.skip_sweep:
        subset = sorted(random.Random(args.seed).sample(images, args.sweep_n))
        gt_sub, _ = build_gt_coco(data_root, "val", subset)
        print(f"\n=== sweep on {len(subset)} images ===")
        best = None
        for cand_slice, cand_overlap in [(640, 0.2), (512, 0.2), (800, 0.2)]:
            tag = f"{cand_slice}/{cand_overlap}"
            print(f"\n--- sweep config: slice {tag} ---")
            dets, _, ms = predict_sahi(args.weights, subset, args.imgsz, cand_slice, cand_overlap)
            m = eval_mode(gt_sub, dets, args.work_dir / f"sweep_{cand_slice}")
            sweep_lines.append(fmt_row(f"SAHI {tag}", m, ms))
            print(f"sweep {tag}: AP={m['AP_all']:.3f} AP_tiny={m['AP_tiny']:.3f} {ms:.0f} ms/img")
            if best is None or m["AP_tiny"] > best[0]:
                best = (m["AP_tiny"], cand_slice, cand_overlap)
        _, slice_size, overlap = best
        print(f"\nchosen config by tiny-bucket AP: slice={slice_size} overlap={overlap}")

    sahi_tag = f"{slice_size}/{overlap}"
    print("\n=== full val: direct ===")
    gt, _ = build_gt_coco(data_root, "val", images)
    d_dets, d_counts, d_ms = predict_direct(args.weights, images, args.imgsz)
    d_metrics = eval_mode(gt, d_dets, args.work_dir / "full_direct")

    print(f"\n=== full val: SAHI {sahi_tag} ===")
    s_dets, s_counts, s_ms = predict_sahi(args.weights, images, args.imgsz, slice_size, overlap)
    s_metrics = eval_mode(gt, s_dets, args.work_dir / "full_sahi")

    d_cap, s_cap = cap_stats(d_counts), cap_stats(s_counts)

    print("\n=== side-by-side figures ===")
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    d_by_img, s_by_img = group_by_image(d_dets), group_by_image(s_dets)
    fig_names = []
    for idx in pick_viz_images(data_root, images):
        out_path = args.figures_dir / f"sahi_vs_direct_{images[idx].stem}.jpg"
        side_by_side(images[idx], d_by_img.get(idx, []), s_by_img.get(idx, []), sahi_tag, out_path)
        fig_names.append(out_path.name)

    # ---- report ----
    lines = ["# Direct vs SAHI sliced inference (yolo26s @ 640, VisDrone val)", ""]
    lines.append(
        f"Protocol: identical for both modes — conf={CONF}, pycocotools with "
        f"maxDets={MAX_DETS} (dense scenes need >100), custom tiny(<16px)/small/"
        f"medium/large buckets. SAHI: GREEDYNMM/IOS/0.5 merge, standard full-image "
        f"pass included (`perform_standard_pred=True`). Latency: per-image wall time "
        f"on RTX 4090, batch=1. All numbers from actual runs."
    )
    lines.append("")
    if sweep_lines:
        lines.append(f"## Parameter sweep ({args.sweep_n}-image subset, seed {args.seed})")
        lines.append("")
        lines.append(TABLE_HEADER)
        lines.extend(sweep_lines)
        lines.append("")
        lines.append(f"Chosen by tiny-bucket AP: **slice {sahi_tag}**.")
        lines.append("")

    lines.append(f"## Full val ({len(images)} images)")
    lines.append("")
    lines.append(TABLE_HEADER)
    lines.append(fmt_row("Direct 640", d_metrics, d_ms))
    lines.append(fmt_row(f"SAHI {sahi_tag}", s_metrics, s_ms))
    lines.append("")
    ap_gain = s_metrics["AP_all"] - d_metrics["AP_all"]
    tiny_gain = s_metrics["AP_tiny"] - d_metrics["AP_tiny"]
    rec_gain = s_metrics["AR500_tiny"] - d_metrics["AR500_tiny"]
    lines.append(
        f"SAHI vs direct: overall AP {ap_gain:+.3f}, tiny-bucket AP {tiny_gain:+.3f} "
        f"({d_metrics['AP_tiny']:.3f} → {s_metrics['AP_tiny']:.3f}), tiny-bucket AR@500 "
        f"{rec_gain:+.3f}, at {s_ms / d_ms:.1f}x the per-image latency."
    )
    lines.append("")

    lines.append("## Detection-count / 300-cap analysis")
    lines.append("")
    lines.append(f"YOLO26's end-to-end head emits at most {E2E_CAP} detections per forward pass.")
    lines.append("")
    lines.append("| mode | mean dets/img | max dets/img | images at cap |")
    lines.append("|------|---------------|--------------|----------------|")
    lines.append(f"| Direct 640 | {d_cap['mean']:.0f} | {d_cap['max']} | {d_cap['n_at_cap']} |")
    lines.append(f"| SAHI {sahi_tag} | {s_cap['mean']:.0f} | {s_cap['max']} | n/a (per-slice cap) |")
    lines.append("")
    lines.append(
        f"{d_cap['n_at_cap']} of {len(images)} direct-mode images saturate the {E2E_CAP}-detection cap "
        f"(at conf≥{CONF}); slicing raises the effective per-image budget to "
        f"{E2E_CAP}×(slices+1), and SAHI's densest output here reached {s_cap['max']} detections."
    )
    lines.append("")

    lines.append("## Side-by-side examples (small-object-heavy scenes)")
    lines.append("")
    for name in fig_names:
        lines.append(f"![sahi vs direct](figures/{name})")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")

    # machine-readable summary for the README table
    summary = {
        "direct": {**d_metrics, "ms_per_img": d_ms, **{f"cap_{k}": v for k, v in d_cap.items()}},
        f"sahi_{slice_size}_{overlap}": {**s_metrics, "ms_per_img": s_ms, **{f"cap_{k}": v for k, v in s_cap.items()}},
    }
    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
