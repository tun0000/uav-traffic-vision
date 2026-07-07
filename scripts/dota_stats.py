"""DOTA-v1.0 (OBB) exploratory analysis.

Scans the raw (pre-tiling) YOLO-OBB-format DOTA-v1.0 dataset (as laid out by
ultralytics' built-in DOTAv1.yaml downloader) and writes reports/dota_stats.md
plus four figures: image resolution histogram, oriented-bbox area histogram,
rotation angle histogram, and aspect-ratio histogram. These numbers motivate
Phase 3's story: DOTA's raw images are far larger than a detector's input size
(hence the mandatory split_dota tiling step) and its objects are genuinely
rotated (hence OBB instead of axis-aligned boxes).

If the dataset is missing, triggers the official ultralytics auto-download
(~2 GB) after pointing the ultralytics `datasets_dir` setting at the parent
of --data-root (persistent, matches this repo's convention).

Usage:
    python scripts/dota_stats.py [--data-root ~/datasets/DOTAv1]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
]

# same bucket boundaries as scripts/dataset_stats.py (VisDrone) for a direct
# cross-dataset size-profile comparison in the README
AREA_BUCKETS = [
    ("tiny (<16^2)", 0, 16**2),
    ("small (16^2-32^2)", 16**2, 32**2),
    ("medium (32^2-96^2)", 32**2, 96**2),
    ("large (>96^2)", 96**2, float("inf")),
]

E2E_DET_CAP = 300  # fixed max detections of YOLO26's end-to-end head


def ensure_dataset(data_root: Path) -> None:
    if (data_root / "images" / "train").exists():
        return
    print(f"{data_root} not found - triggering ultralytics auto-download (~2 GB)")
    import subprocess
    import sys

    from ultralytics import settings

    settings.update({"datasets_dir": str(data_root.parent)})
    # ultralytics freezes DATASETS_DIR at import time, so the download must run
    # in a fresh interpreter that reads the setting updated above
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from ultralytics.data.utils import check_det_dataset; "
            "check_det_dataset('DOTAv1.yaml', autodownload=True)",
        ],
        check=True,
    )


def obb_corners_to_rect(pts: np.ndarray) -> tuple[float, float, float]:
    """(4,2) pixel-space corners -> (w, h, angle_deg).

    Mirrors ultralytics.utils.ops.xyxyxyxy2xywhr's use of cv2.minAreaRect so
    our EDA angle convention matches what the model actually trains on:
    angle normalized to [-45, 135) degrees, w >= h.
    """
    (_, _), (w, h), angle = cv2.minAreaRect(pts.astype(np.float32))
    if w < h:
        w, h = h, w
        angle += 90
    while angle >= 135:
        angle -= 180
    while angle < -45:
        angle += 180
    return w, h, angle


def scan_split(data_root: Path, split: str) -> dict:
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise SystemExit(f"no images found in {img_dir}")

    class_counts: Counter[int] = Counter()
    objects_per_image: list[int] = []
    areas_px: list[float] = []
    angles_deg: list[float] = []
    aspect_ratios: list[float] = []
    resolutions: Counter[tuple[int, int]] = Counter()
    long_sides: list[int] = []
    background = 0

    for img_path in images:
        with Image.open(img_path) as im:
            w, h = im.size
        resolutions[(w, h)] += 1
        long_sides.append(max(w, h))

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        n_obj = 0
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 9:
                    continue
                cls = int(parts[0])
                coords = np.array(parts[1:], dtype=np.float32).reshape(4, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                rw, rh, angle = obb_corners_to_rect(coords)
                if rw <= 0 or rh <= 0:
                    continue
                class_counts[cls] += 1
                areas_px.append(rw * rh)
                angles_deg.append(angle)
                aspect_ratios.append(rw / rh)
                n_obj += 1
        if n_obj == 0:
            background += 1
        objects_per_image.append(n_obj)

    return {
        "n_images": len(images),
        "class_counts": class_counts,
        "objects_per_image": np.array(objects_per_image),
        "areas_px": np.array(areas_px),
        "angles_deg": np.array(angles_deg),
        "aspect_ratios": np.array(aspect_ratios),
        "resolutions": resolutions,
        "long_sides": np.array(long_sides),
        "background": background,
    }


def bucket_counts(areas: np.ndarray) -> list[tuple[str, int, float]]:
    out = []
    for name, lo, hi in AREA_BUCKETS:
        n = int(((areas >= lo) & (areas < hi)).sum())
        out.append((name, n, 100.0 * n / len(areas)))
    return out


def fig_resolution_hist(splits: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, stats in splits.items():
        ax.hist(stats["long_sides"], bins=np.logspace(np.log10(700), np.log10(21000), 50), alpha=0.6, label=f"{split} ({len(stats['long_sides'])} images)")
    ax.set_xscale("log")
    ax.axvline(1024, color="red", ls="--", lw=1.2, label="1024 (split_dota tile size)")
    ax.set_xlabel("image long side [px, log scale]")
    ax.set_ylabel("images")
    ax.set_title("DOTA-v1.0 raw image size distribution (before tiling)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_area_hist(splits: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, stats in splits.items():
        side = np.sqrt(stats["areas_px"])
        ax.hist(side, bins=np.logspace(0, np.log10(3000), 60), alpha=0.6, label=f"{split} ({len(side):,} boxes)")
    ax.set_xscale("log")
    for _, lo, _ in AREA_BUCKETS[1:]:
        ax.axvline(np.sqrt(lo), color="gray", ls="--", lw=0.8)
    ax.set_xlabel("oriented bbox size sqrt(w*h) [px, log scale]")
    ax.set_ylabel("boxes")
    ax.set_title("DOTA-v1.0 oriented-bbox size distribution (native resolution)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_angle_hist(splits: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, stats in splits.items():
        ax.hist(stats["angles_deg"], bins=90, range=(-45, 135), alpha=0.6, label=split)
    ax.set_xlabel("rotation angle [degrees, -45 to 135]")
    ax.set_ylabel("boxes")
    ax.set_title("DOTA-v1.0 oriented-bbox rotation angle distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_aspect_hist(splits: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, stats in splits.items():
        ax.hist(stats["aspect_ratios"], bins=np.logspace(0, np.log10(30), 60), alpha=0.6, label=split)
    ax.set_xscale("log")
    ax.set_xlabel("aspect ratio (long side / short side) [log scale]")
    ax.set_ylabel("boxes")
    ax.set_title("DOTA-v1.0 oriented-bbox aspect ratio distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets" / "DOTAv1")
    parser.add_argument("--out", type=Path, default=Path("reports/dota_stats.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    ensure_dataset(data_root)

    splits = {s: scan_split(data_root, s) for s in ("train", "val")}

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_resolution_hist(splits, args.figures_dir / "dota_resolution_hist.png")
    fig_area_hist(splits, args.figures_dir / "dota_bbox_area_hist.png")
    fig_angle_hist(splits, args.figures_dir / "dota_angle_hist.png")
    fig_aspect_hist(splits, args.figures_dir / "dota_aspect_hist.png")

    lines: list[str] = ["# DOTA-v1.0 (OBB) dataset statistics", ""]
    lines.append("Generated by `scripts/dota_stats.py`; all numbers measured from the raw")
    lines.append("(pre-`split_dota`-tiling) YOLO-OBB labels at native image resolution.")
    lines.append("Test split is excluded (unlabeled).")
    lines.append("")

    lines.append("## Split overview")
    lines.append("")
    lines.append("| split | images | instances | background images | mean obj/img | median | p95 | max |")
    lines.append("|-------|--------|-----------|-------------------|--------------|--------|-----|-----|")
    for split, st in splits.items():
        opi = st["objects_per_image"]
        lines.append(
            f"| {split} | {st['n_images']:,} | {len(st['areas_px']):,} | {st['background']} "
            f"| {opi.mean():.1f} | {int(np.median(opi))} | {int(np.percentile(opi, 95))} | {opi.max()} |"
        )
    lines.append("")

    lines.append(f"## Images above the end-to-end detection cap ({E2E_DET_CAP})")
    lines.append("")
    lines.append("Measured on raw (untiled) images for context; the model never actually sees a")
    lines.append("full raw image at train/inference time (`split_dota` tiles it to 1024x1024 first),")
    lines.append("so this cap applies per-tile, not per-scene -- see the tiling section below.")
    lines.append("")
    for split, st in splits.items():
        opi = st["objects_per_image"]
        n_over = int((opi > E2E_DET_CAP).sum())
        truncated = int(np.clip(opi - E2E_DET_CAP, 0, None).sum())
        lines.append(
            f"- **{split}**: {n_over} images ({100 * n_over / len(opi):.1f}%) have more than "
            f"{E2E_DET_CAP} objects; capping a whole-scene detector at {E2E_DET_CAP} can miss at most "
            f"{truncated:,} ground-truth objects ({100 * truncated / len(st['areas_px']):.2f}% of instances)."
        )
    lines.append("")

    lines.append("## Instances per class")
    lines.append("")
    lines.append("| id | class | train | val |")
    lines.append("|----|-------|-------|-----|")
    seen_ids = sorted(set(splits["train"]["class_counts"]) | set(splits["val"]["class_counts"]))
    for cid in seen_ids:
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"UNEXPECTED({cid})"
        lines.append(f"| {cid} | {name} | {splits['train']['class_counts'][cid]:,} | {splits['val']['class_counts'][cid]:,} |")
    lines.append("")

    lines.append("## Oriented-bbox area buckets (native resolution, same cutoffs as the VisDrone EDA)")
    lines.append("")
    lines.append("| bucket | train boxes | train % | val boxes | val % |")
    lines.append("|--------|-------------|---------|-----------|-------|")
    tb = bucket_counts(splits["train"]["areas_px"])
    vb = bucket_counts(splits["val"]["areas_px"])
    for (name, tn, tp), (_, vn, vp) in zip(tb, vb):
        lines.append(f"| {name} | {tn:,} | {tp:.1f}% | {vn:,} | {vp:.1f}% |")
    lines.append("")

    lines.append("## Rotation angle")
    lines.append("")
    for split, st in splits.items():
        ang = st["angles_deg"]
        near_axis = int(((np.abs(ang) < 5) | (np.abs(ang - 90) < 5) | (np.abs(ang + 45) < 5) | (np.abs(ang - 135) < 5)).sum())
        lines.append(
            f"- **{split}**: {100 * near_axis / len(ang):.1f}% of boxes fall within 5 degrees of an axis-aligned "
            f"orientation (0/90 degrees); the rest are genuinely rotated, which is the core motivation for OBB "
            f"over axis-aligned boxes on this dataset."
        )
    lines.append("")

    lines.append("## Aspect ratio (long side / short side)")
    lines.append("")
    for split, st in splits.items():
        ar = st["aspect_ratios"]
        lines.append(f"- **{split}**: median {np.median(ar):.2f}, p95 {np.percentile(ar, 95):.2f}, max {ar.max():.1f}")
    lines.append("")

    lines.append("## Image resolutions (why tiling is mandatory)")
    lines.append("")
    for split, st in splits.items():
        ls = st["long_sides"]
        over_1024 = int((ls > 1024).sum())
        over_4000 = int((ls > 4000).sum())
        lines.append(
            f"- **{split}**: long side ranges {ls.min()}-{ls.max()}px (median {int(np.median(ls))}px); "
            f"{100 * over_1024 / len(ls):.1f}% of images exceed 1024px, {100 * over_4000 / len(ls):.1f}% exceed 4000px."
        )
    lines.append("")
    lines.append("![image resolution distribution](figures/dota_resolution_hist.png)")
    lines.append("")
    lines.append("![oriented bbox size distribution](figures/dota_bbox_area_hist.png)")
    lines.append("")
    lines.append("![rotation angle distribution](figures/dota_angle_hist.png)")
    lines.append("")
    lines.append("![aspect ratio distribution](figures/dota_aspect_hist.png)")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out} and figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
