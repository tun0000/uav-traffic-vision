"""VisDrone2019-DET exploratory analysis.

Scans the YOLO-format VisDrone dataset (as laid out by ultralytics' built-in
VisDrone.yaml converter) and writes reports/dataset_stats.md plus two figures:
a bbox-area bucket histogram and an objects-per-image histogram. These numbers
motivate the project's small-object story (imgsz 640 -> 1024 -> SAHI) and
quantify how many images exceed the 300-detection cap of YOLO26's end-to-end
head.

If the dataset is missing, triggers the official ultralytics auto-download
(~2.3 GB) after pointing the ultralytics `datasets_dir` setting at the parent
of --data-root (persistent, matches this repo's convention).

Usage:
    python scripts/dataset_stats.py [--data-root ~/datasets/VisDrone]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# pixel-area buckets at native resolution; COCO small/medium/large plus a
# custom "tiny" bucket because VisDrone skews far below COCO's small range
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
    print(f"{data_root} not found - triggering ultralytics auto-download (~2.3 GB)")
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
            "check_det_dataset('VisDrone.yaml', autodownload=True)",
        ],
        check=True,
    )


def scan_split(data_root: Path, split: str) -> dict:
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images found in {img_dir}")

    class_counts: Counter[int] = Counter()
    objects_per_image: list[int] = []
    areas_px: list[float] = []
    resolutions: Counter[tuple[int, int]] = Counter()
    background = 0

    for img_path in images:
        with Image.open(img_path) as im:
            w, h = im.size
        resolutions[(w, h)] += 1

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        n_obj = 0
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                bw, bh = float(parts[3]), float(parts[4])
                class_counts[cls] += 1
                areas_px.append(bw * w * bh * h)
                n_obj += 1
        if n_obj == 0:
            background += 1
        objects_per_image.append(n_obj)

    return {
        "n_images": len(images),
        "class_counts": class_counts,
        "objects_per_image": np.array(objects_per_image),
        "areas_px": np.array(areas_px),
        "resolutions": resolutions,
        "background": background,
    }


def bucket_counts(areas: np.ndarray) -> list[tuple[str, int, float]]:
    out = []
    for name, lo, hi in AREA_BUCKETS:
        n = int(((areas >= lo) & (areas < hi)).sum())
        out.append((name, n, 100.0 * n / len(areas)))
    return out


def fig_area_hist(splits: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, stats in splits.items():
        side = np.sqrt(stats["areas_px"])
        ax.hist(side, bins=np.logspace(0, np.log10(600), 60), alpha=0.6, label=f"{split} ({len(side):,} boxes)")
    ax.set_xscale("log")
    for _, lo, _ in AREA_BUCKETS[1:]:
        ax.axvline(np.sqrt(lo), color="gray", ls="--", lw=0.8)
    ax.text(16, ax.get_ylim()[1] * 0.95, "16px", ha="center", fontsize=8, color="gray")
    ax.text(32, ax.get_ylim()[1] * 0.95, "32px", ha="center", fontsize=8, color="gray")
    ax.text(96, ax.get_ylim()[1] * 0.95, "96px", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("bbox size sqrt(area) [px, log scale]")
    ax.set_ylabel("boxes")
    ax.set_title("VisDrone2019-DET bbox size distribution (native resolution)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_objects_hist(splits: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, stats in splits.items():
        ax.hist(stats["objects_per_image"], bins=60, alpha=0.6, label=split)
    ax.axvline(E2E_DET_CAP, color="red", ls="--", lw=1.2, label=f"e2e head cap ({E2E_DET_CAP})")
    ax.set_xlabel("objects per image")
    ax.set_ylabel("images")
    ax.set_title("VisDrone2019-DET objects per image")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets" / "VisDrone")
    parser.add_argument("--out", type=Path, default=Path("reports/dataset_stats.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    ensure_dataset(data_root)

    splits = {s: scan_split(data_root, s) for s in ("train", "val")}

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_area_hist(splits, args.figures_dir / "bbox_area_hist.png")
    fig_objects_hist(splits, args.figures_dir / "objects_per_image_hist.png")

    lines: list[str] = ["# VisDrone2019-DET dataset statistics", ""]
    lines.append("Generated by `scripts/dataset_stats.py`; all numbers measured from the")
    lines.append("converted YOLO labels at native image resolution.")
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
    for split, st in splits.items():
        opi = st["objects_per_image"]
        n_over = int((opi > E2E_DET_CAP).sum())
        truncated = int(np.clip(opi - E2E_DET_CAP, 0, None).sum())
        lines.append(
            f"- **{split}**: {n_over} images ({100 * n_over / len(opi):.1f}%) have more than "
            f"{E2E_DET_CAP} objects; a full-image detector capped at {E2E_DET_CAP} can miss at most "
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

    lines.append("## Bbox area buckets (native resolution)")
    lines.append("")
    lines.append("| bucket | train boxes | train % | val boxes | val % |")
    lines.append("|--------|-------------|---------|-----------|-------|")
    tb = bucket_counts(splits["train"]["areas_px"])
    vb = bucket_counts(splits["val"]["areas_px"])
    for (name, tn, tp), (_, vn, vp) in zip(tb, vb):
        lines.append(f"| {name} | {tn:,} | {tp:.1f}% | {vn:,} | {vp:.1f}% |")
    lines.append("")

    lines.append("## Image resolutions")
    lines.append("")
    for split, st in splits.items():
        top = st["resolutions"].most_common(5)
        res_str = ", ".join(f"{w}x{h} ({n})" for (w, h), n in top)
        lines.append(f"- **{split}** ({len(st['resolutions'])} distinct): {res_str}")
    lines.append("")
    lines.append("![bbox size distribution](figures/bbox_area_hist.png)")
    lines.append("")
    lines.append("![objects per image](figures/objects_per_image_hist.png)")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out} and figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
