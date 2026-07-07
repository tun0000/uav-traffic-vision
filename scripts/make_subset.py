"""Build a small VisDrone subset for local smoke tests.

Randomly samples images (with their labels) from the converted VisDrone dataset
into a sibling directory and writes a data yaml pointing at it, so a 1-epoch
training run can validate the whole pipeline on the local GPU before spending
Colab credits.

Usage:
    python scripts/make_subset.py [--n-train 300] [--n-val 100]
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def copy_sample(src_root: Path, dst_root: Path, split: str, n: int, seed: int) -> int:
    images = sorted((src_root / "images" / split).glob("*.jpg"))
    if len(images) < n:
        raise SystemExit(f"{split}: requested {n} images but only {len(images)} available")
    picked = random.Random(seed).sample(images, n)

    img_dst = dst_root / "images" / split
    lbl_dst = dst_root / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    copied_labels = 0
    for img in picked:
        shutil.copy2(img, img_dst / img.name)
        lbl = src_root / "labels" / split / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, lbl_dst / lbl.name)
            copied_labels += 1
    return copied_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets" / "VisDrone")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "datasets" / "VisDrone_subset")
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = args.data_root.expanduser()
    dst = args.out_dir.expanduser()
    if dst.exists():
        shutil.rmtree(dst)

    for split, n in (("train", args.n_train), ("val", args.n_val)):
        labels = copy_sample(src, dst, split, n, args.seed)
        print(f"{split}: copied {n} images, {labels} label files -> {dst / 'images' / split}")

    data_yaml = {
        "path": str(dst),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    yaml_path = dst / "visdrone_subset.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
