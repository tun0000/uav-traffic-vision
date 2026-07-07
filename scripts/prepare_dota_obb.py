"""Tile DOTA-v1.0 with split_dota and carve out a small OBB smoke-test subset.

split_dota crops the raw (up to 20000px) DOTA images into overlapping 1024x1024
tiles -- ultralytics' documented preprocessing step before training, since YOLO
can't reasonably ingest a 20000px image directly. This script runs that tiling
locally (single-scale, i.e. split_dota's own defaults: crop=1024, gap=200,
rate=1.0 -- multiscale rates=[0.5,1,1.5] would triple both preprocessing time
and training-set size for further mAP gain, left as a documented option rather
than the default to control Colab compute cost) to produce the full
training-ready dataset, then samples a small subset of the resulting tiles for
a 4090 smoke test before handing full training to Colab.

Usage:
    python scripts/prepare_dota_obb.py
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml
from ultralytics.data.split_dota import split_trainval

CLASS_NAMES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
]


def ensure_split(data_root: Path, split_root: Path) -> None:
    if (split_root / "images" / "train").exists():
        print(f"{split_root} already exists, skipping split_trainval")
        return
    print(f"tiling {data_root} -> {split_root} (single-scale, crop=1024, gap=200)")
    split_trainval(data_root=str(data_root), save_dir=str(split_root))


def copy_subset(split_root: Path, subset_root: Path, n_train: int, n_val: int, seed: int) -> None:
    if subset_root.exists():
        shutil.rmtree(subset_root)

    for split, n in (("train", n_train), ("val", n_val)):
        images = sorted((split_root / "images" / split).glob("*.jpg"))
        if len(images) < n:
            raise SystemExit(f"{split}: requested {n} tiles but only {len(images)} available")
        picked = random.Random(seed).sample(images, n)

        img_dst = subset_root / "images" / split
        lbl_dst = subset_root / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        labeled = 0
        for img in picked:
            shutil.copy2(img, img_dst / img.name)
            lbl = split_root / "labels" / split / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, lbl_dst / lbl.name)
                labeled += 1
        print(f"{split}: copied {n} tiles, {labeled} with labels -> {img_dst}")

    data_yaml = {
        "path": str(subset_root),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    yaml_path = subset_root / "dota_obb_subset.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"wrote {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets" / "DOTAv1")
    parser.add_argument("--split-root", type=Path, default=Path.home() / "datasets" / "DOTAv1-split")
    parser.add_argument("--subset-root", type=Path, default=Path.home() / "datasets" / "DOTAv1_obb_subset")
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    split_root = args.split_root.expanduser()
    subset_root = args.subset_root.expanduser()

    ensure_split(data_root, split_root)
    copy_subset(split_root, subset_root, args.n_train, args.n_val, args.seed)


if __name__ == "__main__":
    main()
