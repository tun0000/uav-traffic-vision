"""Fetch the VisDrone2019-MOT val split and organize it into per-scene sequences.

Downloads the Voxel51/visdrone-mot mirror from Hugging Face (val split, 7
sequences, 2,846 frames, ~1.7 GB, CC BY-SA — attribution: AISKYEYE team,
Tianjin University), reads the FiftyOne samples.json to recover scene ids and
frame order, then:
  - links frames into <out>/sequences/<scene_id>/NNNNNN.jpg
  - encodes a preview MP4 per scene (for quick visual inspection)
  - writes <out>/scenes.json with per-scene frame counts and resolutions

Usage:
    python scripts/fetch_visdrone_mot.py [--out ~/datasets/VisDrone-MOT-val]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

REPO_ID = "Voxel51/visdrone-mot"


def load_samples(repo_dir: Path) -> dict[str, list[tuple[int, str]]]:
    """Return {scene_id: [(frame_number, relative filepath), ...] sorted}."""
    data = json.loads((repo_dir / "samples.json").read_text())
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    scenes: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for s in samples:
        scene = str(s.get("scene_id", s.get("scene", "unknown")))
        frame = int(s.get("frame_number", s.get("frame", 0)))
        scenes[scene].append((frame, s["filepath"]))
    for seq in scenes.values():
        seq.sort()
    return dict(scenes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path.home() / "datasets" / "VisDrone-MOT-val")
    parser.add_argument("--fps", type=int, default=25, help="preview mp4 framerate")
    args = parser.parse_args()

    out = args.out.expanduser()
    repo_dir = out / "hf"
    print(f"downloading {REPO_ID} -> {repo_dir} (~1.7 GB on first run)")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", local_dir=repo_dir,
        ignore_patterns=["*.gif"],  # skip the 43 MB demo gif
    )

    scenes = load_samples(repo_dir)
    print(f"scenes: {len(scenes)}")

    seq_root = out / "sequences"
    summary = {}
    for scene, frames in sorted(scenes.items()):
        seq_dir = seq_root / scene
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, (_, rel_path) in enumerate(frames, start=1):
            src = repo_dir / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
            if not src.exists():  # FiftyOne exports sometimes store bare filenames
                src = repo_dir / "data" / Path(rel_path).name
            dst = seq_dir / f"{i:06d}.jpg"
            if not dst.exists():
                shutil.copy2(src, dst)

        first = next(seq_dir.glob("*.jpg"))
        with Image.open(first) as im:
            w, h = im.size
        mp4 = out / f"{scene}.mp4"
        if not mp4.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                 "-i", str(seq_dir / "%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 # libx264 requires even dimensions; some VisDrone-MOT frames are odd (e.g. 1904x1071)
                 "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                 "-crf", "23", str(mp4)],
                check=True,
            )
        summary[scene] = {"frames": len(frames), "width": w, "height": h, "mp4": str(mp4)}
        print(f"  {scene}: {len(frames)} frames, {w}x{h} -> {mp4.name}")

    (out / "scenes.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'scenes.json'}")


if __name__ == "__main__":
    main()
