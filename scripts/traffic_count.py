"""Traffic flow counting on drone footage: YOLO26 + ByteTrack + a virtual counting line.

Requires (near-)stationary footage — a hovering drone watching a road — since a
counting line fixed in image space is meaningless if the camera pans. Each track
is counted at most once, when its center path crosses the line segment
(proper segment-segment intersection, not infinite-line side flips), with the
crossing direction recorded relative to the line's normal. Vehicle classes are
counted; pedestrian/people/bicycle are drawn but never counted.

Outputs an annotated MP4 (boxes, track trails, live count board), stats.json,
and optionally a README-friendly GIF cut from the annotated video.

Usage:
    python scripts/traffic_count.py --video ~/datasets/VisDrone-MOT-val/<scene>.mp4 \\
        --line 0.05,0.55,0.95,0.55 [--gif-start 10 --gif-len 8]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
VEHICLE_IDS = {3, 4, 5, 6, 7, 8, 9}  # car, van, truck, tricycle, awning-tricycle, bus, motor

# BGR palette, one color per class (same ordering as CLASS_NAMES)
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
]


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, q1, q2) -> bool:
    """Proper segment-segment intersection (touching endpoints count)."""
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0) or d1 == 0 or d2 == 0) and ((d3 > 0) != (d4 > 0) or d3 == 0 or d4 == 0):
        return True
    return False


class LineCounter:
    """Counts each track once when its center path crosses the line segment."""

    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        v = self.p2 - self.p1
        self.normal = np.array([v[1], -v[0]]) / (np.hypot(*v) + 1e-9)  # +normal side
        self.last_pt: dict[int, np.ndarray] = {}
        self.counted: set[int] = set()
        self.counts = {"dir_pos": Counter(), "dir_neg": Counter()}  # class name -> n

    def update(self, track_id: int, cls_id: int, pt: np.ndarray) -> str | None:
        prev = self.last_pt.get(track_id)
        self.last_pt[track_id] = pt
        if prev is None or track_id in self.counted or cls_id not in VEHICLE_IDS:
            return None
        if not segments_intersect(prev, pt, self.p1, self.p2):
            return None
        direction = "dir_pos" if float(np.dot(pt - prev, self.normal)) > 0 else "dir_neg"
        self.counts[direction][CLASS_NAMES[cls_id]] += 1
        self.counted.add(track_id)
        return direction

    def totals(self) -> dict[str, int]:
        return {d: sum(c.values()) for d, c in self.counts.items()}


def parse_line(spec: str, w: int, h: int) -> tuple[tuple[float, float], tuple[float, float]]:
    vals = [float(v) for v in spec.split(",")]
    if len(vals) != 4:
        raise SystemExit("--line expects x1,y1,x2,y2 (normalized 0-1 or pixels)")
    if max(vals) <= 1.5:  # normalized
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    return (vals[0], vals[1]), (vals[2], vals[3])


def draw_frame(
    frame: np.ndarray, boxes, ids, clss, confs,
    counter: LineCounter, trails: dict[int, deque], flash: dict[int, int],
) -> np.ndarray:
    out = frame
    h, w = out.shape[:2]
    thick = max(1, round(min(h, w) / 500))

    # counting line + normal arrows with running totals
    p1, p2 = counter.p1.astype(int), counter.p2.astype(int)
    cv2.line(out, tuple(p1), tuple(p2), (0, 0, 255), thick * 2)
    mid = ((counter.p1 + counter.p2) / 2).astype(int)
    arrow = (counter.normal * 60 * (min(h, w) / 800)).astype(int)
    tot = counter.totals()
    cv2.arrowedLine(out, tuple(mid), tuple(mid + arrow), (0, 220, 0), thick * 2, tipLength=0.35)
    cv2.arrowedLine(out, tuple(mid), tuple(mid - arrow), (0, 165, 255), thick * 2, tipLength=0.35)
    cv2.putText(out, str(tot["dir_pos"]), tuple(mid + arrow * 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * thick / 2, (0, 220, 0), thick, cv2.LINE_AA)
    cv2.putText(out, str(tot["dir_neg"]), tuple(mid - arrow * 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * thick / 2, (0, 165, 255), thick, cv2.LINE_AA)

    for (x0, y0, x1, y1), tid, cls_id, conf in zip(boxes, ids, clss, confs):
        color = PALETTE[cls_id % len(PALETTE)]
        counted_now = flash.get(tid, 0) > 0
        bthick = thick * 3 if counted_now else thick
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, bthick)
        label = f"{tid}"
        cv2.putText(out, label, (int(x0), int(y0) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * thick / 2 + 0.25, color, max(1, thick - 1), cv2.LINE_AA)
        trail = trails.get(tid)
        if trail is not None and len(trail) > 1:
            pts = np.array(trail, dtype=np.int32)
            cv2.polylines(out, [pts], False, color, max(1, thick - 1))

    # count board (top-left)
    counted_classes = [CLASS_NAMES[i] for i in sorted(VEHICLE_IDS)]
    rows = [c for c in counted_classes
            if counter.counts["dir_pos"][c] or counter.counts["dir_neg"][c]]
    line_h = int(26 * min(h, w) / 800)
    pad = line_h // 2
    board_h = (len(rows) + 2) * line_h + pad * 2
    board_w = int(300 * min(h, w) / 800)
    overlay = out[0:board_h, 0:board_w].copy()
    cv2.rectangle(overlay, (0, 0), (board_w, board_h), (25, 25, 25), -1)
    out[0:board_h, 0:board_w] = cv2.addWeighted(overlay, 0.65, out[0:board_h, 0:board_w], 0.35, 0)
    fs = 0.55 * line_h / 26
    y = pad + line_h
    cv2.putText(out, "vehicle counts  +dir / -dir", (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (255, 255, 255), 1, cv2.LINE_AA)
    for c in rows:
        y += line_h
        cv2.putText(out, f"{c}: {counter.counts['dir_pos'][c]} / {counter.counts['dir_neg'][c]}",
                    (pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    PALETTE[CLASS_NAMES.index(c) % len(PALETTE)], 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(out, f"total: {tot['dir_pos']} / {tot['dir_neg']}", (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_gif(mp4: Path, gif: Path, start: float, length: float, width: int = 480, fps: int = 8) -> None:
    # 480px/8fps keeps a dense, busy intersection scene under ~8MB (800/12 -> ~29MB)
    palette = gif.with_suffix(".png")
    common = ["-ss", str(start), "-t", str(length), "-i", str(mp4)]
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *common,
                    "-vf", f"{filters},palettegen", str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *common, "-i", str(palette),
                    "-lavfi", f"{filters} [x]; [x][1:v] paletteuse", str(gif)], check=True)
    palette.unlink()
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("weights/yolo26s_visdrone_640.pt"))
    parser.add_argument("--line", type=str, required=True, help="x1,y1,x2,y2 (normalized 0-1 or px)")
    parser.add_argument("--imgsz", type=int, default=1280, help="inference size (aerial small objects)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--trail", type=int, default=40, help="trail length in frames")
    parser.add_argument("--max-frames", type=int, default=0, help="limit frames (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/traffic"))
    parser.add_argument("--gif", type=Path, default=None, help="optional output GIF path")
    parser.add_argument("--gif-start", type=float, default=0.0)
    parser.add_argument("--gif-len", type=float, default=8.0)
    args = parser.parse_args()

    video = args.video.expanduser()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    p1, p2 = parse_line(args.line, w, h)
    counter = LineCounter(p1, p2)
    print(f"video {video.name}: {w}x{h} @ {fps:.0f} fps, line {p1} -> {p2}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_mp4 = args.out_dir / f"{video.stem}_annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(str(args.weights))
    trails: dict[int, deque] = {}
    flash: dict[int, int] = {}  # track_id -> frames left to highlight after counting
    n_frames = 0
    t0 = time.perf_counter()

    results = model.track(
        source=str(video), stream=True, tracker=args.tracker,
        conf=args.conf, imgsz=args.imgsz, verbose=False,
    )
    for r in results:
        frame = r.orig_img
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
        else:
            boxes, ids, clss, confs = np.zeros((0, 4)), [], [], []

        for (x0, y0, x1, y1), tid, cls_id in zip(boxes, ids, clss):
            center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            trails.setdefault(tid, deque(maxlen=args.trail)).append((int(center[0]), int(center[1])))
            if counter.update(tid, cls_id, center):
                flash[tid] = int(fps)  # highlight for ~1s

        writer.write(draw_frame(frame, boxes, ids, clss, confs, counter, trails, flash))
        flash = {k: v - 1 for k, v in flash.items() if v > 1}
        n_frames += 1
        if args.max_frames and n_frames >= args.max_frames:
            break

    writer.release()
    elapsed = time.perf_counter() - t0
    print(f"processed {n_frames} frames in {elapsed:.1f}s ({n_frames / elapsed:.1f} fps)")

    out_mp4 = args.out_dir / f"{video.stem}_annotated.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_mp4),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_mp4)], check=True)
    raw_mp4.unlink()
    print(f"wrote {out_mp4}")

    stats = {
        "video": video.name,
        "weights": args.weights.name,
        "tracker": args.tracker,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "frames": n_frames,
        "fps": fps,
        "duration_s": round(n_frames / fps, 1),
        "line_px": {"p1": list(p1), "p2": list(p2)},
        "counted_classes": [CLASS_NAMES[i] for i in sorted(VEHICLE_IDS)],
        "counts": {d: dict(c) for d, c in counter.counts.items()},
        "totals": counter.totals(),
        "total_vehicles": sum(counter.totals().values()),
        "note": "each track id counted at most once; dir_pos = crossing along the "
                "green arrow (line normal), dir_neg = along the orange arrow",
    }
    stats_path = args.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")

    if args.gif:
        make_gif(out_mp4, args.gif, args.gif_start, args.gif_len)


if __name__ == "__main__":
    main()
