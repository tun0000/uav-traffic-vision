# UAV Traffic Vision — Drone-View Detection & Traffic Flow Analysis with YOLO26

Small-object detection from drone imagery using
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), trained on
[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) (10 classes:
pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor),
with [SAHI](https://github.com/obss/sahi) sliced inference for dense tiny objects,
ByteTrack-based traffic flow counting, and an edge-deployment benchmark
(ONNX / TensorRT FP16).

> **Status: work in progress** — 640 baseline trained and evaluated; SAHI comparison,
> traffic counting, edge deployment, and publishing still pending.

## Why this matters

<!-- TODO(Phase 2): motivation — Taiwan's growing drone industry: infrastructure
inspection, coast guard patrol, smart traffic monitoring. Small-object detection
from aerial viewpoints is the shared core capability. -->

## Dataset

VisDrone2019-DET — 6,471 train / 548 val images, 10 classes, auto-downloaded and
converted by ultralytics' built-in `VisDrone.yaml`. Key findings from the EDA
([full report](reports/dataset_stats.md)):

- **Tiny objects dominate**: 60.5% of train boxes (68.5% of val) are smaller than
  32×32 px at native resolution — the core motivation for higher input resolution
  and SAHI sliced inference.
- **Dense scenes**: 53 objects per image on average (train), up to 902. Only
  0.2–0.5% of images exceed the 300-detection cap of YOLO26's end-to-end head,
  so the cap matters far less than object size does.
- **Heavy class imbalance**: 144.9k car instances vs 3.2k awning-tricycle.

![bbox size distribution](reports/figures/bbox_area_hist.png)

## Results

yolo26s trained 97 epochs (early-stopped, patience=20) at imgsz=640 on VisDrone2019-DET.
Full breakdown, per-class table, and overlay figures in
[reports/evaluation.md](reports/evaluation.md).

| Setting | mAP50 | mAP50-95 | AP (small) | AR@100 |
|---------|-------|----------|------------|--------|
| yolo26s @ 640, direct | 0.379 | 0.220 | 0.184 | 0.354 |
| yolo26s @ 1024, direct (optional) | TBD | TBD | TBD | TBD |
| yolo26s @ 640 + SAHI | TBD | TBD | TBD | TBD |

Two independent evaluation paths agree closely (ultralytics `model.val()`: mAP50-95
0.220; an independent pycocotools COCO-eval built from scratch: 0.223), cross-validating
the pipeline.

### Small-object breakdown

The dataset's small-object skew (see EDA above) shows up directly in accuracy — AP more
than doubles from the smallest to the largest bucket:

| bucket (bbox side) | AP@[.5:.95] | AR@100 |
|---------------------|-------------|--------|
| tiny (<16px) | 0.075 | 0.184 |
| small (16-32px) | 0.184 | 0.333 |
| medium (32-96px) | 0.318 | 0.462 |
| large (>96px) | 0.480 | 0.595 |

This gap is exactly what the SAHI comparison below is meant to close.

| Dense scene (187 detections) | Small-object-heavy scene |
|---|---|
| ![dense scene](reports/figures/dense_0000295_02400_d_0000033.jpg) | ![small-object scene](reports/figures/tiny_0000242_06010_d_0000017.jpg) |

More examples in [reports/evaluation.md](reports/evaluation.md).

## SAHI sliced inference

<!-- TODO(Phase 2): direct vs SAHI side-by-side visualization + metric deltas on
tiny/small buckets + latency trade-off -->

## Traffic flow counting

<!-- TODO(Phase 2): main GIF — detection + ByteTrack + virtual counting line,
per-class vehicle counts (stats.json) -->

## Edge deployment

<!-- TODO(Phase 2): ONNX CPU / TensorRT FP16 RTX 4090 benchmark table (measured on
a desktop, stated honestly), the path to Jetson-class onboard computers and expected
bottlenecks, and why YOLO26's NMS-free end-to-end head matters for onboard deployment -->

## Demo

<!-- TODO(Phase 2): Gradio demo (CPU ONNX, optional SAHI toggle) + HF Space link -->

## Reproduce

Prerequisites: [uv](https://docs.astral.sh/uv/). The VisDrone2019-DET dataset
(~2.3 GB) is downloaded automatically by ultralytics on first use.

```bash
# 1. install dependencies
uv sync

# 2. dataset EDA (triggers the VisDrone auto-download on first run)
uv run python scripts/dataset_stats.py

# 3. local smoke test (small subset, 1 epoch, sanity check)
uv run python scripts/make_subset.py

# 4. train — open notebooks/train_yolo26_visdrone_colab.ipynb in Google Colab (Runtime -> Run all)

# 5. evaluate a trained checkpoint (overall + per-class + small-object breakdown)
uv run python scripts/evaluate.py --weights weights/yolo26s_visdrone_640.pt
```

## License & dataset attribution

Code is MIT licensed. The [VisDrone2019 dataset](https://github.com/VisDrone/VisDrone-Dataset)
(AISKYEYE team, Tianjin University) is available for **academic / research use only**;
this project uses it for non-commercial portfolio research and does not redistribute
the data. Model weights trained on VisDrone inherit that restriction.
