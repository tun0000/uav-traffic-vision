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

Three settings, one evaluation protocol (pycocotools, conf=0.01, maxDets=500 — dense
scenes overflow COCO's default 100), so every number below is directly comparable.
yolo26s trained 97 epochs at imgsz=640 (early-stopped, patience=20, mAP50-95 0.220 at
convergence) and a second run 74 epochs at imgsz=1024 (early-stopped, mAP50-95 0.293 —
epoch 54 was the actual best checkpoint). Per-class tables and figures:
[reports/evaluation.md](reports/evaluation.md) (640),
[reports/evaluation_1024.md](reports/evaluation_1024.md) (1024),
[reports/sahi_comparison.md](reports/sahi_comparison.md).

| Setting | AP50 | AP@[.5:.95] | AP tiny (<16px) | AP small | AP medium | AP large | ms/img (4090) |
|---------|------|-------------|------------------|---------|-----------|----------|----------------|
| yolo26s @ 640, direct | 0.380 | 0.222 | 0.075 | 0.184 | 0.316 | **0.480** | 10 |
| yolo26s @ 1024, direct | **0.480** | **0.291** | 0.119 | **0.261** | **0.403** | 0.463 | 10 |
| yolo26s @ 640 + SAHI 512/0.2 | 0.465 | 0.269 | **0.134** | 0.248 | 0.349 | 0.455 | 107 |

Batch=1 latency on the RTX 4090 is essentially unchanged from 640 to 1024 (10.2 vs 10.0
ms/img) — at this model scale the GPU isn't compute-saturated at either resolution, so
fixed overhead (preprocessing, kernel launch) dominates over the ~2.6x difference in
pixel count. This makes the 1024 checkpoint a strictly better choice than 640 on this
hardware: higher accuracy at no latency cost. (This may not hold on smaller/edge GPUs —
see Edge deployment below.)

Two independent evaluation paths agree closely at 640 (ultralytics `model.val()`:
mAP50-95 0.220; the pycocotools pipeline above: 0.222) and at 1024 (0.293 vs 0.291),
cross-validating the tooling.

### Small-object breakdown: three techniques, no single winner

No approach dominates every bucket — each wins where its mechanism actually applies:

- **SAHI wins the smallest objects** (AP tiny 0.134 vs 1024's 0.119 vs 640's 0.075):
  slicing shows a tile at closer to native resolution than any single resize can,
  even a resize to 1024.
- **Higher training resolution wins small-to-medium objects** (AP small 0.261, AP
  medium 0.403 — both the best of the three) and does it in a single forward pass,
  no slicing overhead.
- **Plain 640 direct is (barely) best on large objects** (0.480) — both SAHI's tiling
  and 1024's downsampling introduce small costs there that resolution/slicing don't
  pay back.

In short: resolution is the more efficient lever for the bulk of the size distribution,
but SAHI still earns its 10x latency cost specifically on the smallest, hardest objects
that resolution alone doesn't fully recover.

| Dense scene (187 detections) | Small-object-heavy scene |
|---|---|
| ![dense scene](reports/figures/dense_0000295_02400_d_0000033.jpg) | ![small-object scene](reports/figures/tiny_0000242_06010_d_0000017.jpg) |

More examples in [reports/evaluation.md](reports/evaluation.md).

## SAHI sliced inference

[SAHI](https://github.com/obss/sahi) slices each image into overlapping tiles, runs the
detector per tile, and merges the results — so a 15px pedestrian is seen at ~2x scale
instead of shrinking into a few pixels at imgsz=640. A 3-config sweep (512/640/800 tile
sizes, 100-image subset) picked **512×512 tiles, 0.2 overlap** by tiny-bucket AP. Full
report: [reports/sahi_comparison.md](reports/sahi_comparison.md).

On the full val set, SAHI vs direct inference:

- **AP50 +22% relative** (0.380 → 0.465), overall AP +21% (0.222 → 0.269)
- **Tiny-object AP +79% relative** (0.075 → 0.134), tiny-object recall AR@500 +75%
  (0.181 → 0.316) — the buckets that dominate this dataset
- Honest trade-offs: **10.2x per-image latency** (10 → 107 ms on RTX 4090), and large-object
  AP dips slightly (0.480 → 0.455) since tiling can fragment big objects — the built-in
  full-image pass (`perform_standard_pred=True`) recovers most but not all of it

Slicing also side-steps YOLO26's end-to-end 300-detections-per-pass budget: at evaluation
confidence (0.01), **323 of 548 val images saturate the cap in direct mode**, while SAHI's
per-tile budget let the densest scene emit 966 detections. (At practical confidence
thresholds like 0.25 the cap binds far less often — but for dense aerial scenes it is a
real ceiling on recall.)

![sahi vs direct](reports/figures/sahi_vs_direct_0000242_06010_d_0000017.jpg)

*Same image, same model, same confidence threshold — direct 640 finds 10 objects, SAHI
finds 24 (distant vehicles, roadside pedestrians, the truck on the right).*

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

# 6. direct vs SAHI comparison (sweep + full val + side-by-side figures)
uv run python scripts/sahi_compare.py --weights weights/yolo26s_visdrone_640.pt
```

## License & dataset attribution

Code is MIT licensed. The [VisDrone2019 dataset](https://github.com/VisDrone/VisDrone-Dataset)
(AISKYEYE team, Tianjin University) is available for **academic / research use only**;
this project uses it for non-commercial portfolio research and does not redistribute
the data. Model weights trained on VisDrone inherit that restriction.
