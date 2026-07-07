"""Export yolo26s to ONNX and TensorRT FP16, benchmark inference latency.

Exports the end-to-end (NMS-free) ONNX and a TensorRT FP16 engine from the
trained 640 checkpoint, then benchmarks:
  - ONNX on CPU (onnxruntime)
  - TensorRT FP16 on this desktop RTX 4090
  - PyTorch .pt on the RTX 4090, as a reference point

batch=1, imgsz=640, single fixed val image, 10 warmup + 100 timed runs per
backend, reporting mean/p50/p95 wall-clock latency (ms). All numbers measured
on this desktop -- see README for the Jetson migration discussion.

Usage:
    python scripts/export_benchmark.py [--weights weights/yolo26s_visdrone_640.pt]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

WARMUP = 10
N_RUNS = 100


def timed_predict(model: YOLO, image: str, imgsz: int, device: str | int) -> dict[str, float]:
    for _ in range(WARMUP):
        model.predict(image, imgsz=imgsz, device=device, verbose=False)
    times_ms = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model.predict(image, imgsz=imgsz, device=device, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000 / arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=Path("weights/yolo26s_visdrone_640.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets" / "VisDrone")
    parser.add_argument("--out", type=Path, default=Path("reports/edge_benchmark.md"))
    args = parser.parse_args()

    image = str(sorted((args.data_root / "images" / "val").glob("*.jpg"))[0])
    print(f"benchmark image: {image}")

    results: dict[str, dict[str, float]] = {}

    print("\n=== PyTorch .pt (RTX 4090 reference) ===")
    pt_model = YOLO(str(args.weights))
    results["pytorch_4090"] = timed_predict(pt_model, image, args.imgsz, device=0)
    print(results["pytorch_4090"])

    print("\n=== exporting ONNX (end-to-end, NMS-free) ===")
    onnx_path = pt_model.export(format="onnx", imgsz=args.imgsz, batch=1, simplify=True)
    print(f"onnx: {onnx_path}")

    print("\n=== exporting TensorRT FP16 engine (RTX 4090) ===")
    engine_path = pt_model.export(format="engine", imgsz=args.imgsz, batch=1, quantize=16, device=0)
    print(f"engine: {engine_path}")

    print("\n=== benchmark: ONNX on CPU ===")
    onnx_model = YOLO(onnx_path)
    results["onnx_cpu"] = timed_predict(onnx_model, image, args.imgsz, device="cpu")
    print(results["onnx_cpu"])

    print("\n=== benchmark: TensorRT FP16 on RTX 4090 ===")
    engine_model = YOLO(engine_path)
    results["tensorrt_fp16_4090"] = timed_predict(engine_model, image, args.imgsz, device=0)
    print(results["tensorrt_fp16_4090"])

    lines = ["# Edge deployment benchmark", ""]
    lines.append(
        f"Weights: `{args.weights}`, imgsz={args.imgsz}, batch=1, {WARMUP} warmup + {N_RUNS} timed runs "
        "per backend, single fixed val image. All numbers measured on this desktop "
        "(RTX 4090 / host CPU) -- not a Jetson."
    )
    lines.append("")
    lines.append("| backend | mean (ms) | p50 (ms) | p95 (ms) | FPS |")
    lines.append("|---------|-----------|----------|----------|-----|")
    label = {"pytorch_4090": "PyTorch .pt (4090)", "onnx_cpu": "ONNX (CPU)",
              "tensorrt_fp16_4090": "TensorRT FP16 (4090)"}
    for key in ["pytorch_4090", "onnx_cpu", "tensorrt_fp16_4090"]:
        r = results[key]
        lines.append(f"| {label[key]} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['fps']:.1f} |")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    Path(args.out.with_suffix(".json")).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
