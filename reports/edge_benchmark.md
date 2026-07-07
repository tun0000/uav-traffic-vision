# Edge deployment benchmark

Weights: `weights/yolo26s_visdrone_640.pt`, imgsz=640, batch=1, 10 warmup + 100 timed runs per backend, single fixed val image. All numbers measured on this desktop (RTX 4090 / host CPU) -- not a Jetson.

| backend | mean (ms) | p50 (ms) | p95 (ms) | FPS |
|---------|-----------|----------|----------|-----|
| PyTorch .pt (4090) | 13.41 | 13.35 | 14.40 | 74.6 |
| ONNX (CPU) | 48.95 | 49.77 | 53.05 | 20.4 |
| TensorRT FP16 (4090) | 11.52 | 11.34 | 14.11 | 86.8 |
