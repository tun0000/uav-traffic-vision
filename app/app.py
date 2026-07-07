"""Gradio demo: drone-image object detection with yolo26s (VisDrone2019-DET).

Runs on CPU ONNX by design, so it's deployable to a free HF Space (no GPU
required). SAHI sliced inference is available as an opt-in toggle -- it is
meaningfully slower on CPU, so it defaults off with a warning.

Usage:
    python app/app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from ultralytics import YOLO

HF_REPO = "betty0/uav-traffic-vision"
ONNX_FILENAME = "yolo26s_visdrone_640.onnx"
_local_weights = Path(__file__).parent.parent / "weights" / ONNX_FILENAME
if _local_weights.exists():
    WEIGHTS = _local_weights
else:  # HF Space deployment: fetch from the model repo instead of bundling twice
    from huggingface_hub import hf_hub_download

    WEIGHTS = Path(hf_hub_download(repo_id=HF_REPO, filename=ONNX_FILENAME))
IMGSZ = 640
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

_direct_model: YOLO | None = None
_sahi_model = None  # lazy: SAHI wraps its own onnxruntime session


def get_direct_model() -> YOLO:
    global _direct_model
    if _direct_model is None:
        _direct_model = YOLO(str(WEIGHTS))
    return _direct_model


def get_sahi_model():
    global _sahi_model
    if _sahi_model is None:
        _sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(WEIGHTS),
            confidence_threshold=0.25, device="cpu", image_size=IMGSZ,
            category_mapping={str(i): n for i, n in enumerate(CLASS_NAMES)},
        )
    return _sahi_model


def detect(image: Image.Image, conf: float, use_sahi: bool) -> tuple[Image.Image, str]:
    if image is None:
        return None, "Upload an image or pick an example below."

    t0 = time.perf_counter()
    if use_sahi:
        model = get_sahi_model()
        model.confidence_threshold = conf
        result = get_sliced_prediction(
            np.array(image), model, slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0,
        )
        elapsed = time.perf_counter() - t0
        out = visualize_object_predictions(
            np.array(image), object_prediction_list=result.object_prediction_list,
        )["image"]
        out_image = Image.fromarray(out)
        n = len(result.object_prediction_list)
        mode = "SAHI (512x512 tiles, CPU)"
    else:
        model = get_direct_model()
        r = model.predict(image, imgsz=IMGSZ, conf=conf, device="cpu", verbose=False)[0]
        elapsed = time.perf_counter() - t0
        out_image = Image.fromarray(r.plot()[..., ::-1])  # BGR -> RGB
        n = len(r.boxes)
        mode = "Direct (640, CPU)"

    caption = f"{mode} -- {n} detections in {elapsed:.2f}s"
    return out_image, caption


with gr.Blocks(title="UAV Traffic Vision - YOLO26 on VisDrone") as demo:
    gr.Markdown(
        "# UAV Traffic Vision -- drone-view object detection\n"
        "yolo26s trained on [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) "
        "(10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, "
        "awning-tricycle, bus, motor). Runs on **CPU ONNX** here; see the "
        "[GitHub repo](https://github.com/tun0000/uav-traffic-vision) for the SAHI "
        "vs direct comparison, traffic counting, and edge deployment benchmarks. "
        "VisDrone is released for **academic/research use only**."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image (aerial/drone view)")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            sahi_toggle = gr.Checkbox(
                value=False,
                label="Use SAHI sliced inference (finds smaller objects, much slower on CPU)",
            )
            btn = gr.Button("Detect", variant="primary")
            gr.Examples(
                examples=[str(p) for p in sorted(Path(__file__).parent.glob("examples/*.jpg"))],
                inputs=inp,
            )
        with gr.Column():
            out = gr.Image(type="pil", label="Detections")
            caption = gr.Textbox(label="Result", interactive=False)

    btn.click(detect, inputs=[inp, conf, sahi_toggle], outputs=[out, caption])
    inp.change(detect, inputs=[inp, conf, sahi_toggle], outputs=[out, caption])

if __name__ == "__main__":
    demo.launch()
