# Traffic flow counting

Source: [VisDrone2019-MOT val](https://huggingface.co/datasets/Voxel51/visdrone-mot)
(CC BY-SA, AISKYEYE team / Tianjin University), scene `uav0000137_00458_v` — a
near-stationary hover over a dense urban intersection (233 frames, 9.3s @ 25fps,
2688x1512). Chosen over the other 6 val scenes because a fixed counting line only
makes sense with a (near-)stationary camera, and this scene had both the clearest
stationary framing (first/last frame nearly identical) and the richest mixed
traffic (cars, motorcycles, bicycles, pedestrians).

Pipeline: `yolo26s_visdrone_640.pt` detection at imgsz=1280, `bytetrack.yaml` for
identity, then a virtual line counter — proper segment-segment intersection (not
an infinite-line side flip, which would double-count vehicles that hover near the
line), each track ID counted at most once, direction from the line's normal vector.
Only vehicle classes (car/van/truck/tricycle/awning-tricycle/bus/motor) are
counted; pedestrian/people/bicycle are detected and drawn but excluded from counts.

## Results

| direction | car | van | motor | total |
|-----------|-----|-----|-------|-------|
| dir_pos (green arrow) | 21 | 0 | 2 | 23 |
| dir_neg (orange arrow) | 7 | 2 | 2 | 11 |
| **total** | 28 | 2 | 4 | **34** |

Full machine-readable output: [reports/traffic/stats.json](traffic/stats.json).
Spot-checked by sampling the live count overlay at t=2s/6s/9.2s (4/1 -> 11/4 ->
23/11): the running total grows monotonically and the final sampled frame matches
`stats.json` exactly, with no implausible jumps.

**Known limitation**: counting is per track ID, so a track that gets lost and
re-acquired mid-crossing (brief occlusion, e.g. behind another vehicle) gets a new
ID and could be counted twice. Not observed in this clip's totals, but it's a real
failure mode of ID-based line counting worth flagging rather than glossing over.

![traffic counting demo](figures/traffic_demo.gif)
