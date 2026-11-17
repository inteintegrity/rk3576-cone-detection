# Formula Student Cone Detection on reComputer RK3576

Four-class cone detection using the **original MathWorks pretrained YOLOX-tiny
weights**, converted to RKNN without training. FastAPI provides an English video
upload interface, processing previews, per-class counts and downloadable H.264 MP4.

Upstream: [Cone Detection Using Deep Learning for Formula Student Driverless](https://github.com/matlab-deep-learning/Cone-Detection-Using-Deep-Learning-for-Formula-Student-Driverless).
The official test footage is credited upstream to DIAN Racing, Tongji University.

## Repository and deployment assets

```text
rk3576-formula-student-cone-detection/
  github/                 # This clean source tree; suitable for GitHub
    app.py                # FastAPI upload, job status, preview and download API
    cones/                # Actual RKNNLite/ONNX inference, decoding and video jobs
    web/index.html        # English responsive UI; no frontend build required
    tools/                # Original asset download, MAT extraction, RKNN conversion
    scripts/run.sh        # Board server launcher
    tests/                # Geometry, class order and NMS regression checks
  rk-workspace/           # Local-only models, upstream archive, evidence and bundles
```

Model binaries, source videos, Python environments, caches, uploaded files,
intermediate ONNX/MAT files and generated videos are excluded from the source
repository. `tools/package_deploy.py` builds a separate board bundle containing
source and the compiled RKNN. No GitHub repository is created or published by these tools.

## Classes and counts

| ID | Original label | Meaning |
|---|---|---|
| 0 | `greenCone` | Green cone |
| 1 | `redCone` | Red cone |
| 2 | `yellowConeLeft` | Yellow left-side cone class |
| 3 | `yellowConeRight` | Yellow right-side cone class |

The model itself predicts these classes; color labels are not assigned by HSV
rules. Yellow left/right are upstream learned labels and may be confused. Counts
are **per-frame detections**, not unique cones over a video. JSON also reports
per-class peak counts and sums of detection observations (including repeats).
The default confidence threshold is 0.35 and class-aware NMS IoU is 0.45.

The official onboard camera sees two red triangular car markings near the bottom
of the frame. The original model sometimes predicts them as red or yellow cones.
The default **Official onboard** camera profile excludes detections whose box
centers lie in two configured lower vehicle regions. It removed 146 repeated
marker observations in the full test clip without changing the model or raising
the confidence threshold. The normalized polygons in `cones/profiles.py` are
specific to this fixed camera view. Choose **Other camera** in the WebUI (or
`--profile none` on the CLI) for different footage. Raw and excluded predictions
remain in each JSONL trace record. A result URL retains its job ID across page
refreshes while the same server process is running.

## Model conversion, without MATLAB

Use Windows/Linux Python with NumPy, SciPy and ONNX for extraction. Use **x86_64
Linux or WSL** with RKNN Toolkit2 2.3.2 for compilation. The extractor reads the
specific upstream MATLAB v5 MCOS serialization and reconstructs its saved layer
graph, convolution/BN parameters, SiLU, Focus and nearest resize operations.
It checks the source SHA-256 and rejects other MAT files. No weights are retrained.

```bash
python tools/fetch_assets.py --output ../rk-workspace/assets
python tools/export_mat.py \
  --source ../rk-workspace/assets/yoloxConeDetector.mat \
  --output ../rk-workspace/models/cone_yolox.onnx

# In the Linux conversion environment:
python -m pip install -r requirements-convert.txt
python tools/convert_rknn.py \
  --onnx ../rk-workspace/models/cone_yolox.onnx \
  --output ../rk-workspace/models/cone_yolox_rk3576_fp16.rknn
```

Input: raw RGB 0–255, fixed 416×416, batch 1. Letterboxing uses padding 114.
The ONNX includes the original ImageNet z-score normalization; **do not divide
the input by 255 again**. RKNN receives NHWC uint8; ONNX receives NCHW float32.
Outputs are three raw NCHW maps `[1,9,52,52]`, `[1,9,26,26]`, `[1,9,13,13]`:
`tx,ty,tw,th,objectness,4 class logits`. CPU postprocessing applies the YOLOX
stride/grid decode, sigmoid(objectness) × sigmoid(class), coordinate restoration
and NMS. These are not Ultralytics DFL outputs.

Optional `--dataset calibration.txt` enables INT8 conversion. FP16 is the default
validated deployment path; INT8 requires representative, identically letterboxed
calibration RGB images and a separate accuracy comparison.

The extractor has no MATLAB-runtime numerical parity test because MATLAB is not
installed on this workstation. That limitation is separate from ONNX versus
**actual board RKNN** comparisons. See [validation](docs/validation.md).

## Install and run on RK3576

Use Debian/Ubuntu ARM64 with a functioning RKNPU driver, `librknnrt.so`, and the
official matching **RKNN Toolkit Lite2** wheel. Toolkit2 is a conversion tool;
Lite2 is the board runtime. See [Rockchip Toolkit2](https://github.com/airockchip/rknn-toolkit2).

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Install the official aarch64 Lite2 2.3.2 wheel matching your Python version.
python -m pip install /path/to/rknn_toolkit_lite2-2.3.2-cp311-cp311-linux_aarch64.whl
sudo apt-get install ffmpeg

mkdir -p models
# Copy the compiled cone_yolox_rk3576_fp16.rknn into models/.
bash tools/inspect_board.sh
bash scripts/run.sh
```

Open `http://BOARD_IP:8090`, choose a video, then click **Run detection**. The UI
shows previews during processing, the final annotated video and **Download MP4**.
Uploads are limited to 512 MB; decoded dimensions are limited to 3840×2160.
Output has no audio. Odd input dimensions are cropped by at most one pixel.

The server is a single-operator trusted-LAN demo. Run **one worker**. It rejects
concurrent processing with HTTP 409. Jobs run off the request thread. Completed
results remain under `CONE_DATA`; input uploads are deleted after processing.
Old results need manual storage management, especially on small eMMC devices.
Job IDs are held in memory; restarting the server loses the API index while
result files remain on disk. Do not expose this unauthenticated demo to the Internet.

Environment: `CONE_MODEL` (RKNN path), `CONE_DATA` (result directory), `PORT`
(default 8090), `PYTHON` (launcher interpreter). `scripts/run.sh` selects RKNN.
There is **no automatic CPU fallback** when NPU initialization fails.

CLI processing:

```bash
python run_video.py --input test.mp4 --output outputs/test \
  --model models/cone_yolox_rk3576_fp16.rknn --backend rknn
```

## Performance measurements

- **NPU API FPS:** processed frames divided by accumulated synchronous
  `RKNNLite.inference()` wall time, after five real warmup calls. Includes runtime
  and transfer overhead; this is not a device-cycle-only hardware profiler value.
- **Processing FPS:** includes decode, preprocess, NPU call, decode/NMS, overlay,
  JSONL trace, previews and intermediate MP4 writing.
- **End-to-end FPS:** includes all processing plus final browser-compatible H.264
  encoding. Excludes upload, model initialization and warmup. Available at completion.
- The MP4 plays at the original video FPS, independently of processing throughput.

The UI shows a dash for NPU FPS with the optional development-only ONNX backend.
It does not fabricate NPU measurements. Raw timings and metric definitions are
saved in `summary.json`; every processed frame has a `detections.jsonl` record.

## PC development and verification

```bash
python -m pip install -r requirements.txt onnxruntime scipy onnx pytest httpx
python run_video.py --input ../rk-workspace/assets/testConeDetector_416x416.mp4 \
  --model ../rk-workspace/models/cone_yolox.onnx --backend onnx \
  --output ../rk-workspace/validation/onnx-full
python -m pytest tests -q
python tools/smoke_web.py --input /path/to/test.mp4 --model /path/to/model.onnx \
  --backend onnx --output ../rk-workspace/validation/web-smoke
```

To compare hardware and ONNX output, run `tools/validate_outputs.py` once on PC
with `--backend onnx` and once on the board with `--backend rknn`, using the same
video, `--video`, `--model`, and `--output recording.npz`. Copy the board recording
to PC and run `python tools/validate_outputs.py --compare onnx.npz rknn.npz`.
These recordings use actual engines, never an RKNN simulator.

Build a board bundle:

```bash
python tools/package_deploy.py --model ../rk-workspace/models/cone_yolox_rk3576_fp16.rknn \
  --output ../rk-workspace/deploy/cone-deploy.tar.gz
```

## Attribution and limitations

The official model, layer graph and project material retain The MathWorks, Inc.
BSD-3-Clause notice in [LICENSE](LICENSE). The source video is not bundled in
GitHub source. [Rockchip Model Zoo YOLOX](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolox)
was consulted for the RKNN/YOLOX deployment conventions; this project preserves
the MathWorks model's four classes and normalization.

This is a perception demonstration, not an autonomous driving controller. Video
coverage and output consistency are not accuracy metrics; no annotated held-out
benchmark or mAP evaluation has been performed. Small/distant cones, motion blur,
occlusion and changed lighting can cause misses or class errors.
