# Validation record — 2026-11-16

## Model and video provenance

- Model: original `yoloxConeDetector.mat`, 18,996,235 bytes.
- Source SHA-256: `fcd732b620e52b8110e6c3ce869af86dc7d91a18abac2113bbee391d859275bd`.
- Extracted ONNX SHA-256: `8db8ea5c9e3784a9a4cec5e0b61caf3ce3cce723de0418c4b5c43e1481cbe0f9`.
- Compiled FP16 RKNN SHA-256: `13a8465636da332d227afbe1c403eb01ec2203346c9bbd44252c08cd77953357`.
- User-supplied `test.mp4` and upstream `videos/testConeDetector_416x416.mp4`
  are byte-identical: SHA-256 `c62e834eee82666702afa0969fca70a424f3c520eee3adbb4363cd12fe6f85ef`.
- Test video: 416×416, 30 FPS, 747 frames, 24.9 seconds.
- No training and no simulated inference were used.

## Actual board

Seeed reComputer RK3576 devkit, Debian 12 ARM64, kernel
`6.1.115-vendor-seeed-rk3576`. Toolkit2/Lite2 2.3.2, `librknnrt` 2.3.0,
RKNPU driver 0.9.8. Existing Python 3.11.2, OpenCV 4.11.0.86 and NumPy 1.26.4
were reused. The model initialized and ran with `NPU_CORE_AUTO`.

The existing board had about 278 MB free before deployment. FastAPI and its
dependencies were installed in project-local `.deps`. Debian 12 ffmpeg 5.1.9
and missing libavdevice/libavfilter/libpostproc libraries were unpacked under
project-local `.ffmpeg`, without replacing system packages. The tested launcher
uses `/home/seeed/rockchip-yolo11-quant-benchmark/.venv/bin/python`.

The server is deployed at `/home/seeed/rk3576-formula-student-cone-detection`.
Port 8090 was already occupied, so this deployment uses **8096**.

```bash
cd ~/rk3576-formula-student-cone-detection
PORT=8096 PYTHON=/home/seeed/rockchip-yolo11-quant-benchmark/.venv/bin/python \
  bash scripts/run.sh
```

## Full-video NPU run

All 747 frames were processed, with five warmup calls excluded from timing.
The retrieved H.264 MP4 was decoded again on PC and verified to contain
747 frames at 416×416 and 30 FPS. The frame trace contains 747 JSONL records.

| Metric | Measured value |
|---|---:|
| NPU API inference FPS | 34.34 |
| Mean inference call | 29.12 ms |
| Median inference call | 28.60 ms |
| P95 inference call | 33.48 ms |
| Processing pipeline FPS | 18.29 |
| End-to-end FPS including H.264 encode | 16.19 |
| Pipeline time | 40.84 s |
| Final H.264 encode time | 5.29 s |
| Total measured time | 46.13 s |

These describe a single measured offline-video run, not a guaranteed operating
rate. Other workloads, temperature, storage and input resolution can change them.
NPU API FPS includes transfer/runtime overhead, not device-cycle profiling.
End-to-end excludes upload, model load and warmup. MP4 playback remains 30 FPS.

| Class | Peak detections in one frame | Detection observations across all frames |
|---|---:|---:|
| greenCone | 7 | 1656 |
| redCone | 9 | 1615 |
| yellowConeLeft | 2 | 69 |
| yellowConeRight | 2 | 182 |

These counts are predictions, not ground truth or unique-cone counts.

The final run used the `official-onboard` camera profile. Two normalized vehicle
marker regions excluded 146 raw detections of the triangular markings on the car
(34 red and 112 yellow-left observations). Confidence remained 0.35; the model
was unchanged. Raw predictions and exclusion reasons are retained in the trace.
The camera mask is unsuitable for an unrelated camera view; select `none` there.
The unfiltered baseline is preserved separately (NPU API 34.53 FPS, E2E 13.82 FPS).
Throughput variation between runs should not be attributed solely to filtering.

## ONNX versus actual RKNN outputs

The same decoded source frames (0, 120, 360, 600) were passed through PC ONNX
Runtime and the physical board's RKNNLite. Three raw maps were saved per frame.
All output shapes match. Across 12 maps the minimum cosine similarity is
0.999329; maximum absolute raw-logit difference is 1.475. FP16 conversion is
not numerically identical to float32; confidence threshold crossings can differ.

After decoding at confidence 0.35, both backends produced 5, 8, 10 and 1 detections
on those four frames. All 24 ONNX detections had a same-class board match, with
best overlap IoU at least 0.887. This small consistency check does not establish mAP.

The original MATLAB runtime is unavailable. The MAT importer uses saved graph
connections and trained parameters, but direct MATLAB versus ONNX numerical
parity remains untested. No claim of identical MATLAB detector behavior is made.

## Software checks

- Eight regression tests passed, covering empty output, class order/class-aware NMS, box
  restoration, RGB letterboxing, rejection of incompatible output shapes,
  both vehicle-marker regions (including a wide false box), adjacent road cones,
  disabling the camera mask and invalid-profile rejection.
- A real 24-frame FastAPI integration run passed upload, concurrent-job rejection,
  invalid-extension/empty-file rejection, progress polling, preview, MP4/JSON/JSONL
  downloads and downloaded-video frame-count validation. It used ONNX on PC;
  NPU FPS correctly remained null.
- The browser was also used to upload and process all 747 original frames on PC.
- The board UI uses real RKNN inference; completed board browser verification is
  recorded in the local `rk-workspace/validation` evidence.
- After the marker fix, the live board FastAPI endpoint processed all 747 frames
  again. Actual HTTP upload, polling, MP4/JSON/JSONL/preview download and HTTP 206
  MP4 range serving passed. The downloaded MP4 contains 747 frames; all 146
  excluded predictions have recorded reasons, and NPU verification is true.
- Conversion command wall times were approximately 7.16 s for MAT-to-ONNX and
  9.13 s for ONNX-to-RKNN (including interpreter/WSL startup). Downloads and
  validation are excluded. Compiler-internal time was not measured separately.

Raw models, recordings, MP4s, full logs and frame traces are deliberately kept in
`rk-workspace`, outside the GitHub source directory. Only this concise validation
record belongs in the source tree.
