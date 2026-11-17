# Bundled RKNN Lite2 runtime

This directory contains the unmodified official Rockchip RKNN Toolkit Lite2
2.3.2 wheel for **CPython 3.11 on Linux AArch64 (ARM64)**:

`rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`

- Upstream: [Rockchip v2.3.2 packages](https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2/rknn-toolkit-lite2/packages).
- Size: 569,160 bytes.
- Official MD5: `5da9258e0c69779707e8a06411c398da`, checked against upstream `packages.md5sum`.
- SHA-256: `bda74f1179e15fccb8726054a24898982522784b65bb340b20146955d254e800`.
- Upstream copyright notice: [RKNN-Toolkit2-LICENSE](RKNN-Toolkit2-LICENSE).

From the project root, in the board's Python 3.11 environment:

```bash
(cd wheels && sha256sum -c SHA256SUMS)
python -m pip install wheels/rknn_toolkit_lite2-2.3.2-cp311-*.whl
python -c "from rknnlite.api import RKNNLite; print('RKNN Lite2 import OK')"
```

The wheel is the runtime package version used in this project's board validation.
It is not for Windows, x86_64, or another Python ABI. The board must also provide
a compatible RKNPU driver and `librknnrt.so`. This wheel does not install or
replace those system components. The wheel retains Rockchip's upstream terms;
the project's root license does not relicense the wheel.

Only the Lite2 wheel is bundled. Pip may still download its dependencies and
the application packages listed in `requirements.txt`; this is not a complete
offline Python environment.
