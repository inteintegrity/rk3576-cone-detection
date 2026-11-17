#!/usr/bin/env bash
set -eu
echo '=== Board and kernel ==='
uname -a
tr '\0' '\n' < /proc/device-tree/compatible
echo '=== NPU driver ==='
dmesg 2>/dev/null | grep -i -E 'rknpu|rknn' | tail -15 || true
echo '=== NPU utilization (permissions may restrict access) ==='
cat /sys/kernel/debug/rknpu/load 2>/dev/null || true
echo '=== Runtime library ==='
find /usr/lib /usr/local/lib -name 'librknnrt.so*' 2>/dev/null || true
echo '=== Python packages ==='
"${PYTHON:-python3}" - <<'PY'
import importlib.metadata as m
import platform
print(platform.python_version(),platform.machine())
for name in ['rknn-toolkit-lite2','opencv-python','numpy','fastapi','uvicorn','python-multipart']:
    try: print(name,m.version(name))
    except m.PackageNotFoundError:print(name,'not installed in this interpreter')
PY
command -v ffmpeg || true
df -h .
