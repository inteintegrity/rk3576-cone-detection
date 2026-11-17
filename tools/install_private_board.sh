#!/usr/bin/env bash
# Optional reuse of an existing board Python without replacing its dependencies.
# Debian 12 ARM64 only; fresh systems should use the README's venv + apt method.
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import cv2,numpy,rknnlite; print("Existing inference packages found")'
"$PYTHON" -m pip install --no-cache-dir --target .deps \
  fastapi==0.141.1 uvicorn==0.52.3 python-multipart==0.0.32
mkdir -p .ffmpeg
cd .ffmpeg
apt-get download ffmpeg libavdevice59 libavfilter8 libpostproc56
for package in ./*.deb; do dpkg-deb -x "$package" .; done
export LD_LIBRARY_PATH="$PWD/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./usr/bin/ffmpeg -version
