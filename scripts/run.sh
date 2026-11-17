#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
if [ -d "$PWD/.deps" ]; then export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"; fi
if [ -x "$PWD/.ffmpeg/usr/bin/ffmpeg" ]; then
  export PATH="$PWD/.ffmpeg/usr/bin:$PATH"
  export LD_LIBRARY_PATH="$PWD/.ffmpeg/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CONE_BACKEND=rknn
export CONE_MODEL="${CONE_MODEL:-$PWD/models/cone_yolox_rk3576_fp16.rknn}"
exec "${PYTHON:-python3}" -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8090}" --workers 1
