#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${VISTA_WAM_MODEL_GPU:-1}"
PORT="${VISTA_WAM_PORT:-18774}"
mkdir -p "$ROOT/outputs/service"
export CUDA_VISIBLE_DEVICES="$GPU" VISTA_WAM_PORT="$PORT"
exec python -m vista_wam.model.server 2>&1 | tee "$ROOT/outputs/service/server.log"

