#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${VISTA_WAM_PORT:-18774}"
OUT="${1:-$ROOT/outputs/migration_smoke_$(date +%Y%m%d_%H%M%S)}"
python "$ROOT/scripts/verify_install.py"
curl --fail --silent --show-error "http://127.0.0.1:$PORT" > /tmp/vista_wam_model_health.json
python -m vista_wam.evaluation.smoke --port "$PORT" --output "$OUT" --seed 100065 --target front --template explicit --max-actions 16
echo "Migration smoke evidence: $OUT/smoke_result.json"

