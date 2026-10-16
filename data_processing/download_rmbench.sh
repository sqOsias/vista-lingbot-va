#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RMBENCH="${VISTA_WAM_RMBENCH_REPO:-$ROOT/third_party/RMBench}"
[[ -d "$RMBENCH" ]] || { echo "Run data_processing/setup_repositories.sh first"; exit 2; }
cd "$RMBENCH"
bash script/_download_assets.sh
if [[ "${WITH_FULL_DATASET:-0}" == "1" ]]; then bash script/_download_data.sh; fi
echo "Assets ready. Set WITH_FULL_DATASET=1 only for full dataset experiments."

