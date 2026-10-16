#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n vista-wam-model python=3.10
conda activate vista-wam-model
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/environment/model-requirements.txt"
python -m pip install flash-attn --no-build-isolation
python -m pip install -e "$ROOT"

conda create -y -n vista-wam-sim python=3.10
conda activate vista-wam-sim
(cd "$ROOT/third_party/RMBench" && bash script/_install.sh)
python -m pip install -e "$ROOT[test]"

if [[ "${WITH_VLM:-0}" == "1" ]]; then
  conda create -y -n vista-wam-vlm python=3.10
  conda activate vista-wam-vlm
  python -m pip install -r "$ROOT/environment/vlm-requirements.txt"
  python -m pip install -e "$ROOT"
fi

echo "Created vista-wam-model and vista-wam-sim"

