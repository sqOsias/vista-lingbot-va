#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party"

clone_at() {
  local url="$1" destination="$2" revision="$3"
  if [[ ! -d "$destination/.git" ]]; then git clone "$url" "$destination"; fi
  git -C "$destination" fetch --tags origin
  git -C "$destination" checkout --detach "$revision"
}

clone_at https://github.com/Robbyant/lingbot-va.git "$ROOT/third_party/lingbot-va" a85cc1012db8142f8ced2723968f4370708751c0
clone_at https://github.com/RoboTwin-Platform/RMBench.git "$ROOT/third_party/RMBench" 6d36e28ccce86710627edac26b5899bd5b6994c2

if [[ "${WITH_ROBOTWIN:-0}" == "1" ]]; then
  clone_at https://github.com/RoboTwin-Platform/RoboTwin.git "$ROOT/third_party/RoboTwin" 2eeec322
fi

printf 'Pinned third-party repositories are ready under %s\n' "$ROOT/third_party"

