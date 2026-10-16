"""Fail-fast verification for a migrated checkout; does not load large models."""
import json
import subprocess
from pathlib import Path

from vista_wam.config import load_config


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    cfg = load_config(); checks = {}
    expected = {
        "lingbot_va": "a85cc1012db8142f8ced2723968f4370708751c0",
        "rmbench": "6d36e28ccce86710627edac26b5899bd5b6994c2",
    }
    for name, commit in expected.items():
        path = Path(cfg["third_party"][name]); actual = revision(path) if (path/".git").is_dir() else None
        checks[name] = {"path": str(path), "expected": commit, "actual": actual, "ok": actual == commit}
    checkpoint = Path(cfg["models"]["lingbot_posttrain_robotwin"])
    weight_files = list(checkpoint.rglob("*.safetensors")) + list(checkpoint.rglob("*.bin"))
    checks["checkpoint"] = {"path": str(checkpoint), "weight_files": len(weight_files),
                            "ok": checkpoint.is_dir() and bool(weight_files)}
    assets = Path(cfg["third_party"]["rmbench"]) / "assets"
    checks["rmbench_assets"] = {"path": str(assets), "ok": assets.is_dir() and any(assets.iterdir())}
    result = {"status": "passed" if all(v["ok"] for v in checks.values()) else "failed", "checks": checks}
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__": main()

