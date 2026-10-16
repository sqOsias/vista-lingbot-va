"""Load repository-relative configuration with explicit environment overrides."""
from __future__ import annotations

import json
import os
from pathlib import Path


def project_root() -> Path:
    """Return the checked-out repository root."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load JSON config and resolve all repository/model paths.

    Environment variables allow a migrated installation to keep large third-party
    trees and weights outside the Git checkout.
    """
    root = project_root()
    source = Path(path or os.getenv("VISTA_WAM_CONFIG", root / "configs/default.json"))
    cfg = json.loads(source.read_text(encoding="utf-8"))
    overrides = {
        ("third_party", "lingbot_va"): "VISTA_WAM_LINGBOT_REPO",
        ("third_party", "rmbench"): "VISTA_WAM_RMBENCH_REPO",
        ("third_party", "robotwin"): "VISTA_WAM_ROBOTWIN_REPO",
        ("models", "lingbot_posttrain_robotwin"): "VISTA_WAM_LINGBOT_CHECKPOINT",
        ("models", "qwen3_vl"): "VISTA_WAM_QWEN_CHECKPOINT",
    }
    for keys, variable in overrides.items():
        value = os.getenv(variable, cfg[keys[0]][keys[1]])
        path_value = Path(value).expanduser()
        cfg[keys[0]][keys[1]] = str(path_value if path_value.is_absolute() else root / path_value)
    cfg["root"] = str(root)
    cfg["config_file"] = str(source.resolve())
    return cfg

