import json
from pathlib import Path

import numpy as np
from PIL import Image

from vista_wam.config import load_config
from vista_wam.control.robotwin_contract import decode_raw, encode_raw
from vista_wam.core.evidence import ActionQueue, EvidenceStore
from vista_wam.evaluation.target_interface import draw_target_box


def test_config_paths_are_absolute():
    config = load_config()
    assert Path(config["third_party"]["rmbench"]).is_absolute()
    assert Path(config["models"]["lingbot_posttrain_robotwin"]).is_absolute()


def test_evidence_integrity_and_causality(tmp_path):
    store = EvidenceStore(tmp_path)
    image = Image.fromarray(np.zeros((24, 32, 3), dtype=np.uint8))
    row = store.add(0, "head_camera", image)
    assert len(row["sha256"]) == 64
    assert store.retrieve(0, "head_camera", as_of=0).is_file()
    try:
        store.retrieve(1, "head_camera", as_of=0)
        assert False, "future evidence should be rejected"
    except ValueError:
        pass


def test_action_queue_invalidates_stale_plan():
    queue = ActionQueue(); queue.set_subgoal("a"); queue.fill([[1]])
    queue.set_subgoal("b")
    assert queue.cancelled == 1


def test_robotwin_numeric_roundtrip():
    origin = np.array([0,0,0,0,0,0,1,1, 1,2,3,0,0,0,1,0], dtype=float)
    absolute = origin.copy(); absolute[[0,1,8,9]] += [.1,-.2,.05,.03]
    encoded = encode_raw(absolute, origin)
    decoded = decode_raw(encoded, origin)
    assert np.allclose(decoded, absolute)


def test_visual_outline_only_changes_outline():
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    marked = draw_target_box(rgb, (10, 8, 30, 24))
    assert np.count_nonzero(marked) > 0
    assert np.array_equal(marked[0:7], rgb[0:7])

