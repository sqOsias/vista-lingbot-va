"""Text and visual target-conditioning helpers."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

TEMPLATES = {
    "suffix": "Move the block back to the {target} mat.",
    "explicit": "Pick up the block from the center and place it on the {target} mat.",
    "visual": "Pick up the block from the center and place it on the highlighted mat.",
}


def draw_target_box(rgb, box):
    """Return a copy with the fixed two-pixel magenta target outline."""
    image = Image.fromarray(np.asarray(rgb).copy())
    ImageDraw.Draw(image).rectangle(tuple(box), outline=(255, 0, 255), width=2)
    return np.asarray(image)


def instruction(kind: str, target: str) -> str:
    return TEMPLATES[kind].format(target=target)

