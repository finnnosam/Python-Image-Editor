"""Discoverable layer blend modes and their shared alpha compositor."""

from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image


def _load_modes():
    modes = {}
    preferred_order = {
        "normal": 0,
        "multiply": 1,
        "xor": 2,
        "set_brightness": 3,
    }
    paths = sorted(Path(__file__).parent.glob("*.py"),
                   key=lambda path: (preferred_order.get(path.stem, 100),
                                     path.stem))
    for path in paths:
        if path.stem.startswith("_"):
            continue
        module = import_module(f"{__name__}.{path.stem}")
        modes[str(module.ID).strip().lower()] = module
    if "normal" not in modes:
        raise RuntimeError("The required Normal blend mode is missing")
    return modes


MODES = _load_modes()


def mode_labels():
    return {key: module.LABEL for key, module in MODES.items()}


def normalize_mode(mode):
    mode = str(mode).lower()
    return mode if mode in MODES else "normal"


def composite(backdrop, source, mode="normal"):
    """Composite an RGBA source over a same-sized RGBA backdrop."""
    mode = normalize_mode(mode)
    if mode == "normal":
        result = backdrop.copy()
        result.alpha_composite(source)
        return result

    # Blend modes cannot affect pixels where the source alpha is zero. Most
    # paint layers are sparse, so avoid allocating multiple full-canvas float
    # arrays merely to process a small stroke or object.
    source_bounds = source.getchannel("A").getbbox()
    if source_bounds is None:
        return backdrop.copy()
    if source_bounds != (0, 0, source.width, source.height):
        result = backdrop.copy()
        blended_region = composite(
            backdrop.crop(source_bounds), source.crop(source_bounds), mode)
        result.paste(blended_region, source_bounds)
        return result

    destination = np.asarray(backdrop.convert("RGBA"), dtype=np.float32) / 255.0
    foreground = np.asarray(source.convert("RGBA"), dtype=np.float32) / 255.0
    cb, ab = destination[..., :3], destination[..., 3:4]
    cs, source_alpha = foreground[..., :3], foreground[..., 3:4]
    blended = MODES[mode].blend_rgb(cb, cs)
    alpha = source_alpha + ab * (1.0 - source_alpha)
    premultiplied = (source_alpha * (1.0 - ab) * cs
                     + ab * (1.0 - source_alpha) * cb
                     + source_alpha * ab * blended)
    rgb = np.divide(premultiplied, alpha, out=np.zeros_like(premultiplied),
                    where=alpha > 0)
    pixels = np.concatenate((rgb, alpha), axis=2)
    return Image.fromarray(np.uint8(np.clip(pixels * 255.0 + 0.5, 0, 255)),
                           "RGBA")
