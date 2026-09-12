"""Push backdrop colors darker or lighter from a neutral midpoint."""

import numpy as np


ID = "set_brightness"
LABEL = "Push Brightness"

_NEUTRAL = np.float32(128.0 / 255.0)

def _luminance(rgb):
    # Spell this out to avoid allocating an extra RGB-sized product array.
    return (rgb[..., 0:1] * 0.2126 + rgb[..., 1:2] * 0.7152
            + rgb[..., 2:3] * 0.0722)


def blend_rgb(backdrop, source):
    """Use source luminance as a signed adjustment around 128 gray.

    Black maps the backdrop to black, 128 gray leaves it unchanged, and white
    maps it to white. Intermediate values move proportionally toward the
    corresponding endpoint while retaining the backdrop's color character.
    """
    control = _luminance(source)
    dark_scale = control / _NEUTRAL
    light_amount = (control - _NEUTRAL) / (1.0 - _NEUTRAL)
    darker = backdrop * dark_scale
    lighter = backdrop + (1.0 - backdrop) * light_amount
    return np.where(control <= _NEUTRAL, darker, lighter)
