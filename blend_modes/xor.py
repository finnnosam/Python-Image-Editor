import numpy as np


ID = "xor"
LABEL = "XOR"


def blend_rgb(backdrop, source):
    backdrop_bytes = np.uint8(np.clip(backdrop * 255.0 + 0.5, 0, 255))
    source_bytes = np.uint8(np.clip(source * 255.0 + 0.5, 0, 255))
    return np.bitwise_xor(backdrop_bytes, source_bytes).astype(np.float32) / 255.0
