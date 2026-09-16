ID = "screen"
LABEL = "Screen"


def blend_rgb(backdrop, source):
    return 1.0 - (1.0 - backdrop) * (1.0 - source)
