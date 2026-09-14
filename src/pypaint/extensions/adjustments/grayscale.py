from PIL import ImageOps

from pypaint.extensions.registry import Operation


def grayscale(image, parameters):
    result = ImageOps.grayscale(image).convert("RGBA")
    result.putalpha(image.getchannel("A"))
    return result


OPERATION = Operation("grayscale", "Grayscale", "adjustment", grayscale, lambda box, p: box)
