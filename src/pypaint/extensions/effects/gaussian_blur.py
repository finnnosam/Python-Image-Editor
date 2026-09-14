import math

from PIL import ImageFilter

from pypaint.extensions.registry import Operation, Parameter


def input_region(box, parameters):
    halo = math.ceil(parameters["radius"] * 3) + 2
    return box[0] - halo, box[1] - halo, box[2] + halo, box[3] + halo


OPERATION = Operation(
    "gaussian_blur",
    "Gaussian Blur",
    "effect",
    lambda image, p: image.filter(ImageFilter.GaussianBlur(p["radius"])),
    input_region,
    (Parameter("radius", 4, 0, 32),),
)
