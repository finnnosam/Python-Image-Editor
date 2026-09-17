"""Raster codecs, legacy project migration and document format dispatch."""

import base64
import io
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from pypaint.jobs import Cancellation
from pypaint.layer import Layer
from pypaint.native import (
    NativeError,
    read_native,
    validate_dimensions,
    validate_layer,
    validate_vectors,
)
from pypaint.pdn import read_pdn
from pypaint.state import Document
from pypaint.surface import TiledSurface
from pypaint.vectors import VectorLayer

MAX_PIXELS = 32 * 1024**2
RESERVATION = 512 * 1024**2


def validate_size(size):
    if min(size) < 1 or size[0] * size[1] > MAX_PIXELS:
        raise ValueError(
            "Pillow image import/export is limited to 32 megapixels; use native tiled projects for larger images."
        )


def read_raster(filename, cancellation=None):
    token = cancellation or Cancellation()
    with Image.open(filename) as source:
        validate_size(source.size)
        token.check()
        pixels = ImageOps.exif_transpose(source).convert("RGBA")
        token.check()
        result = TiledSurface("RGBA", pixels.size)
        for coordinate in result.coordinates():
            token.check()
            box = result.tile_box(coordinate)
            result.paste(pixels.crop(box), box)
        result.publish()
        return result


def export_raster(snapshot, filename, renderer, cancellation=None):
    token = cancellation or Cancellation()
    validate_size(snapshot.size)
    destination = Path(filename)
    format_name = Image.registered_extensions().get(destination.suffix.lower())
    if format_name not in ("PNG", "JPEG", "BMP", "TIFF", "WEBP"):
        raise ValueError("Unsupported raster export extension")
    pixels = Image.new("RGBA", snapshot.size)
    for box, image in renderer.render_tiles(snapshot):
        token.check()
        pixels.paste(image, box)
    if format_name == "JPEG":
        flattened = Image.new("RGB", snapshot.size, "white")
        flattened.paste(pixels, mask=pixels.getchannel("A"))
        pixels = flattened
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + "-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w+b") as output:
            token.check()
            pixels.save(output, format=format_name)
            output.flush()
            os.fsync(output.fileno())
        token.check()
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


# Strict legacy JSON migration; unsupported editable content is refused explicitly.


def read_legacy(filename):
    if Path(filename).stat().st_size > 512 * 1024**2:
        raise NativeError("Legacy JSON import is limited to 512 MiB")
    with open(filename, encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("version") not in ("1.0", "2.0"):
        raise NativeError("Unsupported legacy version")
    width, height = data.get("document_width"), data.get("document_height")
    validate_dimensions(width, height)
    layers = data.get("layers")
    if (
        not isinstance(layers, list)
        or not 1 <= len(layers) <= 4096
        or width * height * 4 * len(layers) > 512 * 1024**2
    ):
        raise NativeError("Legacy decoded document exceeds 512 MiB")
    doc = Document(doc_w=width, doc_h=height)
    for record in layers:
        kind = record.get("layer_type", "raster")
        layer = Layer(width, height, record["name"], "raster" if kind == "mask" else kind)
        layer.visible = record["visible"]
        layer.opacity = max(0, min(100, float(record.get("opacity", 100))))
        layer.blend_mode = record.get("blend_mode", "normal")
        layer.masked = bool(record.get("masked", kind == "mask"))
        layer.anti_mask = bool(record.get("anti_mask", False))
        layer.mask_mode = record.get("mask_mode", "layers_underneath")
        layer.mask_visibility = record.get(
            "mask_visibility", "all_below" if layer.mask_mode == "layer_below" else "visible_only"
        )
        validate_layer(vars(layer))
        if layer.layer_type == "vector":
            validate_vectors(record.get("vector_data"))
            layer.vector_data = VectorLayer.from_dict(record["vector_data"], width, height)
        else:
            encoded = base64.b64decode(record["image_data"], validate=True)
            with Image.open(io.BytesIO(encoded)) as image:
                if image.size != (width, height):
                    raise NativeError("Legacy image dimensions differ from document")
                layer.image = image.convert("RGBA")
            layer.reset_mipmaps()
        doc.layers.append(layer)
    doc.active_layer = max(0, min(int(data.get("active_layer", 0)), len(layers) - 1))
    doc.saved_state_id = doc.state_id
    return doc


# File adapter dispatch, independent of dialogs and document tabs.


def read_document(filename, token):
    suffix = Path(filename).suffix.lower()
    if suffix == ".pypaint":
        document = read_native(filename, token)
    elif suffix == ".pdn":
        width, height, records = read_pdn(filename)
        token.check()
        document = Document(doc_w=width, doc_h=height)
        for record in records:
            token.check()
            layer = Layer(width, height, record.name)
            layer.image = record.image
            layer.visible, layer.opacity = record.visible, record.opacity * 100 / 255
            document.layers.append(layer)
        document.active_layer = len(document.layers) - 1
        document.saved_state_id = document.state_id
    else:
        surface = read_raster(filename, token)
        document = Document(doc_w=surface.width, doc_h=surface.height)
        layer = Layer(*surface.size, Path(filename).stem)
        layer.image = surface
        document.layers = [layer]
    # Opening establishes the on-disk pixels as the saved baseline, including
    # raster formats that do not retain an editable project filename.
    document.saved_state_id = document.state_id
    document.current_file = str(filename) if suffix in (".pypaint", ".pdn") else None
    return document
