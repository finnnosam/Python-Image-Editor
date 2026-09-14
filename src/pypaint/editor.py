"""Document resize, layer operations and transactional regional effects."""

from dataclasses import replace

from PIL import Image

from pypaint.history import LAYER_FIELDS, LayerRecord, Transaction
from pypaint.state import ChangeKind
from pypaint.surface import TiledSurface
from pypaint.vectors import VectorSnapshot, scale_vector, translate_vector


def resize_nearest(source, size, token):
    result = TiledSurface(source.mode, size, source.fill, store=source.store)
    for coordinate in result.coordinates():
        token.check()
        box = result.tile_box(coordinate)
        extent = (
            box[0] * source.width / size[0],
            box[1] * source.height / size[1],
            box[2] * source.width / size[0],
            box[3] * source.height / size[1],
        )
        result.paste(
            source.transform((box[2] - box[0], box[3] - box[1]), Image.Transform.EXTENT, extent),
            box,
        )
    return result.snapshot()


def resized(snapshot, size, selection, token):
    if min(size) < 1 or max(size) > 1_000_000:
        raise ValueError("Invalid resize dimensions")
    layers = []
    for record in snapshot.layers:
        token.check()
        metadata = dict(zip(LAYER_FIELDS, record.metadata), width=size[0], height=size[1])
        if record.vector_records is not None:
            layer = record.restore()
            for obj in layer.vector_data.objects:
                token.check()
                scale_vector(obj, size[0] / snapshot.size[0], size[1] / snapshot.size[1])
            layer.vector_data.width, layer.vector_data.height = size
            vectors = VectorSnapshot.capture(layer.vector_data)
            surface = None
        else:
            vectors = None
            surface = resize_nearest(record.surface, size, token)
        layers.append(LayerRecord(tuple(metadata[key] for key in LAYER_FIELDS), surface, vectors))
    return replace(snapshot, size=tuple(size), layers=tuple(layers)), resize_nearest(
        selection, size, token
    )


def translated_surface(source, size, offset, token):
    result = TiledSurface(source.mode, size, store=source.store)
    if source.fill in (0, (0, 0, 0, 0)):
        tiles = source.iter_tiles()
    else:
        tiles = (
            (coordinate, source.tile_box(coordinate), source._read_tile(coordinate))
            for coordinate in source.coordinates()
        )
    for _, box, pixels in tiles:
        token.check()
        result.paste(pixels, (box[0] + offset[0], box[1] + offset[1]))
    return result.snapshot()


def canvas_resized(snapshot, size, offset, selection, token):
    layers = []
    for record in snapshot.layers:
        token.check()
        metadata = dict(zip(LAYER_FIELDS, record.metadata), width=size[0], height=size[1])
        if record.vector_records is not None:
            layer = record.restore()
            for obj in layer.vector_data.objects:
                token.check()
                translate_vector(obj, *offset)
            layer.vector_data.width, layer.vector_data.height = size
            vectors = VectorSnapshot.capture(layer.vector_data)
            surface = None
        else:
            vectors = None
            surface = translated_surface(record.surface, size, offset, token)
        layers.append(LayerRecord(tuple(metadata[key] for key in LAYER_FIELDS), surface, vectors))
    return replace(snapshot, size=tuple(size), layers=tuple(layers)), translated_surface(
        selection, size, offset, token
    )


def apply_snapshot(document, expected_generation, snapshot, name):
    if document.closed or document.generation != expected_generation:
        raise ValueError("Document changed before publication")
    tx = Transaction(document, name)
    state_id = document.state_id
    try:
        snapshot.restore(document)
        document.state_id = state_id
        document.change(ChangeKind.DIMENSIONS)
        tx.commit()
    except Exception:
        tx.cancel()
        raise


# Bounded layer operations reuse the shared compositor's mask stage.


def baked_mask(snapshot, index, renderer, cancellation=None):
    result = TiledSurface("RGBA", snapshot.size)
    for coordinate in result.coordinates():
        if cancellation:
            cancellation.check()
        box = result.tile_box(coordinate)
        with renderer.lock:
            pixels = renderer.layer_pixels(snapshot, box, visible_only=False)[index]
        result.paste(pixels, box)
    return result.snapshot()


# Detached regional previews, selected application and one transactional history step.


def compute(source, selection, operation, parameters, cancellation):
    parameters = operation.validate_parameters(parameters)
    result = source.copy()
    for coordinate in source.coordinates():
        cancellation.check()
        box = source.tile_box(coordinate)
        mask = selection.crop(box) if selection is not None else None
        if mask is not None and mask.getbbox() is None:
            continue
        region = operation.input_region(box, parameters)
        region = (
            max(0, region[0]),
            max(0, region[1]),
            min(source.width, region[2]),
            min(source.height, region[3]),
        )
        pixels = operation.function(source.crop(region), parameters)
        expected = (region[2] - region[0], region[3] - region[1])
        if pixels.mode != "RGBA" or pixels.size != expected:
            raise ValueError("Extension returned an incompatible region")
        patch = pixels.crop(
            (box[0] - region[0], box[1] - region[1], box[2] - region[0], box[3] - region[1])
        )
        if patch.tobytes() != source.crop(box).tobytes():
            result.paste(patch, box, mask)
    result.publish()
    return result.snapshot()


def apply(document, layer_id, generation, surface, name):
    if document.closed or document.generation != generation:
        raise ValueError("Document changed while the operation was being computed")
    layer = next(layer for layer in document.layers if layer.id == layer_id)
    transaction = Transaction(document, name)
    try:
        layer.image = surface.copy()
        layer.reset_mipmaps()
        document.change(ChangeKind.RASTER, layer_id, ((0, 0, document.doc_w, document.doc_h),))
        transaction.commit()
    except Exception:
        transaction.cancel()
        raise
