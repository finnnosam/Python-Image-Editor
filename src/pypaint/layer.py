"""Raster/vector layers, mipmap updates and tiled selection masks."""

import json
import math
from collections import defaultdict
from uuid import uuid4

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageOps

from pypaint import storage as accounting
from pypaint.storage import entries
from pypaint.surface import TiledSurface
from pypaint.vectors import VectorLayer


class RegionalDraw:
    """Legacy point-drawing adapter; application tools use bounded kernels."""

    def __init__(self, surface):
        self.surface = surface

    def point(self, xy, fill=None):
        self.surface.putpixel(xy, fill)


class Layer:
    MASK_LAYERS_UNDERNEATH = "layers_underneath"
    MASK_LAYER_BELOW = "layer_below"
    MASK_VISIBLE_ONLY = "visible_only"
    MASK_ALL_BELOW = "all_below"

    def __init__(self, width, height, name, layer_type="raster"):
        self.id = uuid4().hex
        self.name = name
        self.visible = True
        self.opacity = 100
        self.blend_mode = "normal"
        self.layer_type = layer_type  # "raster" or "vector"
        self.masked = False
        self.anti_mask = False
        self.mask_mode = self.MASK_LAYERS_UNDERNEATH
        self.mask_visibility = self.MASK_VISIBLE_ONLY
        self.width = width
        self.height = height

        if layer_type == "raster":
            self.image = TiledSurface("RGBA", (width, height))
            self.draw = None
            self.vector_data = None
        else:  # vector
            self._image = None
            self.draw = None
            self.vector_data = VectorLayer(name, width, height)

        # Level zero is always the editable image.  Smaller levels are built
        # lazily as zooming needs them, rather than for every opened image.
        self._mipmaps = [self._image] if self._image is not None else []
        self._mipmap_revision = 0

    @property
    def draw(self):
        return RegionalDraw(self.image) if isinstance(self.image, TiledSurface) else self._draw

    @draw.setter
    def draw(self, draw):
        self._draw = draw

    @property
    def image(self):
        if self._image is None:
            self.render_vector()
        return self._image

    @image.setter
    def image(self, image):
        if self.layer_type == "raster" and not isinstance(image, TiledSurface):
            image = TiledSurface.from_image(image.convert("RGBA"))
        self._image = image

    def reset_mipmaps(self):
        """Discard reduced previews after replacing the whole layer image."""
        if self.layer_type == "vector" and self._image is None:
            self._mipmaps = []
            self._mipmap_revision += 1
            return
        self._mipmaps = [self.image]
        self._mipmap_revision += 1
        if getattr(self, "_vector_render_image", None) is not self.image:
            self._vector_render_image = None

    @property
    def is_raster(self):
        return self.layer_type == "raster"

    def get_mipmap(self, level):
        """Return a cached 2**level reduction of this layer."""
        if not self._mipmaps or self._mipmaps[0] is not self.image:
            self.reset_mipmaps()
        while len(self._mipmaps) <= level:
            previous = self._mipmaps[-1]
            size = (max(1, (previous.width + 1) // 2), max(1, (previous.height + 1) // 2))
            self._mipmaps.append(previous.resize(size, Image.Resampling.BOX))
        return self._mipmaps[level]

    def update_mipmaps(self, box):
        """Incrementally refresh cached levels touched by a raster edit."""
        self._mipmap_revision += 1
        if not self._mipmaps or self._mipmaps[0] is not self.image:
            self.reset_mipmaps()
            return

        try:
            self._update_cached_mipmaps(box)
        except (OSError, MemoryError):
            # Reduced previews are disposable. Allocation/spill failure here
            # must not cancel a valid edit to the authoritative layer pixels.
            # The normal preview job can rebuild them when resources permit.
            self.reset_mipmaps()

    def _update_cached_mipmaps(self, box):
        left, top, right, bottom = box
        for level in range(1, len(self._mipmaps)):
            previous = self._mipmaps[level - 1]
            current = self._mipmaps[level]
            if isinstance(current, TiledSurface):
                accounting.working.register_reclaimer(current.publish)
            # Include a pixel of context for BOX filtering at edit boundaries.
            dl = max(0, int(math.floor(left / 2)) - 1)
            dt = max(0, int(math.floor(top / 2)) - 1)
            dr = min(current.width, int(math.ceil(right / 2)) + 1)
            db = min(current.height, int(math.ceil(bottom / 2)) + 1)
            if dr <= dl or db <= dt:
                return
            source_box = (dl * 2, dt * 2, min(previous.width, dr * 2), min(previous.height, db * 2))
            reduced = previous.crop(source_box).resize((dr - dl, db - dt), Image.Resampling.BOX)
            current.paste(reduced, (dl, dt))
            left, top, right, bottom = dl, dt, dr, db

    def render_vector(self, regional=False):
        """Render vector objects to the raster image"""
        if self.layer_type == "vector" and self.vector_data:
            if regional:
                if getattr(self, "_regional_revision", None) != self.vector_data.revision:
                    self._image = None
                    self._mipmaps = []
                    self._mipmap_revision += 1
                    self._regional_revision = self.vector_data.revision
                return
            # Objects can be edited through handles, settings, or the table.
            # A value key catches all those mutations without rerasterizing
            # unchanged geometry on every viewport/globe/thumbnail request.
            key = (self.width, self.height, self.vector_data.revision)
            if (
                getattr(self, "_vector_render_key", None) == key
                and getattr(self, "_vector_render_image", None) is self._image
                and self._image is not None
            ):
                return
            # Clear the image
            if self.width * self.height * 4 > TiledSurface.MAX_MATERIALIZED_BYTES:
                raise ValueError(
                    "Whole-vector compatibility rendering exceeds 512 MiB; use regional rendering"
                )
            self.image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            self.draw = ImageDraw.Draw(self.image)
            self.vector_data.render(self.image)
            self.reset_mipmaps()
            self._vector_render_key = key
            self._vector_render_image = self.image

    def image_with_opacity(self, image=None):
        """Return a compositing copy with this layer's opacity applied."""
        source = image if image is not None else self.image
        if isinstance(source, TiledSurface):
            source = source.materialize()
        if self.opacity >= 100:
            return source
        adjusted = source.copy()
        alpha = adjusted.getchannel("A").point(
            lambda value: int((value * self.opacity + 50) // 100)
        )
        adjusted.putalpha(alpha)
        return adjusted


# Sparse selection operations with explicit absent-pixel semantics (zero).


def combine(first, second, operation="lighter", cancellation=None):
    if not isinstance(first, TiledSurface) and not isinstance(second, TiledSurface):
        return getattr(ImageChops, operation)(first, second)
    if first.size != second.size:
        raise ValueError("Selection dimensions differ")
    result = TiledSurface("L", first.size)
    for coordinate in result.coordinates():
        if cancellation:
            cancellation.check()
        box = result.tile_box(coordinate)
        image = getattr(ImageChops, operation)(first.crop(box), second.crop(box))
        if image.getbbox():
            result.paste(image, box)
    result.publish()
    return result


def outline(mask):
    """Exact edge runs using one tile plus neighbor pixels, never a full mask array."""
    if not isinstance(mask, TiledSurface):
        mask = TiledSurface.from_image(mask)
    mask.publish()
    if mask.fill and mask._root is None:
        w, h = mask.size
        return [(0, 0, w, 0), (0, h, w, h), (0, 0, 0, h), (w, 0, w, h)]
    lines = defaultdict(list)
    coordinates = {coordinate for coordinate, _ in entries(mask._root)}
    if mask.fill:
        cols = (mask.width + mask.tile_size - 1) // mask.tile_size
        rows = (mask.height + mask.tile_size - 1) // mask.tile_size
        coordinates.update((x, y) for x in range(cols) for y in (0, rows - 1))
        coordinates.update((x, y) for y in range(rows) for x in (0, cols - 1))
    for coordinate in coordinates:
        box = mask.tile_box(coordinate)
        left, top, right, bottom = box
        pixels = np.asarray(mask.crop((left - 1, top - 1, right + 1, bottom + 1))) > 0
        horizontal = pixels[:-1, 1:-1] != pixels[1:, 1:-1]
        vertical = pixels[1:-1, :-1] != pixels[1:-1, 1:]
        for axis, values in (("h", horizontal), ("v", vertical.T)):
            for offset in np.flatnonzero(values.any(axis=1)):
                row = values[offset]
                transitions = np.flatnonzero(
                    np.r_[False, row, False][1:] != np.r_[False, row, False][:-1]
                )
                coordinate = (top if axis == "h" else left) + offset
                origin = left if axis == "h" else top
                lines[(axis, coordinate)].extend(
                    (origin + int(a), origin + int(b))
                    for a, b in zip(transitions[::2], transitions[1::2])
                )
    edges = []
    for (axis, coordinate), intervals in sorted(lines.items()):
        merged = []
        for start, end in sorted(set(intervals)):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        edges.extend(
            (start, coordinate, end, coordinate)
            if axis == "h"
            else (coordinate, start, coordinate, end)
            for start, end in merged
        )
    return edges
