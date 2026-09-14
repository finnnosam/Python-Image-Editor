"""Regional raster/vector composition, sampling, mipmaps and thumbnails."""

import json
import math
import weakref
from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageOps

from pypaint.extensions.blend_modes import composite
from pypaint.history import LAYER_FIELDS
from pypaint.layer import Layer
from pypaint.storage import default_store
from pypaint.surface import TiledSurface
from pypaint.tools import _apply_hardness_to_alpha, _brush_ellipse_box, _composite_brush_shape
from pypaint.vectors import *
from pypaint.vectors import SpatialIndex, VectorLayer, VectorObject, object_bounds


class ByteCache:
    def __init__(self, limit=64 * 1024**2):
        self.limit, self.bytes = limit, 0
        self.hits = self.misses = self.evictions = 0
        self.entries = OrderedDict()
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            found = self.entries.get(key)
            if found is None:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return found[0].copy()

    def put(self, key, image):
        size = image.width * image.height * len(image.getbands()) + 256
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous:
                self.bytes -= previous[1]
            if size > self.limit:
                return
            while self.entries and self.bytes + size > self.limit:
                _, (_, amount) = self.entries.popitem(last=False)
                self.bytes -= amount
                self.evictions += 1
            self.entries[key] = (image.copy(), size)
            self.bytes += size

    def clear(self):
        with self.lock:
            self.evictions += len(self.entries)
            self.entries.clear()
            self.bytes = 0


# Document-space target for exact regional legacy vector kernels.


class RegionTarget:
    def __init__(self, size, box):
        self.width, self.height = size
        self.size, self.box = size, box
        self.image = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]))

    def crop(self, box):
        x, y = self.box[:2]
        return self.image.crop((box[0] - x, box[1] - y, box[2] - x, box[3] - y))

    def alpha_composite(self, image, dest=(0, 0)):
        self.image.alpha_composite(image, (dest[0] - self.box[0], dest[1] - self.box[1]))

    def paste(self, image, box=None, mask=None):
        box = box or (0, 0)
        offset = (box[0] - self.box[0], box[1] - self.box[1])
        self.image.paste(image, offset, mask)

    def polygon(self, points, fill):
        ImageDraw.Draw(self.image).polygon(
            [(x - self.box[0], y - self.box[1]) for x, y in points], fill=fill
        )


def _compositing_layer_indices(layers):
    """Include visible layers and the hidden layers their masks require."""
    required = set()

    def include(index, apply_mask=True):
        already_included = index in expanded
        required.add(index)
        if not apply_mask or already_included:
            return
        expanded.add(index)
        layer = layers[index]
        if not layer.masked:
            return
        sources = (
            ([index - 1] if index else [])
            if (layer.mask_mode == Layer.MASK_LAYER_BELOW)
            else range(index)
        )
        for source in sources:
            if layer.mask_visibility == Layer.MASK_VISIBLE_ONLY and not layers[source].visible:
                continue
            include(source, layer.mask_mode != Layer.MASK_LAYER_BELOW)

    expanded = set()
    for index, layer in enumerate(layers):
        if layer.visible:
            include(index)
    return required


def _apply_layer_masks(layers, rendered_layers, visible_only=False):
    """Apply each layer's independent depth and visibility mask scopes."""
    rendered_layers = [
        image.materialize() if isinstance(image, TiledSurface) else image
        for image in rendered_layers
    ]
    alpha_cache = {}

    def masked_alpha(index):
        if index in alpha_cache:
            return alpha_cache[index]
        layer = layers[index]
        alpha = rendered_layers[index].getchannel("A")
        if layer.masked:
            if layer.mask_mode == Layer.MASK_LAYER_BELOW:
                source_indices = [index - 1] if index else []
            else:
                source_indices = range(index)
            source_alpha = Image.new("L", alpha.size, 0)
            for source_index in source_indices:
                source = layers[source_index]
                if layer.mask_visibility == Layer.MASK_VISIBLE_ONLY and not source.visible:
                    continue
                source_layer_alpha = (
                    rendered_layers[source_index].getchannel("A")
                    if layer.mask_mode == Layer.MASK_LAYER_BELOW
                    else masked_alpha(source_index)
                )
                source_alpha = ImageChops.lighter(source_alpha, source_layer_alpha)
            if layer.anti_mask:
                source_alpha = ImageOps.invert(source_alpha)
            alpha = ImageChops.darker(alpha, source_alpha)
        alpha_cache[index] = alpha
        return alpha

    masked = []
    for index, rendered in enumerate(rendered_layers):
        if (
            rendered is not None
            and layers[index].masked
            and (not visible_only or layers[index].visible)
        ):
            capped = rendered.copy()
            capped.putalpha(masked_alpha(index))
            masked.append(capped)
        else:
            masked.append(rendered)
    return masked


# Nearest sampling on Pillow's coordinate grid with bounded tile inputs.


def sample_tiles(document_size, size, extent, mode, tile_size, read_tile, sampling_grid=None):
    width, height = document_size
    left, top, right, bottom = extent
    if sampling_grid is None:
        xmap = Image.fromarray(np.arange(width, dtype=np.int32)[None, :])
        ymap = Image.fromarray(np.arange(height, dtype=np.int32)[:, None])
        xs = np.asarray(
            xmap.transform((size[0], 1), Image.Transform.EXTENT, (left, 0, right, 1), fillcolor=-1)
        )[0]
        ys = np.asarray(
            ymap.transform((1, size[1]), Image.Transform.EXTENT, (0, top, 1, bottom), fillcolor=-1)
        )[:, 0]
    else:
        # Screen-pixel centers on one document-anchored lattice. Recomputing an
        # EXTENT slope for each exposed strip can pick a different neighbor at
        # fractional zoom ties and leave seams between old and new pixels.
        zoom, x, y = sampling_grid
        xs = np.floor((np.arange(size[0], dtype=np.float64) + x + 0.5) / zoom).astype(np.int32)
        ys = np.floor((np.arange(size[1], dtype=np.float64) + y + 0.5) / zoom).astype(np.int32)
    shape = (size[1], size[0], 4) if mode == "RGBA" else (size[1], size[0])
    output = np.zeros(shape, dtype=np.uint8)
    columns = [
        (int(tx), np.flatnonzero((xs // tile_size == tx) & (xs >= 0) & (xs < width)))
        for tx in np.unique(xs // tile_size)
    ]
    for ty in np.unique(ys // tile_size):
        rows = np.flatnonzero((ys // tile_size == ty) & (ys >= 0) & (ys < height))
        if not len(rows):
            continue
        for tx, cols in columns:
            if not len(cols):
                continue
            image = read_tile((tx, int(ty)))
            if image is None:
                continue
            pixels = np.asarray(image)
            # An affine EXTENT maps each source tile to a contiguous output
            # rectangle, even for reversed axes. Assign that rectangle directly
            # instead of scattering every magnified output pixel through NumPy
            # advanced indexing. Keep Pillow's exact coordinate maps above.
            output[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = pixels.take(
                ys[rows] % tile_size, axis=0
            ).take(xs[cols] % tile_size, axis=1)
    return Image.fromarray(output)


# Exact Pillow separable resizing with bounded full-axis strips.
#
# Preserve global filter coefficients and the single premultiply/unpremultiply pair.
# Cropping source boxes first changes float coefficient rounding on odd dimensions.


def resize_surface(source, size, resample, cancellation=None):
    from pypaint.surface import TiledSurface

    if min(size) < 1:
        raise ValueError("Invalid resize dimensions")
    working_mode = "RGBa" if source.mode == "RGBA" else source.mode
    horizontal = TiledSurface(source.mode, (size[0], source.height), store=source.store)
    rows = max(1, min(256, (16 * 1024**2) // (max(source.width, size[0]) * 4)))
    for top in range(0, source.height, rows):
        if cancellation:
            cancellation.check()
        bottom = min(source.height, top + rows)
        strip = source.crop((0, top, source.width, bottom)).convert(working_mode)
        strip = strip.resize((size[0], bottom - top), resample)
        # Internal storage carries premultiplied bytes without interpreting them.
        horizontal.paste(Image.frombytes(source.mode, strip.size, strip.tobytes()), (0, top))
    horizontal.publish()
    result = TiledSurface(source.mode, size, store=source.store)
    columns = max(1, min(256, (16 * 1024**2) // (max(source.height, size[1]) * 4)))
    for left in range(0, size[0], columns):
        if cancellation:
            cancellation.check()
        right = min(size[0], left + columns)
        raw = horizontal.crop((left, 0, right, source.height))
        strip = Image.frombytes(working_mode, raw.size, raw.tobytes())
        strip = strip.resize((right - left, size[1]), resample).convert(source.mode)
        result.paste(strip, (left, 0))
    result.publish()
    horizontal.close()
    return result


# Immutable inputs and bounded, cooperatively cancellable pyramid construction.


def build_pyramids(inputs, level, cancellation, renderer):
    results = []
    for layer_id, revision, size, record, cached in inputs:
        cancellation.check()
        if record.surface is not None:
            base = record.surface
        else:
            base = TiledSurface("RGBA", size)
            for coordinate in base.coordinates():
                cancellation.check()
                box = base.tile_box(coordinate)
                with renderer.lock:
                    image = renderer._vector(record, size, box)
                base.paste(image, box)
            base = base.snapshot()
        pyramid = [base, *cached]
        while len(pyramid) <= level:
            previous = pyramid[-1]
            size = (max(1, (previous.width + 1) // 2), max(1, (previous.height + 1) // 2))
            image = resize_surface(previous, size, Image.Resampling.BOX, cancellation)
            pyramid.append(image.snapshot())
        results.append((layer_id, revision, pyramid))
    return results


def _draw_vector_path(
    image, points, color, width, antialias=True, hardness=75, square_corners=False
):
    """Stroke a path with consistent geometry and optional edge smoothing."""
    if len(points) < 2:
        return

    softness_scale = width * 0.25
    blur_radius = softness_scale * (75 - hardness) / 75 if antialias and hardness < 75 else 0
    padding = width / (math.sqrt(2) if square_corners else 2) + 2 + math.ceil(blur_radius * 3)
    left = max(0, math.floor(min(x for x, _ in points) - padding))
    top = max(0, math.floor(min(y for _, y in points) - padding))
    right = min(image.width, math.ceil(max(x for x, _ in points) + padding + 1))
    bottom = min(image.height, math.ceil(max(y for _, y in points) + padding + 1))
    if right <= left or bottom <= top:
        return

    tile_width, tile_height = right - left, bottom - top
    # Supersampling stabilizes shallow lines in both modes. Aliased vectors
    # retain hard pixel edges by using nearest-neighbor reduction.
    # factor only for exceptionally large paths so a document-sized ellipse
    # cannot allocate an excessive temporary image.
    scale = 8 if antialias else 4
    while scale > 2 and tile_width * tile_height * scale * scale > 32_000_000:
        scale -= 1

    if hasattr(image, "box"):
        # Select supersampling using the original clipped object bounds above,
        # then intersect with the requested region plus filter support.
        halo = 4 + math.ceil(blur_radius * 3)
        left = max(left, image.box[0] - halo)
        top = max(top, image.box[1] - halo)
        right = min(right, image.box[2] + halo)
        bottom = min(bottom, image.box[3] + halo)
        if right <= left or bottom <= top:
            return
        tile_width, tile_height = right - left, bottom - top

    tile = Image.new("RGBA", (tile_width * scale, tile_height * scale), (0, 0, 0, 0))
    scaled_points = [((x - left) * scale, (y - top) * scale) for x, y in points]
    draw = ImageDraw.Draw(tile)
    if square_corners:
        # Rectangle edges meet at right angles. Square-capped strips form
        # exact miter joins, including the closing vertex, in one alpha layer.
        half = width * scale / 2
        for (ax, ay), (bx, by) in zip(scaled_points, scaled_points[1:]):
            length = math.hypot(bx - ax, by - ay)
            if length == 0:
                continue
            ux, uy = (bx - ax) / length * half, (by - ay) / length * half
            nx, ny = -uy, ux
            draw.polygon(
                [
                    (ax - ux + nx, ay - uy + ny),
                    (bx + ux + nx, by + uy + ny),
                    (bx + ux - nx, by + uy - ny),
                    (ax - ux - nx, ay - uy - ny),
                ],
                fill=color,
            )
    else:
        draw.line(scaled_points, fill=color, width=max(1, round(width * scale)), joint="curve")
    resampling = Image.Resampling.LANCZOS if antialias else Image.Resampling.NEAREST
    tile = tile.resize((tile_width, tile_height), resampling)
    if antialias and hardness != 75:
        tile.putalpha(_apply_hardness_to_alpha(tile.getchannel("A"), hardness, softness_scale))
    image.alpha_composite(tile, (left, top))


def render_vector_object(image, obj, document_width, document_height):
    """Render one vector object with an anti-aliased, uniform-width stroke."""
    if isinstance(obj, Point):
        radius = obj.width / 2
        bounds = (obj.x - radius, obj.y - radius, obj.x + radius, obj.y + radius)
        _composite_brush_shape(
            image,
            bounds,
            obj.color,
            lambda draw, left, top, scale: draw.ellipse(
                _brush_ellipse_box(bounds, left, top, scale), fill=255
            ),
            antialias=obj.antialias,
        )
        return
    if isinstance(obj, Line):
        points = obj.sampled_points(document_width, document_height)
        offsets = (-document_width, 0, document_width) if obj.space == "globe" else (0,)
        for offset in offsets:
            _draw_vector_path(
                image,
                [(x + offset, y) for x, y in points],
                obj.color,
                obj.width,
                obj.antialias,
                getattr(obj, "hardness", 75),
            )
        return

    if isinstance(obj, Shape):
        points = obj._outline(document_width, document_height)
        globe = any(line.space == "globe" for line in obj.lines)
        if obj.fill and len(points) >= 3 and obj.filled_side == "inside":
            if hasattr(image, "box") and globe:
                mask = obj._spherical_fill_mask(document_width, document_height, image.box)
                # Match ImageDraw.bitmap's replacement semantics for RGBA fill.
                ImageDraw.Draw(image.image).bitmap((0, 0), mask, fill=obj.fill)
            elif hasattr(image, "box"):
                image.polygon(points, obj.fill)
            elif globe:
                draw = ImageDraw.Draw(image)
                draw.bitmap(
                    (0, 0), obj._spherical_fill_mask(document_width, document_height), fill=obj.fill
                )
            else:
                draw = ImageDraw.Draw(image)
                draw.polygon(points, fill=obj.fill)
        offsets = (-document_width, 0, document_width) if globe else (0,)
        for offset in offsets:
            _draw_vector_path(
                image,
                [(x + offset, y) for x, y in points],
                obj.color,
                obj.width,
                obj.antialias,
                getattr(obj, "hardness", 75),
                square_corners=obj.preset == "rect" and not globe,
            )
        return

    # Compatibility for any legacy in-memory vector object.
    obj.draw(ImageDraw.Draw(image), document_width, document_height)


# One snapshot renderer for regions, textures, previews, thumbnails and exports.
#
# Quality policies are explicit: export composites at document resolution; interactive
# reduces each layer independently before nonlinear blending. Checkerboards are UI-owned.


class SceneObjects:
    """Decode visible objects lazily; dense offscreen scenes retain small records."""

    def __init__(self, records, limit=2 * 1024**2):
        self.records, self.limit = records, limit
        self.cache = OrderedDict()
        self.bytes = 0

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index][0]
        payload = self.records[index]
        obj = VectorObject.from_dict(json.loads(payload.read()))
        amount = payload.size * 8 + 256
        if amount <= self.limit:
            while self.cache and self.bytes + amount > self.limit:
                _, (_, size) = self.cache.popitem(last=False)
                self.bytes -= size
            self.cache[index] = (obj, amount)
            self.bytes += amount
        return obj


class Renderer:
    def __init__(self, cache_bytes=64 * 1024**2):
        self.cache = ByteCache(cache_bytes)
        default_store().evict_caches.append(weakref.WeakMethod(self.cache.clear))
        self.scenes = OrderedDict()
        self.scene_bytes = 0
        self.scene_limit = max(1024, cache_bytes // 4)
        self.lock = RLock()

    def _scene(self, record, size):
        key = record.vector_records
        scene = self.scenes.get(key)
        if scene is None:
            scene = SimpleNamespace(objects=SceneObjects(key.objects))
            scene._bounds = (
                key.bounds
                if key.bounds is not None and key.size == size
                else [object_bounds(scene.objects[i], size) for i in range(len(key.objects))]
            )
            scene._index = SpatialIndex(scene._bounds)
            amount = (
                len(key.objects) * 400
                + sum(len(values) * 8 for values in scene._index.cells.values())
                + scene.objects.limit
            )
            if amount <= self.scene_limit:
                while self.scenes and self.scene_bytes + amount > self.scene_limit:
                    _, old = self.scenes.popitem(last=False)
                    self.scene_bytes -= old._bytes
                scene._bytes = amount
                self.scenes[key] = scene
                self.scene_bytes += amount
        else:
            self.scenes.move_to_end(key)
        return scene

    @staticmethod
    def _bounds(obj, size):
        lines = getattr(obj, "lines", (obj,))
        if any(getattr(line, "space", "flat") == "globe" for line in lines):
            return (0, 0, *size)
        points = obj.get_points()
        if hasattr(obj, "sampled_points"):
            points = obj.sampled_points(*size)
        elif hasattr(obj, "_outline"):
            points = obj._outline(*size)
        padding = obj.width * 2 + 8
        return (
            (
                math.floor(min(x for x, y in points) - padding),
                math.floor(min(y for x, y in points) - padding),
                math.ceil(max(x for x, y in points) + padding),
                math.ceil(max(y for x, y in points) + padding),
            )
            if points
            else (0, 0, 0, 0)
        )

    @staticmethod
    def _intersects(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    def _vector(self, record, size, box):
        target = RegionTarget(size, box)
        scene = self._scene(record, size)
        for index in scene._index.query(box):
            render_vector_object(target, scene.objects[index], *size)
        return target.image

    def layer_pixels(self, snapshot, box, visible_only=True):
        """Shared opacity and masking stage for canvas, layer baking and export."""
        layers = [
            SimpleNamespace(**dict(zip(LAYER_FIELDS, record.metadata)))
            for record in snapshot.layers
        ]
        required = _compositing_layer_indices(layers) if visible_only else range(len(layers))
        rendered = []
        for index, (record, layer) in enumerate(zip(snapshot.layers, layers)):
            if index not in required:
                rendered.append(None)
                continue
            image = (
                self._vector(record, snapshot.size, box)
                if record.vector_records is not None
                else record.surface.crop(box)
            )
            if layer.opacity < 100:
                image.putalpha(
                    image.getchannel("A").point(
                        lambda value: int((value * layer.opacity + 50) // 100)
                    )
                )
            rendered.append(image)
        return _apply_layer_masks(layers, rendered, visible_only=visible_only)

    def _tile(self, snapshot, box, quality):
        layers = [SimpleNamespace(**dict(zip(LAYER_FIELDS, r.metadata))) for r in snapshot.layers]
        required = _compositing_layer_indices(layers)

        def content_signature(record):
            if record.vector_records is None:
                return record.surface.signature(box)
            scene = self._scene(record, snapshot.size)
            return tuple(record.vector_records.objects[index] for index in scene._index.query(box))

        signature = tuple(
            (r.metadata, content_signature(r))
            for i, r in enumerate(snapshot.layers)
            if i in required
        )
        key = (snapshot.document_id, snapshot.size, box, quality, snapshot.background, signature)
        found = self.cache.get(key)
        if found is not None:
            return found
        rendered = self.layer_pixels(snapshot, box)
        result = Image.new(
            "RGBA",
            (box[2] - box[0], box[3] - box[1]),
            snapshot.background if quality == "export" else (0, 0, 0, 0),
        )
        for layer, image in zip(layers, rendered):
            if layer.visible:
                result = composite(result, image, layer.blend_mode)
        self.cache.put(key, result)
        return result

    def render(
        self,
        snapshot,
        box=None,
        output_size=None,
        quality="export",
        pyramids=None,
        sampling_grid=None,
    ):
        box = box or (0, 0, *snapshot.size)
        if all(float(value).is_integer() for value in box):
            box = tuple(map(int, box))
        size = output_size or (box[2] - box[0], box[3] - box[1])
        if quality not in ("export", "interactive") or min(size) < 1:
            raise ValueError("Invalid rendering policy or output size")
        if size[0] * size[1] * 4 > 512 * 1024**2:
            raise ValueError("Use render_tiles for output exceeding 512 MiB")
        with self.lock:  # also deduplicates concurrent same-work requests
            if sampling_grid is not None and quality == "interactive":
                return self._interactive(snapshot, box, size, pyramids, sampling_grid)
            if (
                size != (box[2] - box[0], box[3] - box[1]) or any(type(v) is not int for v in box)
            ) and quality == "interactive":
                return self._interactive(snapshot, box, size, pyramids)
            if (box[2] - box[0]) * (box[3] - box[1]) * 4 > 512 * 1024**2:
                raise ValueError("Full-resolution intermediate exceeds 512 MiB; use render_tiles")
            output = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]))
            for tile_box, image in self.render_tiles(snapshot, box, quality):
                overlap = (
                    max(box[0], tile_box[0]),
                    max(box[1], tile_box[1]),
                    min(box[2], tile_box[2]),
                    min(box[3], tile_box[3]),
                )
                output.paste(
                    image.crop(
                        (
                            overlap[0] - tile_box[0],
                            overlap[1] - tile_box[1],
                            overlap[2] - tile_box[0],
                            overlap[3] - tile_box[1],
                        )
                    ),
                    (overlap[0] - box[0], overlap[1] - box[1]),
                )
            return output if output.size == size else output.resize(size, Image.Resampling.LANCZOS)

    def render_tiles(self, snapshot, box=None, quality="export"):
        box = box or (0, 0, *snapshot.size)
        with self.lock:
            for y in range(max(0, box[1]) // 256 * 256, min(snapshot.size[1], box[3]), 256):
                for x in range(max(0, box[0]) // 256 * 256, min(snapshot.size[0], box[2]), 256):
                    region = (x, y, min(snapshot.size[0], x + 256), min(snapshot.size[1], y + 256))
                    yield region, self._tile(snapshot, region, quality)

    def _interactive(self, snapshot, box, size, pyramids=None, sampling_grid=None):
        # Nearest magnification commutes with our pointwise opacity, mask and
        # blend operations. Composite cached document tiles once, then sample
        # the result, instead of blending viewport-sized layers on every pan.
        # Reduction keeps its separate legacy layer-first quality policy.
        if box[2] - box[0] <= size[0] and box[3] - box[1] <= size[1]:

            def read_composite(coordinate):
                x, y = coordinate
                region = (
                    x * 256,
                    y * 256,
                    min(snapshot.size[0], (x + 1) * 256),
                    min(snapshot.size[1], (y + 1) * 256),
                )
                return self._tile(snapshot, region, "interactive")

            return sample_tiles(
                snapshot.size, size, box, "RGBA", 256, read_composite, sampling_grid
            )
        return self._interactive_layers(snapshot, box, size, pyramids, sampling_grid)

    def _interactive_layers(self, snapshot, box, size, pyramids=None, sampling_grid=None):
        # The raster sampling path is bounded by visible output. Vector output
        # is assembled from only intersecting regional requests.
        layers = [SimpleNamespace(**dict(zip(LAYER_FIELDS, r.metadata))) for r in snapshot.layers]
        required = _compositing_layer_indices(layers)
        reduction = max((box[2] - box[0]) / size[0], (box[3] - box[1]) / size[1])
        desired = max(0, int(math.log2(reduction))) if reduction > 1 else 0
        level = min([desired] + [len(pyramids[i]) - 1 for i in required]) if pyramids else 0
        level = max(0, level)
        rendered = []
        for index, (record, layer) in enumerate(zip(snapshot.layers, layers)):
            if index not in required:
                rendered.append(None)
                continue
            if level > 0:
                factor = 1 << level
                source = pyramids[index][level]
                extent = tuple(value / factor for value in box)
                grid = (sampling_grid[0] * factor, *sampling_grid[1:]) if sampling_grid else None
                if grid is not None and isinstance(source, Image.Image):
                    image = sample_tiles(
                        source.size,
                        size,
                        extent,
                        "RGBA",
                        256,
                        lambda xy: source.crop(
                            (xy[0] * 256, xy[1] * 256, (xy[0] + 1) * 256, (xy[1] + 1) * 256)
                        ),
                        grid,
                    )
                else:
                    image = source.transform(
                        size,
                        Image.Transform.EXTENT,
                        extent,
                        **({"sampling_grid": grid} if grid else {}),
                    )
            elif record.vector_records is None:
                image = record.surface.transform(
                    size, Image.Transform.EXTENT, box, sampling_grid=sampling_grid
                )
            else:
                scene = self._scene(record, snapshot.size)

                def read_vector_tile(coordinate):
                    x, y = coordinate
                    region = (
                        x * 256,
                        y * 256,
                        min(snapshot.size[0], (x + 1) * 256),
                        min(snapshot.size[1], (y + 1) * 256),
                    )
                    if not scene._index.query(region):
                        return None
                    key = (
                        "vector",
                        snapshot.size,
                        tuple(
                            record.vector_records.objects[index]
                            for index in scene._index.query(region)
                        ),
                        region,
                    )
                    pixels = self.cache.get(key)
                    if pixels is None:
                        pixels = self._vector(record, snapshot.size, region)
                        self.cache.put(key, pixels)
                    return pixels

                image = sample_tiles(
                    snapshot.size, size, box, "RGBA", 256, read_vector_tile, sampling_grid
                )
            if layer.opacity < 100:
                image.putalpha(
                    image.getchannel("A").point(
                        lambda value: int((value * layer.opacity + 50) // 100)
                    )
                )
            rendered.append(image)
        rendered = _apply_layer_masks(layers, rendered, visible_only=True)
        result = Image.new("RGBA", size)
        for layer, image in zip(layers, rendered):
            if layer.visible:
                result = composite(result, image, layer.blend_mode)
        return result


# Snapshot-only thumbnail composition; callers own the Tk upload.


def render_thumbnail(snapshot, size=(48, 34)):
    width, height = size
    preview = Image.new("RGBA", size, "#eeeeee")
    draw = ImageDraw.Draw(preview)
    for y in range(0, height, 5):
        for x in range(0, width, 5):
            if (x // 5 + y // 5) % 2:
                draw.rectangle((x, y, x + 4, y + 4), fill="#cccccc")
    ratio = min((width - 2) / snapshot.size[0], (height - 2) / snapshot.size[1])
    target = (max(1, round(snapshot.size[0] * ratio)), max(1, round(snapshot.size[1] * ratio)))
    pixels = Renderer(8 * 1024**2).render(snapshot, output_size=target, quality="interactive")
    preview.alpha_composite(pixels, ((width - target[0]) // 2, (height - target[1]) // 2))
    draw.rectangle((0, 0, width - 1, height - 1), outline="#888888")
    return preview
