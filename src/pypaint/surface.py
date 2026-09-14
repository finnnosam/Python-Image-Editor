"""Regional pixel contract and sparse copy-on-write RGBA8/L surfaces.

`crop` always returns owned Pillow pixels; `snapshot` shares immutable versions.
Full codec/NumPy adapters are explicit and limited to 512 MiB. Interactive kernels
use crop/paste, never that compatibility adapter.
"""

import math
import sys
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from pypaint import storage as tiles
from pypaint.storage import default_store, metadata, working


class Surface(Protocol):
    size: tuple[int, int]
    mode: str

    def crop(self, box): ...
    def paste(self, image, box=None, mask=None): ...
    def snapshot(self): ...
    def close(self): ...


class PillowSurface:
    """Reference adapter used to verify extraction independently of tiling."""

    def __init__(self, image):
        self._image = image.copy()
        self.mode, self.size = image.mode, image.size
        self._readonly = False

    def crop(self, box):
        return self._image.crop(box)

    def paste(self, image, box=None, mask=None):
        if self._readonly:
            raise TypeError("An immutable snapshot cannot be edited")
        self._image.paste(image, box, mask)

    def snapshot(self):
        result = PillowSurface(self._image)
        result._readonly = True
        return result

    def close(self):
        self._image.close()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Tile:
    payload: object
    bounds: tuple | None

    def __post_init__(self):
        metadata.track(
            self, sys.getsizeof(self) + (sys.getsizeof(self.bounds) if self.bounds else 0)
        )


class TiledSurface:
    MAX_MATERIALIZED_BYTES = 512 * 1024**2

    def __init__(self, mode, size, fill=0, *, tile_size=256, store=None):
        if mode not in ("RGBA", "L") or min(size) < 1 or max(size) > 1_000_000:
            raise ValueError("Unsupported surface format or dimensions")
        if tile_size not in (128, 256, 512):
            raise ValueError("Tile size must be 128, 256, or 512")
        self.mode, self.size, self.tile_size = mode, tuple(size), tile_size
        self.width, self.height = self.size
        self.fill = Image.new(mode, (1, 1), fill).getpixel((0, 0))
        self.store = store if store is not None else default_store()
        self._root = None
        self._working = {}
        self._working_bytes = 0
        self.revision = 0
        self.copied_bytes = 0
        self._readonly = False
        self._closed = False

    @classmethod
    def from_image(cls, image, **kwargs):
        result = cls(image.mode, image.size, **kwargs)
        result.paste(image)
        result.publish()
        return result

    def _check(self, write=False):
        if self._closed:
            raise RuntimeError("Surface has been released")
        if write and self._readonly:
            raise TypeError("An immutable snapshot cannot be edited")

    def tile_box(self, coordinate):
        x, y = coordinate
        n = self.tile_size
        return x * n, y * n, min(self.width, (x + 1) * n), min(self.height, (y + 1) * n)

    def coordinates(self, box=None):
        left, top, right, bottom = box or (0, 0, self.width, self.height)
        n = self.tile_size
        for y in range(max(0, top) // n, (min(bottom, self.height) + n - 1) // n):
            for x in range(max(0, left) // n, (min(right, self.width) + n - 1) // n):
                yield x, y

    def _read_tile(self, coordinate):
        self._check()
        if coordinate in self._working:
            return self._working[coordinate].copy()
        tile = tiles.get(self._root, tiles.tile_key(*coordinate))
        box = self.tile_box(coordinate)
        size = box[2] - box[0], box[3] - box[1]
        if tile is None:
            return Image.new(self.mode, size, self.fill)
        return Image.frombytes(self.mode, size, tile.payload.read())

    def _writable_tile(self, coordinate):
        if coordinate not in self._working:
            image = self._read_tile(coordinate)
            amount = image.width * image.height * len(image.getbands())
            try:
                working.track(image, amount)
            except OSError:
                # Other surfaces share this budget. Freeze our own pending
                # tiles before rejecting a write; their pixels remain available
                # through immutable, spillable payloads. Never publish a foreign
                # surface that may belong to a worker thread.
                if not self._working:
                    raise
                self.publish()
                working.track(image, amount)
            self._working[coordinate] = image
            self.copied_bytes += amount
            self._working_bytes += amount
        return self._working[coordinate]

    def publish(self):
        """Freeze all changed tiles atomically; failure leaves the previous root intact."""
        self._check()
        root = self._root
        for coordinate, image in self._working.items():
            data = image.tobytes()
            default = Image.new(self.mode, image.size, self.fill).tobytes()
            tile = None if data == default else Tile(self.store.put(data), image.getbbox())
            root = tiles.set_entry(root, tiles.tile_key(*coordinate), tile)
        self._root = root
        self._working.clear()
        self._working_bytes = 0

    def snapshot(self):
        self.publish()
        result = self.copy()
        result._readonly = True
        return result

    def copy(self):
        self.publish()
        result = TiledSurface(
            self.mode, self.size, self.fill, tile_size=self.tile_size, store=self.store
        )
        result._root, result.revision = self._root, self.revision
        return result

    def __deepcopy__(self, memo):
        result = self.copy()
        memo[id(self)] = result
        return result

    def crop(self, box=None):
        self._check()
        box = tuple(round(value) for value in (box or (0, 0, self.width, self.height)))
        left, top, right, bottom = box
        if right < left or bottom < top:
            raise ValueError("Invalid region")
        if (right - left) * (bottom - top) * Image.getmodebands(
            self.mode
        ) > self.MAX_MATERIALIZED_BYTES:
            raise ValueError("Region exceeds the 512 MiB Pillow allocation limit")
        result = Image.new(self.mode, (right - left, bottom - top), 0)
        for coordinate in self.coordinates(box):
            region = self.tile_box(coordinate)
            x0, y0, x1, y1 = (
                max(left, region[0]),
                max(top, region[1]),
                min(right, region[2]),
                min(bottom, region[3]),
            )
            tile = self._read_tile(coordinate)
            result.paste(
                tile.crop((x0 - region[0], y0 - region[1], x1 - region[0], y1 - region[1])),
                (x0 - left, y0 - top),
            )
        return result

    def paste(self, image, box=None, mask=None):
        self._check(write=True)
        if box is None:
            box = (0, 0)
        if len(box) == 2:
            size = image.size if hasattr(image, "size") else self.size
            box = (*box, box[0] + size[0], box[1] + size[1])
        box = tuple(map(int, box))
        if image is self:
            image = self.snapshot()
        for coordinate in self.coordinates(box):
            if self._working_bytes >= 16 * 1024**2:
                self.publish()
            region = self.tile_box(coordinate)
            x0, y0, x1, y1 = (
                max(box[0], region[0]),
                max(box[1], region[1]),
                min(box[2], region[2]),
                min(box[3], region[3]),
            )
            source_box = x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1]
            source = image.crop(source_box) if hasattr(image, "crop") else image
            local_mask = mask.crop(source_box) if mask is not None else None
            self._writable_tile(coordinate).paste(
                source, (x0 - region[0], y0 - region[1], x1 - region[0], y1 - region[1]), local_mask
            )
        self.revision += 1

    def alpha_composite(self, image, dest=(0, 0), source=(0, 0)):
        if len(source) == 2:
            source = (*source, image.width, image.height)
        box = (*dest, dest[0] + source[2] - source[0], dest[1] + source[3] - source[1])
        for coordinate in self.coordinates(box):
            region = self.tile_box(coordinate)
            overlap = (
                max(box[0], region[0]),
                max(box[1], region[1]),
                min(box[2], region[2]),
                min(box[3], region[3]),
            )
            patch = image.crop(
                (
                    overlap[0] - dest[0] + source[0],
                    overlap[1] - dest[1] + source[1],
                    overlap[2] - dest[0] + source[0],
                    overlap[3] - dest[1] + source[1],
                )
            )
            result = self.crop(overlap)
            result.alpha_composite(patch)
            self.paste(result, overlap)

    def region_surface(self, box):
        """Owned sparse region for large floating selections and codec staging."""
        result = TiledSurface(
            self.mode,
            (box[2] - box[0], box[3] - box[1]),
            tile_size=self.tile_size,
            store=self.store,
        )
        for coordinate in result.coordinates():
            region = result.tile_box(coordinate)
            result.paste(
                self.crop(
                    (region[0] + box[0], region[1] + box[1], region[2] + box[0], region[3] + box[1])
                ),
                region,
            )
        result.publish()
        return result

    def getpixel(self, xy):
        x, y = xy
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("Pixel outside surface")
        return self.crop((x, y, x + 1, y + 1)).getpixel((0, 0))

    def putpixel(self, xy, value):
        self.paste(value, (*xy, xy[0] + 1, xy[1] + 1))

    def getbbox(self, *, alpha_only=True):
        self.publish()
        if self.fill != (0, 0, 0, 0) and self.fill != 0:
            # Default-filled selections remain sparse. Read only their boundary
            # tiles until each nonempty extent is known.
            if self._root is None:
                return Image.new(self.mode, (1, 1), self.fill).getbbox(alpha_only=alpha_only) and (
                    0,
                    0,
                    self.width,
                    self.height,
                )
            bounds = None
            for coordinate in self.coordinates():
                x, y = coordinate
                if bounds and (
                    bounds[0] <= x * self.tile_size
                    and bounds[1] <= y * self.tile_size
                    and bounds[2] >= min(self.width, (x + 1) * self.tile_size)
                    and bounds[3] >= min(self.height, (y + 1) * self.tile_size)
                ):
                    continue
                local = self._read_tile(coordinate).getbbox(alpha_only=alpha_only)
                if local:
                    current = (
                        local[0] + x * self.tile_size,
                        local[1] + y * self.tile_size,
                        local[2] + x * self.tile_size,
                        local[3] + y * self.tile_size,
                    )
                    bounds = (
                        current
                        if bounds is None
                        else (
                            min(bounds[0], current[0]),
                            min(bounds[1], current[1]),
                            max(bounds[2], current[2]),
                            max(bounds[3], current[3]),
                        )
                    )
                if bounds == (0, 0, self.width, self.height):
                    return bounds
            return bounds
        bounds = []
        for (x, y), tile in tiles.entries(self._root):
            box = tile.bounds if alpha_only else self._read_tile((x, y)).getbbox(alpha_only=False)
            if box:
                bounds.append(
                    (
                        box[0] + x * self.tile_size,
                        box[1] + y * self.tile_size,
                        box[2] + x * self.tile_size,
                        box[3] + y * self.tile_size,
                    )
                )
        return (
            (
                min(b[0] for b in bounds),
                min(b[1] for b in bounds),
                max(b[2] for b in bounds),
                max(b[3] for b in bounds),
            )
            if bounds
            else None
        )

    def iter_tiles(self):
        self.publish()
        for coordinate, tile in tiles.entries(self._root):
            yield coordinate, self.tile_box(coordinate), self._read_tile(coordinate)

    def signature(self, box):
        self.publish()
        return tuple(
            (coordinate, tiles.get(self._root, tiles.tile_key(*coordinate)))
            for coordinate in self.coordinates(box)
        )

    def materialize(self):
        if (
            self.width * self.height * (4 if self.mode == "RGBA" else 1)
            > self.MAX_MATERIALIZED_BYTES
        ):
            raise ValueError("This Pillow adapter is limited to 512 MiB; use tiled native storage.")
        return self.crop((0, 0, self.width, self.height))

    def transform(self, size, method, data, resample=Image.Resampling.NEAREST, **kwargs):
        if method != Image.Transform.EXTENT or resample != Image.Resampling.NEAREST:
            raise ValueError("Regional surface transform supports nearest EXTENT only")
        from pypaint.rendering import sample_tiles

        def read_tile(coordinate):
            if (
                coordinate not in self._working
                and self.fill in (0, (0, 0, 0, 0))
                and tiles.get(self._root, tiles.tile_key(*coordinate)) is None
            ):
                return None
            return self._read_tile(coordinate)

        return sample_tiles(
            self.size,
            size,
            data,
            self.mode,
            self.tile_size,
            read_tile,
            sampling_grid=kwargs.get("sampling_grid"),
        )

    def resize(self, size, resample=Image.Resampling.BICUBIC, box=None, reducing_gap=None):
        if resample == Image.Resampling.NEAREST:
            return self.transform(
                size, Image.Transform.EXTENT, box or (0, 0, self.width, self.height)
            )
        if box is not None:
            return self.crop(box).resize(size, resample, reducing_gap=reducing_gap)
        from pypaint.rendering import resize_surface

        return resize_surface(self, size, resample)

    def getchannel(self, channel):
        result = TiledSurface(
            "L",
            self.size,
            Image.new(self.mode, (1, 1), self.fill).getchannel(channel).getpixel((0, 0)),
            tile_size=self.tile_size,
            store=self.store,
        )
        for _, box, image in self.iter_tiles():
            result.paste(image.getchannel(channel), box)
        return result

    def point(self, lut, mode=None):
        result = TiledSurface(
            mode or self.mode,
            self.size,
            Image.new(self.mode, (1, 1), self.fill).point(lut, mode).getpixel((0, 0)),
            tile_size=self.tile_size,
            store=self.store,
        )
        for _, box, image in self.iter_tiles():
            result.paste(image.point(lut, mode), box)
        return result

    def filter(self, kernel, cancellation=None):
        radius = getattr(kernel, "radius", None)
        if radius is None:
            raise ValueError("Surface filters must declare a bounded radius")
        if not isinstance(radius, (float, int)):
            radius = max(radius)
        halo = math.ceil(radius * 3) + 2
        result = TiledSurface(
            self.mode, self.size, self.fill, tile_size=self.tile_size, store=self.store
        )
        for coordinate in self.coordinates():
            if cancellation:
                cancellation.check()
            box = self.tile_box(coordinate)
            region = (
                max(0, box[0] - halo),
                max(0, box[1] - halo),
                min(self.width, box[2] + halo),
                min(self.height, box[3] + halo),
            )
            image = self.crop(region).filter(kernel)
            result.paste(
                image.crop(
                    (box[0] - region[0], box[1] - region[1], box[2] - region[0], box[3] - region[1])
                ),
                box,
            )
        result.publish()
        return result

    def putalpha(self, alpha):
        self._check(write=True)
        for coordinate in self.coordinates():
            if self._working_bytes >= 16 * 1024**2:
                self.publish()
            box = self.tile_box(coordinate)
            image = self._writable_tile(coordinate)
            image.putalpha(alpha.crop(box) if hasattr(alpha, "crop") else alpha)
        self.revision += 1

    def convert(self, mode=None, *args, **kwargs):
        return self.materialize().convert(mode, *args, **kwargs)

    def split(self):
        return self.materialize().split()

    def tobytes(self):
        return self.materialize().tobytes()

    def save(self, *args, **kwargs):
        return self.materialize().save(*args, **kwargs)

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.materialize(), dtype=dtype)
        return result.copy() if copy else result

    def close(self):
        self._root = None
        self._working.clear()
        self._closed = True
