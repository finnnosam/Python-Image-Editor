"""Brush, fill and clone kernels with cancellable raster gestures."""

import math
from typing import Protocol

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from pypaint.history import Transaction
from pypaint.jobs import Cancellation
from pypaint.surface import TiledSurface


class Gesture(Protocol):
    def update(self, image, box, mask=None): ...
    def commit(self): ...
    def cancel(self): ...


# UI-independent legacy brush kernels. Pixel conventions are characterized.


def _apply_hardness_to_alpha(alpha, hardness, softness_scale):
    """Adjust edge falloff while keeping 75% bit-identical to current output."""
    hardness = max(0, min(100, int(hardness)))
    if hardness == 75:
        return alpha
    if hardness < 75:
        blur_radius = softness_scale * (75 - hardness) / 75
        return alpha.filter(ImageFilter.GaussianBlur(blur_radius))
    if hardness == 100:
        return alpha.point(lambda value: 255 if value >= 128 else 0)
    exponent = 1 / (1 + 3 * (hardness - 75) / 25)
    return alpha.point(lambda value: round(255 * ((value / 255) ** exponent)))


def _build_up_opacity(opacity):
    """Apply a gentler response curve to build-up brush opacity."""
    opacity = max(0, min(255, int(opacity)))
    if opacity == 0:
        return 0
    return max(1, round(255 * ((opacity / 255) ** 3)))


def _accumulate_build_up_mask(existing, dab):
    """Screen a dab into coverage without an 8-bit rounding ceiling."""
    merged = ImageChops.screen(existing, dab)
    stalled = ImageChops.difference(merged, existing).point(
        lambda value: 255 if value == 0 else 0
    )
    stalled = ImageChops.multiply(stalled, dab.point(lambda value: 255 if value else 0))
    stalled = ImageChops.multiply(
        stalled, ImageOps.invert(existing).point(lambda value: 255 if value else 0)
    )
    stalled = stalled.point(lambda value: 1 if value else 0)
    return ImageChops.add(merged, stalled)


def _brush_shape_mask(image, bounds, paint_mask, antialias=False, hardness=75, softness_scale=None):
    """Return the clipped document box and coverage mask for a brush shape."""
    if softness_scale is None:
        softness_scale = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.25
    blur_radius = softness_scale * (75 - hardness) / 75 if antialias and hardness < 75 else 0
    padding = (1 + math.ceil(blur_radius * 3)) if antialias else 0
    left = max(0, math.floor(bounds[0]) - padding)
    top = max(0, math.floor(bounds[1]) - padding)
    right = min(image.width, math.ceil(bounds[2]) + 1 + padding)
    bottom = min(image.height, math.ceil(bounds[3]) + 1 + padding)
    if right <= left or bottom <= top:
        return None, None

    size = (right - left, bottom - top)
    # Tiny brushes need finer subpixel precision; larger stamps use 4x to
    # keep interactive painting responsive.
    scale = (64 if max(size) <= 6 else 4) if antialias else 1
    mask = Image.new("L", (size[0] * scale, size[1] * scale), 0)
    paint_mask(ImageDraw.Draw(mask), left, top, scale)
    if antialias:
        # BOX computes area coverage.  LANCZOS looks smooth too, but its
        # ringing can create faint pixels outside the actual brush footprint.
        mask = mask.resize(size, Image.Resampling.BOX)
        mask = _apply_hardness_to_alpha(mask, hardness, softness_scale)
    return (left, top, right, bottom), mask


def _brush_ellipse_box(bounds, left, top, scale):
    """Return a Pillow-safe local ellipse box for a brush dab.

    A two-pixel brush has a half-pixel raster radius, so without
    antialiasing its transformed endpoints should be identical. Floating
    point rounding can instead leave x1 or y1 infinitesimally below x0/y0,
    which Pillow rejects as an inverted ellipse.
    """
    x0 = (bounds[0] - left + 0.5) * scale
    y0 = (bounds[1] - top + 0.5) * scale
    # Pillow includes both endpoints. At scale 1 that inclusive far edge is
    # what gives a diameter-N brush N pixels. Supersampled masks, however,
    # need the final subpixel removed before being reduced to document size.
    far_edge_adjustment = 1 if scale > 1 else 0
    x1 = (bounds[2] - left + 0.5) * scale - far_edge_adjustment
    y1 = (bounds[3] - top + 0.5) * scale - far_edge_adjustment
    return (x0, y0, max(x0, x1), max(y0, y1))


def _composite_brush_shape(image, bounds, color, paint_mask, antialias=False):
    """Source-over composite a solid brush color through a shape mask."""
    box, mask = _brush_shape_mask(image, bounds, paint_mask, antialias=antialias)
    if box is None:
        return None

    source = Image.new("RGBA", mask.size, color)
    source_alpha = source.getchannel("A")
    source.putalpha(ImageChops.multiply(source_alpha, mask))
    image.alpha_composite(source, (box[0], box[1]))
    return box


def _pencil_path(x0, y0, x1, y1):
    """Yield a thin, four-connected staircase between two pixel centers."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx - dy
    yield x0, y0
    while (x0, y0) != (x1, y1):
        previous_x, previous_y = x0, y0
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x0 += sx
        if twice < dx:
            error += dx
            y0 += sy
        if x0 != previous_x and y0 != previous_y:
            yield ((previous_x, y0) if sy > 0 else (x0, previous_y))
        yield x0, y0


# Scanline fill reading bounded rows and storing sparse selection tiles.


def flood_region(image, x, y, tolerance=0, cancellation=None):
    token = cancellation or Cancellation()
    if image.width * image.height <= 1024 * 1024:
        # Bounded small-image fast path retains the original native-array scan.

        token.check()
        pixels = np.asarray(image.crop((0, 0, image.width, image.height)))
        target = pixels[y, x]
        if tolerance == 0:
            matches = np.all(pixels == target, axis=2)
        elif tolerance == 100:
            matches = np.ones(pixels.shape[:2], dtype=bool)
        else:
            difference = pixels.astype(np.int32) - target.astype(np.int32)
            matches = (
                np.sum(difference * difference, axis=2)
                <= (math.sqrt(4 * 255 * 255) * tolerance / 100) ** 2
            )
        result = _connected_region_mask(matches, x, y)
        token.check()
        return TiledSurface.from_image(result)
    target = np.asarray(image.crop((x, y, x + 1, y + 1)))[0, 0]
    result = TiledSurface("L", image.size)
    pending = [(x, y)]

    def matching_row(row):
        pixels = np.asarray(image.crop((0, row, image.width, row + 1)))[0]
        if tolerance == 0:
            return np.all(pixels == target, axis=1)
        if tolerance == 100:
            return np.ones(image.width, dtype=bool)
        distance = np.zeros(image.width, dtype=np.int32)
        for channel in range(4):
            difference = pixels[:, channel].astype(np.int32) - int(target[channel])
            distance += difference * difference
        return distance <= (math.sqrt(4 * 255 * 255) * tolerance / 100) ** 2

    while pending:
        token.check()
        seed, row = pending.pop()
        visited = np.asarray(result.crop((0, row, image.width, row + 1)))[0] > 0
        matches = matching_row(row) & ~visited
        if not matches[seed]:
            continue
        left = seed
        while left > 0 and matches[left - 1]:
            left -= 1
        right = seed + 1
        while right < image.width and matches[right]:
            right += 1
        result.paste(255, (left, row, right, row + 1))
        for neighbor in (row - 1, row + 1):
            if 0 <= neighbor < image.height:
                available = matching_row(neighbor)[left:right]
                available &= np.asarray(result.crop((left, neighbor, right, neighbor + 1)))[0] == 0
                starts = np.flatnonzero(available & ~np.r_[False, available[:-1]])
                if len(pending) + len(starts) > 1_000_000:
                    raise ValueError("Fill frontier exceeds the supported working-set limit")
                pending.extend((left + int(start), neighbor) for start in starts)
    return result


def _connected_region_mask(matches, pixel_x, pixel_y):
    """Flood four-connected runs, without a Python set entry per pixel."""
    height, width = matches.shape
    pending = [(pixel_x, pixel_y)]
    remaining = bytearray(matches.astype(np.uint8).tobytes())
    result = Image.new("L", (width, height), 0)
    while pending:
        x, y = pending.pop()
        row = y * width
        if not remaining[row + x]:
            continue
        left = max(row, remaining.rfind(b"\x00", row, row + x) + 1)
        right = remaining.find(b"\x00", row + x, row + width)
        if right < 0:
            right = row + width
        remaining[left:right] = b"\x00" * (right - left)
        left -= row
        right -= row
        result.paste(255, (left, y, right, y + 1))
        for neighbor in (y - 1, y + 1):
            if not 0 <= neighbor < height:
                continue
            offset = neighbor * width
            end = offset + right
            start = remaining.find(b"\x01", offset + left, end)
            while start >= 0:
                pending.append((start - offset, neighbor))
                stop = remaining.find(b"\x00", start, end)
                if stop < 0:
                    break
                start = remaining.find(b"\x01", stop, end)
    return result


# Bounded, cancellable bucket processing from an immutable source version.


def compute(source, seed, tolerance, antialias, hardness, color, selection, token):
    mask = flood_region(source, *seed, tolerance, token)
    if antialias:
        mask = mask.filter(ImageFilter.GaussianBlur(0.65), cancellation=token)
        if hardness < 75:
            mask = mask.filter(
                ImageFilter.GaussianBlur(2 * (75 - hardness) / 75), cancellation=token
            )
        else:
            mask = _apply_hardness_to_alpha(mask, hardness, 2)
    result = source.copy()
    bounds = mask.getbbox()
    if bounds is None:
        return result.snapshot(), None
    for coordinate in source.coordinates(bounds):
        token.check()
        box = source.tile_box(coordinate)
        local = mask.crop(box)
        selected = selection.crop(box) if selection is not None else None
        if selected is not None:
            local = ImageChops.multiply(local, selected)
        if not local.getbbox():
            continue
        paint = Image.new("RGBA", local.size, color)
        paint.putalpha(ImageChops.multiply(paint.getchannel("A"), local))
        patch = source.crop(box)
        patch.alpha_composite(paint)
        # Retain the original selection's second write-mask application.
        result.paste(patch, box, selected)
    return result.snapshot(), bounds


class RasterGesture:
    def __init__(self, document, layer, name="Brush"):
        self.transaction = Transaction(document, name)
        self.layer = layer
        self.baseline = layer.image.snapshot()
        self.coverage = TiledSurface("L", layer.image.size, store=layer.image.store)

    def update(self, image, box, mask=None):
        return self.transaction.write(self.layer, image, box, mask)

    def commit(self):
        return self.transaction.commit()

    def cancel(self):
        self.transaction.cancel()


# Clone gesture owns a fixed source, capped baseline, and one transaction.


class CloneGesture(RasterGesture):
    def __init__(self, document, layer):
        super().__init__(document, layer, "Clone stroke")
        self.source = self.baseline
