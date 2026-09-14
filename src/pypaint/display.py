"""Retained Tk display pixels and cooperative navigation prefetch."""

import math
import time
from types import SimpleNamespace

from PIL import Image, ImageTk


class DisplayPrefetch:
    MAX_BYTES = 32 * 1024**2

    def __init__(self, canvas):
        self.canvas = canvas
        self.after_id = None
        self.pending = self.ready = None
        self.times = []
        self.error = None
        self.failed_key = None

    def cancel(self):
        if self.after_id is not None:
            self.canvas.after_cancel(self.after_id)
        self.after_id = self.pending = self.ready = None

    def validate(self, key, layout, dirty):
        state = self.pending or self.ready
        if state is not None and (state.key != key or state.layout != layout or dirty):
            self.cancel()

    def start(self, key, layout, box, source_box, source, render, zoom):

        width, height = box[2] - box[0], box[3] - box[1]
        # Account conservatively for Pillow's padded RGB storage, alpha, RGBA,
        # and the optional Tk RGB photo (excluding transient render patches).
        if width * height * 14 > self.MAX_BYTES or key == self.failed_key:
            return
        state = self.pending or self.ready
        if state is not None:
            return
        overlap = intersection(box, source_box)
        # Keep each request inside a document tile: a wide strip can otherwise
        # synchronously composite many cold, multilayer tiles in one callback.
        rectangles = []
        for l, t, r, b in exposed_boxes(box, overlap):
            y = t
            while y < b:
                edge_y = math.ceil((math.floor((y + 0.5) / (256 * zoom)) + 1) * 256 * zoom - 0.5)
                bottom = min(b, y + 64, max(y + 1, edge_y))
                x = l
                while x < r:
                    edge_x = math.ceil(
                        (math.floor((x + 0.5) / (256 * zoom)) + 1) * 256 * zoom - 0.5
                    )
                    right = min(r, x + 128, max(x + 1, edge_x))
                    rectangles.append((x, y, right, bottom))
                    x = right
                y = bottom
        self.pending = SimpleNamespace(
            key=key,
            layout=layout,
            box=box,
            overlap=overlap,
            source_box=source_box,
            source=source,
            render=render,
            zoom=zoom,
            rectangles=iter(rectangles),
            rgba=None,
            rgb=None,
            mask=None,
            photo=None,
            stage="allocate",
        )
        self._schedule()

    def _schedule(self):
        self.after_id = self.canvas.after(2, self._step)

    def _step(self):
        self.after_id = None
        state = self.pending
        if state is None:
            return
        start = time.perf_counter()
        try:
            while self.pending is state:
                l, t, r, b = state.box
                if state.stage == "allocate":
                    state.rgba = Image.new("RGBA", (r - l, b - t))
                    if state.overlap is not None:
                        x0, y0, x1, y1 = state.overlap
                        sx, sy = state.source_box[:2]
                        state.rgba.paste(
                            state.source.crop((x0 - sx, y0 - sy, x1 - sx, y1 - sy)),
                            (x0 - l, y0 - t),
                        )
                    state.source = None
                    state.stage = "render"
                elif state.stage == "render":
                    rect = next(state.rectangles, None)
                    if rect is None:
                        state.stage = "rgb"
                    else:
                        x0, y0, x1, y1 = rect
                        pixels = state.render(
                            tuple(v / state.zoom for v in rect), (x1 - x0, y1 - y0)
                        )
                        state.rgba.paste(pixels, (x0 - l, y0 - t))
                elif state.stage == "rgb":
                    state.rgb = state.rgba.convert("RGB")
                    state.stage = "alpha"
                elif state.stage == "alpha":
                    state.mask = state.rgba.getchannel("A")
                    state.alpha = state.mask.getextrema()
                    state.stage = "photo" if state.alpha == (255, 255) else "ready"
                elif state.stage == "photo":
                    # Unbound RGB photos avoid the fragmented-alpha Tk repaint path.
                    state.photo = ImageTk.PhotoImage(state.rgb)
                    state.stage = "ready"
                if state.stage == "ready":
                    state.render = None
                    self.ready, self.pending = state, None
                    break
                if time.perf_counter() - start >= 0.002:
                    break
            if self.pending is not None:
                self._schedule()
        except Exception as error:
            # This cache is optional. A failed prefetch must not spoil the
            # currently displayed image or a valid edit; the draw path can retry.
            self.error = str(error)
            self.failed_key = state.key
            self.cancel()
        finally:
            self.times.append((time.perf_counter() - start) * 1000)
            del self.times[:-256]


# Retain visible pixels in Tk and redraw only strips exposed by panning.


def intersection(a, b):
    box = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return box if box[2] > box[0] and box[3] > box[1] else None


def exposed_boxes(box, overlap):
    if overlap is None:
        return [box]
    l, t, r, b = box
    x0, y0, x1, y1 = overlap
    return [
        rect
        for rect in ((l, t, r, y0), (l, y1, r, b), (l, y0, x0, y1), (x1, y0, r, y1))
        if rect[2] > rect[0] and rect[3] > rect[1]
    ]


def visible_box(size, viewport, zoom, offset):
    # Integer coordinates in the zoomed document give sampling a stable origin.
    return intersection(
        (0, 0, math.ceil(size[0] * zoom), math.ceil(size[1] * zoom)),
        (
            math.floor(-offset[0]),
            math.floor(-offset[1]),
            math.ceil(viewport[0] - offset[0]),
            math.ceil(viewport[1] - offset[1]),
        ),
    )


def rounded(value):
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


class DisplaySurface:
    def __init__(self, canvas):
        self.canvas = canvas
        self.photo = self.item = self.box = self.rgba = None
        self.rgb = self.mask = None
        self.display_box = self.display_offset = None
        self.alpha = (0, 0)
        self.key = self.layout = None
        self.checker = self.checker_photo = None
        self.backdrop_items = []
        self.backdrop_geometry = None
        self.last_stats = {}
        self.prefetch = DisplayPrefetch(canvas)

    def clear(self):
        self.prefetch.cancel()
        self.canvas.delete("document-display")
        self.photo = self.item = self.box = self.rgba = None
        self.rgb = self.mask = None
        self.display_box = self.display_offset = None
        self.alpha = (0, 0)
        self.backdrop_items.clear()
        self.checker = self.checker_photo = None
        self.key = self.layout = self.backdrop_geometry = None

    def draw(self, size, viewport, zoom, offset, key, layout, dirty_box, checker, render):
        box = visible_box(size, viewport, zoom, offset)
        visible = box
        self.prefetch.validate(key, layout, dirty_box is not None)
        ready = self.prefetch.ready
        if ready is not None and (
            ready.box[2] - ready.box[0] > viewport[0] + 258
            or ready.box[3] - ready.box[1] > viewport[1] + 258
        ):
            self.prefetch.cancel()
            ready = None
        if ready is not None and box is not None and intersection(box, ready.box) == box:
            self.rgba, self.rgb, self.mask = ready.rgba, ready.rgb, ready.mask
            self.box, self.alpha = ready.box, ready.alpha
            if ready.photo is not None and self.item is not None:
                self.photo = ready.photo
                self.canvas.itemconfigure(self.item, image=self.photo)
                self.display_box = ready.box
            self.prefetch.ready = None
        checker_changed = self.checker is not checker
        self._backdrop(size, viewport, zoom, offset, checker)
        full = self.layout != layout or (self.key != key and dirty_box is None)
        if box is not None and self.layout == layout and self.box is not None:
            margin = 128
            bounded = (
                self.box[2] - self.box[0] <= viewport[0] + margin * 2 + 2
                and self.box[3] - self.box[1] <= viewport[1] + margin * 2 + 2
            )
            if bounded and intersection(box, self.box) == box:
                box = self.box
            else:
                # A narrow fallback bridges timer granularity while the larger
                # margin is prepared cooperatively. Avoid a full 128px refill here.
                can_prefetch = (viewport[0] + 256) * (
                    viewport[1] + 256
                ) * 14 <= self.prefetch.MAX_BYTES and key != self.prefetch.failed_key
                if (
                    self.display_offset is not None
                    and max(
                        abs(offset[0] - self.display_offset[0]),
                        abs(offset[1] - self.display_offset[1]),
                    )
                    > 48
                ):
                    can_prefetch = False
                fallback = 32 if can_prefetch else 128
                box = intersection(
                    (box[0] - fallback, box[1] - fallback, box[2] + fallback, box[3] + fallback),
                    (0, 0, math.ceil(size[0] * zoom), math.ceil(size[1] * zoom)),
                )
        self.key, self.layout = key, layout
        if box is None:
            self.prefetch.cancel()
            if self.item is not None:
                self.canvas.delete(self.item)
            self.photo = self.item = self.box = self.rgba = None
            self.rgb = self.mask = None
            self.display_box = self.display_offset = None
            self.alpha = (0, 0)
            self.last_stats = dict(rendered_regions=0, uploaded_pixels=0, retained_pixels=0)
            return
        overlap = intersection(box, self.box) if self.box is not None and not full else None
        regions = exposed_boxes(box, overlap)
        if dirty_box is not None and overlap is not None:
            dirty = (
                math.floor(dirty_box[0] * zoom),
                math.floor(dirty_box[1] * zoom),
                math.ceil(dirty_box[2] * zoom),
                math.ceil(dirty_box[3] * zoom),
            )
            dirty = intersection(dirty, overlap)
            if dirty is not None:
                regions.append(dirty)
        width, height = box[2] - box[0], box[3] - box[1]
        previous_box = self.box
        rebased = box != previous_box or overlap is None
        if rebased:
            previous = self.rgba
            self.rgba = Image.new("RGBA", (width, height))
            if overlap is not None:
                l, t, r, b = overlap
                self.rgba.paste(
                    previous.crop(
                        (
                            l - previous_box[0],
                            t - previous_box[1],
                            r - previous_box[0],
                            b - previous_box[1],
                        )
                    ),
                    (l - box[0], t - box[1]),
                )
        for l, t, r, b in regions:
            patch = render(tuple(value / zoom for value in (l, t, r, b)), (r - l, b - t))
            self.rgba.paste(patch, (l - box[0], t - box[1]))
            if not rebased:
                mask = patch.getchannel("A")
                self.rgb.paste(patch.convert("RGB"), (l - box[0], t - box[1]))
                self.mask.paste(mask, (l - box[0], t - box[1]))
                lo, hi = mask.getextrema()
                self.alpha = min(self.alpha[0], lo), max(self.alpha[1], hi)
        if rebased:
            self.rgb = self.rgba.convert("RGB")
            self.mask = self.rgba.getchannel("A")
            self.alpha = self.mask.getextrema()
        self.box = box
        uploaded = 0
        if self.alpha == (0, 0):
            if self.item is not None:
                self.canvas.delete(self.item)
            self.photo = self.item = self.display_box = None
        else:
            opaque = self.alpha == (255, 255)
            target = box if opaque else visible_box(size, viewport, zoom, offset)
            position = (offset[0] + target[0], offset[1] + target[1])
            changed_position = self.display_offset != offset

            # Never hand Tk a fragmented alpha image: Windows Tk can spend
            # seconds constructing its transparency region. Flatten to RGB first.
            def pixels(rect):
                l, t, r, b = rect
                if opaque:
                    return self.rgb.crop((l - box[0], t - box[1], r - box[0], b - box[1]))
                sx, sy = rounded(offset[0] + l), rounded(offset[1] + t)
                background = self.checker_rgb.crop((sx, sy, sx + r - l, sy + b - t))
                # The checker is opaque: an RGB masked paste is exactly the
                # same blend, without RGBA conversion or cropped source copies.
                background.paste(self.rgb, (box[0] - l, box[1] - t), self.mask)
                # EXTENT can enclose one pixel beyond the actual Tk viewport.
                # Preserve the old transparent-outside-checker crop semantics.
                bounds = (0, 0, r - l, b - t)
                covered = intersection(bounds, (-sx, -sy, checker.width - sx, checker.height - sy))
                for x0, y0, x1, y1 in exposed_boxes(bounds, covered):
                    outside = self.rgba.crop(
                        (l - box[0] + x0, t - box[1] + y0, l - box[0] + x1, t - box[1] + y1)
                    )
                    empty = Image.new("RGBA", outside.size)
                    background.paste(Image.alpha_composite(empty, outside).convert("RGB"), (x0, y0))
                return background

            complete = (
                self.photo is None
                or target != self.display_box
                or full
                or (not opaque and (changed_position or checker_changed))
            )
            if complete:
                image = pixels(target)
                resized = (
                    self.photo is None or (self.photo.width(), self.photo.height()) != image.size
                )
                if resized:
                    self.photo = ImageTk.PhotoImage(image)
                else:
                    self.photo.paste(image)
                uploaded = image.width * image.height
            else:
                resized = False
                for rect in regions:
                    rect = intersection(rect, target)
                    if rect is None:
                        continue
                    patch = ImageTk.PhotoImage(pixels(rect))
                    self.canvas.tk.call(
                        str(self.photo),
                        "copy",
                        str(patch),
                        "-to",
                        rect[0] - target[0],
                        rect[1] - target[1],
                        "-compositingrule",
                        "set",
                    )
                    uploaded += (rect[2] - rect[0]) * (rect[3] - rect[1])
            if self.item is None:
                self.item = self.canvas.create_image(
                    *position, image=self.photo, anchor="nw", tags=("document-display",)
                )
            else:
                self.canvas.coords(self.item, *position)
                if resized:
                    self.canvas.itemconfigure(self.item, image=self.photo)
            self.display_box = target
        previous_offset = self.display_offset
        self.display_offset = offset
        self.last_stats = dict(
            rendered_regions=len(regions),
            rendered_pixels=sum((r - l) * (b - t) for l, t, r, b in regions),
            uploaded_pixels=uploaded,
            retained_pixels=width * height,
        )
        if (
            not full
            and dirty_box is None
            and previous_offset is not None
            and offset != previous_offset
        ):
            self._prefetch_next(
                size, viewport, zoom, offset, previous_offset, visible, key, layout, render
            )

    def _prefetch_next(self, size, viewport, zoom, offset, previous, visible, key, layout, render):
        dx, dy = previous[0] - offset[0], previous[1] - offset[1]
        if max(abs(dx), abs(dy)) > 48:
            # Rapid jumps can outrun the cooperative producer. Spend the UI
            # budget on the current view rather than repeatedly cancelled work.
            self.prefetch.cancel()
            return
        # Start before the leading edge exhausts the current buffer. Cancel work
        # after a direction reversal or when its target no longer covers the view.
        state = self.prefetch.pending or self.prefetch.ready
        if state is not None:
            sx, sy = getattr(state, "direction", (dx, dy))
            if dx * sx < 0 or dy * sy < 0 or intersection(visible, state.box) != visible:
                self.prefetch.cancel()
            else:
                return
        xgap = self.box[2] - visible[2] if dx > 0 else visible[0] - self.box[0]
        ygap = self.box[3] - visible[3] if dy > 0 else visible[1] - self.box[1]
        if (dx == 0 or xgap > 96) and (dy == 0 or ygap > 96):
            return
        lead_x = 64 if dx > 0 else -64 if dx < 0 else 0
        lead_y = 64 if dy > 0 else -64 if dy < 0 else 0
        target = intersection(
            (
                visible[0] - 128 + lead_x,
                visible[1] - 128 + lead_y,
                visible[2] + 128 + lead_x,
                visible[3] + 128 + lead_y,
            ),
            (0, 0, math.ceil(size[0] * zoom), math.ceil(size[1] * zoom)),
        )
        if target is not None and target != self.box:
            self.prefetch.start(key, layout, target, self.box, self.rgba, render, zoom)
            if self.prefetch.pending is not None:
                self.prefetch.pending.direction = dx, dy

    def _backdrop(self, size, viewport, zoom, offset, checker):
        # Four rectangles clip the fixed checkerboard to the document footprint.
        if self.checker is not checker:
            for item in self.backdrop_items:
                self.canvas.delete(item)
            self.checker = checker
            self.checker_rgb = checker.convert("RGB")
            self.checker_photo = ImageTk.PhotoImage(self.checker_rgb)
            background = self.canvas.cget("background")
            self.backdrop_items = [
                self.canvas.create_image(
                    0, 0, image=self.checker_photo, anchor="nw", tags=("document-display",)
                )
            ]
            self.backdrop_items += [
                self.canvas.create_rectangle(
                    0, 0, 0, 0, fill=background, outline="", tags=("document-display",)
                )
                for _ in range(4)
            ]
            for item in reversed(self.backdrop_items):
                self.canvas.tag_lower(item)
            self.backdrop_geometry = None
        cw, ch = viewport

        left, top = map(rounded, offset)
        right, bottom = left + math.ceil(size[0] * zoom), top + math.ceil(size[1] * zoom)
        left, right = max(0, min(cw, left)), max(0, min(cw, right))
        top, bottom = max(0, min(ch, top)), max(0, min(ch, bottom))
        geometry = (cw, ch, left, top, right, bottom)
        if geometry != self.backdrop_geometry:
            boxes = (
                (0, 0, cw, top),
                (0, bottom, cw, ch),
                (0, top, left, bottom),
                (right, top, cw, bottom),
            )
            for item, box in zip(self.backdrop_items[1:], boxes):
                self.canvas.coords(item, *box)
            self.backdrop_geometry = geometry
