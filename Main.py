import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageChops, ImageDraw, ImageTk, ImageOps, ImageFilter
import numpy as np
import math
import colorsys
import copy
import io
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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
    return alpha.point(
        lambda value: round(255 * ((value / 255) ** exponent)))


def _brush_shape_mask(image, bounds, paint_mask, antialias=False,
                      hardness=75):
    """Return the clipped document box and coverage mask for a brush shape."""
    softness_scale = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.25
    blur_radius = (softness_scale * (75 - hardness) / 75
                   if antialias and hardness < 75 else 0)
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
    x1 = ((bounds[2] - left + 0.5) * scale - far_edge_adjustment)
    y1 = ((bounds[3] - top + 0.5) * scale - far_edge_adjustment)
    return (x0, y0, max(x0, x1), max(y0, y1))


def _composite_brush_shape(image, bounds, color, paint_mask, antialias=False):
    """Source-over composite a solid brush color through a shape mask."""
    box, mask = _brush_shape_mask(
        image, bounds, paint_mask, antialias=antialias)
    if box is None:
        return None

    source = Image.new("RGBA", mask.size, color)
    source_alpha = source.getchannel("A")
    source.putalpha(ImageChops.multiply(source_alpha, mask))
    image.alpha_composite(source, (box[0], box[1]))
    return box


class VectorObject:
    """Base class for vector objects"""
    def __init__(self, color="#000000", width=2, antialias=True, hardness=75):
        self.color = color
        self.width = width
        self.antialias = antialias
        self.hardness = hardness
        self.selected = False
        
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'type': self.__class__.__name__,
            'color': self.color,
            'width': self.width,
            'antialias': self.antialias,
            'hardness': self.hardness,
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create object from dictionary"""
        data = dict(data)  # avoid mutating the original
        obj_type = data.pop('type')
        if obj_type == 'Line':
            return Line.from_dict(data)
        elif obj_type == 'Shape':
            return Shape.from_dict(data)
        elif obj_type == 'Rectangle':
            return Shape.from_legacy_rectangle(data)
        elif obj_type == 'Ellipse':
            return Shape.from_legacy_ellipse(data)
        return None

class Line(VectorObject):
    def __init__(self, x1=0, y1=0, x2=100, y2=100, color="#000000", width=2,
                 curve=None, space="flat", antialias=True, hardness=75):
        super().__init__(color, width, antialias, hardness)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.curve = curve
        self.space = space
        
    def to_dict(self):
        data = super().to_dict()
        data.update({
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'curve': self.curve,
            'space': self.space,
        })
        return data
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['x1'], data['y1'], data['x2'], data['y2'],
                   data['color'], data['width'], data.get('curve'),
                   data.get('space', 'flat'), data.get('antialias', True),
                   data.get('hardness', 75))
    
    def sampled_points(self, document_width=1024, document_height=512, steps=32):
        """Return map points. Globe lines follow the shortest great-circle arc."""
        if self.space == "globe":
            from sphere_math import uv_to_vec, arc_to_uv
            a = uv_to_vec((1.0 - self.x1 / document_width) % 1.0,
                          self.y1 / document_height)
            b = uv_to_vec((1.0 - self.x2 / document_width) % 1.0,
                          self.y2 / document_height)
            points = arc_to_uv(a, b, step_radians=math.pi / max(8, steps))
            mapped = [((1.0 - u) % 1.0 * document_width, v * document_height)
                      for u, v in points]
            # Unwrap longitude so PIL draws across the seam, not across the map.
            for i in range(1, len(mapped)):
                px = mapped[i - 1][0]
                x, y = mapped[i]
                while x - px > document_width / 2: x -= document_width
                while px - x > document_width / 2: x += document_width
                mapped[i] = (x, y)
            return mapped
        if not self.curve:
            return [(self.x1, self.y1), (self.x2, self.y2)]
        controls = self.curve
        if len(controls) == 2:
            cx, cy = controls[0], controls[1]
            return [((1-t)**2*self.x1 + 2*(1-t)*t*cx + t*t*self.x2,
                     (1-t)**2*self.y1 + 2*(1-t)*t*cy + t*t*self.y2)
                    for t in (i / steps for i in range(steps + 1))]
        c1x, c1y, c2x, c2y = controls
        return [((1-t)**3*self.x1 + 3*(1-t)**2*t*c1x + 3*(1-t)*t*t*c2x + t**3*self.x2,
                 (1-t)**3*self.y1 + 3*(1-t)**2*t*c1y + 3*(1-t)*t*t*c2y + t**3*self.y2)
                for t in (i / steps for i in range(steps + 1))]

    def draw(self, draw, document_width=1024, document_height=512):
        points = self.sampled_points(document_width, document_height)
        offsets = (-document_width, 0, document_width) if self.space == "globe" else (0,)
        for offset in offsets:
            draw.line([(x + offset, y) for x, y in points], fill=self.color, width=self.width)
        
    def get_points(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]
    
    def update_point(self, index, x, y):
        if index == 0:
            self.x1, self.y1 = x, y
        elif index == 1:
            self.x2, self.y2 = x, y


class Shape(VectorObject):
    """A closed/open preset made exclusively from Line primitives."""
    def __init__(self, lines=None, color="#000000", width=2, fill=None,
                 filled_side="inside", preset="custom", antialias=True,
                 hardness=75):
        super().__init__(color, width, antialias, hardness)
        self.lines = lines or []
        self.fill = fill
        self.filled_side = filled_side
        self.preset = preset
        self._spherical_fill_cache = None
        for line in self.lines:
            line.color, line.width = color, width
            line.antialias = antialias
            line.hardness = hardness

    def to_dict(self):
        data = super().to_dict()
        data.update(lines=[line.to_dict() for line in self.lines], fill=self.fill,
                    filled_side=self.filled_side, preset=self.preset)
        return data

    @classmethod
    def from_dict(cls, data):
        lines = [Line.from_dict({k: v for k, v in item.items() if k != 'type'})
                 for item in data.get('lines', [])]
        return cls(lines, data['color'], data['width'], data.get('fill'),
                   data.get('filled_side', 'inside'), data.get('preset', 'custom'),
                   data.get('antialias', True), data.get('hardness', 75))

    @classmethod
    def from_legacy_rectangle(cls, data):
        x, y, w, h = data['x'], data['y'], data['w'], data['h']
        antialias = data.get('antialias', True)
        hardness = data.get('hardness', 75)
        vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        lines = [Line(*vertices[i], *vertices[(i+1) % 4], data['color'],
                      data['width'], antialias=antialias, hardness=hardness)
                 for i in range(4)]
        return cls(lines, data['color'], data['width'], data.get('fill'),
                   preset='rect', antialias=antialias, hardness=hardness)

    @classmethod
    def from_legacy_ellipse(cls, data):
        cx, cy, rx, ry = data['x'], data['y'], data['rx'], data['ry']
        antialias = data.get('antialias', True)
        hardness = data.get('hardness', 75)
        k = 0.5522847498
        vertices = [(cx+rx,cy),(cx,cy+ry),(cx-rx,cy),(cx,cy-ry)]
        controls = [(cx+rx,cy+k*ry,cx+k*rx,cy+ry),
                    (cx-k*rx,cy+ry,cx-rx,cy+k*ry),
                    (cx-rx,cy-k*ry,cx-k*rx,cy-ry),
                    (cx+k*rx,cy-ry,cx+rx,cy-k*ry)]
        lines = [Line(*vertices[i], *vertices[(i+1) % 4], data['color'],
                      data['width'], curve=controls[i], antialias=antialias,
                      hardness=hardness)
                 for i in range(4)]
        return cls(lines, data['color'], data['width'], data.get('fill'),
                   preset='ellipse', antialias=antialias, hardness=hardness)

    def _outline(self, width, height):
        points = []
        for line in self.lines:
            segment = line.sampled_points(width, height)
            if points and segment:
                shift = round((points[-1][0] - segment[0][0]) / width) * width
                segment = [(x + shift, y) for x, y in segment]
            points.extend(segment if not points else segment[1:])
        return points

    def _spherical_fill_mask(self, width, height):
        """Rasterize the smaller spherical interior, including across a pole."""
        vertices_xy = [(line.x1, line.y1) for line in self.lines]
        cache_key = (width, height, tuple(vertices_xy))
        if self._spherical_fill_cache and self._spherical_fill_cache[0] == cache_key:
            return self._spherical_fill_cache[1]

        # Physical sphere coordinates use the opposite longitude direction to
        # the displayed texture, matching globe_view's texture transform.
        uv = np.asarray([((1.0 - x / width) % 1.0, y / height)
                         for x, y in vertices_xy], dtype=np.float64)
        lon = uv[:, 0] * (2 * np.pi) - np.pi
        lat = (0.5 - uv[:, 1]) * np.pi
        vertices = np.column_stack((np.cos(lat) * np.cos(lon),
                                    np.sin(lat),
                                    np.cos(lat) * np.sin(lon)))

        # A spherical winding has an antipodal counterpart with the opposite
        # direction.  Determine which direction belongs to the shape at its
        # own centre so the far side is not filled as a second copy.
        centre = np.sum(vertices, axis=0)
        centre_length = np.linalg.norm(centre)
        if centre_length < 1e-12:
            centre = vertices[0]
        else:
            centre /= centre_length
        centre_tangents = []
        for vertex in vertices:
            tangent = vertex - np.dot(centre, vertex) * centre
            tangent /= max(np.linalg.norm(tangent), 1e-12)
            centre_tangents.append(tangent)
        centre_winding = 0.0
        for i, tangent in enumerate(centre_tangents):
            following = centre_tangents[(i + 1) % len(centre_tangents)]
            centre_winding += np.arctan2(
                np.dot(centre, np.cross(tangent, following)),
                np.dot(tangent, following))
        inside_direction = 1.0 if centre_winding >= 0.0 else -1.0

        mask = np.zeros((height, width), dtype=np.uint8)
        pixel_lon = (1.0 - (np.arange(width) + 0.5) / width) * (2*np.pi) - np.pi
        cos_lon, sin_lon = np.cos(pixel_lon), np.sin(pixel_lon)

        # Work in strips to keep temporary tangent arrays bounded for ellipses.
        for y0 in range(0, height, 32):
            y1 = min(height, y0 + 32)
            pixel_lat = (0.5 - (np.arange(y0, y1) + 0.5) / height) * np.pi
            cos_lat = np.cos(pixel_lat)[:, None]
            points = np.stack(np.broadcast_arrays(
                cos_lat * cos_lon[None, :],
                np.sin(pixel_lat)[:, None],
                cos_lat * sin_lon[None, :]), axis=-1)
            winding = np.zeros(points.shape[:2], dtype=np.float64)
            tangents = []
            for vertex in vertices:
                tangent = vertex - np.sum(points * vertex, axis=-1)[..., None] * points
                tangent /= np.maximum(np.linalg.norm(tangent, axis=-1)[..., None], 1e-12)
                tangents.append(tangent)
            for i, tangent in enumerate(tangents):
                following = tangents[(i + 1) % len(tangents)]
                sine = np.sum(points * np.cross(tangent, following), axis=-1)
                cosine = np.sum(tangent * following, axis=-1)
                winding += np.arctan2(sine, cosine)
            mask[y0:y1] = (winding * inside_direction > np.pi).astype(np.uint8) * 255

        result = Image.fromarray(mask, mode="L")
        self._spherical_fill_cache = (cache_key, result)
        return result

    def draw(self, draw, document_width=1024, document_height=512):
        points = self._outline(document_width, document_height)
        globe = any(line.space == "globe" for line in self.lines)
        offsets = (-document_width, 0, document_width) if globe else (0,)
        if self.fill and len(points) >= 3 and self.filled_side == "inside":
            if globe:
                draw.bitmap((0, 0), self._spherical_fill_mask(
                    document_width, document_height), fill=self.fill)
            else:
                draw.polygon(points, fill=self.fill)
        # Stroke the complete outline once. Drawing each constituent segment
        # separately makes shared endpoints overlap, producing visibly thicker
        # corners and uneven joins.
        if len(points) >= 2:
            for offset in offsets:
                draw.line(
                    [(x + offset, y) for x, y in points],
                    fill=self.color, width=self.width, joint="curve")

    def get_points(self):
        return [(line.x1, line.y1) for line in self.lines]

    def update_point(self, index, x, y):
        if not (0 <= index < len(self.lines)):
            return
        old = (self.lines[index].x1, self.lines[index].y1)
        self._spherical_fill_cache = None
        for line in self.lines:
            if (line.x1, line.y1) == old: line.x1, line.y1 = x, y
            if (line.x2, line.y2) == old: line.x2, line.y2 = x, y

class Rectangle(VectorObject):
    def __init__(self, x=0, y=0, w=100, h=100, color="#000000", width=2, fill=None):
        super().__init__(color, width)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.fill = fill
        
    def to_dict(self):
        data = super().to_dict()
        data.update({
            'x': self.x,
            'y': self.y,
            'w': self.w,
            'h': self.h,
            'fill': self.fill
        })
        return data
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['x'], data['y'], data['w'], data['h'], data['color'], data['width'], data['fill'])
    
    def draw(self, draw, document_width=1024, document_height=512):
        draw.rectangle([(self.x, self.y), (self.x + self.w, self.y + self.h)], 
                       outline=self.color, fill=self.fill, width=self.width)
        
    def get_points(self):
        return [(self.x, self.y), (self.x + self.w, self.y), 
                (self.x + self.w, self.y + self.h), (self.x, self.y + self.h)]

class Ellipse(VectorObject):
    def __init__(self, x=0, y=0, rx=50, ry=50, color="#000000", width=2, fill=None):
        super().__init__(color, width)
        self.x = x
        self.y = y
        self.rx = rx
        self.ry = ry
        self.fill = fill
        
    def to_dict(self):
        data = super().to_dict()
        data.update({
            'x': self.x,
            'y': self.y,
            'rx': self.rx,
            'ry': self.ry,
            'fill': self.fill
        })
        return data
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['x'], data['y'], data['rx'], data['ry'], data['color'], data['width'], data['fill'])
    
    def draw(self, draw, document_width=1024, document_height=512):
        draw.ellipse([(self.x - self.rx, self.y - self.ry), (self.x + self.rx, self.y + self.ry)], 
                     outline=self.color, fill=self.fill, width=self.width)
        
    def get_points(self):
        return [(self.x - self.rx, self.y), (self.x + self.rx, self.y), 
                (self.x, self.y - self.ry), (self.x, self.y + self.ry)]


def _draw_vector_path(image, points, color, width, antialias=True, hardness=75):
    """Stroke a path with consistent geometry and optional edge smoothing."""
    if len(points) < 2:
        return

    softness_scale = width * 0.25
    blur_radius = (softness_scale * (75 - hardness) / 75
                   if antialias and hardness < 75 else 0)
    padding = width / 2 + 2 + math.ceil(blur_radius * 3)
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

    tile = Image.new("RGBA", (tile_width * scale, tile_height * scale),
                     (0, 0, 0, 0))
    scaled_points = [((x - left) * scale, (y - top) * scale)
                     for x, y in points]
    ImageDraw.Draw(tile).line(
        scaled_points, fill=color, width=max(1, round(width * scale)),
        joint="curve")
    resampling = (Image.Resampling.LANCZOS if antialias
                  else Image.Resampling.NEAREST)
    tile = tile.resize((tile_width, tile_height), resampling)
    if antialias and hardness != 75:
        tile.putalpha(_apply_hardness_to_alpha(
            tile.getchannel("A"), hardness, softness_scale))
    image.alpha_composite(tile, (left, top))


def render_vector_object(image, obj, document_width, document_height):
    """Render one vector object with an anti-aliased, uniform-width stroke."""
    if isinstance(obj, Line):
        points = obj.sampled_points(document_width, document_height)
        offsets = (-document_width, 0, document_width) \
            if obj.space == "globe" else (0,)
        for offset in offsets:
            _draw_vector_path(
                image, [(x + offset, y) for x, y in points],
                obj.color, obj.width, obj.antialias,
                getattr(obj, "hardness", 75))
        return

    if isinstance(obj, Shape):
        points = obj._outline(document_width, document_height)
        globe = any(line.space == "globe" for line in obj.lines)
        if obj.fill and len(points) >= 3 and obj.filled_side == "inside":
            draw = ImageDraw.Draw(image)
            if globe:
                draw.bitmap((0, 0), obj._spherical_fill_mask(
                    document_width, document_height), fill=obj.fill)
            else:
                draw.polygon(points, fill=obj.fill)
        offsets = (-document_width, 0, document_width) if globe else (0,)
        for offset in offsets:
            _draw_vector_path(
                image, [(x + offset, y) for x, y in points],
                obj.color, obj.width, obj.antialias,
                getattr(obj, "hardness", 75))
        return

    # Compatibility for any legacy in-memory vector object.
    obj.draw(ImageDraw.Draw(image), document_width, document_height)

class VectorLayer:
    def __init__(self, name, width, height):
        self.name = name
        self.visible = True
        self.objects = []
        self.selected_object = None
        self.selected_point = None
        self.width = width
        self.height = height
        
    def add_object(self, obj):
        self.objects.append(obj)
        
    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)
            
    def render(self, image):
        for obj in self.objects:
            render_vector_object(image, obj, self.width, self.height)
            
    def get_object_at(self, x, y, tolerance=10):
        """Find object at position (for selection)"""
        # Check in reverse order (top objects first)
        for obj in reversed(self.objects):
            points = obj.get_points()
            for px, py in points:
                if abs(px - x) <= tolerance and abs(py - y) <= tolerance:
                    return obj, points.index((px, py))
        return None, None
    
    def to_dict(self):
        return {
            'name': self.name,
            'visible': self.visible,
            'objects': [obj.to_dict() for obj in self.objects]
        }
    
    @classmethod
    def from_dict(cls, data, width, height):
        layer = cls(data['name'], width, height)
        layer.visible = data['visible']
        for obj_data in data['objects']:
            obj = VectorObject.from_dict(obj_data)
            if obj:
                layer.objects.append(obj)
        return layer

class Layer:
    def __init__(self, width, height, name, layer_type="raster"):
        self.name = name
        self.visible = True
        self.opacity = 100
        self.layer_type = layer_type  # "raster" or "vector"
        self.masked = False
        self.width = width
        self.height = height
        
        if layer_type == "raster":
            self.image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            self.draw = ImageDraw.Draw(self.image)
            self.vector_data = None
        else:  # vector
            self.image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            self.draw = ImageDraw.Draw(self.image)
            self.vector_data = VectorLayer(name, width, height)

        # Level zero is always the editable image.  Smaller levels are built
        # lazily as zooming needs them, rather than for every opened image.
        self._mipmaps = [self.image]
        self._mipmap_revision = 0

    def reset_mipmaps(self):
        """Discard reduced previews after replacing the whole layer image."""
        self._mipmaps = [self.image]
        self._mipmap_revision += 1

    @property
    def is_raster(self):
        return self.layer_type == "raster"

    def get_mipmap(self, level):
        """Return a cached 2**level reduction of this layer."""
        if not self._mipmaps or self._mipmaps[0] is not self.image:
            self.reset_mipmaps()
        while len(self._mipmaps) <= level:
            previous = self._mipmaps[-1]
            size = (max(1, (previous.width + 1) // 2),
                    max(1, (previous.height + 1) // 2))
            self._mipmaps.append(previous.resize(size, Image.Resampling.BOX))
        return self._mipmaps[level]

    def update_mipmaps(self, box):
        """Incrementally refresh cached levels touched by a raster edit."""
        self._mipmap_revision += 1
        if not self._mipmaps or self._mipmaps[0] is not self.image:
            self.reset_mipmaps()
            return

        left, top, right, bottom = box
        for level in range(1, len(self._mipmaps)):
            previous = self._mipmaps[level - 1]
            current = self._mipmaps[level]
            # Include a pixel of context for BOX filtering at edit boundaries.
            dl = max(0, int(math.floor(left / 2)) - 1)
            dt = max(0, int(math.floor(top / 2)) - 1)
            dr = min(current.width, int(math.ceil(right / 2)) + 1)
            db = min(current.height, int(math.ceil(bottom / 2)) + 1)
            if dr <= dl or db <= dt:
                return
            source_box = (dl * 2, dt * 2,
                          min(previous.width, dr * 2),
                          min(previous.height, db * 2))
            reduced = previous.crop(source_box).resize(
                (dr - dl, db - dt), Image.Resampling.BOX)
            current.paste(reduced, (dl, dt))
            left, top, right, bottom = dl, dt, dr, db

    def render_vector(self):
        """Render vector objects to the raster image"""
        if self.layer_type == "vector" and self.vector_data:
            # Clear the image
            self.image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            self.draw = ImageDraw.Draw(self.image)
            self.vector_data.render(self.image)
            self.reset_mipmaps()

    def image_with_opacity(self, image=None):
        """Return a compositing copy with this layer's opacity applied."""
        source = image if image is not None else self.image
        if self.opacity >= 100:
            return source
        adjusted = source.copy()
        alpha = adjusted.getchannel("A").point(
            lambda value: (value * self.opacity + 50) // 100)
        adjusted.putalpha(alpha)
        return adjusted

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyPaint")

        self.doc_w = 1024   #this is what it launches with
        self.doc_h = 512
        self.current_file = None

        self.redraw_pending = False
        self.last_redraw = 0.0
        self.redraw_after_id = None
        self.mipmap_after_id = None
        self.pending_mipmap_level = None
        self.mipmap_future = None
        self.mipmap_build_level = None
        self.mipmap_executor = ThreadPoolExecutor(max_workers=1,
                                                  thread_name_prefix="mipmap")
        self.main_view_dirty = False
        self.target_frame_time = 1 / 60.0    #FPS

        self.layers = [Layer(self.doc_w, self.doc_h, "Background", "raster")]
        self.active_layer = 0
        self.undo_stack = []

        self.zoom = 1.0
        self.offset_x = 20
        self.offset_y = 20

        self.tool = "brush"
        self.primary_color = "#000000"
        self.secondary_color = "#ffffff"
        self.primary_opacity = 255
        self.secondary_opacity = 255
        self._stroke_base_image = None
        self._stroke_coverage = None
        self.active_color_slot = "primary"
        self.picker_hue = 0.0
        self._color_controls_updating = False
        self.color = self.primary_color  # Keep for compatibility
        # Empty document pixels stay transparent.  The checkerboard is a view
        # backdrop, not document content, so compositing (including the globe
        # texture) must not begin on opaque white.
        self.bg_color = (255, 255, 255, 0)
        self.last_button = 1  # Track which mouse button was pressed

        # Checkerboard "absent pixel" backdrop (lives behind the canvas content,
        # does not pan/zoom with it - rendered once per canvas size).
        self.checker_size = 18
        self.checker_light = (235, 235, 235, 255)
        self.checker_dark = (210, 210, 210, 255)
        self._checker_pil = None        # cached full-viewport tiled PIL image
        self._checker_pil_dims = None   # (cw, ch) it was built for
        self._canvas_image_id = None

        # The brush cursor artwork lives beside the application instead of
        # being drawn as a Canvas primitive.  Keep the source image and cache
        # only the currently displayed size.
        cursor_path = Path(__file__).resolve().parent / "icons" / "brush-outline.png"
        with Image.open(cursor_path) as cursor_image:
            self._brush_cursor_source = cursor_image.convert("RGBA")
        self._brush_cursor_tkimg = None
        self._brush_cursor_diameter = None

        self.last_x = None
        self.last_y = None
        self.mouse_x = 0
        self.mouse_y = 0
        self.pan_x = 0
        self.pan_y = 0

        # Vector tool states
        self.vector_start_x = None
        self.vector_start_y = None
        self.current_vector_obj = None
        self.selected_vector_obj = None
        self.selected_point_index = None
        self.is_dragging_point = False
        self.selection_start = None
        self.selection_bounds = None
        self.selection_mask = Image.new("L", (self.doc_w, self.doc_h), 0)
        self.selection_operation = None
        self.selection_base_mask = None
        self.selection_edges = []
        self.selection_dash_offset = 0
        self.selection_animation_id = None
        self.move_start = None
        self.move_source_box = None
        self.move_pixels = None
        self.move_mask = None
        self.move_base_image = None
        self.move_selection_bounds = None
        self.move_offset = (0, 0)
        self.move_drag_origin_offset = (0, 0)
        self.selection_move_start = None
        self.selection_move_bounds = None
        self.selection_move_mask = None
        self.selection_brush_last = None
        self.selection_brush_remove = False
        self.clone_source_center = None
        self.clone_offset = None
        self.clone_last = None
        self.clone_stroke_source = None
        self.clone_stroke_base = None
        self.clone_stroke_coverage = None
        self.bucket_pending = None

        # Drag and drop variables
        self.drag_start_index = None
        self.drag_start_y = None

        # UI scale (1.5 = launch default)
        self.ui_scale = 1.5

        self.build_ui()
        # Tk does not normally move keyboard focus when a label, frame, or
        # canvas background is clicked. Treat those clicks as "click away"
        # so entries commit through their existing <FocusOut> handlers.
        self.root.bind_all("<Button-1>", self._commit_active_entry, add="+")
        self.root.bind_all(
            "<Button-1>", self._commit_pending_bucket_on_click, add="+")
        self.root.bind_all(
            "<Button-3>", self._commit_pending_bucket_on_click, add="+")
        self.root.bind_all("<Return>", self._finish_bucket_preview, add="+")
        self.apply_ui_scale(self.ui_scale)  # A: apply 1.5× on launch
        self.refresh_layers()
        self.redraw()
        self.update_title()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _commit_active_entry(self, event):
        """Commit and unfocus an entry when the user clicks outside it."""
        focused = self.root.focus_get()
        if focused is None or focused.winfo_class() not in {
                "Entry", "TEntry", "Spinbox", "TSpinbox"}:
            return

        clicked = event.widget
        while clicked is not None:
            if clicked == focused:
                return
            clicked = getattr(clicked, "master", None)

        # Moving focus fires the field's FocusOut callback synchronously,
        # which validates and applies the edited value.
        self.root.focus_set()

    def _commit_pending_bucket_on_click(self, event):
        """Commit a bucket preview when clicking outside its settings."""
        if self.bucket_pending is None:
            return
        if event.serial == self.bucket_pending.get("event_serial"):
            return
        clicked = event.widget
        while clicked is not None:
            if clicked == self.bucket_settings_frame:
                return
            clicked = getattr(clicked, "master", None)
        self._finish_bucket_preview()

    def build_ui(self):
        # ── Top bar: file & edit actions ──────────────────────────────────
        top = tk.Frame(self.root, bd=1, relief="raised")
        top.pack(fill="x", side="top")

        tk.Button(top, text="New",        command=self.new_project).pack(side="left", padx=2, pady=2)
        tk.Button(top, text="Open",       command=self.open_project).pack(side="left", padx=2, pady=2)
        tk.Button(top, text="Save",       command=self.save_project).pack(side="left", padx=2, pady=2)
        tk.Button(top, text="Save As",    command=self.save_project_as).pack(side="left", padx=2, pady=2)
        tk.Button(top, text="Export PNG", command=self.save_image).pack(side="left", padx=2, pady=2)
        tk.Button(top, text="Undo",       command=self.undo).pack(side="left", padx=2, pady=2)
        tk.Button(top, text="Globe View", command=self.open_globe_view).pack(side="left", padx=2, pady=2)
        tk.Button(top, text="Settings",   command=self.open_settings).pack(side="right", padx=2, pady=2)

        # ── Main area: tools | canvas | layers ────────────────────────────
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # ── Left panel: tools ─────────────────────────────────────────────
        left = tk.Frame(main, width=190, bd=1, relief="sunken")
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="Tools", font=("TkDefaultFont", 9, "bold")).pack(pady=(6, 2))

        tool_frame = tk.Frame(left)
        tool_frame.pack(fill="x", padx=4)

        tools = [
            ("Selection",  "selection",   None),
            ("Brush Selection", "brush selection", None),
            ("Move",       "move",        None),
            ("Move Selection", "move selection", None),
            ("Pan",        "pan",         None),
            ("Color Picker", "color picker", None),
            ("Brush",      "brush",       "brush.png"),
            ("Eraser",     "eraser",      "eraser.png"),
            ("Clone",      "clone",       None),
            ("Paint Bucket", "paint bucket", None),
            ("Vector Edit", "vector edit", "vector-edit.png"),
            ("Line",       "line",        "line.png"),
            ("Rectangle",  "rect",        "rect.png"),
            ("Ellipse",    "ellipse",     "ellipse.png"),
        ]
        icon_dir = Path(__file__).resolve().parent / "icons"
        self.tool_icons = {}
        self.tool_buttons = {}
        self.tools_by_layer_type = {
            "raster": ("selection", "move", "move selection",
                       "brush selection", "pan", "color picker", "brush",
                       "eraser", "clone", "paint bucket"),
            "vector": ("pan", "color picker", "vector edit", "line", "rect", "ellipse"),
        }
        self.tool_hint_var = tk.StringVar(value="Brush")
        for index, (label, tool, filename) in enumerate(tools):
            if filename is None:
                icon_image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
                icon_draw = ImageDraw.Draw(icon_image)
                if tool == "selection":
                    icon_draw.rectangle(
                        (5, 6, 27, 25), outline="#202020", width=2)
                    icon_draw.rectangle(
                        (8, 9, 24, 22), outline="#d8d8d8", width=1)
                    # Small gaps suggest a marquee without depending on a
                    # platform-specific dashed-line renderer.
                    for x, y in ((5, 6), (16, 6), (27, 6), (5, 16),
                                 (27, 16), (5, 25), (16, 25), (27, 25)):
                        icon_draw.rectangle(
                            (x - 1, y - 1, x + 1, y + 1), fill="#f2f2f2")
                elif tool == "move":
                    icon_draw.rectangle(
                        (5, 5, 20, 20), outline="#202020", width=2)
                    icon_draw.line((14, 14, 27, 27), fill="#202020", width=3)
                    icon_draw.polygon(
                        (28, 28, 20, 26, 26, 20), fill="#202020")
                elif tool == "move selection":
                    icon_draw.rectangle(
                        (4, 5, 20, 21), outline="#202020", width=2)
                    icon_draw.line((16, 17, 27, 28), fill="#787878", width=2)
                    icon_draw.polygon(
                        (28, 29, 21, 27, 27, 21), fill="#787878")
                elif tool == "brush selection":
                    icon_draw.ellipse(
                        (5, 5, 25, 25), outline="#202020", width=2)
                    icon_draw.ellipse((11, 11, 19, 19), fill="#787878")
                    icon_draw.line((22, 22, 28, 28), fill="#202020", width=3)
                elif tool == "clone":
                    icon_draw.ellipse((6, 5, 19, 18), outline="#202020", width=2)
                    icon_draw.ellipse((13, 13, 26, 26), outline="#787878", width=2)
                    icon_draw.line((17, 9, 23, 15), fill="#202020", width=2)
                    icon_draw.polygon((25, 17, 19, 15, 23, 11), fill="#202020")
                elif tool == "paint bucket":
                    icon_draw.polygon(
                        (7, 12, 17, 5, 26, 15, 16, 24),
                        outline="#202020", fill="#d8d8d8")
                    icon_draw.line((9, 12, 18, 21), fill="#202020", width=2)
                    icon_draw.ellipse((22, 23, 27, 29), fill="#202020")
                elif tool == "pan":
                    icon_draw.line((6, 16, 26, 16), fill="#202020", width=3)
                    icon_draw.line((16, 6, 16, 26), fill="#202020", width=3)
                    icon_draw.polygon((3, 16, 9, 11, 9, 21), fill="#202020")
                    icon_draw.polygon((29, 16, 23, 11, 23, 21), fill="#202020")
                    icon_draw.polygon((16, 3, 11, 9, 21, 9), fill="#202020")
                    icon_draw.polygon((16, 29, 11, 23, 21, 23), fill="#202020")
                else:
                    icon_draw.line((8, 25, 23, 10), fill="#202020", width=5)
                    icon_draw.line((11, 28, 26, 13), fill="#d8d8d8", width=3)
                    icon_draw.polygon((22, 5, 28, 11, 24, 15, 18, 9),
                                      fill="#202020")
                    icon_draw.rectangle((5, 25, 10, 29), outline="#202020")
            else:
                with Image.open(icon_dir / filename) as source_image:
                    icon_image = source_image.convert("RGBA").resize(
                        (32, 32), Image.Resampling.LANCZOS)
            icon = ImageTk.PhotoImage(icon_image)
            self.tool_icons[tool] = icon
            button = tk.Button(
                tool_frame, image=icon, width=42, height=42,
                command=lambda t=tool: self.set_tool(t),
                relief="sunken" if tool == self.tool else "raised",
                takefocus=True)
            if tool == "selection":
                button.grid(row=0, column=1, padx=2, pady=2, sticky="w")
            elif tool == "move":
                button.grid(row=2, column=1, padx=2, pady=2, sticky="w")
            elif tool == "move selection":
                button.grid(row=3, column=1, padx=2, pady=2, sticky="w")
            elif tool == "brush selection":
                button.grid(row=1, column=1, padx=2, pady=2, sticky="w")
            else:
                button.grid(row=index - 4, column=0, padx=2, pady=2, sticky="w")
            button.bind(
                "<Enter>",
                lambda event, name=label: self.tool_hint_var.set(name))
            button.bind(
                "<Leave>",
                lambda event: self.tool_hint_var.set(self.tool.title()))
            self.tool_buttons[tool] = button
        self.brush_build_up_var = tk.BooleanVar(value=False)
        self.brush_antialias_var = tk.BooleanVar(value=True)
        self.brush_hardness_var = tk.IntVar(value=75)
        self.brush_spacing_var = tk.StringVar(value="12.5")
        # Same-named tool options are universal. Each tool presents its own
        # relevant controls, but all of those controls reference these shared
        # variables and therefore stay synchronized automatically.
        self.clone_antialias_var = self.brush_antialias_var
        self.clone_build_up_var = self.brush_build_up_var
        self.clone_hardness_var = self.brush_hardness_var
        self.clone_spacing_var = self.brush_spacing_var
        self.bucket_antialias_var = self.brush_antialias_var
        self.bucket_hardness_var = self.brush_hardness_var
        self.bucket_tolerance_var = tk.IntVar(value=0)
        self.picker_sample_area_var = tk.BooleanVar(value=False)
        self.vector_antialias_var = self.brush_antialias_var
        self.vector_hardness_var = self.brush_hardness_var
        tk.Label(left, textvariable=self.tool_hint_var).pack(pady=(2, 0))

        # ── Color Selector ─────────────────────────────────────────────
        # Keep the whole group anchored to the bottom of the sidebar. The
        # unused height between Tools and Colors expands with the window.
        color_panel = tk.Frame(left)
        color_panel.pack(side="bottom", fill="x", pady=(0, 6))

        ttk.Separator(color_panel, orient="horizontal").pack(
            fill="x", padx=4, pady=(0, 12))
        tk.Label(color_panel, text="Colors",
                 font=("TkDefaultFont", 9, "bold")).pack(pady=(0, 2))

        color_frame = tk.Frame(color_panel)
        color_frame.pack(pady=4)

        # Primary color square (left)
        self.primary_square = tk.Canvas(color_frame, width=30, height=30, bg=self.primary_color,
                                        highlightthickness=3, highlightbackground="#2878d7")
        self.primary_square.pack(side="left", padx=2)
        self.primary_square.bind("<Button-1>", lambda e: self.choose_primary_color())

        # Secondary color square (right)
        self.secondary_square = tk.Canvas(color_frame, width=30, height=30, bg=self.secondary_color,
                                          highlightthickness=3, highlightbackground="black")
        self.secondary_square.pack(side="left", padx=2)
        self.secondary_square.bind("<Button-1>", lambda e: self.choose_secondary_color())

        # Swap colors button
        tk.Button(color_panel, text="↔", width=3,
                  command=self.swap_colors).pack(pady=(0, 6))

        controls = tk.Frame(color_panel)
        controls.pack(fill="x", padx=8, pady=(4, 0))
        self.rgb_vars = [tk.IntVar(value=0) for _ in range(3)]
        self.hsv_vars = [tk.IntVar(value=0), tk.IntVar(value=0),
                         tk.IntVar(value=0)]
        self.opacity_var = tk.IntVar(value=255)
        self.hex_var = tk.StringVar(value="000000")
        self.color_sliders = []

        tk.Label(controls, text="RGB", anchor="w").pack(fill="x")
        for label, variable in zip(("R", "G", "B"), self.rgb_vars):
            self._add_color_slider(controls, label, variable, 0, 255,
                                   self._rgb_controls_changed)

        hex_row = tk.Frame(controls)
        hex_row.pack(fill="x", pady=(2, 3))
        tk.Label(hex_row, text="Hex:", width=4, anchor="w").pack(side="left")
        self.hex_entry = tk.Entry(hex_row, textvariable=self.hex_var,
                                  width=8, justify="right")
        self.hex_entry.pack(side="right")
        self.hex_entry.bind("<Return>", self._hex_control_changed)
        self.hex_entry.bind("<FocusOut>", self._hex_control_changed)

        tk.Label(controls, text="HSV", anchor="w").pack(fill="x")
        for index, (label, variable, maximum) in enumerate(zip(
                ("H", "S", "V"), self.hsv_vars, (359, 100, 100))):
            self._add_color_slider(controls, label, variable, 0, maximum,
                                   lambda component=index: self._hsv_controls_changed(component))

        tk.Label(controls, text="Opacity", anchor="w").pack(fill="x", pady=(3, 0))
        self._add_color_slider(controls, "A", self.opacity_var, 0, 255,
                               self._opacity_control_changed)

        self._sync_picker_to_active_color()

        self.size_var = tk.StringVar(value="2")

        # Update size when Enter is pressed or focus is lost
        def update_size(event=None):
            try:
                val = int(self.size_var.get())
                if val < 1:
                    val = 1
                elif val > 999:
                    val = 999
                self.size_var.set(str(val))
            except ValueError:
                self.size_var.set("2")  # revert to default on invalid input

        def adjust_size(delta):
            """Adjust tool size by delta while preserving its valid range."""
            try:
                current = int(self.size_var.get())
            except ValueError:
                current = 2
            self.size_var.set(str(max(1, min(999, current + delta))))
            self.request_redraw()

        def adjust_size_with_control(delta):
            def handler(event):
                adjust_size(delta * 5)
                # Suppress the button's normal command for this click.
                return "break"
            return handler

        # ── Centre: reusable view workspace ──────────────────────────────
        # Tools and layers live outside this frame, so every view shares them.
        self.view_workspace = tk.Frame(main)
        self.view_workspace.pack(side="left", fill="both", expand=True)

        self.view_tabs = tk.Frame(self.view_workspace, bd=1, relief="raised")
        self.view_tabs.pack(side="top", fill="x")

        # Keep tool settings in a stable horizontal strip below the view tabs.
        # The fixed-height outer frame remains visible even for tools without
        # options, preventing the canvas from changing size as tools change.
        self.tool_settings_bar = tk.Frame(
            self.view_workspace, height=44, bd=1, relief="groove")
        self.tool_settings_bar.pack(side="top", fill="x")
        self.tool_settings_bar.pack_propagate(False)
        tk.Label(
            self.tool_settings_bar, text="Tool settings:",
            font=("TkDefaultFont", 9, "bold")
        ).pack(side="left", padx=(8, 10))

        self.size_frame = tk.Frame(self.tool_settings_bar)
        self.size_label = tk.Label(self.size_frame, text="Size:")
        self.size_label.pack(side="left")
        self.size_entry = tk.Entry(
            self.size_frame, width=6, textvariable=self.size_var)
        self.size_entry.pack(side="left", padx=(4, 3))
        self.size_entry.bind("<Return>", update_size)
        self.size_entry.bind("<FocusOut>", update_size)
        self.size_minus_button = tk.Button(
            self.size_frame, text="−", width=2,
            command=lambda: adjust_size(-1))
        self.size_minus_button.pack(side="left")
        self.size_minus_button.bind(
            "<Control-Button-1>", adjust_size_with_control(-1))
        self.size_plus_button = tk.Button(
            self.size_frame, text="+", width=2,
            command=lambda: adjust_size(1))
        self.size_plus_button.pack(side="left", padx=(2, 10))
        self.size_plus_button.bind(
            "<Control-Button-1>", adjust_size_with_control(1))

        def scroll_size(event):
            if event.delta:
                adjust_size(1 if event.delta > 0 else -1)
            return "break"

        for widget in (self.size_frame, self.size_label, self.size_entry,
                       self.size_minus_button, self.size_plus_button):
            widget.bind("<MouseWheel>", scroll_size)

        self.brush_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.brush_settings_frame, text="Build up",
            variable=self.brush_build_up_var
        ).pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            self.brush_settings_frame, text="Anti-alias",
            variable=self.brush_antialias_var
        ).pack(side="left")
        tk.Label(self.brush_settings_frame, text="Hardness:").pack(
            side="left", padx=(10, 3))
        tk.Scale(
            self.brush_settings_frame, from_=0, to=100, orient="horizontal",
            length=110, variable=self.brush_hardness_var
        ).pack(side="left")
        tk.Button(
            self.brush_settings_frame, text="Reset",
            command=lambda: self.brush_hardness_var.set(75)
        ).pack(side="left", padx=(3, 0))

        self.clone_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.clone_settings_frame, text="Build up",
            variable=self.clone_build_up_var
        ).pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            self.clone_settings_frame, text="Anti-alias",
            variable=self.clone_antialias_var
        ).pack(side="left")
        tk.Label(self.clone_settings_frame, text="Hardness:").pack(
            side="left", padx=(10, 3))
        tk.Scale(
            self.clone_settings_frame, from_=0, to=100, orient="horizontal",
            length=110, variable=self.clone_hardness_var
        ).pack(side="left")
        tk.Button(
            self.clone_settings_frame, text="Reset",
            command=lambda: self.clone_hardness_var.set(75)
        ).pack(side="left", padx=(3, 0))

        self.bucket_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.bucket_settings_frame, text="Anti-alias",
            variable=self.bucket_antialias_var,
            command=self._refresh_bucket_preview
        ).pack(side="left")
        tk.Label(self.bucket_settings_frame, text="Hardness:").pack(
            side="left", padx=(10, 3))
        tk.Scale(
            self.bucket_settings_frame, from_=0, to=100, orient="horizontal",
            length=110, variable=self.bucket_hardness_var,
            command=lambda value: self._refresh_bucket_preview()
        ).pack(side="left")
        tk.Button(
            self.bucket_settings_frame, text="Reset",
            command=self._reset_bucket_hardness
        ).pack(side="left", padx=(3, 0))
        tk.Label(self.bucket_settings_frame, text="Tolerance:").pack(
            side="left", padx=(10, 3))
        tk.Scale(
            self.bucket_settings_frame, from_=0, to=100, orient="horizontal",
            length=110, variable=self.bucket_tolerance_var,
            command=lambda value: self._refresh_bucket_preview()
        ).pack(side="left")
        tk.Button(
            self.bucket_settings_frame, text="Reset",
            command=self._reset_bucket_tolerance
        ).pack(side="left", padx=(3, 0))
        tk.Label(self.clone_settings_frame, text="Spacing:").pack(
            side="left", padx=(10, 3))
        self.clone_spacing_entry = tk.Entry(
            self.clone_settings_frame, width=5,
            textvariable=self.clone_spacing_var)
        self.clone_spacing_entry.pack(side="left")
        tk.Label(self.clone_settings_frame, text="%").pack(side="left")
        tk.Label(self.brush_settings_frame, text="Spacing:").pack(
            side="left", padx=(10, 3))
        self.brush_spacing_entry = tk.Entry(
            self.brush_settings_frame, width=5,
            textvariable=self.brush_spacing_var)
        self.brush_spacing_entry.pack(side="left")
        tk.Label(self.brush_settings_frame, text="%").pack(side="left")

        self.picker_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.picker_settings_frame, text="Sample area",
            variable=self.picker_sample_area_var,
            command=self.update_tool_settings_visibility
        ).pack(side="left")

        self.vector_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.vector_settings_frame, text="Anti-alias",
            variable=self.vector_antialias_var
        ).pack(side="left")
        tk.Label(self.vector_settings_frame, text="Hardness:").pack(
            side="left", padx=(10, 3))
        tk.Scale(
            self.vector_settings_frame, from_=0, to=100,
            orient="horizontal", length=110,
            variable=self.vector_hardness_var
        ).pack(side="left")
        tk.Button(
            self.vector_settings_frame, text="Reset",
            command=lambda: self.vector_hardness_var.set(75)
        ).pack(side="left", padx=(3, 0))

        def validate_spacing(variable):
            try:
                value = max(0.1, min(1000, float(
                    variable.get())))
            except ValueError:
                value = 12.5
            variable.set(f"{value:g}")

        for entry, variable in (
                (self.brush_spacing_entry, self.brush_spacing_var),
                (self.clone_spacing_entry, self.clone_spacing_var)):
            entry.bind("<Return>", lambda event, var=variable:
                       validate_spacing(var))
            entry.bind("<FocusOut>", lambda event, var=variable:
                       validate_spacing(var))

        self.view_host = tk.Frame(self.view_workspace)
        self.view_host.pack(side="top", fill="both", expand=True)

        self.views = {}
        self.view_tab_widgets = {}
        self.active_view = None

        flat_view = tk.Frame(self.view_host)
        # Use a centered plus pointer so brush/eraser strokes land at the
        # intersection while the separate overlay still shows brush size.
        self.canvas = tk.Canvas(flat_view, bg="gray25", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.register_view("main", "Main", flat_view, closable=False)

        self.canvas.bind("<Button-1>",        self.on_mouse_down)
        self.canvas.bind("<B1-Motion>",       self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>",          self.mouse_move)
        self.canvas.bind("<Button-2>",        self.start_pan)
        self.canvas.bind("<B2-Motion>",       self.pan)
        self.canvas.bind("<Button-3>",        self.on_mouse_down)
        self.canvas.bind("<B3-Motion>",       self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>",      self.on_mousewheel)  # Plain scroll for panning
        self.canvas.bind("<Control-MouseWheel>", self.zoom_mouse)  # Ctrl+scroll for zoom
        # Hotkeys
        self.canvas.bind("p", lambda e: self.set_tool("pan"))
        self.canvas.bind("s", self.select_selection_tool)
        self.canvas.bind("m", self.select_move_tool)
        self.canvas.bind("i", lambda e: self.set_tool("color picker"))
        self.canvas.bind("b", lambda e: self.set_tool("brush"))
        self.canvas.bind("e", lambda e: self.set_tool("eraser"))
        self.canvas.bind("c", lambda e: self.set_tool("clone"))
        self.canvas.bind("f", lambda e: self.set_tool("paint bucket"))
        self.canvas.bind("v", lambda e: self.set_tool("vector edit"))
        self.canvas.bind("l", lambda e: self.set_tool("line"))
        self.canvas.bind("r", lambda e: self.set_tool("rect"))
        self.canvas.bind("o", lambda e: self.set_tool("ellipse"))
        # Bind size keys at the window level so they keep working after a
        # settings entry or toolbar button has temporarily taken focus.
        self.root.bind("<KeyPress-plus>", lambda e: adjust_size(1))
        self.root.bind("<KeyPress-equal>", lambda e: adjust_size(1))
        self.root.bind("<Shift-KeyPress-equal>", lambda e: adjust_size(1))
        self.root.bind("<KeyPress-minus>", lambda e: adjust_size(-1))
        self.root.bind("<KeyPress-KP_Add>", lambda e: adjust_size(1))
        self.root.bind("<KeyPress-KP_Subtract>", lambda e: adjust_size(-1))
        self.canvas.bind("<Control-z>", lambda e: self.undo())
        self.canvas.bind("<Control-a>", self.select_all)
        self.canvas.bind("<Control-b>", self.zoom_to_selection)
        self.canvas.bind("<Control-s>", lambda e: self.save_project())
        self.canvas.bind("<Control-n>", lambda e: self.new_project())
        self.canvas.focus_set()
        self.update_tool_settings_visibility()
        self.switch_view("main")

        # ── Right panel: layers ───────────────────────────────────────────
        right = tk.Frame(main, width=250, bd=1, relief="sunken")
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Layers", font=("TkDefaultFont", 9, "bold")).pack(pady=(6, 2))

        self.layer_style = ttk.Style()
        self.layer_style.configure("Layer.Treeview", rowheight=32)
        self.layer_selected_vector_color = (
            self.layer_style.lookup(
                "Treeview", "background", ("selected",)) or "#4a6984")
        self.layer_selected_text_color = (
            self.layer_style.lookup(
                "Treeview", "foreground", ("selected",)) or "#ffffff")
        self.layer_selected_raster_color = "#c94f4f"
        self.layer_list = ttk.Treeview(
            right, show="tree", selectmode="browse", style="Layer.Treeview")
        self.layer_list.pack(fill="both", expand=True, padx=4)
        self.layer_list.column("#0", stretch=True, width=220)
        self.layer_list.tag_configure("raster", background="#f7dddd")
        self.layer_list.tag_configure("vector", background="#dcecff")
        self.layer_list.bind("<<TreeviewSelect>>", self.select_layer)
        self.layer_list.bind("<Button-1>",         self.on_layer_pointer_down)
        self.layer_list.bind("<Button-3>",         self.show_layer_properties)
        self.layer_list.bind("<B1-Motion>",        self.on_layer_drag)
        self.layer_list.bind("<ButtonRelease-1>",  self.on_layer_drag_end)

        self.layer_row_icons = {}
        row_sources = {}
        for name in ("layer-visible", "layer-hidden",
                     "layer-raster", "layer-vector"):
            with Image.open(icon_dir / f"{name}.png") as row_icon:
                row_sources[name] = row_icon.convert("RGBA").resize(
                    (24, 24), Image.Resampling.LANCZOS)
        for visible in (True, False):
            for layer_type in ("raster", "vector"):
                row_image = Image.new("RGBA", (58, 28), (0, 0, 0, 0))
                row_draw = ImageDraw.Draw(row_image)
                row_draw.rounded_rectangle(
                    (0, 0, 27, 27), radius=4,
                    fill=(232, 232, 232, 255), outline=(135, 135, 135, 255))
                visibility_name = "layer-visible" if visible else "layer-hidden"
                row_image.alpha_composite(row_sources[visibility_name], (2, 2))
                row_image.alpha_composite(row_sources[f"layer-{layer_type}"],
                                          (34, 2))
                self.layer_row_icons[(visible, layer_type)] = \
                    ImageTk.PhotoImage(row_image)

        layer_actions = [
            ("Add Raster Layer", "add-raster-layer.png",
             lambda: self.add_layer("raster")),
            ("Add Vector Layer", "add-vector-layer.png",
             lambda: self.add_layer("vector")),
            ("Delete Layer", "delete-layer.png", self.delete_layer),
            ("Toggle Visibility", "toggle-visibility.png",
             self.toggle_visibility),
            ("Move Layer Up", "move-layer-up.png", self.move_layer_up),
            ("Move Layer Down", "move-layer-down.png", self.move_layer_down),
        ]
        action_frame = tk.Frame(right)
        action_frame.pack(padx=4, pady=(4, 0))
        self.layer_action_icons = {}
        self.layer_action_hint = tk.StringVar(value="Layer Actions")
        for index, (label, filename, command) in enumerate(layer_actions):
            with Image.open(icon_dir / filename) as icon_image:
                icon_image = icon_image.convert("RGBA").resize(
                    (28, 28), Image.Resampling.LANCZOS)
            icon = ImageTk.PhotoImage(icon_image)
            self.layer_action_icons[filename] = icon
            button = tk.Button(
                action_frame, image=icon, width=34, height=34,
                command=command, takefocus=True)
            button.grid(row=0, column=index, padx=1, pady=1)
            button.bind(
                "<Enter>",
                lambda event, name=label: self.layer_action_hint.set(name))
            button.bind(
                "<Leave>",
                lambda event: self.layer_action_hint.set("Layer Actions"))
        tk.Label(right, textvariable=self.layer_action_hint).pack(pady=(1, 4))

    def request_redraw(self, defer=False):
        if hasattr(self, "active_view") and self.active_view != "main":
            self.main_view_dirty = True
            return

        if self.redraw_after_id is not None:
            return

        now = time.perf_counter()
        frame_time = 1 / 30.0 if defer else self.target_frame_time
        delay = max(0.0, frame_time - (now - self.last_redraw))
        if defer:
            delay = max(delay, 0.008)
        if delay == 0:
            self.last_redraw = now
            self.redraw()
        else:
            # Keep a trailing redraw.  Merely discarding requests received
            # inside the frame interval makes fast drags look jerky and can
            # leave the canvas one event behind the document.
            self.redraw_after_id = self.root.after(
                max(1, math.ceil(delay * 1000)), self._scheduled_redraw)

    def _scheduled_redraw(self):
        self.redraw_after_id = None
        self.last_redraw = time.perf_counter()
        self.redraw()

    def request_mipmap_level(self, level):
        """Build a missing zoom level after wheel input has settled.

        Constructing a large level in the wheel callback makes zooming hitch.
        Until this callback runs, compositing uses the nearest cached level.
        """
        if self.mipmap_future is not None:
            self.pending_mipmap_level = max(
                level, self.pending_mipmap_level or 0)
            return
        if (self.mipmap_after_id is not None and
                self.pending_mipmap_level == level):
            return
        if self.mipmap_after_id is not None:
            self.root.after_cancel(self.mipmap_after_id)
        self.pending_mipmap_level = level
        self.mipmap_after_id = self.root.after(140, self._build_pending_mipmaps)

    def _build_pending_mipmaps(self):
        level = self.pending_mipmap_level
        self.mipmap_after_id = None
        self.pending_mipmap_level = None
        if level is None:
            return
        jobs = [(layer, layer._mipmap_revision, layer.image)
                for layer in self.layers if layer.visible]

        def build_levels():
            results = []
            for layer, revision, base in jobs:
                pyramid = [base]
                while len(pyramid) <= level:
                    previous = pyramid[-1]
                    size = (max(1, (previous.width + 1) // 2),
                            max(1, (previous.height + 1) // 2))
                    pyramid.append(previous.resize(size, Image.Resampling.BOX))
                results.append((layer, revision, base, pyramid))
            return results

        self.mipmap_build_level = level
        self.mipmap_future = self.mipmap_executor.submit(build_levels)
        self.root.after(8, self._poll_mipmap_build)

    def _poll_mipmap_build(self):
        future = self.mipmap_future
        if future is None:
            return
        if not future.done():
            self.root.after(8, self._poll_mipmap_build)
            return
        self.mipmap_future = None
        built_level = self.mipmap_build_level
        self.mipmap_build_level = None
        try:
            results = future.result()
        except Exception:
            return
        for layer, revision, base, pyramid in results:
            if layer._mipmap_revision == revision and layer.image is base:
                layer._mipmaps = pyramid
        self.request_redraw(defer=True)
        if (self.pending_mipmap_level is not None and
                self.pending_mipmap_level > (built_level or 0)):
            pending = self.pending_mipmap_level
            self.pending_mipmap_level = None
            self.request_mipmap_level(pending)

    def choose_primary_color(self):
        self.active_color_slot = "primary"
        self._sync_picker_to_active_color()

    def choose_secondary_color(self):
        self.active_color_slot = "secondary"
        self._sync_picker_to_active_color()

    @staticmethod
    def _hex_to_rgb(color):
        color = color.lstrip("#")
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _add_color_slider(self, parent, label, variable, minimum, maximum,
                          callback):
        row = tk.Frame(parent)
        row.pack(fill="x")
        tk.Label(row, text=f"{label}:", width=2, anchor="w").pack(side="left")
        if label == "H":
            slider = tk.Canvas(row, width=92, height=14, bd=0,
                               highlightthickness=0, cursor="sb_h_double_arrow")
            slider.pack(side="left", fill="x", expand=True)
            slider.bind(
                "<Button-1>",
                lambda event, control=slider, value=variable, changed=callback:
                    self._set_hue_slider_from_pointer(
                        control, value, event, changed))
            slider.bind(
                "<B1-Motion>",
                lambda event, control=slider, value=variable, changed=callback:
                    self._set_hue_slider_from_pointer(
                        control, value, event, changed))
            variable.trace_add(
                "write", lambda *args, control=slider, value=variable:
                    self._render_hue_slider(control, value))
            slider.bind(
                "<Configure>",
                lambda event, control=slider, value=variable:
                    self._render_hue_slider(control, value))
            slider.after_idle(lambda: self._render_hue_slider(slider, variable))
        else:
            slider = tk.Scale(row, variable=variable, from_=minimum, to=maximum,
                              orient="horizontal", showvalue=False, length=92,
                              resolution=1, bd=0, highlightthickness=0,
                              sliderlength=6, width=8,
                              command=lambda value: callback())
            slider.pack(side="left", fill="x", expand=True)
            slider.bind(
                "<Button-1>",
                lambda event, control=slider, changed=callback:
                    self._set_slider_from_pointer(control, event, changed))
            slider.bind(
                "<B1-Motion>",
                lambda event, control=slider, changed=callback:
                    self._set_slider_from_pointer(control, event, changed))
        spinner = tk.Spinbox(row, textvariable=variable, from_=minimum,
                             to=maximum, width=4, justify="right",
                             command=callback)
        spinner.pack(side="right")
        spinner.bind("<Return>", lambda event: callback())
        spinner.bind("<FocusOut>", lambda event: callback())
        self.color_sliders.append(slider)

    def _render_hue_slider(self, slider, variable):
        width = max(2, slider.winfo_width())
        height = max(2, slider.winfo_height())
        image = Image.new("RGB", (width, height))
        pixels = image.load()
        for x in range(width):
            rgb = colorsys.hsv_to_rgb(x / (width - 1), 1, 1)
            color = tuple(round(channel * 255) for channel in rgb)
            for y in range(height):
                pixels[x, y] = color
        slider._hue_image = ImageTk.PhotoImage(image)
        slider.delete("all")
        slider.create_image(0, 0, image=slider._hue_image, anchor="nw")
        try:
            marker_x = int(variable.get()) / 359 * (width - 1)
        except (tk.TclError, ValueError):
            marker_x = 0
        slider.create_line(marker_x, 0, marker_x, height,
                           fill="white", width=3)
        slider.create_line(marker_x, 0, marker_x, height, fill="black")

    def _set_hue_slider_from_pointer(self, slider, variable, event, callback):
        width = max(2, slider.winfo_width())
        fraction = min(1.0, max(0.0, event.x / (width - 1)))
        variable.set(round(fraction * 359))
        callback()
        return "break"

    def _set_slider_from_pointer(self, slider, event, callback):
        """Snap a compact scale's handle directly beneath the pointer."""
        handle_width = float(slider.cget("sliderlength"))
        usable_width = max(1.0, slider.winfo_width() - handle_width)
        fraction = (event.x - handle_width / 2) / usable_width
        fraction = min(1.0, max(0.0, fraction))
        minimum = float(slider.cget("from"))
        maximum = float(slider.cget("to"))
        slider.set(round(minimum + fraction * (maximum - minimum)))
        callback()
        return "break"

    @staticmethod
    def _clamp_control(variable, minimum, maximum):
        try:
            value = int(variable.get())
        except (tk.TclError, ValueError):
            value = minimum
        value = min(maximum, max(minimum, value))
        variable.set(value)
        return value

    def _rgb_controls_changed(self):
        if self._color_controls_updating:
            return
        rgb = tuple(self._clamp_control(variable, 0, 255)
                    for variable in self.rgb_vars)
        self._set_selected_color(self._rgb_to_hex(rgb))

    def _hsv_controls_changed(self, component=None):
        if self._color_controls_updating:
            return
        hue = self.picker_hue
        saturation = self.picker_saturation
        value = self.picker_value
        if component in (None, 0):
            hue = self._clamp_control(self.hsv_vars[0], 0, 359) / 360
        if component in (None, 1):
            saturation = self._clamp_control(
                self.hsv_vars[1], 0, 100) / 100
        if component in (None, 2):
            value = self._clamp_control(self.hsv_vars[2], 0, 100) / 100
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        self.picker_hue = hue
        self.picker_saturation = saturation
        self.picker_value = value
        self._set_selected_color(
            self._rgb_to_hex(tuple(round(channel * 255) for channel in rgb)),
            preserve_hsv=True)

    def _hex_control_changed(self, event=None):
        if self._color_controls_updating:
            return
        value = self.hex_var.get().strip().lstrip("#")
        if len(value) == 3:
            value = "".join(character * 2 for character in value)
        try:
            if len(value) != 6:
                raise ValueError
            int(value, 16)
        except ValueError:
            self._sync_picker_to_active_color()
            return
        self._set_selected_color("#" + value.lower())

    def _opacity_control_changed(self):
        if self._color_controls_updating:
            return
        opacity = self._clamp_control(self.opacity_var, 0, 255)
        if self.active_color_slot == "primary":
            self.primary_opacity = opacity
        else:
            self.secondary_opacity = opacity
        self.request_redraw()

    def _set_selected_color(self, color, preserve_hsv=False):
        if self.active_color_slot == "primary":
            self.primary_color = color
            self.color = color
            self.primary_square.config(bg=color)
        else:
            self.secondary_color = color
            self.secondary_square.config(bg=color)
        self._sync_picker_to_active_color(
            preserve_hsv=preserve_hsv)
        self.request_redraw()

    def _color_with_opacity(self, slot):
        if slot == "primary":
            color, opacity = self.primary_color, self.primary_opacity
        else:
            color, opacity = self.secondary_color, self.secondary_opacity
        return color if opacity == 255 else f"{color}{opacity:02x}"

    def _sync_picker_to_active_color(self, preserve_hsv=False):
        color = (self.primary_color if self.active_color_slot == "primary"
                 else self.secondary_color)
        r, g, b = (channel / 255 for channel in self._hex_to_rgb(color))
        hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)
        # HSV controls have more precision than the stored 8-bit RGB color.
        # Preserve their exact state after an HSV edit instead of converting
        # the rounded RGB value back to HSV, which makes H and S slowly drift
        # while V is dragged back and forth.
        if not preserve_hsv:
            # Hue is undefined for grayscale colors, so retain the last useful
            # hue when syncing from RGB, hex, or a selected swatch.
            if saturation > 0:
                self.picker_hue = hue
            self.picker_saturation = saturation
            self.picker_value = value
        self.primary_square.config(
            highlightbackground="#2878d7" if self.active_color_slot == "primary" else "black")
        self.secondary_square.config(
            highlightbackground="#2878d7" if self.active_color_slot == "secondary" else "black")
        opacity = (self.primary_opacity if self.active_color_slot == "primary"
                   else self.secondary_opacity)
        self._color_controls_updating = True
        try:
            for variable, channel in zip(self.rgb_vars, self._hex_to_rgb(color)):
                variable.set(channel)
            self.hsv_vars[0].set(round(self.picker_hue * 360) % 360)
            self.hsv_vars[1].set(round(self.picker_saturation * 100))
            self.hsv_vars[2].set(round(self.picker_value * 100))
            self.opacity_var.set(opacity)
            self.hex_var.set(color.lstrip("#").upper())
        finally:
            self._color_controls_updating = False

    def swap_colors(self):
        self.primary_color, self.secondary_color = self.secondary_color, self.primary_color
        self.primary_opacity, self.secondary_opacity = (
            self.secondary_opacity, self.primary_opacity)
        self.color = self.primary_color
        self.primary_square.config(bg=self.primary_color)
        self.secondary_square.config(bg=self.secondary_color)
        self._sync_picker_to_active_color()
        self.request_redraw()

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.resizable(False, False)
        win.grab_set()  # modal

        tk.Label(win, text="UI Scale", font=("TkDefaultFont", 9, "bold")).pack(pady=(12, 2))
        tk.Label(win, text="Adjusts the size of all text and widgets.\nTakes effect immediately.",
                 justify="center").pack(padx=16)

        scale_var = tk.DoubleVar(value=self.ui_scale)
        slider = tk.Scale(win, variable=scale_var, from_=0.5, to=2.5,
                          resolution=0.05, orient="horizontal", length=260,
                          label="Scale factor")
        slider.pack(padx=16, pady=8)

        preview_label = tk.Label(win, text="1.00×")
        preview_label.pack()

        def on_change(val):
            preview_label.config(text=f"{float(val):.2f}×")

        slider.config(command=on_change)

        btn_row = tk.Frame(win)
        btn_row.pack(pady=(4, 12))

        def apply():
            self.apply_ui_scale(scale_var.get())

        def ok():
            apply()
            win.destroy()

        tk.Button(btn_row, text="Apply",  command=apply).pack(side="left", padx=4)
        tk.Button(btn_row, text="OK",     command=ok).pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="left", padx=4)

    def register_view(self, view_id, label, widget, closable=True):
        """Add an embedded workspace view and its compact tab."""
        self.views[view_id] = widget
        tab = tk.Frame(self.view_tabs)
        tk.Button(tab, text=label, bd=0,
                  command=lambda key=view_id: self.switch_view(key)).pack(side="left")
        if closable:
            tk.Button(tab, text="×", bd=0, padx=4,
                      command=lambda key=view_id: self.close_view(key)).pack(side="left")
        tab.pack(side="left", padx=2, pady=2)
        self.view_tab_widgets[view_id] = tab

    def switch_view(self, view_id):
        """Show one view without disturbing the shared tools or layers."""
        if view_id not in self.views:
            return
        if self.active_view in self.views:
            old_view = self.views[self.active_view]
            if hasattr(old_view, "on_hidden"):
                old_view.on_hidden()
            old_view.pack_forget()
        self.active_view = view_id
        new_view = self.views[view_id]
        new_view.pack(fill="both", expand=True)
        if hasattr(new_view, "on_shown"):
            new_view.on_shown()
        if view_id == "main":
            if self.main_view_dirty:
                self.main_view_dirty = False
                self.redraw()
            self.canvas.focus_set()

    def close_view(self, view_id):
        """Remove an optional view and return to the main canvas."""
        if view_id == "main" or view_id not in self.views:
            return
        widget = self.views.pop(view_id)
        tab = self.view_tab_widgets.pop(view_id)
        if self.active_view == view_id:
            self.active_view = None
            self.switch_view("main")
        tab.destroy()
        widget.destroy()
        if view_id == "globe":
            self.globe_window = None

    def apply_ui_scale(self, scale):
        self.ui_scale = scale
        # Compute an absolute font size from the scale (base size = 9pt)
        size = max(7, round(9 * scale))
        bold_size = max(7, round(9 * scale))

        # Update the named fonts tkinter uses by default
        import tkinter.font as tkfont
        for font_name in tkfont.names():
            try:
                f = tkfont.nametofont(font_name)
                # Scale relative to 9pt base; preserve sign (negative = pixels)
                f.configure(size=size)
            except Exception:
                pass

        # Force a geometry update so widgets reflow to their new sizes
        self.root.update_idletasks()

    def on_close(self):
        if self.undo_stack:
            if not messagebox.askyesno("Unsaved Changes",
                                       "You have unsaved changes. Quit anyway?"):
                return
        if self.selection_animation_id is not None:
            self.root.after_cancel(self.selection_animation_id)
            self.selection_animation_id = None
        self.mipmap_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def update_title(self):
        if self.current_file:
            self.root.title(f"PyPaint - {self.current_file}")
        else:
            self.root.title("PyPaint - Untitled")

    def notify_globe_document_changed(self):
        """Refresh an open globe view after document content changes."""
        globe = getattr(self, "globe_window", None)
        if globe is None:
            return
        try:
            if globe.winfo_exists():
                globe.notify_document_changed()
        except tk.TclError:
            self.globe_window = None

    def can_paint_from_globe(self):
        layer = self.layers[self.active_layer]
        return ((layer.is_raster and self.tool in ("brush", "eraser")) or
                (layer.layer_type == "vector" and self.tool in ("line", "rect", "ellipse")))

    def can_draw_vector_from_globe(self):
        return (self.layers[self.active_layer].layer_type == "vector" and
                self.tool in ("line", "rect", "ellipse"))

    def snapshot(self):
        snap = []
        for l in self.layers:
            n = Layer(self.doc_w, self.doc_h, l.name, l.layer_type)
            n.visible = l.visible
            n.opacity = l.opacity
            n.masked = l.masked
            n.image = l.image.copy()
            n.draw = ImageDraw.Draw(n.image)
            n.reset_mipmaps()
            if l.layer_type == "vector" and l.vector_data:
                n.vector_data = copy.deepcopy(l.vector_data)
            snap.append(n)
        self.undo_stack.append((snap, self.active_layer))
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def undo(self):
        self._finish_bucket_preview()
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        if not self.undo_stack:
            messagebox.showinfo("Undo", "Nothing to undo")
            return
        self.layers, self.active_layer = self.undo_stack.pop()
        # Recreate draw objects
        for l in self.layers:
            l.draw = ImageDraw.Draw(l.image)
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def new_project(self):
        if self.undo_stack:
            if not messagebox.askyesno("Unsaved Changes", 
                                       "You have unsaved changes. Create new project anyway?"):
                return
        self._finish_bucket_preview()
        
        self.layers = [Layer(self.doc_w, self.doc_h, "Background", "raster")]
        self.active_layer = 0
        self.current_file = None
        self.undo_stack = []
        self.selection_mask = Image.new("L", (self.doc_w, self.doc_h), 0)
        self._update_selection_geometry()
        self.clone_source_center = None
        self.clone_offset = None
        self.refresh_layers()
        self.redraw()
        self.update_title()
        self.notify_globe_document_changed()

    def save_project(self):
        self._finish_bucket_preview()
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_project_as()

    def save_project_as(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".pypaint",
            filetypes=[("PyPaint files", "*.pypaint"), ("All files", "*.*")]
        )
        if filename:
            self._save_to_file(filename)
            self.current_file = filename
            self.update_title()
            messagebox.showinfo("Success", f"Project saved to {filename}")

    def _save_to_file(self, filename):
        try:
            layer_data = []
            for layer in self.layers:
                # Convert image to bytes
                img_bytes = io.BytesIO()
                layer.image.save(img_bytes, format='PNG')
                img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                
                layer_info = {
                    'name': layer.name,
                    'visible': layer.visible,
                    'opacity': layer.opacity,
                    'masked': layer.masked,
                    'layer_type': layer.layer_type,
                    'image_data': img_base64,
                    'width': self.doc_w,
                    'height': self.doc_h
                }
                
                if layer.layer_type == "vector" and layer.vector_data:
                    layer_info['vector_data'] = layer.vector_data.to_dict()
                
                layer_data.append(layer_info)
            
            project_data = {
                'version': '2.0',
                'document_width': self.doc_w,
                'document_height': self.doc_h,
                'layers': layer_data,
                'active_layer': self.active_layer
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(project_data, f)
            
            self.undo_stack = []
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save project: {e}")

    def open_project(self):
        if self.undo_stack:
            if not messagebox.askyesno("Unsaved Changes", 
                                       "You have unsaved changes. Open project anyway?"):
                return
        
        filename = filedialog.askopenfilename(
            filetypes=[
                ("All files", "*.*"),
                ("PyPaint projects", "*.pypaint"),
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
            ]
        )
        if not filename:
            return
        
        try:
            if Path(filename).suffix.lower() == ".pypaint":
                self._open_pypaint_file(filename)
                messagebox.showinfo("Success", f"Project loaded from {filename}")
            else:
                self._open_image_file(filename)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")

    def _open_pypaint_file(self, filename):
        """Load a native, editable PyPaint project."""
        with open(filename, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        if 'version' not in project_data or 'layers' not in project_data:
            raise ValueError("Invalid project file format")

        loaded_layers = []

        for layer_info in project_data['layers']:
            img_bytes = base64.b64decode(layer_info['image_data'])
            with Image.open(io.BytesIO(img_bytes)) as source:
                img = source.convert("RGBA")

            saved_type = layer_info.get('layer_type', 'raster')
            # Mask used to be a dedicated raster layer type. Treat legacy
            # mask layers as ordinary raster layers with per-layer masking.
            layer_type = 'raster' if saved_type == 'mask' else saved_type
            layer = Layer(project_data['document_width'],
                          project_data['document_height'],
                          layer_info['name'], layer_type)
            layer.image = img
            layer.visible = layer_info['visible']
            layer.opacity = max(0, min(100, int(layer_info.get('opacity', 100))))
            layer.masked = bool(layer_info.get('masked', saved_type == 'mask'))
            layer.draw = ImageDraw.Draw(layer.image)
            layer.reset_mipmaps()

            if layer.layer_type == "vector" and 'vector_data' in layer_info:
                layer.vector_data = VectorLayer.from_dict(
                    layer_info['vector_data'],
                    project_data['document_width'],
                    project_data['document_height']
                )

            loaded_layers.append(layer)

        if not loaded_layers:
            raise ValueError("Project contains no layers")

        self.doc_w = project_data['document_width']
        self.doc_h = project_data['document_height']
        self.layers = loaded_layers
        self.active_layer = min(project_data.get('active_layer', 0),
                                len(self.layers) - 1)
        self.current_file = filename
        self._finish_open()

    def _open_image_file(self, filename):
        """Import an ordinary image as a new, unsaved raster document."""
        with Image.open(filename) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")

        self.doc_w, self.doc_h = image.size
        layer = Layer(self.doc_w, self.doc_h, Path(filename).stem, "raster")
        layer.image = image
        layer.draw = ImageDraw.Draw(layer.image)
        layer.reset_mipmaps()
        self.layers = [layer]
        self.active_layer = 0

        # An imported image is not a native project yet.  This ensures Save
        # opens Save As instead of replacing (for example) a PNG with JSON.
        self.current_file = None
        self._finish_open()

    def _finish_open(self):
        """Refresh shared UI state after either kind of file is opened."""
        self.undo_stack = []
        self.selection_mask = Image.new("L", (self.doc_w, self.doc_h), 0)
        self._update_selection_geometry()
        self.clone_source_center = None
        self.clone_offset = None
        self.refresh_layers()
        self.redraw()
        self.update_title()
        self.notify_globe_document_changed()

    def set_tool(self, tool):
        if (hasattr(self, "tools_by_layer_type") and self.layers and
                tool not in self.tools_by_layer_type[
                    self.layers[self.active_layer].layer_type]):
            return
        self._finish_raster_stroke()
        if tool != self.tool:
            self._finish_bucket_preview()
            self._finish_clone_stroke()
            self._finish_selection_move()
            self._finish_selection_boundary_move()
            self.selection_brush_last = None
            self.selection_brush_remove = False
        self.tool = tool
        if hasattr(self, "canvas"):
            cursor = ("fleur" if tool in ("pan", "move", "move selection")
                      else "crosshair")
            try:
                self.canvas.configure(cursor=cursor)
            except tk.TclError:
                self.canvas.configure(cursor="crosshair")
        if hasattr(self, "tool_buttons"):
            for name, button in self.tool_buttons.items():
                button.configure(relief="sunken" if name == tool else "raised")
            self.tool_hint_var.set(tool.title())
        self.update_tool_settings_visibility()
        # Reset vector drawing state
        self.vector_start_x = None
        self.vector_start_y = None
        self.current_vector_obj = None
        self.request_redraw()

    def select_move_tool(self, event=None):
        """Select Move, or Move Selection when Move is already active."""
        next_tool = "move selection" if self.tool == "move" else "move"
        self.set_tool(next_tool)

    def select_selection_tool(self, event=None):
        """Select Selection, or Brush Selection when it is already active."""
        next_tool = ("brush selection" if self.tool == "selection"
                     else "selection")
        self.set_tool(next_tool)

    def select_all(self, event=None):
        """Select every pixel in the document."""
        self._finish_bucket_preview()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        self.selection_brush_last = None
        self.selection_mask = Image.new(
            "L", (self.doc_w, self.doc_h), 255)
        self._update_selection_geometry()
        self._ensure_selection_animation()
        self.request_redraw()
        return "break"

    def zoom_to_selection(self, event=None):
        """Fit the selection, or the full document when empty, in the canvas."""
        bounds = self._selection_pixel_box()
        if bounds is None:
            bounds = (0, 0, self.doc_w, self.doc_h)
        left, top, right, bottom = bounds
        width = max(1, right - left)
        height = max(1, bottom - top)
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        margin = 20
        usable_width = max(1, canvas_width - margin * 2)
        usable_height = max(1, canvas_height - margin * 2)
        self.zoom = max(
            0.01, min(20, usable_width / width, usable_height / height))
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        self.offset_x = canvas_width / 2 - center_x * self.zoom
        self.offset_y = canvas_height / 2 - center_y * self.zoom
        self.request_redraw()
        return "break"

    def update_tools_for_active_layer(self):
        """Show and select only tools supported by the active layer type."""
        if not hasattr(self, "tool_buttons") or not self.layers:
            return

        layer_type = self.layers[self.active_layer].layer_type
        available_tools = self.tools_by_layer_type[layer_type]
        for name, button in self.tool_buttons.items():
            if name in available_tools:
                button.grid()
            else:
                button.grid_remove()

        if self.tool not in available_tools:
            self.set_tool(available_tools[0])
        else:
            self.update_tool_settings_visibility()

    def update_tool_settings_visibility(self):
        """Show settings that are meaningful for the currently selected tool."""
        if not hasattr(self, "size_frame"):
            return

        for frame in (self.size_frame, self.picker_settings_frame,
                      self.brush_settings_frame, self.clone_settings_frame,
                      self.bucket_settings_frame, self.vector_settings_frame):
            frame.pack_forget()

        uses_size = (self.tool not in
                     ("pan", "selection", "move", "move selection",
                      "paint bucket") and
                     (self.tool != "color picker" or
                      self.picker_sample_area_var.get()))
        if uses_size:
            self.size_frame.pack(side="left")

        if self.tool == "color picker":
            self.picker_settings_frame.pack(side="left")
        elif (self.tool in ("brush", "eraser") and self.layers and
              self.layers[self.active_layer].is_raster):
            self.brush_settings_frame.pack(side="left")
        elif (self.tool == "clone" and self.layers and
              self.layers[self.active_layer].is_raster):
            self.clone_settings_frame.pack(side="left")
        elif (self.tool == "paint bucket" and self.layers and
              self.layers[self.active_layer].is_raster):
            self.bucket_settings_frame.pack(side="left")
        elif self.tool in ("line", "rect", "ellipse"):
            self.vector_settings_frame.pack(side="left")
        self.request_redraw()

    def add_layer(self, layer_type="raster"):
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        self.snapshot()
        name = f"{layer_type.capitalize()} Layer {len([l for l in self.layers if l.layer_type == layer_type]) + 1}"
        self.layers.append(Layer(self.doc_w, self.doc_h, name, layer_type))
        self.active_layer = len(self.layers) - 1
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def delete_layer(self):
        if len(self.layers) == 1:
            return
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        self.snapshot()
        del self.layers[self.active_layer]
        self.active_layer = max(0, self.active_layer - 1)
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def toggle_visibility(self):
        self.layers[self.active_layer].visible = not self.layers[self.active_layer].visible
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def move_layer_up(self):
        if self.active_layer >= len(self.layers) - 1:
            return
        self.snapshot()
        self.layers[self.active_layer], self.layers[self.active_layer + 1] = \
            self.layers[self.active_layer + 1], self.layers[self.active_layer]
        self.active_layer += 1
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def move_layer_down(self):
        if self.active_layer <= 0:
            return
        self.snapshot()
        self.layers[self.active_layer], self.layers[self.active_layer - 1] = \
            self.layers[self.active_layer - 1], self.layers[self.active_layer]
        self.active_layer -= 1
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def select_layer(self, event=None):
        sel = self.layer_list.selection()
        if sel:
            display_index = self.layer_list.index(sel[0])
            selected_layer = len(self.layers) - 1 - display_index
            if selected_layer != self.active_layer:
                self._finish_clone_stroke()
                self._finish_selection_move()
                self._finish_selection_boundary_move()
            self.active_layer = selected_layer
            self.update_layer_selection_style()
            self.update_tools_for_active_layer()
            self.request_redraw()

    def update_layer_selection_style(self):
        """Match the selected-row color to the active layer's type."""
        layer_type = self.layers[self.active_layer].layer_type
        selected_color = {
            "raster": self.layer_selected_raster_color,
            "vector": self.layer_selected_vector_color,
        }[layer_type]
        self.layer_style.map(
            "Layer.Treeview",
            background=[("selected", selected_color)],
            foreground=[("selected", self.layer_selected_text_color)])

    def show_layer_properties(self, event):
        """Open the properties editor for the layer under the pointer."""
        row = self.layer_list.identify_row(event.y)
        if not row:
            return "break"

        display_index = self.layer_list.index(row)
        layer_index = len(self.layers) - 1 - display_index
        layer = self.layers[layer_index]
        original_name = layer.name
        original_opacity = layer.opacity
        original_masked = layer.masked
        self.active_layer = layer_index
        self.layer_list.selection_set(row)
        self.layer_list.focus(row)
        self.update_layer_selection_style()
        self.update_tools_for_active_layer()

        dialog = tk.Toplevel(self.root)
        dialog.title("Layer Properties")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        body = ttk.Frame(dialog, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="Name:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 10))
        name_var = tk.StringVar(value=layer.name)
        name_entry = ttk.Entry(body, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 10))

        masked_var = tk.BooleanVar(value=layer.masked)
        masked_check = ttk.Checkbutton(
            body, text="Masked by layers underneath", variable=masked_var)
        masked_check.grid(
            row=1, column=1, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(body, text="Opacity:").grid(
            row=2, column=0, sticky="w", padx=(0, 8))
        opacity_var = tk.IntVar(value=layer.opacity)
        opacity_scale = ttk.Scale(
            body, from_=0, to=100, orient="horizontal", length=190)
        opacity_scale.set(layer.opacity)
        opacity_scale.grid(row=2, column=1, sticky="ew")

        opacity_spinbox = ttk.Spinbox(
            body, from_=0, to=100, textvariable=opacity_var,
            width=5, justify="right")
        opacity_spinbox.grid(row=2, column=2, padx=(8, 0))
        ttk.Label(body, text="%").grid(row=2, column=3, sticky="w", padx=(3, 0))

        syncing = False
        preview_after_id = None

        def schedule_preview():
            """Coalesce rapid slider/key events into a modest refresh rate."""
            nonlocal preview_after_id
            if preview_after_id is not None:
                self.root.after_cancel(preview_after_id)
            preview_after_id = self.root.after(75, apply_preview)

        def apply_preview():
            nonlocal preview_after_id
            preview_after_id = None
            self.request_redraw()
            self.notify_globe_document_changed()

        def name_changed(*_args):
            layer.name = name_var.get()
            if layer.vector_data:
                layer.vector_data.name = layer.name
            self.refresh_layers()

        def scale_changed(value):
            nonlocal syncing
            if syncing:
                return
            value = round(float(value))
            syncing = True
            opacity_var.set(value)
            syncing = False
            layer.opacity = value
            schedule_preview()

        def number_changed(*_args):
            nonlocal syncing
            if syncing:
                return
            try:
                entered_value = int(opacity_var.get())
                value = max(0, min(100, entered_value))
            except (tk.TclError, ValueError):
                return
            syncing = True
            if entered_value != value:
                opacity_var.set(value)
            opacity_scale.set(value)
            syncing = False
            layer.opacity = value
            schedule_preview()

        opacity_scale.configure(command=scale_changed)
        opacity_var.trace_add("write", number_changed)
        name_var.trace_add("write", name_changed)

        def masked_changed():
            layer.masked = masked_var.get()
            self.refresh_layers()
            schedule_preview()

        masked_check.configure(command=masked_changed)

        def accept(_event=None):
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning(
                    "Layer Properties", "Layer name cannot be empty.", parent=dialog)
                name_entry.focus_set()
                return
            try:
                opacity = max(0, min(100, int(opacity_var.get())))
            except (tk.TclError, ValueError):
                messagebox.showwarning(
                    "Layer Properties", "Opacity must be a number from 0 to 100.",
                    parent=dialog)
                opacity_spinbox.focus_set()
                return

            # The controls have already previewed their values. Temporarily
            # restore the originals so Undo records the pre-dialog state.
            new_masked = masked_var.get()
            layer.name = original_name
            layer.opacity = original_opacity
            layer.masked = original_masked
            if layer.vector_data:
                layer.vector_data.name = original_name
            self.snapshot()
            layer.name = name
            layer.opacity = opacity
            layer.masked = new_masked
            if layer.vector_data:
                layer.vector_data.name = name
            self.refresh_layers()
            self.request_redraw()
            self.notify_globe_document_changed()
            dialog.destroy()

        def cancel(_event=None):
            nonlocal preview_after_id
            if preview_after_id is not None:
                self.root.after_cancel(preview_after_id)
                preview_after_id = None
            layer.name = original_name
            layer.opacity = original_opacity
            layer.masked = original_masked
            if layer.vector_data:
                layer.vector_data.name = original_name
            self.refresh_layers()
            self.request_redraw()
            self.notify_globe_document_changed()
            dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=3, column=0, columnspan=4, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="Cancel", command=cancel).pack(
            side="right", padx=(6, 0))
        ttk.Button(buttons, text="OK", command=accept).pack(side="right")

        dialog.bind("<Return>", accept)
        dialog.bind("<Escape>", cancel)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.update_idletasks()
        x = event.x_root - dialog.winfo_reqwidth()
        y = event.y_root
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        name_entry.focus_set()
        name_entry.selection_range(0, "end")
        dialog.grab_set()
        return "break"

    def refresh_layers(self):
        self.layer_list.delete(*self.layer_list.get_children())
        for i in range(len(self.layers) - 1, -1, -1):
            l = self.layers[i]
            self.layer_list.insert(
                "", "end", text=f"{l.name}{' [Masked]' if l.masked else ''}",
                image=self.layer_row_icons[(l.visible, l.layer_type)],
                tags=(l.layer_type,))

        self.update_layer_selection_style()
        self.update_tools_for_active_layer()
        
        display_index = len(self.layers) - 1 - self.active_layer
        rows = self.layer_list.get_children()
        if 0 <= display_index < len(rows):
            self.layer_list.selection_set(rows[display_index])
            self.layer_list.focus(rows[display_index])

    def on_layer_pointer_down(self, event):
        row = self.layer_list.identify_row(event.y)
        if not row:
            return
        display_index = self.layer_list.index(row)

        # The first icon in each row is a button-shaped visibility control.
        row_box = self.layer_list.bbox(row, "#0")
        # Treeview reserves a small indent before the row image; the button
        # occupies the first 28 pixels of that image.
        if row_box and event.x < row_box[0] + 50:
            layer_index = len(self.layers) - 1 - display_index
            self.layers[layer_index].visible = not self.layers[layer_index].visible
            self.refresh_layers()
            self.request_redraw()
            self.notify_globe_document_changed()
            return "break"

        self.layer_list.selection_set(row)
        self.layer_list.focus(row)
        self.active_layer = len(self.layers) - 1 - display_index
        self.update_layer_selection_style()
        self.update_tools_for_active_layer()
        self.drag_start_index = display_index
        self.drag_start_y = event.y
        self.snapshot()  # snapshot once at the start of the drag
        self.request_redraw()
        return "break"

    def on_layer_drag(self, event):
        if self.drag_start_index is None:
            return

        row = self.layer_list.identify_row(event.y)
        if not row:
            return
        current_index = self.layer_list.index(row)
        
        if current_index == self.drag_start_index:
            return
        
        from_idx = len(self.layers) - 1 - self.drag_start_index
        to_idx = len(self.layers) - 1 - current_index
        
        layer = self.layers.pop(from_idx)
        self.layers.insert(to_idx, layer)
        self.active_layer = to_idx
        self.drag_start_index = current_index
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def on_layer_drag_end(self, event):
        self.drag_start_index = None
        self.drag_start_y = None

    def image_coords(self, sx, sy):
        return ((sx - self.offset_x) / self.zoom,
                (sy - self.offset_y) / self.zoom)

    def raster_image_coords(self, sx, sy):
        """Map a screen point to Pillow's pixel-centre coordinate system."""
        x, y = self.image_coords(sx, sy)
        # image_coords() maps to pixel boundaries (pixel 0 spans 0..1), while
        # Pillow's raster primitives address pixel centres at integer values.
        # Without this conversion every dab is biased half a pixel right/down.
        return x - 0.5, y - 0.5

    def screen_coords(self, ix, iy):
        return (ix * self.zoom + self.offset_x,
                iy * self.zoom + self.offset_y)

    def on_mouse_down(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.last_button = event.num  # Track which button (1=left, 3=right)
        if self.tool == "pan":
            self.start_pan(event)
            return
        if self.tool == "color picker":
            self.pick_color(event)
            return
        x, y = self.image_coords(event.x, event.y)
        if self.tool == "selection":
            control_down = bool(event.state & 0x4)
            operation = ("subtract" if control_down and event.num == 3 else
                         "add" if control_down and event.num == 1 else
                         "replace" if event.num == 1 else None)
            if operation is not None:
                self.selection_start = (x, y)
                self.selection_bounds = (x, y, x, y)
                self.selection_operation = operation
                self.selection_base_mask = self.selection_mask.copy()
                self._ensure_selection_animation()
                self.request_redraw()
            return
        if self.tool == "move":
            if event.num == 1:
                self._start_selection_move(x, y)
            return
        if self.tool == "move selection":
            if event.num == 1:
                self._start_selection_boundary_move(x, y)
            return
        if self.tool == "brush selection":
            if event.num in (1, 3):
                self.selection_brush_remove = event.num == 3
                self.selection_brush_last = self.raster_image_coords(
                    event.x, event.y)
                self._paint_selection_brush(*self.selection_brush_last)
            return
        if self.tool == "clone":
            clone_x, clone_y = self.raster_image_coords(event.x, event.y)
            if event.state & 0x4:
                self.clone_source_center = (clone_x, clone_y)
                self.clone_offset = None
                self.request_redraw()
            elif event.num in (1, 3) and self.clone_source_center is not None:
                if self.clone_offset is None:
                    self.clone_offset = (
                        round(self.clone_source_center[0] - clone_x),
                        round(self.clone_source_center[1] - clone_y))
                self.snapshot()
                self.clone_stroke_source = \
                    self.layers[self.active_layer].image.copy()
                if self.clone_build_up_var.get():
                    self.clone_stroke_base = None
                    self.clone_stroke_coverage = None
                else:
                    self.clone_stroke_base = self.clone_stroke_source
                    self.clone_stroke_coverage = Image.new(
                        "L", (self.doc_w, self.doc_h), 0)
                self.clone_last = (clone_x, clone_y)
                self._paint_clone(clone_x, clone_y)
            return
        if self.tool == "paint bucket":
            if self.bucket_pending is not None:
                self._finish_bucket_preview()
                return
            self._begin_bucket_preview(
                math.floor(x), math.floor(y), event.num, event.serial)
            return
        current_layer = self.layers[self.active_layer]
        
        if current_layer.is_raster:
            self.start_raster_draw(event)
        else:  # vector layer
            self.start_vector_operation(event, x, y)

    def start_raster_draw(self, event):
        self.snapshot()
        self._prepare_raster_stroke()
        self.last_x, self.last_y = self.raster_image_coords(event.x, event.y)
        # Stamp the initial point immediately so a click/tap without any
        # motion produces a dot, just as it does in the globe view.
        self.raster_paint_image(self.last_x, self.last_y)

    def start_vector_operation(self, event, x, y):
        if self.tool == "vector edit":
            # Try to edit a vector object
            if self.layers[self.active_layer].vector_data:
                obj, point_idx = self.layers[self.active_layer].vector_data.get_object_at(x, y)
                if obj:
                    self.is_dragging_point = True
                    self.selected_vector_obj = obj
                    self.selected_point_index = point_idx
                    self.snapshot()
                else:
                    self.selected_vector_obj = None
                    self.selected_point_index = None
        elif self.tool in ["line", "rect", "ellipse"]:
            # Start drawing a new vector object
            self.vector_start_x = x
            self.vector_start_y = y
            self.snapshot()

    def on_mouse_move(self, event):
        # Tk dispatches B1/B3-Motion to this handler instead of mouse_move(),
        # so keep the brush outline position current during a stroke too.
        self.mouse_x = event.x
        self.mouse_y = event.y
        if self.tool == "pan":
            self.pan(event)
            return
        if self.tool == "color picker":
            self.pick_color(event)
            return
        x, y = self.image_coords(event.x, event.y)
        if self.tool == "selection":
            if self.selection_start is not None:
                start_x, start_y = self.selection_start
                self.selection_bounds = (start_x, start_y, x, y)
                self.request_redraw()
            return
        if self.tool == "move":
            self._update_selection_move(x, y)
            return
        if self.tool == "move selection":
            self._update_selection_boundary_move(x, y)
            return
        if self.tool == "brush selection":
            if self.selection_brush_last is not None:
                brush_x, brush_y = self.raster_image_coords(event.x, event.y)
                self._paint_selection_brush(brush_x, brush_y)
            return
        if self.tool == "clone":
            if self.clone_last is not None:
                clone_x, clone_y = self.raster_image_coords(event.x, event.y)
                self._paint_clone(clone_x, clone_y)
            return
        if self.tool == "paint bucket":
            return
        current_layer = self.layers[self.active_layer]
        
        if current_layer.is_raster:
            self.raster_paint(event)
        else:  # vector layer
            self.vector_operation(event, x, y)

    def on_mousewheel(self, event):
        """Handle plain scroll wheel for panning up/down and shift+scroll for left/right"""
        # Get scroll amount (cross-platform)
        if hasattr(event, 'delta'):
            delta = event.delta
        elif hasattr(event, 'num'):
            # Linux mouse wheel
            if event.num == 4:
                delta = 120  # scroll up
            elif event.num == 5:
                delta = -120 # scroll down
            else:
                return
        else:
            return
        
        pan_amount = delta * 0.5
        
        if event.state & 0x1:  # Shift key is pressed
            # Pan left/right
            self.offset_x += pan_amount
        else:
            # Pan up/down
            self.offset_y += pan_amount
        
        # Wheel input arrives in dense bursts (especially from touchpads and
        # high-resolution wheels).  Rendering synchronously for the first
        # event in each burst blocks Tk from consuming the remaining deltas,
        # which makes both vertical and Shift+horizontal panning trail behind.
        # The offsets above still accumulate every event; only the expensive
        # canvas refresh is coalesced to the most recent position.
        self.request_redraw(defer=True)

    def begin_external_raster_draw(self, x, y, button=1):
        """
        Begin a brush stroke from an external input source
        (such as the globe window).
        """
        self.snapshot()
        self._prepare_raster_stroke()
        self.last_x = x
        self.last_y = y
        self.last_button = button


    def end_external_raster_draw(self):
        """
        Finish an externally-driven brush stroke.
        """
        self.last_x = None
        self.last_y = None
        self._finish_raster_stroke()

    def _prepare_raster_stroke(self):
        """Capture the state needed to cap opacity within one brush gesture."""
        self._stroke_base_image = None
        self._stroke_coverage = None
        if (self.tool in ("brush", "eraser") and
                not self.brush_build_up_var.get() and self.undo_stack):
            snapshot_layers, snapshot_active = self.undo_stack[-1]
            if snapshot_active == self.active_layer:
                self._stroke_base_image = snapshot_layers[snapshot_active].image
                self._stroke_coverage = Image.new(
                    "L", (self.doc_w, self.doc_h), 0)

    def _finish_raster_stroke(self):
        self._stroke_base_image = None
        self._stroke_coverage = None

    def brush_hardness(self):
        """Return the brush edge hardness as a validated percentage."""
        try:
            return max(0, min(100, int(self.brush_hardness_var.get())))
        except (tk.TclError, ValueError):
            return 75

    def brush_spacing(self):
        """Return brush stamp spacing as a percentage of its diameter."""
        try:
            return max(0.1, min(1000, float(self.brush_spacing_var.get())))
        except (tk.TclError, ValueError):
            return 12.5

    def clone_hardness(self):
        try:
            return max(0, min(100, int(self.clone_hardness_var.get())))
        except (tk.TclError, ValueError):
            return 75

    def clone_spacing(self):
        try:
            return max(0.1, min(1000, float(self.clone_spacing_var.get())))
        except (tk.TclError, ValueError):
            return 12.5

    def bucket_hardness(self):
        try:
            return max(0, min(100, int(self.bucket_hardness_var.get())))
        except (tk.TclError, ValueError):
            return 75

    def bucket_tolerance(self):
        try:
            return max(0, min(100, int(self.bucket_tolerance_var.get())))
        except (tk.TclError, ValueError):
            return 0

    def _reset_bucket_hardness(self):
        self.bucket_hardness_var.set(75)
        self._refresh_bucket_preview()

    def _reset_bucket_tolerance(self):
        self.bucket_tolerance_var.set(0)
        self._refresh_bucket_preview()

    def _begin_bucket_preview(self, pixel_x, pixel_y, button, event_serial=None):
        """Start an undoable bucket preview from an untouched source image."""
        if not (0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h):
            return
        layer = self.layers[self.active_layer]
        if not layer.is_raster:
            return
        self.snapshot()
        self.bucket_pending = {
            "layer_index": self.active_layer,
            "original": layer.image.copy(),
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "button": button,
            "event_serial": event_serial,
        }
        self._refresh_bucket_preview()

    def _refresh_bucket_preview(self):
        """Recalculate a pending fill after one of its settings changes."""
        if self.bucket_pending is None:
            return
        pending = self.bucket_pending
        layer = self.layers[pending["layer_index"]]
        layer.image.paste(pending["original"])
        layer.reset_mipmaps()
        pixel_x, pixel_y = pending["pixel_x"], pending["pixel_y"]

        pixels = np.asarray(pending["original"])
        target = pixels[pixel_y, pixel_x]
        tolerance = self.bucket_tolerance()
        if tolerance == 0:
            matches = np.all(pixels == target, axis=2)
        elif tolerance == 100:
            matches = np.ones((self.doc_h, self.doc_w), dtype=bool)
        else:
            differences = (pixels.astype(np.int32) -
                           target.astype(np.int32))
            distance_squared = np.sum(
                differences * differences, axis=2)
            maximum_distance = math.sqrt(4 * 255 * 255)
            threshold = maximum_distance * tolerance / 100
            matches = distance_squared <= threshold * threshold
        flood_source = Image.fromarray(
            np.where(matches, 0, 255).astype(np.uint8)).copy()
        ImageDraw.floodfill(flood_source, (pixel_x, pixel_y), 128)
        fill_mask = flood_source.point(
            lambda value: 255 if value == 128 else 0)

        if self.bucket_antialias_var.get():
            fill_mask = fill_mask.filter(ImageFilter.GaussianBlur(0.65))
            fill_mask = _apply_hardness_to_alpha(
                fill_mask, self.bucket_hardness(), 2)
        fill_mask = self._clip_raster_mask_to_selection(
            (0, 0, self.doc_w, self.doc_h), fill_mask)
        dirty_box = fill_mask.getbbox()
        if dirty_box is None:
            self.request_redraw()
            return

        local_mask = fill_mask.crop(dirty_box)
        color = self._color_with_opacity(
            "primary" if pending["button"] == 1 else "secondary")
        source = Image.new("RGBA", local_mask.size, color)
        source.putalpha(ImageChops.multiply(
            source.getchannel("A"), local_mask))
        result = layer.image.crop(dirty_box)
        result.alpha_composite(source)
        self.apply_raster_result(layer, result, dirty_box)
        layer.update_mipmaps(dirty_box)
        self.request_redraw()

    def _finish_bucket_preview(self, event=None):
        """Commit the currently displayed bucket preview."""
        if self.bucket_pending is None:
            return
        self.bucket_pending = None
        self.notify_globe_document_changed()
        return "break"

    def _stamp_clone(self, x, y):
        """Copy one source-aligned circular sample to the destination."""
        if self.clone_stroke_source is None or self.clone_offset is None:
            return None
        radius = max(0.5, int(self.size_var.get()) / 2)
        raster_radius = max(0, radius - 0.5)
        antialias = self.clone_antialias_var.get()
        if raster_radius == 0 and not antialias:
            pixel_x, pixel_y = round(x), round(y)
            bounds = (pixel_x, pixel_y, pixel_x, pixel_y)
            paint_mask = lambda draw, left, top, scale: draw.rectangle(
                ((pixel_x - left) * scale, (pixel_y - top) * scale,
                 (pixel_x - left + 1) * scale - 1,
                 (pixel_y - top + 1) * scale - 1), fill=255)
        else:
            effective_radius = radius if raster_radius == 0 else raster_radius
            bounds = (x - effective_radius, y - effective_radius,
                      x + effective_radius, y + effective_radius)
            paint_mask = lambda draw, left, top, scale: draw.ellipse(
                _brush_ellipse_box(bounds, left, top, scale),
                fill=255, outline=255)
        box, dab_mask = _brush_shape_mask(
            self.layers[self.active_layer].image, bounds, paint_mask,
            antialias=antialias,
            hardness=self.clone_hardness())
        if box is None:
            return None
        dab_mask = self._clip_raster_mask_to_selection(box, dab_mask)
        opacity = (self.primary_opacity if self.last_button == 1
                   else self.secondary_opacity)
        if opacity < 255:
            dab_mask = dab_mask.point(
                lambda value: (value * opacity + 127) // 255)

        if self.clone_stroke_coverage is not None:
            coverage = ImageChops.lighter(
                self.clone_stroke_coverage.crop(box), dab_mask)
            self.clone_stroke_coverage.paste(
                coverage, (box[0], box[1]))
            composite_mask = coverage
        else:
            composite_mask = dab_mask

        offset_x, offset_y = self.clone_offset
        source_left, source_top = box[0] + offset_x, box[1] + offset_y
        source_right = source_left + dab_mask.width
        source_bottom = source_top + dab_mask.height
        clipped_left = max(0, source_left)
        clipped_top = max(0, source_top)
        clipped_right = min(self.doc_w, source_right)
        clipped_bottom = min(self.doc_h, source_bottom)
        source = Image.new("RGBA", dab_mask.size, (0, 0, 0, 0))
        if clipped_right > clipped_left and clipped_bottom > clipped_top:
            source.paste(
                self.clone_stroke_source.crop(
                    (clipped_left, clipped_top, clipped_right, clipped_bottom)),
                (clipped_left - source_left, clipped_top - source_top))
        source.putalpha(ImageChops.multiply(
            source.getchannel("A"), composite_mask))
        result_base = (self.clone_stroke_base
                       if self.clone_stroke_base is not None
                       else self.layers[self.active_layer].image)
        result = result_base.crop(box)
        result.alpha_composite(source)
        self.apply_raster_result(self.layers[self.active_layer], result, box)
        return box

    def _paint_clone(self, x, y):
        """Interpolate clone stamps using the configured brush spacing."""
        last_x, last_y = self.clone_last or (x, y)
        dx, dy = x - last_x, y - last_y
        radius = max(0.5, int(self.size_var.get()) / 2)
        spacing = max(1, radius * 2 * self.clone_spacing() / 100)
        steps = max(1, math.ceil(math.hypot(dx, dy) / spacing))
        dirty_boxes = []
        for index in range(steps + 1):
            amount = index / steps
            dirty = self._stamp_clone(
                last_x + dx * amount, last_y + dy * amount)
            if dirty is not None:
                dirty_boxes.append(dirty)
        self.clone_last = (x, y)
        if dirty_boxes:
            dirty = (min(box[0] for box in dirty_boxes),
                     min(box[1] for box in dirty_boxes),
                     max(box[2] for box in dirty_boxes),
                     max(box[3] for box in dirty_boxes))
            self.layers[self.active_layer].update_mipmaps(dirty)
        self.request_redraw()

    def _finish_clone_stroke(self):
        if self.clone_stroke_source is None:
            return
        self.clone_last = None
        self.clone_stroke_source = None
        self.clone_stroke_base = None
        self.clone_stroke_coverage = None
        self.notify_globe_document_changed()

    def _ensure_selection_animation(self):
        """Start the marquee timer if it is not already running."""
        if self.selection_animation_id is None:
            self.selection_animation_id = self.root.after(
                70, self._animate_selection_marquee)

    def _start_selection_move(self, x, y):
        """Capture the selected raster pixels for an interactive move."""
        if self.selection_bounds is None:
            return
        if not self._point_in_selection(x, y):
            return

        # A released move remains floating. Starting another drag reuses the
        # original pixels and cleared base, preserving one continuous edit.
        if self.move_pixels is not None:
            self.move_start = (x, y)
            self.move_drag_origin_offset = self.move_offset
            return

        box = self._selection_pixel_box()
        if box is None:
            return
        box = (max(0, box[0]), max(0, box[1]),
               min(self.doc_w, box[2]), min(self.doc_h, box[3]))
        if box[2] <= box[0] or box[3] <= box[1]:
            return

        layer = self.layers[self.active_layer]
        if not layer.is_raster:
            return
        self.snapshot()
        self.move_start = (x, y)
        self.move_source_box = box
        self.move_pixels = layer.image.crop(box)
        self.move_mask = self.selection_mask.crop(box)
        moved_alpha = ImageChops.multiply(
            self.move_pixels.getchannel("A"), self.move_mask)
        self.move_pixels.putalpha(moved_alpha)
        self.move_base_image = layer.image.copy()
        self.move_base_image.paste((0, 0, 0, 0), box, self.move_mask)
        self.move_selection_bounds = self.selection_bounds
        self.move_offset = (0, 0)
        self.move_drag_origin_offset = (0, 0)

    def _update_selection_move(self, x, y):
        """Preview selected pixels at the current drag position."""
        if self.move_start is None:
            return
        source_left, source_top, source_right, source_bottom = \
            self.move_source_box
        dx = (self.move_drag_origin_offset[0] +
              round(x - self.move_start[0]))
        dy = (self.move_drag_origin_offset[1] +
              round(y - self.move_start[1]))
        dx = max(-source_left, min(self.doc_w - source_right, dx))
        dy = max(-source_top, min(self.doc_h - source_bottom, dy))

        layer = self.layers[self.active_layer]
        layer.image.paste(self.move_base_image)
        layer.image.alpha_composite(
            self.move_pixels, (source_left + dx, source_top + dy))
        layer.reset_mipmaps()

        self.selection_mask = Image.new("L", (self.doc_w, self.doc_h), 0)
        self.selection_mask.paste(
            self.move_mask, (source_left + dx, source_top + dy))
        self._update_selection_geometry()
        self.move_offset = (dx, dy)
        self.request_redraw()

    def _release_selection_move(self):
        """End one drag while keeping the selected pixels floating."""
        if self.move_pixels is not None:
            self.move_start = None
            self.move_drag_origin_offset = self.move_offset

    def _finish_selection_move(self):
        """Finish an interactive move and release its temporary images."""
        if self.move_pixels is None:
            return
        moved = self.move_offset != (0, 0)
        self.move_start = None
        self.move_source_box = None
        self.move_pixels = None
        self.move_mask = None
        self.move_base_image = None
        self.move_selection_bounds = None
        self.move_offset = (0, 0)
        self.move_drag_origin_offset = (0, 0)
        if not moved and self.undo_stack:
            # Avoid adding an undo step for a click without movement.
            self.undo_stack.pop()
        if moved:
            self.notify_globe_document_changed()

    def _start_selection_boundary_move(self, x, y):
        """Begin moving only the selection marquee, leaving pixels untouched."""
        if self.selection_bounds is None:
            return
        if self._point_in_selection(x, y):
            self.selection_move_start = (x, y)
            self.selection_move_bounds = self.selection_bounds
            self.selection_move_mask = self.selection_mask.copy()

    def _update_selection_boundary_move(self, x, y):
        """Move the selection rectangle within the document bounds."""
        if self.selection_move_start is None:
            return
        left, top, right, bottom = self.selection_move_bounds
        dx = round(x - self.selection_move_start[0])
        dy = round(y - self.selection_move_start[1])
        dx = max(-left, min(self.doc_w - right, dx))
        dy = max(-top, min(self.doc_h - bottom, dy))
        self.selection_mask = Image.new("L", (self.doc_w, self.doc_h), 0)
        self.selection_mask.paste(
            self.selection_move_mask, (round(dx), round(dy)))
        self._update_selection_geometry()
        self.request_redraw()

    def _finish_selection_boundary_move(self):
        """Finish moving the marquee without changing document pixels."""
        self.selection_move_start = None
        self.selection_move_bounds = None
        self.selection_move_mask = None

    def _paint_selection_brush(self, x, y):
        """Add an interpolated circular brush stroke to the selection mask."""
        try:
            radius = max(0.5, min(999, int(self.size_var.get())) / 2)
        except ValueError:
            radius = 1
        last_x, last_y = self.selection_brush_last or (x, y)
        dx, dy = x - last_x, y - last_y
        distance = math.hypot(dx, dy)
        steps = max(1, math.ceil(distance / max(1, radius * 0.25)))

        for index in range(steps + 1):
            amount = index / steps
            center_x = last_x + dx * amount
            center_y = last_y + dy * amount
            raster_radius = max(0, radius - 0.5)
            bounds = (center_x - raster_radius,
                      center_y - raster_radius,
                      center_x + raster_radius,
                      center_y + raster_radius)
            box, dab = _brush_shape_mask(
                self.selection_mask, bounds,
                lambda draw, left, top, scale, shape=bounds: draw.ellipse(
                    _brush_ellipse_box(shape, left, top, scale),
                    fill=255, outline=255),
                antialias=False)
            if box is not None:
                existing = self.selection_mask.crop(box)
                if self.selection_brush_remove:
                    combined = ImageChops.multiply(
                        existing, ImageOps.invert(dab))
                else:
                    combined = ImageChops.lighter(existing, dab)
                self.selection_mask.paste(combined, (box[0], box[1]))

        self.selection_brush_last = (x, y)
        self._update_selection_geometry()
        self._ensure_selection_animation()
        self.request_redraw()

    def _animate_selection_marquee(self):
        """Advance the selection dashes without rerendering the document."""
        self.selection_animation_id = None
        if self.selection_bounds is None:
            return
        self.selection_dash_offset = (self.selection_dash_offset + 1) % 10
        try:
            # Recreate only the lightweight outline items so the animation is
            # independent of Tk's platform-specific dashed-line repainting.
            self.canvas.delete("selection_marquee")
            self._draw_selection_marquee()
        except tk.TclError:
            return
        self._ensure_selection_animation()

    def _draw_selection_marquee(self):
        """Draw the active selection outline over the current raster view."""
        if (self.selection_bounds is None or not self.layers or
                not self.layers[self.active_layer].is_raster):
            return
        tags = ("overlay", "selection_marquee")
        edges = list(self.selection_edges)
        if self.selection_start is not None:
            left, top, right, bottom = self.selection_bounds
            edges.extend(((left, top, right, top),
                          (right, top, right, bottom),
                          (right, bottom, left, bottom),
                          (left, bottom, left, top)))

        # Draw the bright portions as actual moving segments rather than a
        # Tk dash pattern. Windows Tk can cache dashed rectangles and ignore
        # dashoffset changes, whereas changing line coordinates always paints.
        dash_length = 6
        period = 10

        def draw_moving_edge(start_x, start_y, end_x, end_y):
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.hypot(dx, dy)
            if length <= 0:
                return
            unit_x, unit_y = dx / length, dy / length
            distance = self.selection_dash_offset - period
            while distance < length:
                segment_start = max(0, distance)
                segment_end = min(length, distance + dash_length)
                if segment_end > segment_start:
                    self.canvas.create_line(
                        start_x + unit_x * segment_start,
                        start_y + unit_y * segment_start,
                        start_x + unit_x * segment_end,
                        start_y + unit_y * segment_end,
                        fill="white", width=1, tags=tags)
                distance += period

        for left, top, right, bottom in edges:
            x0, y0 = self.screen_coords(left, top)
            x1, y1 = self.screen_coords(right, bottom)
            self.canvas.create_line(
                x0, y0, x1, y1, fill="black", width=3, tags=tags)
            draw_moving_edge(x0, y0, x1, y1)

    def _pixel_box_from_bounds(self, bounds):
        """Convert continuous document bounds to selected pixel boundaries."""
        left, top, right, bottom = bounds
        # A pixel belongs to the selection when its center lies within the
        # marquee. This keeps clipping aligned with the displayed boundary at
        # every zoom level, including selections made at fractional positions.
        return (math.ceil(left - 0.5), math.ceil(top - 0.5),
                math.floor(right - 0.5) + 1,
                math.floor(bottom - 0.5) + 1)

    def _update_selection_geometry(self):
        """Cache the exact outline segments for the current selection mask."""
        self.selection_bounds = self.selection_mask.getbbox()
        self.selection_edges = []
        if self.selection_bounds is None:
            return
        left, top, right, bottom = self.selection_bounds
        selected = np.asarray(
            self.selection_mask.crop(self.selection_bounds), dtype=np.uint8) > 0
        height, width = selected.shape

        def add_runs(values, make_segment):
            start = None
            for index, value in enumerate(np.append(values, False)):
                if value and start is None:
                    start = index
                elif not value and start is not None:
                    self.selection_edges.append(make_segment(start, index))
                    start = None

        for row in range(height + 1):
            above = selected[row - 1] if row > 0 else np.zeros(width, bool)
            below = selected[row] if row < height else np.zeros(width, bool)
            add_runs(
                above != below,
                lambda start, end, y=top + row:
                    (left + start, y, left + end, y))
        for column in range(width + 1):
            before = (selected[:, column - 1] if column > 0
                      else np.zeros(height, bool))
            after = (selected[:, column] if column < width
                     else np.zeros(height, bool))
            add_runs(
                before != after,
                lambda start, end, x=left + column:
                    (x, top + start, x, top + end))

    def _selection_pixel_box(self):
        """Return the active selection's exact nonempty pixel extent."""
        if self.selection_mask.size != (self.doc_w, self.doc_h):
            self.selection_mask = Image.new("L", (self.doc_w, self.doc_h), 0)
        return self.selection_mask.getbbox()

    def _point_in_selection(self, x, y):
        """Return whether a document point lies in an actually selected pixel."""
        pixel_x, pixel_y = math.floor(x), math.floor(y)
        return (0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h and
                self.selection_mask.getpixel((pixel_x, pixel_y)) > 0)

    def _selection_mask_for_box(self, box):
        """Build a selection mask local to a document-space patch box."""
        mask = Image.new("L", (box[2] - box[0], box[3] - box[1]), 0)
        selection_box = self._selection_pixel_box()
        if selection_box is None:
            return Image.new("L", mask.size, 255)
        return self.selection_mask.crop(box)

    def apply_raster_result(self, layer, result, box=None, mask=None):
        """Apply a raster tool's result through the active selection.

        This is the common write boundary for raster tools and image-wide
        functions. ``result`` may be a full document image or a patch matching
        ``box``. An optional tool mask is combined with the selection mask.
        New raster operations should use this method instead of writing or
        pasting directly into ``layer.image``.
        """
        if box is None:
            box = (0, 0, self.doc_w, self.doc_h)
        expected_size = (box[2] - box[0], box[3] - box[1])
        if result.size != expected_size:
            raise ValueError("Raster result size must match its destination box")

        write_mask = self._selection_mask_for_box(box)
        if mask is not None:
            if mask.size != expected_size:
                raise ValueError("Raster edit mask must match its destination box")
            write_mask = ImageChops.multiply(write_mask, mask.convert("L"))
        layer.image.paste(result, (box[0], box[1]), write_mask)
        return write_mask.getbbox() is not None

    def _clip_raster_mask_to_selection(self, box, mask):
        """Restrict a raster tool mask to the active selection."""
        return ImageChops.multiply(mask, self._selection_mask_for_box(box))

    def _paint_brush_shape(self, layer, bounds, color, paint_mask):
        """Paint one dab, optionally capping coverage for the current stroke."""
        antialias = self.brush_antialias_var.get()
        box, dab_mask = _brush_shape_mask(
            layer.image, bounds, paint_mask, antialias=antialias,
            hardness=self.brush_hardness())
        if box is None:
            return None
        dab_mask = self._clip_raster_mask_to_selection(box, dab_mask)

        if self._stroke_base_image is None or self._stroke_coverage is None:
            source = Image.new("RGBA", dab_mask.size, color)
            source.putalpha(ImageChops.multiply(
                source.getchannel("A"), dab_mask))
            result = layer.image.crop(box)
            result.alpha_composite(source)
            self.apply_raster_result(layer, result, box)
            return box

        coverage = self._stroke_coverage.crop(box)
        coverage = ImageChops.lighter(coverage, dab_mask)
        self._stroke_coverage.paste(coverage, (box[0], box[1]))

        result = self._stroke_base_image.crop(box)
        source = Image.new("RGBA", coverage.size, color)
        source.putalpha(ImageChops.multiply(source.getchannel("A"), coverage))
        result.alpha_composite(source)
        self.apply_raster_result(layer, result, box)
        return box

    def _erase_brush_shape(self, layer, bounds, paint_mask):
        """Erase through a hard or anti-aliased mask with stroke buildup rules."""
        box, dab_mask = _brush_shape_mask(
            layer.image, bounds, paint_mask,
            antialias=self.brush_antialias_var.get(),
            hardness=self.brush_hardness())
        if box is None:
            return None
        dab_mask = self._clip_raster_mask_to_selection(box, dab_mask)

        # Eraser strength follows the opacity of the color slot associated
        # with the mouse button, just like Brush chooses its paint opacity.
        eraser_opacity = (self.primary_opacity if self.last_button == 1
                          else self.secondary_opacity)
        if eraser_opacity < 255:
            dab_mask = dab_mask.point(
                lambda value: (value * eraser_opacity + 127) // 255)

        if self._stroke_base_image is not None and self._stroke_coverage is not None:
            coverage = self._stroke_coverage.crop(box)
            coverage = ImageChops.lighter(coverage, dab_mask)
            self._stroke_coverage.paste(coverage, (box[0], box[1]))
            result = self._stroke_base_image.crop(box)
            source_alpha = result.getchannel("A")
            result.putalpha(ImageChops.multiply(
                source_alpha, ImageOps.invert(coverage)))
        else:
            result = layer.image.crop(box)
            source_alpha = result.getchannel("A")
            result.putalpha(ImageChops.multiply(
                source_alpha, ImageOps.invert(dab_mask)))

        self.apply_raster_result(layer, result, box)
        return box


    def raster_paint_image(self, x, y):
        """
        Paint using image coordinates instead of a Tk mouse event.
        """

        if self.last_x is None or self.last_y is None:
            self.last_x = x
            self.last_y = y

        # The entry is validated on Return/focus-out, but painting can begin
        # while it temporarily contains 0 during editing.  Keep every dab at
        # least one pixel wide so Pillow never receives an inverted ellipse.
        radius = max(0.5, int(self.size_var.get()) / 2)

        if self.tool == "eraser":
            color = (0, 0, 0, 0)
        else:
            color = self._color_with_opacity(
                "primary" if self.last_button == 1 else "secondary")

        dx = x - self.last_x
        dy = y - self.last_y

        dist = math.hypot(dx, dy)

        spacing = max(1, radius * 2 * self.brush_spacing() / 100)
        # Round upward so the distance between adjacent stamps never exceeds
        # the requested spacing. Flooring this value could leave one- or
        # two-pixel holes in thin strokes between mouse-motion events.
        steps = max(1, math.ceil(dist / spacing))

        for i in range(steps + 1):

            t = i / steps

            px = self.last_x + dx * t
            py = self.last_y + dy * t

            self.draw_circle(
                px,
                py,
                radius,
                color
            )

        self.last_x = x
        self.last_y = y

        self.request_redraw()
        self.notify_globe_document_changed()

    def stamp_external_raster(self, x, y, refresh=True):
        """Stamp one globe brush sample, wrapping it at the map seam.

        Globe strokes can contain several samples for one mouse event.  Callers
        can defer the display refresh until the whole group has been stamped.
        """
        if not self.can_paint_from_globe():
            return

        radius = max(0.5, int(self.size_var.get()) / 2)
        color = ((0, 0, 0, 0) if self.tool == "eraser" else
                 self._color_with_opacity(
                     "primary" if self.last_button == 1 else "secondary"))
        self.draw_circle(x, y, radius, color)
        self.draw_circle(x - self.doc_w, y, radius, color)
        self.draw_circle(x + self.doc_w, y, radius, color)
        self.last_x, self.last_y = x, y
        if refresh:
            self.request_redraw()
            self.notify_globe_document_changed()

    def stamp_external_spherical_raster(self, footprint_uv, center_x, center_y,
                                        refresh=True):
        """Fill a globe-relative brush footprint on the equirectangular map.

        ``footprint_uv`` is the spherical brush boundary expressed in texture
        coordinates.  Its U values may extend beyond the map edges so the same
        shape can be drawn cleanly across the longitude seam.
        """
        if not self.can_paint_from_globe() or not footprint_uv:
            return

        color = ((0, 0, 0, 0) if self.tool == "eraser" else
                 self._color_with_opacity(
                     "primary" if self.last_button == 1 else "secondary"))
        polygon = [(u * self.doc_w, v * self.doc_h) for u, v in footprint_uv]

        # Repeat the unwrapped polygon on both sides of the texture.  PIL clips
        # each copy to the image, preserving a brush that straddles the seam.
        layer = self.layers[self.active_layer]
        for offset in (-self.doc_w, 0, self.doc_w):
            shifted = [(x + offset, y) for x, y in polygon]
            xs = [point[0] for point in shifted]
            ys = [point[1] for point in shifted]
            bounds = (min(xs), min(ys), max(xs), max(ys))
            if self.tool == "eraser":
                dirty_box = self._erase_brush_shape(
                    layer,
                    bounds,
                    lambda draw, left, top, scale, points=shifted: draw.polygon(
                        [((x - left + 0.5) * scale,
                          (y - top + 0.5) * scale)
                         for x, y in points], fill=255),
                )
            else:
                dirty_box = self._paint_brush_shape(
                    layer,
                    bounds,
                    color,
                    lambda draw, left, top, scale, points=shifted: draw.polygon(
                        [((x - left + 0.5) * scale,
                          (y - top + 0.5) * scale)
                         for x, y in points], fill=255),
                )
            if dirty_box is not None:
                layer.update_mipmaps(dirty_box)

        self.last_x, self.last_y = center_x, center_y
        if refresh:
            self.request_redraw()
            self.notify_globe_document_changed()

    def raster_paint(self, event):

        x, y = self.raster_image_coords(
            event.x,
            event.y
        )

        self.raster_paint_image(
            x,
            y
        )

    def open_globe_view(self):
        if "globe" in self.views:
            self.switch_view("globe")
            return

        if self.doc_w != self.doc_h * 2:
            should_continue = messagebox.askyesno(
                "Globe View Aspect Ratio",
                "Globe View is designed for 2:1 equirectangular documents.\n\n"
                f"This document is {self.doc_w} x {self.doc_h}. Continue anyway?",
            )
            if not should_continue:
                return

        from globe_view import GlobeView

        self.globe_window = GlobeView(self.view_host, self)
        self.register_view("globe", "Globe", self.globe_window)
        self.switch_view("globe")

    def draw_circle(self, x, y, radius, color):
        layer = self.layers[self.active_layer]
        # Also enforce the invariant here for callers that supply a radius
        # directly instead of reading the validated size control.
        radius = max(0.5, radius)
        # Pillow includes both ends of an ellipse's bounding box. Reduce the
        # raster radius by half a pixel so a requested diameter of 1 paints
        # one pixel (and diameter N spans N pixels), rather than N + 1.
        raster_radius = max(0, radius - 0.5)
        if raster_radius == 0:
            # A 1 px anti-aliased brush is still a geometric disc centered at
            # the pointer's fractional document coordinate.  Snapping it to a
            # point first would throw away the subpixel position and give the
            # main pixel the same alpha everywhere along a stroke.
            if self.brush_antialias_var.get():
                bounds = (x - radius, y - radius,
                          x + radius, y + radius)
                paint_mask = lambda draw, left, top, scale: draw.ellipse(
                    _brush_ellipse_box(bounds, left, top, scale),
                    fill=255, outline=255)
                if self.tool == "eraser":
                    dirty_box = self._erase_brush_shape(
                        layer, bounds, paint_mask)
                else:
                    dirty_box = self._paint_brush_shape(
                        layer, bounds, color, paint_mask)
                if dirty_box is not None:
                    layer.update_mipmaps(dirty_box)
                return

            px, py = round(x), round(y)
            if self.tool == "eraser":
                dirty_box = self._erase_brush_shape(
                    layer,
                    (px, py, px, py),
                    lambda draw, left, top, scale: draw.rectangle(
                        ((px - left) * scale, (py - top) * scale,
                         (px - left + 1) * scale - 1,
                         (py - top + 1) * scale - 1), fill=255),
                )
            else:
                dirty_box = self._paint_brush_shape(
                    layer,
                    (px, py, px, py),
                    color,
                    lambda draw, left, top, scale: draw.rectangle(
                        ((px - left) * scale, (py - top) * scale,
                         (px - left + 1) * scale - 1,
                         (py - top + 1) * scale - 1), fill=255),
                )
            if dirty_box is not None:
                layer.update_mipmaps(dirty_box)
            return
        bounds = (x - raster_radius, y - raster_radius,
                  x + raster_radius, y + raster_radius)
        if self.tool == "eraser":
            dirty_box = self._erase_brush_shape(
                layer,
                bounds,
                lambda draw, left, top, scale: draw.ellipse(
                    _brush_ellipse_box(bounds, left, top, scale),
                    fill=255, outline=255),
            )
        else:
            dirty_box = self._paint_brush_shape(
                layer,
                bounds,
                color,
                lambda draw, left, top, scale: draw.ellipse(
                    _brush_ellipse_box(bounds, left, top, scale),
                    fill=255, outline=255),
            )
        if dirty_box is not None:
            layer.update_mipmaps(dirty_box)

    def vector_operation(self, event, x, y):
        if self.is_dragging_point and self.selected_vector_obj:
            # Update the point position
            self.selected_vector_obj.update_point(self.selected_point_index, x, y)
            self.layers[self.active_layer].render_vector()
            self.request_redraw()
            self.notify_globe_document_changed()
        elif self.tool in ["line", "rect", "ellipse"] and self.vector_start_x is not None:
            # Preview the shape (by redrawing)
            self.layers[self.active_layer].render_vector()
            self.request_redraw()
            # Draw temporary preview
            self.draw_vector_preview(self.vector_start_x, self.vector_start_y, x, y)

    def draw_vector_preview(self, x1, y1, x2, y2):
        """Draw a temporary preview of the vector object being created"""
        layer = self.layers[self.active_layer]
        
        preview_img = layer.image.copy()
        preview_object = self.make_vector_object(
            self.tool, [(x1, y1), (x2, y2)], "flat")
        if preview_object:
            render_vector_object(
                preview_img, preview_object, self.doc_w, self.doc_h)

        # Preview the active vector layer in its normal place in the complete
        # layer stack.  Displaying preview_img directly hides every raster
        # layer for the duration of the drag.
        preview_composite = Image.new(
            "RGBA", (self.doc_w, self.doc_h), (0, 0, 0, 0))
        underlying_alpha = Image.new("L", (self.doc_w, self.doc_h), 0)
        for candidate in self.layers:
            if not candidate.visible:
                continue
            candidate_image = preview_img if candidate is layer else candidate.image
            rendered = candidate.image_with_opacity(candidate_image)
            rendered = self._cap_masked_layer(
                candidate, rendered, underlying_alpha)
            preview_composite.alpha_composite(rendered)
            underlying_alpha = ImageChops.lighter(
                underlying_alpha, rendered.getchannel("A"))

        self.display_image(preview_composite)

    def create_vector_object(self, x1, y1, x2, y2):
        """Create a vector object and add it to the current layer"""
        layer = self.layers[self.active_layer]
        if layer.layer_type != "vector":
            return
        
        obj = self.make_vector_object(
            self.tool, [(x1, y1), (x2, y2)], "flat")
        
        if obj:
            layer.vector_data.add_object(obj)

    def vector_line_width(self):
        """Return the shared Size control as a valid vector stroke width."""
        try:
            return max(1, min(999, int(self.size_var.get())))
        except (tk.TclError, ValueError):
            return 2

    def vector_antialias_enabled(self):
        """Capture the vector Anti-alias toggle for a new vector object."""
        try:
            return bool(self.vector_antialias_var.get())
        except (tk.TclError, AttributeError):
            return True

    def vector_hardness(self):
        """Capture the vector edge hardness for a new vector object."""
        try:
            return max(0, min(100, int(self.vector_hardness_var.get())))
        except (tk.TclError, ValueError):
            return 75

    def make_vector_object(self, preset, points, space="flat"):
        """Build the same vector object for previews and finalized gestures."""
        if len(points) < 2:
            return None
        if preset == "line":
            return Line(*points[0], *points[-1],
                        self._color_with_opacity("primary"),
                        self.vector_line_width(), space=space,
                        antialias=self.vector_antialias_enabled(),
                        hardness=self.vector_hardness())
        if preset in ("rect", "ellipse"):
            return self.make_shape_preset(preset, points, space)
        return None

    def make_shape_preset(self, preset, points, space="flat"):
        """Turn a UI shape preset into lines; presets are never special render objects."""
        color = self._color_with_opacity("primary")
        width = self.vector_line_width()
        antialias = self.vector_antialias_enabled()
        hardness = self.vector_hardness()
        if preset == "rect":
            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                vertices = points[:4]
            lines = [Line(*vertices[i], *vertices[(i + 1) % 4], color, width,
                          space=space, antialias=antialias,
                          hardness=hardness)
                     for i in range(4)]
        else:
            if space == "flat":
                (x1, y1), (x2, y2) = points
                cx, cy = (x1+x2)/2, (y1+y2)/2
                rx, ry = abs(x2-x1)/2, abs(y2-y1)/2
                k = 0.5522847498
                verts = [(cx+rx,cy),(cx,cy+ry),(cx-rx,cy),(cx,cy-ry)]
                controls = [(cx+rx,cy+k*ry,cx+k*rx,cy+ry),
                            (cx-k*rx,cy+ry,cx-rx,cy+k*ry),
                            (cx-rx,cy-k*ry,cx-k*rx,cy-ry),
                            (cx+k*rx,cy-ry,cx+rx,cy-k*ry)]
                lines = [Line(*verts[i], *verts[(i+1)%4], color, width,
                              curve=controls[i], space=space,
                              antialias=antialias, hardness=hardness)
                         for i in range(4)]
            else:
                vertices = points
                lines = [Line(*vertices[i], *vertices[(i+1)%len(vertices)], color,
                              width, space=space, antialias=antialias,
                              hardness=hardness)
                         for i in range(len(vertices))]
        # The secondary colour represents the shape's filled (inside) side.
        return Shape(lines, color, width,
                     self._color_with_opacity("secondary"), "inside", preset,
                     antialias, hardness)

    def create_globe_vector(self, preset, image_points):
        if not self.can_draw_vector_from_globe() or len(image_points) < 2:
            return
        self.snapshot()
        obj = self.make_vector_object(preset, image_points, "globe")
        if obj is None:
            return
        self.layers[self.active_layer].vector_data.add_object(obj)
        self.layers[self.active_layer].render_vector()
        self.request_redraw()
        self.notify_globe_document_changed()

    def on_mouse_up(self, event):
        if self.tool in ("pan", "color picker"):
            return
        x, y = self.image_coords(event.x, event.y)
        if self.tool == "selection":
            if self.selection_start is not None:
                start_x, start_y = self.selection_start
                left, right = sorted((start_x, x))
                top, bottom = sorted((start_y, y))
                left = max(0, min(self.doc_w, left))
                right = max(0, min(self.doc_w, right))
                top = max(0, min(self.doc_h, top))
                bottom = max(0, min(self.doc_h, bottom))
                if right - left >= 1 and bottom - top >= 1:
                    rectangle = Image.new(
                        "L", (self.doc_w, self.doc_h), 0)
                    pixel_box = self._pixel_box_from_bounds(
                        (left, top, right, bottom))
                    if pixel_box[2] > pixel_box[0] and pixel_box[3] > pixel_box[1]:
                        ImageDraw.Draw(rectangle).rectangle(
                            (pixel_box[0], pixel_box[1],
                             pixel_box[2] - 1, pixel_box[3] - 1), fill=255)
                    if self.selection_operation == "add":
                        self.selection_mask = ImageChops.lighter(
                            self.selection_base_mask, rectangle)
                    elif self.selection_operation == "subtract":
                        self.selection_mask = ImageChops.multiply(
                            self.selection_base_mask, ImageOps.invert(rectangle))
                    else:
                        self.selection_mask = rectangle
                elif self.selection_operation == "replace":
                    self.selection_mask = Image.new(
                        "L", (self.doc_w, self.doc_h), 0)
                self._update_selection_geometry()
                self.selection_start = None
                self.selection_operation = None
                self.selection_base_mask = None
                self.request_redraw()
            return
        if self.tool == "move":
            self._update_selection_move(x, y)
            self._release_selection_move()
            return
        if self.tool == "move selection":
            self._update_selection_boundary_move(x, y)
            self._finish_selection_boundary_move()
            return
        if self.tool == "brush selection":
            self.selection_brush_last = None
            self.selection_brush_remove = False
            return
        if self.tool == "clone":
            self._finish_clone_stroke()
            return
        if self.tool == "paint bucket":
            return
        current_layer = self.layers[self.active_layer]
        
        if current_layer.is_raster:
            self.last_x = None
            self.last_y = None
            self._finish_raster_stroke()
        else:  # vector layer
            if self.is_dragging_point:
                self.is_dragging_point = False
                self.selected_vector_obj = None
                self.selected_point_index = None
            elif self.tool in ["line", "rect", "ellipse"] and self.vector_start_x is not None:
                # Only create if there's a significant size
                if abs(x - self.vector_start_x) > 2 or abs(y - self.vector_start_y) > 2:
                    self.create_vector_object(self.vector_start_x, self.vector_start_y, x, y)
                self.vector_start_x = None
                self.vector_start_y = None
                self.layers[self.active_layer].render_vector()
                self.request_redraw()
                self.notify_globe_document_changed()

    def start_pan(self, event):
        self.pan_x = event.x
        self.pan_y = event.y

    def pan(self, event):
        self.offset_x += event.x - self.pan_x
        self.offset_y += event.y - self.pan_y
        self.pan_x = event.x
        self.pan_y = event.y
        self.request_redraw()

    def pick_color(self, event):
        """Sample one document pixel into the left or right color slot."""
        image_x, image_y = self.image_coords(event.x, event.y)
        pixel_x, pixel_y = math.floor(image_x), math.floor(image_y)
        if not (0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h):
            return

        composite = bool(event.state & 0x4)
        if composite:  # Ctrl: sample the final visible composite.
            for layer in self.layers:
                if layer.visible and layer.layer_type == "vector" and layer.vector_data:
                    layer.render_vector()
        else:
            layer = self.layers[self.active_layer]
            if layer.layer_type == "vector" and layer.vector_data:
                layer.render_vector()

        if self.picker_sample_area_var.get():
            diameter = max(1, int(self.size_var.get()))
            radius = diameter / 2
            center_x, center_y = pixel_x + 0.5, pixel_y + 0.5
            left = max(0, math.floor(center_x - radius))
            top = max(0, math.floor(center_y - radius))
            right = min(self.doc_w, math.ceil(center_x + radius))
            bottom = min(self.doc_h, math.ceil(center_y + radius))
            box = (left, top, right, bottom)
            source = (self.composite_region(box) if composite
                      else layer.image.crop(box))
            totals = [0, 0, 0, 0]
            count = 0
            radius_squared = radius * radius
            for y in range(top, bottom):
                for x in range(left, right):
                    if ((x + 0.5 - center_x) ** 2 +
                            (y + 0.5 - center_y) ** 2 > radius_squared):
                        continue
                    sample = source.getpixel((x - left, y - top))
                    for channel in range(4):
                        totals[channel] += sample[channel]
                    count += 1
            rgba = tuple(round(total / count) for total in totals)
        elif composite:
            rgba = self.composite_region(
                (pixel_x, pixel_y, pixel_x + 1, pixel_y + 1)).getpixel((0, 0))
        else:
            rgba = layer.image.getpixel((pixel_x, pixel_y))

        self.active_color_slot = (
            "secondary" if self.last_button == 3 else "primary")
        if self.active_color_slot == "primary":
            self.primary_opacity = rgba[3]
        else:
            self.secondary_opacity = rgba[3]
        self._set_selected_color(self._rgb_to_hex(rgba[:3]))

    def zoom_mouse(self, event):
        old = self.zoom
        self.zoom *= 1.1 if event.delta > 0 else (1 / 1.1)
        self.zoom = max(0.1, min(20, self.zoom))

        # Do not rebuild an identical frame when the wheel keeps moving after
        # reaching either zoom limit.
        if self.zoom == old:
            return

        ix = (event.x - self.offset_x) / old
        iy = (event.y - self.offset_y) / old

        self.offset_x = event.x - ix * self.zoom
        self.offset_y = event.y - iy * self.zoom
        # Wheel events arrive in bursts.  Always schedule their redraw so the
        # input queue can coalesce several events before expensive Tk upload.
        self.request_redraw(defer=True)

    def mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.request_redraw()

    def save_image(self):
        name = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files", "*.png"),
                                                       ("JPEG files", "*.jpg"),
                                                       ("All files", "*.*")])
        if not name:
            return
        self.composite_image().save(name)

    def composite_image(self):
        result = Image.new("RGBA", (self.doc_w, self.doc_h), self.bg_color)
        underlying_alpha = Image.new("L", (self.doc_w, self.doc_h), 0)
        for layer in self.layers:
            if layer.visible:
                if layer.layer_type == "vector" and layer.vector_data:
                    layer.render_vector()
                rendered = layer.image_with_opacity()
                rendered = self._cap_masked_layer(
                    layer, rendered, underlying_alpha)
                result.alpha_composite(rendered)
                underlying_alpha = ImageChops.lighter(
                    underlying_alpha, rendered.getchannel("A"))
        return result

    @staticmethod
    def _cap_masked_layer(layer, rendered, underlying_alpha):
        """Cap one masked layer by the strongest visible layer below it."""
        if not layer.masked:
            return rendered
        capped = rendered.copy()
        capped.putalpha(ImageChops.darker(
            rendered.getchannel("A"), underlying_alpha))
        return capped

    def composite_region(self, box, output_size=None):
        """Composite only a document-space rectangle for interactive display.

        A full 8192 x 4096 RGBA composite is 128 MiB.  Building it before
        cropping to a roughly window-sized viewport made every mouse event do
        work proportional to document size instead of screen size.
        """
        left, top, right, bottom = box
        source_size = (right - left, bottom - top)
        size = output_size or source_size

        # Pick the nearest pyramid level that still has at least one source
        # pixel per output pixel.  At 12.5% zoom, for example, display reads a
        # cached 1/8-size layer instead of all 33 million original pixels.
        reduction = max(source_size[0] / size[0], source_size[1] / size[1])
        desired_level = (max(0, int(math.floor(math.log2(reduction))))
                         if reduction > 1 else 0)
        max_level = int(math.floor(math.log2(max(1, min(self.doc_w, self.doc_h)))))
        desired_level = min(desired_level, max_level)

        available_level = desired_level
        for layer in self.layers:
            if layer.visible:
                if not layer._mipmaps or layer._mipmaps[0] is not layer.image:
                    layer.reset_mipmaps()
                available_level = min(available_level, len(layer._mipmaps) - 1)
        if available_level < desired_level:
            self.request_mipmap_level(desired_level)
        level = available_level
        factor = 1 << level

        # Preserve document transparency here.  The Main view adds its
        # checkerboard after compositing the layers; beginning with an opaque
        # white image would permanently cover that backdrop.
        result = Image.new("RGBA", size, (0, 0, 0, 0))
        underlying_alpha = Image.new("L", size, 0)
        for layer in self.layers:
            if layer.visible:
                source = layer.get_mipmap(level)
                extent = (left / factor, top / factor,
                          right / factor, bottom / factor)
                rendered = source.transform(
                    size, Image.Transform.EXTENT, extent,
                    resample=Image.Resampling.NEAREST)
                rendered = layer.image_with_opacity(rendered)
                rendered = self._cap_masked_layer(
                    layer, rendered, underlying_alpha)
                result.alpha_composite(rendered)
                underlying_alpha = ImageChops.lighter(
                    underlying_alpha, rendered.getchannel("A"))
        return result

    def get_checker_backdrop_pil(self, cw, ch):
        """Build (and cache) the full-viewport repeating checkerboard PIL
        image. This is the expensive part (tiling) and only happens when the
        canvas viewport size changes - never on pan/zoom."""
        if self._checker_pil is not None and self._checker_pil_dims == (cw, ch):
            return self._checker_pil

        s = self.checker_size
        tile = Image.new("RGBA", (s * 2, s * 2), self.checker_light)
        tdraw = ImageDraw.Draw(tile)
        tdraw.rectangle((s, 0, s * 2, s), fill=self.checker_dark)
        tdraw.rectangle((0, s, s, s * 2), fill=self.checker_dark)

        backdrop = Image.new("RGBA", (cw, ch))
        for y in range(0, ch, s * 2):
            for x in range(0, cw, s * 2):
                backdrop.paste(tile, (x, y))

        self._checker_pil = backdrop
        self._checker_pil_dims = (cw, ch)
        return backdrop
    def display_image(self, img):
        """Display an image on the canvas"""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        self.canvas.delete("all")
        self._canvas_image_id = None

        left = max(0, (-self.offset_x) / self.zoom)
        top = max(0, (-self.offset_y) / self.zoom)
        right = min(self.doc_w, (cw - self.offset_x) / self.zoom)
        bottom = min(self.doc_h, (ch - self.offset_y) / self.zoom)
        
        if right <= left or bottom <= top:
            return

        # Static checkerboard backdrop - only behind the document's on-screen
        # footprint (so the canvas's own blank-area color still shows through
        # everywhere else), and it never pans/zooms with the content.
        # Crop on document-pixel boundaries and position that same boundary
        # through the screen transform.  Mixing a truncated source crop with
        # the fractional visible bounds shifts painted pixels away from the
        # pointer after panning or at fractional zoom levels.
        crop_left = max(0, math.floor(left))
        crop_top = max(0, math.floor(top))
        crop_right = min(self.doc_w, math.ceil(right))
        crop_bottom = min(self.doc_h, math.ceil(bottom))
        sx = self.offset_x + crop_left * self.zoom
        sy = self.offset_y + crop_top * self.zoom
        doc_sx = math.floor(sx)
        doc_sy = math.floor(sy)
        crop = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        sw = max(1, round((crop_right - crop_left) * self.zoom))
        sh = max(1, round((crop_bottom - crop_top) * self.zoom))
        crop = crop.resize((sw, sh), Image.Resampling.NEAREST)

        # Keep the backdrop and document in one tracked canvas item.  The old
        # preview path created a separate, untracked checkerboard item which
        # survived the next redraw and appeared as a second locked pattern.
        checker = self.get_checker_backdrop_pil(cw, ch).crop(
            (doc_sx, doc_sy, doc_sx + sw, doc_sy + sh))
        checker.alpha_composite(crop)
        self.tkimg = ImageTk.PhotoImage(checker.convert("RGB"))
        
        self._canvas_image_id = self.canvas.create_image(
            sx, sy, image=self.tkimg, anchor="nw")

    def redraw(self):
        # Keep inactive views lazy.  In particular, globe painting used to
        # render this hidden canvas as well as the visible globe every frame.
        if hasattr(self, "active_view") and self.active_view != "main":
            self.main_view_dirty = True
            return
        
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        left = max(0, (-self.offset_x) / self.zoom)
        top = max(0, (-self.offset_y) / self.zoom)
        right = min(self.doc_w, (cw - self.offset_x) / self.zoom)
        bottom = min(self.doc_h, (ch - self.offset_y) / self.zoom)

        self.canvas.delete("overlay")

        if right <= left or bottom <= top:
            if self._canvas_image_id is not None:
                self.canvas.itemconfigure(self._canvas_image_id, state="hidden")
            return

        # Static checkerboard backdrop - only behind the document's on-screen
        # footprint, leaving the canvas's own blank-area color visible
        # elsewhere. Never pans/zooms with the content.
        crop_left = max(0, math.floor(left))
        crop_top = max(0, math.floor(top))
        crop_right = min(self.doc_w, math.ceil(right))
        crop_bottom = min(self.doc_h, math.ceil(bottom))
        sx = self.offset_x + crop_left * self.zoom
        sy = self.offset_y + crop_top * self.zoom
        doc_sx = math.floor(sx)
        doc_sy = math.floor(sy)
        sw = max(1, round((crop_right - crop_left) * self.zoom))
        sh = max(1, round((crop_bottom - crop_top) * self.zoom))
        crop_box = (crop_left, crop_top, crop_right, crop_bottom)
        crop = self.composite_region(crop_box, (sw, sh))

        checker = self.get_checker_backdrop_pil(cw, ch).crop(
            (doc_sx, doc_sy, doc_sx + sw, doc_sy + sh))
        checker.alpha_composite(crop)
        self.tkimg = ImageTk.PhotoImage(checker.convert("RGB"))

        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(
                sx, sy, image=self.tkimg, anchor="nw")
        else:
            self.canvas.coords(self._canvas_image_id, sx, sy)
            self.canvas.itemconfigure(self._canvas_image_id,
                                      image=self.tkimg, state="normal")

        # Draw brush cursor for raster layers
        current_layer = self.layers[self.active_layer]
        if self.tool == "color picker":
            if self.picker_sample_area_var.get():
                image_x, image_y = self.image_coords(self.mouse_x, self.mouse_y)
                pixel_x, pixel_y = math.floor(image_x), math.floor(image_y)
                if 0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h:
                    diameter = max(1, round(int(self.size_var.get()) * self.zoom))
                    if diameter != self._brush_cursor_diameter:
                        cursor_image = self._brush_cursor_source.resize(
                            (diameter, diameter), Image.Resampling.LANCZOS)
                        self._brush_cursor_tkimg = ImageTk.PhotoImage(cursor_image)
                        self._brush_cursor_diameter = diameter
                    center_x, center_y = self.screen_coords(
                        pixel_x + 0.5, pixel_y + 0.5)
                    self.canvas.create_image(
                        center_x, center_y, image=self._brush_cursor_tkimg,
                        anchor="center", tags=("overlay",))
            else:
                image_x, image_y = self.image_coords(self.mouse_x, self.mouse_y)
                pixel_x, pixel_y = math.floor(image_x), math.floor(image_y)
                if 0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h:
                    x0, y0 = self.screen_coords(pixel_x, pixel_y)
                    x1, y1 = self.screen_coords(pixel_x + 1, pixel_y + 1)
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, outline="black", width=3,
                        tags=("overlay",))
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, outline="white", width=1,
                        tags=("overlay",))

        if (current_layer.is_raster and
                self.tool in ("brush", "eraser", "brush selection", "clone") and
                0 <= self.mouse_x < cw and 0 <= self.mouse_y < ch):
            diameter = max(1, round(int(self.size_var.get()) * self.zoom))
            if diameter != self._brush_cursor_diameter:
                cursor_image = self._brush_cursor_source.resize(
                    (diameter, diameter), Image.Resampling.LANCZOS)
                self._brush_cursor_tkimg = ImageTk.PhotoImage(cursor_image)
                self._brush_cursor_diameter = diameter
            self.canvas.create_image(
                self.mouse_x, self.mouse_y,
                image=self._brush_cursor_tkimg,
                anchor="center", tags=("overlay",))

        if self.tool == "clone" and self.clone_source_center is not None:
            if self.clone_offset is None:
                source_x, source_y = self.clone_source_center
            else:
                hover_x, hover_y = self.raster_image_coords(
                    self.mouse_x, self.mouse_y)
                source_x = hover_x + self.clone_offset[0]
                source_y = hover_y + self.clone_offset[1]
            center_x, center_y = self.screen_coords(source_x + 0.5,
                                                    source_y + 0.5)
            radius = max(1, int(self.size_var.get()) * self.zoom / 2)
            self.canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                outline="#00ff80", width=2, tags=("overlay",))
            self.canvas.create_line(
                center_x - 5, center_y, center_x + 5, center_y,
                fill="#00ff80", width=1, tags=("overlay",))
            self.canvas.create_line(
                center_x, center_y - 5, center_x, center_y + 5,
                fill="#00ff80", width=1, tags=("overlay",))
        
        # Draw vector handles if in proper mode and on vector layer
        if self.tool == "vector edit" and current_layer.layer_type == "vector" and current_layer.vector_data:
            for obj in current_layer.vector_data.objects:
                points = obj.get_points()
                for px, py in points:
                    sx, sy = self.screen_coords(px, py)
                    self.canvas.create_rectangle(sx - 3, sy - 3, sx + 3, sy + 3,
                                               outline="cyan", fill="cyan", width=1,
                                               tags=("overlay",))

        # A raster selection is document-space state, so it stays aligned as
        # the canvas pans and zooms. Draw a contrasting marquee over the image.
        if current_layer.is_raster and self.selection_bounds is not None:
            self._draw_selection_marquee()


if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")   # maximized on Windows/macOS
    try:
        root.attributes("-zoomed", True)  # maximized on Linux
    except tk.TclError:
        pass
    PaintApp(root)
    root.mainloop()
