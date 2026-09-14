"""Editable vector objects, geometry queries, transforms and immutable snapshots."""

import json
import math
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from pypaint.state import Revisioned
from pypaint.storage import default_store, metadata


class VectorObject(Revisioned):
    """Base class for vector objects"""

    def __init__(self, color="#000000", width=2, antialias=True, hardness=75):
        self._init_revision()
        self.name = None
        self.color = color
        self.width = width
        self.antialias = antialias
        self.hardness = hardness
        self.selected = False

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "type": self.__class__.__name__,
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "width": self.width,
            "antialias": self.antialias,
            "hardness": self.hardness,
        }

    @classmethod
    def from_dict(cls, data):
        """Create object from dictionary"""
        data = dict(data)  # avoid mutating the original
        obj_type = data.pop("type")
        if obj_type == "Point":
            obj = Point(
                data["x"],
                data["y"],
                data["color"],
                data["width"],
                data.get("antialias", True),
                data.get("hardness", 75),
            )
        elif obj_type == "Line":
            obj = Line.from_dict(data)
        elif obj_type == "Shape":
            obj = Shape.from_dict(data)
        elif obj_type == "Rectangle":
            obj = Shape.from_legacy_rectangle(data)
        elif obj_type == "Ellipse":
            obj = Shape.from_legacy_ellipse(data)
        else:
            raise ValueError(f"Unsupported vector type: {obj_type}")
        obj.id = data.get("id", obj.id)
        obj.name = data.get("name")
        return obj


class Point(VectorObject):
    """A movable vector dot; width is its diameter."""

    def __init__(self, x=0, y=0, color="#000000", width=2, antialias=True, hardness=75):
        super().__init__(color, width, antialias, hardness)
        self.x, self.y = x, y

    def to_dict(self):
        return dict(super().to_dict(), x=self.x, y=self.y)

    def get_points(self):
        return [(self.x, self.y)]

    def update_point(self, index, x, y):
        self.x, self.y = x, y


def snap_line_endpoint(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    angle = round(math.atan2(dy, dx) / (math.pi / 12)) * (math.pi / 12)
    length = math.hypot(dx, dy)
    return start[0] + length * math.cos(angle), start[1] + length * math.sin(angle)


def transform_vector(obj, transform):
    """Transform endpoints and Bezier controls together."""
    if isinstance(obj, Shape):
        for line in obj.lines:
            transform_vector(line, transform)
        obj._spherical_fill_cache = None
    elif isinstance(obj, Line):
        obj.x1, obj.y1 = transform(obj.x1, obj.y1)
        obj.x2, obj.y2 = transform(obj.x2, obj.y2)
        if obj.curve:
            obj.curve = [
                v for i in range(0, len(obj.curve), 2) for v in transform(*obj.curve[i : i + 2])
            ]
    elif isinstance(obj, Point):
        obj.x, obj.y = transform(obj.x, obj.y)


def vector_center(obj):
    points = obj.get_points()
    return (sum(x for x, y in points) / len(points), sum(y for x, y in points) / len(points))


def rotation_handle(obj, zoom):
    cx, cy = vector_center(obj)
    return cx, min(y for x, y in obj.get_points()) - 28 / zoom


class Line(VectorObject):
    def __init__(
        self,
        x1=0,
        y1=0,
        x2=100,
        y2=100,
        color="#000000",
        width=2,
        curve=None,
        space="flat",
        antialias=True,
        hardness=75,
    ):
        super().__init__(color, width, antialias, hardness)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.curve = curve
        self.space = space

    def to_dict(self):
        data = super().to_dict()
        data.update(
            {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
                "curve": self.curve,
                "space": self.space,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            data["x1"],
            data["y1"],
            data["x2"],
            data["y2"],
            data["color"],
            data["width"],
            data.get("curve"),
            data.get("space", "flat"),
            data.get("antialias", True),
            data.get("hardness", 75),
        )
        obj.id = data.get("id", obj.id)
        obj.name = data.get("name")
        return obj

    def sampled_points(self, document_width=1024, document_height=512, steps=32):
        """Return map points. Globe lines follow the shortest great-circle arc."""
        if self.space == "globe":
            from pypaint.sphere import arc_to_uv, uv_to_vec

            a = uv_to_vec((1.0 - self.x1 / document_width) % 1.0, self.y1 / document_height)
            b = uv_to_vec((1.0 - self.x2 / document_width) % 1.0, self.y2 / document_height)
            points = arc_to_uv(a, b, step_radians=math.pi / max(8, steps))
            mapped = [((1.0 - u) % 1.0 * document_width, v * document_height) for u, v in points]
            # Unwrap longitude so PIL draws across the seam, not across the map.
            for i in range(1, len(mapped)):
                px = mapped[i - 1][0]
                x, y = mapped[i]
                while x - px > document_width / 2:
                    x -= document_width
                while px - x > document_width / 2:
                    x += document_width
                mapped[i] = (x, y)
            return mapped
        if not self.curve:
            return [(self.x1, self.y1), (self.x2, self.y2)]
        controls = self.curve
        if len(controls) == 2:
            cx, cy = controls[0], controls[1]
            return [
                (
                    (1 - t) ** 2 * self.x1 + 2 * (1 - t) * t * cx + t * t * self.x2,
                    (1 - t) ** 2 * self.y1 + 2 * (1 - t) * t * cy + t * t * self.y2,
                )
                for t in (i / steps for i in range(steps + 1))
            ]
        c1x, c1y, c2x, c2y = controls
        return [
            (
                (1 - t) ** 3 * self.x1
                + 3 * (1 - t) ** 2 * t * c1x
                + 3 * (1 - t) * t * t * c2x
                + t**3 * self.x2,
                (1 - t) ** 3 * self.y1
                + 3 * (1 - t) ** 2 * t * c1y
                + 3 * (1 - t) * t * t * c2y
                + t**3 * self.y2,
            )
            for t in (i / steps for i in range(steps + 1))
        ]

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

    def __init__(
        self,
        lines=None,
        color="#000000",
        width=2,
        fill=None,
        filled_side="inside",
        preset="custom",
        antialias=True,
        hardness=75,
    ):
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
        data.update(
            lines=[line.to_dict() for line in self.lines],
            fill=self.fill,
            filled_side=self.filled_side,
            preset=self.preset,
        )
        return data

    @classmethod
    def from_dict(cls, data):
        lines = [
            Line.from_dict({k: v for k, v in item.items() if k != "type"})
            for item in data.get("lines", [])
        ]
        obj = cls(
            lines,
            data["color"],
            data["width"],
            data.get("fill"),
            data.get("filled_side", "inside"),
            data.get("preset", "custom"),
            data.get("antialias", True),
            data.get("hardness", 75),
        )
        # Construction applies the shape's initial style. Decoding must retain
        # independently edited child styles for a lossless native round trip.
        for line, record in zip(obj.lines, data.get("lines", [])):
            for name in ("color", "width", "antialias", "hardness"):
                if name in record:
                    setattr(line, name, record[name])
        obj.id = data.get("id", obj.id)
        obj.name = data.get("name")
        return obj

    @classmethod
    def from_legacy_rectangle(cls, data):
        x, y, w, h = data["x"], data["y"], data["w"], data["h"]
        antialias = data.get("antialias", True)
        hardness = data.get("hardness", 75)
        vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        lines = [
            Line(
                *vertices[i],
                *vertices[(i + 1) % 4],
                data["color"],
                data["width"],
                antialias=antialias,
                hardness=hardness,
            )
            for i in range(4)
        ]
        return cls(
            lines,
            data["color"],
            data["width"],
            data.get("fill"),
            preset="rect",
            antialias=antialias,
            hardness=hardness,
        )

    @classmethod
    def from_legacy_ellipse(cls, data):
        cx, cy, rx, ry = data["x"], data["y"], data["rx"], data["ry"]
        antialias = data.get("antialias", True)
        hardness = data.get("hardness", 75)
        k = 0.5522847498
        vertices = [(cx + rx, cy), (cx, cy + ry), (cx - rx, cy), (cx, cy - ry)]
        controls = [
            (cx + rx, cy + k * ry, cx + k * rx, cy + ry),
            (cx - k * rx, cy + ry, cx - rx, cy + k * ry),
            (cx - rx, cy - k * ry, cx - k * rx, cy - ry),
            (cx + k * rx, cy - ry, cx + rx, cy - k * ry),
        ]
        lines = [
            Line(
                *vertices[i],
                *vertices[(i + 1) % 4],
                data["color"],
                data["width"],
                curve=controls[i],
                antialias=antialias,
                hardness=hardness,
            )
            for i in range(4)
        ]
        return cls(
            lines,
            data["color"],
            data["width"],
            data.get("fill"),
            preset="ellipse",
            antialias=antialias,
            hardness=hardness,
        )

    def _outline(self, width, height):
        points = []
        for line in self.lines:
            segment = line.sampled_points(width, height)
            if points and segment and line.space == "globe":
                shift = round((points[-1][0] - segment[0][0]) / width) * width
                segment = [(x + shift, y) for x, y in segment]
            points.extend(segment if not points else segment[1:])
        return points

    def _spherical_fill_mask(self, width, height, box=None):
        """Rasterize the smaller spherical interior, including across a pole."""
        vertices_xy = [(line.x1, line.y1) for line in self.lines]
        box = box or (0, 0, width, height)
        cache_key = (width, height, box, tuple(vertices_xy))
        if self._spherical_fill_cache and self._spherical_fill_cache[0] == cache_key:
            return self._spherical_fill_cache[1]

        # Physical sphere coordinates use the opposite longitude direction to
        # the displayed texture, matching globe_view's texture transform.
        uv = np.asarray(
            [((1.0 - x / width) % 1.0, y / height) for x, y in vertices_xy], dtype=np.float64
        )
        lon = uv[:, 0] * (2 * np.pi) - np.pi
        lat = (0.5 - uv[:, 1]) * np.pi
        vertices = np.column_stack(
            (np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon))
        )

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
                np.dot(centre, np.cross(tangent, following)), np.dot(tangent, following)
            )
        inside_direction = 1.0 if centre_winding >= 0.0 else -1.0

        mask = np.zeros((box[3] - box[1], box[2] - box[0]), dtype=np.uint8)
        pixel_lon = (1.0 - (np.arange(box[0], box[2]) + 0.5) / width) * (2 * np.pi) - np.pi
        cos_lon, sin_lon = np.cos(pixel_lon), np.sin(pixel_lon)

        # Work in strips to keep temporary tangent arrays bounded for ellipses.
        for y0 in range(box[1], box[3], 32):
            y1 = min(box[3], y0 + 32)
            pixel_lat = (0.5 - (np.arange(y0, y1) + 0.5) / height) * np.pi
            cos_lat = np.cos(pixel_lat)[:, None]
            points = np.stack(
                np.broadcast_arrays(
                    cos_lat * cos_lon[None, :],
                    np.sin(pixel_lat)[:, None],
                    cos_lat * sin_lon[None, :],
                ),
                axis=-1,
            )
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
            mask[y0 - box[1] : y1 - box[1]] = (winding * inside_direction > np.pi).astype(
                np.uint8
            ) * 255

        result = Image.fromarray(mask, mode="L")
        self._spherical_fill_cache = (cache_key, result)
        return result

    def draw(self, draw, document_width=1024, document_height=512):
        points = self._outline(document_width, document_height)
        globe = any(line.space == "globe" for line in self.lines)
        offsets = (-document_width, 0, document_width) if globe else (0,)
        if self.fill and len(points) >= 3 and self.filled_side == "inside":
            if globe:
                draw.bitmap(
                    (0, 0),
                    self._spherical_fill_mask(document_width, document_height),
                    fill=self.fill,
                )
            else:
                draw.polygon(points, fill=self.fill)
        # Stroke the complete outline once. Drawing each constituent segment
        # separately makes shared endpoints overlap, producing visibly thicker
        # corners and uneven joins.
        if len(points) >= 2:
            for offset in offsets:
                draw.line(
                    [(x + offset, y) for x, y in points],
                    fill=self.color,
                    width=self.width,
                    joint="curve",
                )

    def get_points(self):
        if (
            self.preset == "ellipse"
            and len(self.lines) == 4
            and all(line.space == "flat" for line in self.lines)
        ):
            right, bottom, left, top = [(line.x1, line.y1) for line in self.lines]
            cx, cy = (right[0] + left[0]) / 2, (right[1] + left[1]) / 2
            ux, uy = right[0] - cx, right[1] - cy
            vx, vy = bottom[0] - cx, bottom[1] - cy
            return [
                (cx - ux - vx, cy - uy - vy),
                (cx + ux - vx, cy + uy - vy),
                (cx + ux + vx, cy + uy + vy),
                (cx - ux + vx, cy - uy + vy),
            ]
        return [(line.x1, line.y1) for line in self.lines]

    def update_point(self, index, x, y, square=False):
        if (
            self.preset in ("rect", "ellipse")
            and len(self.lines) == 4
            and all(line.space == "flat" for line in self.lines)
        ):
            points = self.get_points()
            anchor = points[(index + 2) % 4]
            a, b = points[(index + 1) % 4], points[(index - 1) % 4]
            u = (a[0] - anchor[0], a[1] - anchor[1])
            v = (b[0] - anchor[0], b[1] - anchor[1])
            determinant = u[0] * v[1] - u[1] * v[0]
            if abs(determinant) < 1e-10:
                return
            dx, dy = x - anchor[0], y - anchor[1]
            su = (dx * v[1] - dy * v[0]) / determinant
            sv = (u[0] * dy - u[1] * dx) / determinant
            if square:
                ul, vl = math.hypot(*u), math.hypot(*v)
                side = max(abs(su) * ul, abs(sv) * vl, 1e-4)
                su = math.copysign(side / ul, su)
                sv = math.copysign(side / vl, sv)
            su = math.copysign(max(abs(su), 1e-4), su)
            sv = math.copysign(max(abs(sv), 1e-4), sv)

            def resize(px, py):
                dx, dy = px - anchor[0], py - anchor[1]
                cu = (dx * v[1] - dy * v[0]) / determinant
                cv = (u[0] * dy - u[1] * dx) / determinant
                return (
                    anchor[0] + cu * su * u[0] + cv * sv * v[0],
                    anchor[1] + cu * su * u[1] + cv * sv * v[1],
                )

            transform_vector(self, resize)
            return
        if not (0 <= index < len(self.lines)):
            return
        old = (self.lines[index].x1, self.lines[index].y1)
        self._spherical_fill_cache = None
        for line in self.lines:
            if (line.x1, line.y1) == old:
                line.x1, line.y1 = x, y
            if (line.x2, line.y2) == old:
                line.x2, line.y2 = x, y


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
        data.update({"x": self.x, "y": self.y, "w": self.w, "h": self.h, "fill": self.fill})
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["x"], data["y"], data["w"], data["h"], data["color"], data["width"], data["fill"]
        )

    def draw(self, draw, document_width=1024, document_height=512):
        draw.rectangle(
            [(self.x, self.y), (self.x + self.w, self.y + self.h)],
            outline=self.color,
            fill=self.fill,
            width=self.width,
        )

    def get_points(self):
        return [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h),
        ]


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
        data.update({"x": self.x, "y": self.y, "rx": self.rx, "ry": self.ry, "fill": self.fill})
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["x"], data["y"], data["rx"], data["ry"], data["color"], data["width"], data["fill"]
        )

    def draw(self, draw, document_width=1024, document_height=512):
        draw.ellipse(
            [(self.x - self.rx, self.y - self.ry), (self.x + self.rx, self.y + self.ry)],
            outline=self.color,
            fill=self.fill,
            width=self.width,
        )

    def get_points(self):
        return [
            (self.x - self.rx, self.y),
            (self.x + self.rx, self.y),
            (self.x, self.y - self.ry),
            (self.x, self.y + self.ry),
        ]


class VectorLayer(Revisioned):
    def __init__(self, name, width, height):
        self._init_revision()
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
        from pypaint.rendering import render_vector_object

        for obj in self.objects:
            render_vector_object(image, obj, self.width, self.height)

    def get_object_at(self, x, y, tolerance=10):
        """Find object at position (for selection)"""
        # Check in reverse order (top objects first)
        for obj in self.objects_in_region(
            (x - tolerance, y - tolerance, x + tolerance, y + tolerance), reverse=True
        ):
            points = obj.get_points()
            for px, py in points:
                if abs(px - x) <= tolerance and abs(py - y) <= tolerance:
                    return obj, points.index((px, py))
        return None, None

    def objects_in_region(self, box, reverse=False):

        key = (self.revision, self.width, self.height)
        if getattr(self, "_hit_index_key", None) != key:
            self._hit_index = SpatialIndex(
                [object_bounds(obj, (self.width, self.height)) for obj in self.objects]
            )
            self._hit_index_key = key
        indices = self._hit_index.query(box)
        return [self.objects[index] for index in (reversed(indices) if reverse else indices)]

    def get_object_near(self, x, y, tolerance=10):
        """Return the topmost object whose rendered path is near a point."""
        tolerance_sq = tolerance * tolerance
        for obj in self.objects_in_region(
            (x - tolerance, y - tolerance, x + tolerance, y + tolerance), reverse=True
        ):
            if isinstance(obj, Point):
                if math.hypot(x - obj.x, y - obj.y) <= tolerance + obj.width / 2:
                    return obj
                continue
            if isinstance(obj, Line):
                points = obj.sampled_points(self.width, self.height)
            elif isinstance(obj, Shape):
                points = obj._outline(self.width, self.height)
            else:
                points = obj.get_points()
            if any(
                self._point_segment_distance_sq(x, y, a, b) <= tolerance_sq
                for a, b in zip(points, points[1:])
            ):
                return obj
        return None

    @staticmethod
    def _point_segment_distance_sq(x, y, start, end):
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return (x - x1) ** 2 + (y - y1) ** 2
        amount = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
        nearest_x, nearest_y = x1 + amount * dx, y1 + amount * dy
        return (x - nearest_x) ** 2 + (y - nearest_y) ** 2

    def to_dict(self):
        return {
            "name": self.name,
            "visible": self.visible,
            "objects": [obj.to_dict() for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, data, width, height):
        layer = cls(data["name"], width, height)
        layer.visible = data["visible"]
        for obj_data in data["objects"]:
            obj = VectorObject.from_dict(obj_data)
            if obj:
                layer.objects.append(obj)
        return layer


# Conservative uniform-grid queries retaining original draw/hit-test order.


def object_bounds(obj, size):
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


class SpatialIndex:
    def __init__(self, bounds, cell_size=256):
        self.bounds = bounds
        self.cell_size = cell_size
        self.cells = defaultdict(list)
        self.global_indices = []
        for index, box in enumerate(bounds):
            left, top, right, bottom = self._cells(box)
            if (right - left + 1) * (bottom - top + 1) > 64:
                self.global_indices.append(index)
                continue
            for y in range(top, bottom + 1):
                for x in range(left, right + 1):
                    self.cells[(x, y)].append(index)

    def _cells(self, box):
        return tuple(math.floor(value / self.cell_size) for value in box)

    def query(self, box):
        left, top, right, bottom = self._cells(box)
        if (right - left + 1) * (bottom - top + 1) > 65536:
            candidates = range(len(self.bounds))
        else:
            candidates = set(self.global_indices)
            for y in range(top, bottom + 1):
                for x in range(left, right + 1):
                    candidates.update(self.cells.get((x, y), ()))
        return [index for index in sorted(candidates) if self.intersects(box, self.bounds[index])]

    @staticmethod
    def intersects(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# Pure editable geometry transforms shared by use cases and tools.


def scale_vector(obj, scale_x, scale_y):
    """Scale editable vector geometry with a resized image."""
    lines = obj.lines if isinstance(obj, Shape) else ([obj] if isinstance(obj, Line) else [])
    for line in lines:
        line.x1 *= scale_x
        line.y1 *= scale_y
        line.x2 *= scale_x
        line.y2 *= scale_y
        line.width = max(1, round(line.width * math.sqrt(scale_x * scale_y)))
        if line.curve:
            line.curve = tuple(
                value * (scale_x if index % 2 == 0 else scale_y)
                for index, value in enumerate(line.curve)
            )
    if isinstance(obj, Shape):
        obj.width = max(1, round(obj.width * math.sqrt(scale_x * scale_y)))
        obj._spherical_fill_cache = None
    elif isinstance(obj, (Rectangle, Ellipse, Point)):
        obj.x *= scale_x
        obj.y *= scale_y
        obj.width = max(1, round(obj.width * math.sqrt(scale_x * scale_y)))
        if isinstance(obj, Rectangle):
            obj.w *= scale_x
            obj.h *= scale_y
        elif isinstance(obj, Ellipse):
            obj.rx *= scale_x
            obj.ry *= scale_y


def translate_vector(obj, x, y):
    """Move an editable vector object with its newly anchored canvas."""
    lines = obj.lines if isinstance(obj, Shape) else ([obj] if isinstance(obj, Line) else [])
    for line in lines:
        line.x1 += x
        line.y1 += y
        line.x2 += x
        line.y2 += y
        if line.curve:
            line.curve = tuple(
                value + (x if index % 2 == 0 else y) for index, value in enumerate(line.curve)
            )
    if isinstance(obj, Shape):
        obj._spherical_fill_cache = None
    elif isinstance(obj, (Rectangle, Ellipse, Point)):
        obj.x += x
        obj.y += y


# Immutable per-object vector records; unchanged objects share spillable payloads.


class VectorSnapshot:
    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other or (
            isinstance(other, VectorSnapshot)
            and (self.name, self.visible, self.size, self.objects)
            == (other.name, other.visible, other.size, other.objects)
        )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Vector snapshots are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, name, visible, objects, bounds=None, size=None):
        self.name, self.visible, self.objects = name, visible, tuple(objects)
        self.bounds, self.size = bounds, size
        self._hash = hash((name, visible, size, self.objects))
        metadata.track(
            self,
            sys.getsizeof(self)
            + sys.getsizeof(self.objects)
            + (sys.getsizeof(bounds) if bounds is not None else 0),
        )
        self._sealed = True

    @classmethod
    def capture(cls, vector_layer):
        records = []
        bounds = []
        size = (vector_layer.width, vector_layer.height)
        for obj in vector_layer.objects:
            if getattr(obj, "_snapshot_revision", None) != obj.revision:
                obj._snapshot_payload = default_store().put(
                    json.dumps(obj.to_dict(), allow_nan=False).encode()
                )
                obj._snapshot_revision = obj.revision
            records.append(obj._snapshot_payload)
            if getattr(obj, "_bounds_revision", None) != (obj.revision, size):
                obj._snapshot_bounds = object_bounds(obj, size)
                obj._bounds_revision = (obj.revision, size)
            bounds.append(obj._snapshot_bounds)
        return cls(vector_layer.name, vector_layer.visible, records, tuple(bounds), size)

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["name"],
            data["visible"],
            [
                default_store().put(json.dumps(obj, allow_nan=False).encode())
                for obj in data["objects"]
            ],
        )

    def as_dict(self):
        return dict(
            name=self.name,
            visible=self.visible,
            objects=[json.loads(payload.read()) for payload in self.objects],
        )

    def encoded(self):
        return json.dumps(self.as_dict(), allow_nan=False).encode()

    def appended(self, obj):
        """Preview one object without decoding or serializing the existing scene."""
        payload = default_store().put(json.dumps(obj.to_dict(), allow_nan=False).encode())
        bounds = (*self.bounds, object_bounds(obj, self.size)) if self.bounds is not None else None
        return VectorSnapshot(self.name, self.visible, (*self.objects, payload), bounds, self.size)
