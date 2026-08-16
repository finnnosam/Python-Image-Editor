import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk, ImageOps
import numpy as np
import math
import copy
import io
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class VectorObject:
    """Base class for vector objects"""
    def __init__(self, color="#000000", width=2):
        self.color = color
        self.width = width
        self.selected = False
        
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'type': self.__class__.__name__,
            'color': self.color,
            'width': self.width
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
                 curve=None, space="flat"):
        super().__init__(color, width)
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
                   data.get('space', 'flat'))
    
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
                 filled_side="inside", preset="custom"):
        super().__init__(color, width)
        self.lines = lines or []
        self.fill = fill
        self.filled_side = filled_side
        self.preset = preset
        self._spherical_fill_cache = None
        for line in self.lines:
            line.color, line.width = color, width

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
                   data.get('filled_side', 'inside'), data.get('preset', 'custom'))

    @classmethod
    def from_legacy_rectangle(cls, data):
        x, y, w, h = data['x'], data['y'], data['w'], data['h']
        vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        lines = [Line(*vertices[i], *vertices[(i+1) % 4], data['color'], data['width'])
                 for i in range(4)]
        return cls(lines, data['color'], data['width'], data.get('fill'), preset='rect')

    @classmethod
    def from_legacy_ellipse(cls, data):
        cx, cy, rx, ry = data['x'], data['y'], data['rx'], data['ry']
        k = 0.5522847498
        vertices = [(cx+rx,cy),(cx,cy+ry),(cx-rx,cy),(cx,cy-ry)]
        controls = [(cx+rx,cy+k*ry,cx+k*rx,cy+ry),
                    (cx-k*rx,cy+ry,cx-rx,cy+k*ry),
                    (cx-rx,cy-k*ry,cx-k*rx,cy-ry),
                    (cx+k*rx,cy-ry,cx+rx,cy-k*ry)]
        lines = [Line(*vertices[i], *vertices[(i+1) % 4], data['color'],
                      data['width'], curve=controls[i]) for i in range(4)]
        return cls(lines, data['color'], data['width'], data.get('fill'), preset='ellipse')

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
        for line in self.lines:
            line.draw(draw, document_width, document_height)

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
            
    def render(self, draw):
        for obj in self.objects:
            obj.draw(draw, self.width, self.height)
            
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
        self.layer_type = layer_type  # "raster" or "vector"
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
            self.vector_data.render(self.draw)
            self.reset_mipmaps()

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

        # Drag and drop variables
        self.drag_start_index = None
        self.drag_start_y = None

        # UI scale (1.5 = launch default)
        self.ui_scale = 1.5

        self.build_ui()
        self.apply_ui_scale(self.ui_scale)  # A: apply 1.5× on launch
        self.refresh_layers()
        self.redraw()
        self.update_title()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
            ("Brush",      "brush",       "brush.png"),
            ("Eraser",     "eraser",      "eraser.png"),
            ("Vector Edit", "vector edit", "vector-edit.png"),
            ("Line",       "line",        "line.png"),
            ("Rectangle",  "rect",        "rect.png"),
            ("Ellipse",    "ellipse",     "ellipse.png"),
        ]
        icon_dir = Path(__file__).resolve().parent / "icons"
        self.tool_icons = {}
        self.tool_buttons = {}
        self.tool_hint_var = tk.StringVar(value="Brush")
        for index, (label, tool, filename) in enumerate(tools):
            with Image.open(icon_dir / filename) as icon_image:
                icon_image = icon_image.convert("RGBA").resize(
                    (32, 32), Image.Resampling.LANCZOS)
            icon = ImageTk.PhotoImage(icon_image)
            self.tool_icons[tool] = icon
            button = tk.Button(
                tool_frame, image=icon, width=42, height=42,
                command=lambda t=tool: self.set_tool(t),
                relief="sunken" if tool == self.tool else "raised",
                takefocus=True)
            # Column 1 is intentionally left open for per-tool settings.
            button.grid(row=index, column=0, padx=2, pady=2)
            button.bind(
                "<Enter>",
                lambda event, name=label: self.tool_hint_var.set(name))
            button.bind(
                "<Leave>",
                lambda event: self.tool_hint_var.set(self.tool.title()))
            self.tool_buttons[tool] = button
        tk.Label(left, textvariable=self.tool_hint_var).pack(pady=(2, 0))

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=4, pady=6)

        # ── Color Selector ─────────────────────────────────────────────
        tk.Label(left, text="Colors", font=("TkDefaultFont", 9, "bold")).pack(pady=(6, 2))

        color_frame = tk.Frame(left)
        color_frame.pack(pady=4)

        # Primary color square (left)
        self.primary_square = tk.Canvas(color_frame, width=30, height=30, bg=self.primary_color, highlightthickness=1, highlightbackground="black")
        self.primary_square.pack(side="left", padx=2)
        self.primary_square.bind("<Button-1>", lambda e: self.choose_primary_color())

        # Secondary color square (right)
        self.secondary_square = tk.Canvas(color_frame, width=30, height=30, bg=self.secondary_color, highlightthickness=1, highlightbackground="black")
        self.secondary_square.pack(side="left", padx=2)
        self.secondary_square.bind("<Button-1>", lambda e: self.choose_secondary_color())

        # Swap colors button
        tk.Button(left, text="↔", width=3, command=self.swap_colors).pack(pady=2)

        # Size is shared by tools, so it remains below the tool column.  The
        # second column stays available for future per-tool settings.
        self.size_frame = tk.Frame(tool_frame)
        self.size_frame.grid(row=len(tools), column=0, padx=2, pady=(6, 2))

        tk.Label(self.size_frame, text="Size:").pack(side="left")

        self.size_var = tk.StringVar(value="20")
        self.size_entry = tk.Entry(
            self.size_frame, width=6, textvariable=self.size_var)
        self.size_entry.pack(side="left", padx=4)

        # Update size when Enter is pressed or focus is lost
        def update_size(event=None):
            try:
                val = int(self.size_var.get())
                if val < 1:
                    val = 1
                elif val > 100:
                    val = 100
                self.size_var.set(str(val))
            except ValueError:
                self.size_var.set("20")  # revert to default on invalid input

        self.size_entry.bind("<Return>", update_size)
        self.size_entry.bind("<FocusOut>", update_size)

        # ── Centre: reusable view workspace ──────────────────────────────
        # Tools and layers live outside this frame, so every view shares them.
        self.view_workspace = tk.Frame(main)
        self.view_workspace.pack(side="left", fill="both", expand=True)

        self.view_tabs = tk.Frame(self.view_workspace, bd=1, relief="raised")
        self.view_tabs.pack(side="top", fill="x")
        self.view_host = tk.Frame(self.view_workspace)
        self.view_host.pack(side="top", fill="both", expand=True)

        self.views = {}
        self.view_tab_widgets = {}
        self.active_view = None

        flat_view = tk.Frame(self.view_host)
        self.canvas = tk.Canvas(flat_view, bg="gray25")
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
        self.canvas.bind("b", lambda e: self.set_tool("brush"))
        self.canvas.bind("e", lambda e: self.set_tool("eraser"))
        self.canvas.bind("v", lambda e: self.set_tool("vector edit"))
        self.canvas.bind("l", lambda e: self.set_tool("line"))
        self.canvas.bind("r", lambda e: self.set_tool("rect"))
        self.canvas.bind("o", lambda e: self.set_tool("ellipse"))
        self.canvas.bind("<Control-z>", lambda e: self.undo())
        self.canvas.bind("<Control-s>", lambda e: self.save_project())
        self.canvas.bind("<Control-n>", lambda e: self.new_project())
        self.canvas.focus_set()
        self.switch_view("main")

        # ── Right panel: layers ───────────────────────────────────────────
        right = tk.Frame(main, width=250, bd=1, relief="sunken")
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Layers", font=("TkDefaultFont", 9, "bold")).pack(pady=(6, 2))

        layer_style = ttk.Style()
        layer_style.configure("Layer.Treeview", rowheight=32)
        self.layer_list = ttk.Treeview(
            right, show="tree", selectmode="browse", style="Layer.Treeview")
        self.layer_list.pack(fill="both", expand=True, padx=4)
        self.layer_list.column("#0", stretch=True, width=220)
        self.layer_list.bind("<<TreeviewSelect>>", self.select_layer)
        self.layer_list.bind("<Button-1>",         self.on_layer_pointer_down)
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
        c = colorchooser.askcolor(self.primary_color)[1]
        if c:
            self.primary_color = c
            self.color = c  # Update current color for compatibility
            self.primary_square.config(bg=c)
            self.request_redraw()

    def choose_secondary_color(self):
        c = colorchooser.askcolor(self.secondary_color)[1]
        if c:
            self.secondary_color = c
            self.secondary_square.config(bg=c)

    def swap_colors(self):
        self.primary_color, self.secondary_color = self.secondary_color, self.primary_color
        self.color = self.primary_color
        self.primary_square.config(bg=self.primary_color)
        self.secondary_square.config(bg=self.secondary_color)
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
        return ((layer.layer_type == "raster" and self.tool in ("brush", "eraser")) or
                (layer.layer_type == "vector" and self.tool in ("line", "rect", "ellipse")))

    def can_draw_vector_from_globe(self):
        return (self.layers[self.active_layer].layer_type == "vector" and
                self.tool in ("line", "rect", "ellipse"))

    def snapshot(self):
        snap = []
        for l in self.layers:
            n = Layer(self.doc_w, self.doc_h, l.name, l.layer_type)
            n.visible = l.visible
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
        
        self.layers = [Layer(self.doc_w, self.doc_h, "Background", "raster")]
        self.active_layer = 0
        self.current_file = None
        self.undo_stack = []
        self.refresh_layers()
        self.redraw()
        self.update_title()
        self.notify_globe_document_changed()

    def save_project(self):
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

            layer = Layer(project_data['document_width'],
                          project_data['document_height'],
                          layer_info['name'],
                          layer_info.get('layer_type', 'raster'))
            layer.image = img
            layer.visible = layer_info['visible']
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
        self.refresh_layers()
        self.redraw()
        self.update_title()
        self.notify_globe_document_changed()

    def set_tool(self, tool):
        self.tool = tool
        if hasattr(self, "tool_buttons"):
            for name, button in self.tool_buttons.items():
                button.configure(relief="sunken" if name == tool else "raised")
            self.tool_hint_var.set(tool.title())
        # Reset vector drawing state
        self.vector_start_x = None
        self.vector_start_y = None
        self.current_vector_obj = None
        self.request_redraw()

    def add_layer(self, layer_type="raster"):
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
            self.active_layer = len(self.layers) - 1 - display_index
            self.request_redraw()

    def refresh_layers(self):
        self.layer_list.delete(*self.layer_list.get_children())
        for i in range(len(self.layers) - 1, -1, -1):
            l = self.layers[i]
            self.layer_list.insert(
                "", "end", text=l.name,
                image=self.layer_row_icons[(l.visible, l.layer_type)])
        
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

    def screen_coords(self, ix, iy):
        return (ix * self.zoom + self.offset_x,
                iy * self.zoom + self.offset_y)

    def on_mouse_down(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.last_button = event.num  # Track which button (1=left, 3=right)
        x, y = self.image_coords(event.x, event.y)
        current_layer = self.layers[self.active_layer]
        
        if current_layer.layer_type == "raster":
            self.start_raster_draw(event)
        else:  # vector layer
            self.start_vector_operation(event, x, y)

    def start_raster_draw(self, event):
        self.snapshot()
        self.last_x, self.last_y = self.image_coords(event.x, event.y)
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
        x, y = self.image_coords(event.x, event.y)
        current_layer = self.layers[self.active_layer]
        
        if current_layer.layer_type == "raster":
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
        self.last_x = x
        self.last_y = y
        self.last_button = button


    def end_external_raster_draw(self):
        """
        Finish an externally-driven brush stroke.
        """
        self.last_x = None
        self.last_y = None


    def raster_paint_image(self, x, y):
        """
        Paint using image coordinates instead of a Tk mouse event.
        """

        if self.last_x is None or self.last_y is None:
            self.last_x = x
            self.last_y = y

        radius = int(self.size_var.get()) / 2

        if self.tool == "eraser":
            color = (0, 0, 0, 0)
        else:
            color = (
                self.primary_color
                if self.last_button == 1
                else self.secondary_color
            )

        dx = x - self.last_x
        dy = y - self.last_y

        dist = math.hypot(dx, dy)

        spacing = max(1, radius * 0.25)
        steps = max(1, int(dist / spacing))

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

        radius = int(self.size_var.get()) / 2
        color = (0, 0, 0, 0) if self.tool == "eraser" else (
            self.primary_color if self.last_button == 1 else self.secondary_color
        )
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

        color = (0, 0, 0, 0) if self.tool == "eraser" else (
            self.primary_color if self.last_button == 1 else self.secondary_color
        )
        polygon = [(u * self.doc_w, v * self.doc_h) for u, v in footprint_uv]

        # Repeat the unwrapped polygon on both sides of the texture.  PIL clips
        # each copy to the image, preserving a brush that straddles the seam.
        layer = self.layers[self.active_layer]
        for offset in (-self.doc_w, 0, self.doc_w):
            shifted = [(x + offset, y) for x, y in polygon]
            layer.draw.polygon(
                shifted,
                fill=color,
            )
            xs = [point[0] for point in shifted]
            ys = [point[1] for point in shifted]
            layer.update_mipmaps((min(xs), min(ys), max(xs) + 1, max(ys) + 1))

        self.last_x, self.last_y = center_x, center_y
        if refresh:
            self.request_redraw()
            self.notify_globe_document_changed()

    def raster_paint(self, event):

        x, y = self.image_coords(
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
        layer.draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                           fill=color, outline=color)
        layer.update_mipmaps((x - radius, y - radius,
                              x + radius + 1, y + radius + 1))

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
        preview_draw = ImageDraw.Draw(preview_img)
        
        if self.tool == "line":
            preview_draw.line([(x1, y1), (x2, y2)], fill=self.primary_color, width=2)
        elif self.tool == "rect":
            bounds = [(min(x1, x2), min(y1, y2)),
                      (max(x1, x2), max(y1, y2))]
            preview_draw.rectangle(bounds, outline=self.primary_color, width=2)
        elif self.tool == "ellipse":
            bounds = [(min(x1, x2), min(y1, y2)),
                      (max(x1, x2), max(y1, y2))]
            preview_draw.ellipse(bounds, outline=self.primary_color, width=2)

        # Preview the active vector layer in its normal place in the complete
        # layer stack.  Displaying preview_img directly hides every raster
        # layer for the duration of the drag.
        preview_composite = Image.new(
            "RGBA", (self.doc_w, self.doc_h), (0, 0, 0, 0))
        for candidate in self.layers:
            if not candidate.visible:
                continue
            candidate_image = preview_img if candidate is layer else candidate.image
            preview_composite.alpha_composite(candidate_image)

        self.display_image(preview_composite)

    def create_vector_object(self, x1, y1, x2, y2):
        """Create a vector object and add it to the current layer"""
        layer = self.layers[self.active_layer]
        if layer.layer_type != "vector":
            return
        
        # Use primary color for vector objects
        obj = None
        if self.tool == "line":
            obj = Line(x1, y1, x2, y2, self.primary_color, 2)
        elif self.tool in ("rect", "ellipse"):
            obj = self.make_shape_preset(self.tool, [(x1, y1), (x2, y2)], "flat")
        
        if obj:
            layer.vector_data.add_object(obj)

    def make_shape_preset(self, preset, points, space="flat"):
        """Turn a UI shape preset into lines; presets are never special render objects."""
        color, width = self.primary_color, 2
        if preset == "rect":
            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                vertices = points[:4]
            lines = [Line(*vertices[i], *vertices[(i + 1) % 4], color, width, space=space)
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
                              curve=controls[i], space=space) for i in range(4)]
            else:
                vertices = points
                lines = [Line(*vertices[i], *vertices[(i+1)%len(vertices)], color,
                              width, space=space) for i in range(len(vertices))]
        # The secondary colour represents the shape's filled (inside) side.
        return Shape(lines, color, width, self.secondary_color, "inside", preset)

    def create_globe_vector(self, preset, image_points):
        if not self.can_draw_vector_from_globe() or len(image_points) < 2:
            return
        self.snapshot()
        if preset == "line":
            obj = Line(*image_points[0], *image_points[-1], self.primary_color, 2,
                       space="globe")
        else:
            obj = self.make_shape_preset(preset, image_points, "globe")
        self.layers[self.active_layer].vector_data.add_object(obj)
        self.layers[self.active_layer].render_vector()
        self.request_redraw()
        self.notify_globe_document_changed()

    def on_mouse_up(self, event):
        x, y = self.image_coords(event.x, event.y)
        current_layer = self.layers[self.active_layer]
        
        if current_layer.layer_type == "raster":
            self.last_x = None
            self.last_y = None
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
        for layer in self.layers:
            if layer.visible:
                if layer.layer_type == "vector" and layer.vector_data:
                    layer.render_vector()
                result.alpha_composite(layer.image)
        return result

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
        for layer in self.layers:
            if layer.visible:
                source = layer.get_mipmap(level)
                extent = (left / factor, top / factor,
                          right / factor, bottom / factor)
                rendered = source.transform(
                    size, Image.Transform.EXTENT, extent,
                    resample=Image.Resampling.NEAREST)
                result.alpha_composite(rendered)
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
        doc_sx = max(0, int(self.offset_x + left * self.zoom))
        doc_sy = max(0, int(self.offset_y + top * self.zoom))
        crop = img.crop((int(left), int(top), int(right), int(bottom)))
        sw = max(1, int((right - left) * self.zoom))
        sh = max(1, int((bottom - top) * self.zoom))
        crop = crop.resize((sw, sh), Image.Resampling.NEAREST)

        # Keep the backdrop and document in one tracked canvas item.  The old
        # preview path created a separate, untracked checkerboard item which
        # survived the next redraw and appeared as a second locked pattern.
        checker = self.get_checker_backdrop_pil(cw, ch).crop(
            (doc_sx, doc_sy, doc_sx + sw, doc_sy + sh))
        checker.alpha_composite(crop)
        self.tkimg = ImageTk.PhotoImage(checker.convert("RGB"))
        
        sx = self.offset_x + left * self.zoom
        sy = self.offset_y + top * self.zoom
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
        doc_sx = max(0, int(self.offset_x + left * self.zoom))
        doc_sy = max(0, int(self.offset_y + top * self.zoom))
        sw = max(1, int((right - left) * self.zoom))
        sh = max(1, int((bottom - top) * self.zoom))
        crop_box = (int(left), int(top), int(right), int(bottom))
        crop = self.composite_region(crop_box, (sw, sh))

        checker = self.get_checker_backdrop_pil(cw, ch).crop(
            (doc_sx, doc_sy, doc_sx + sw, doc_sy + sh))
        checker.alpha_composite(crop)
        self.tkimg = ImageTk.PhotoImage(checker.convert("RGB"))

        sx = self.offset_x + left * self.zoom
        sy = self.offset_y + top * self.zoom

        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(
                sx, sy, image=self.tkimg, anchor="nw")
        else:
            self.canvas.coords(self._canvas_image_id, sx, sy)
            self.canvas.itemconfigure(self._canvas_image_id,
                                      image=self.tkimg, state="normal")

        # Draw brush cursor for raster layers
        current_layer = self.layers[self.active_layer]
        if (current_layer.layer_type == "raster" and
                self.tool in ("brush", "eraser") and
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
        
        # Draw vector handles if in proper mode and on vector layer
        if self.tool == "vector edit" and current_layer.layer_type == "vector" and current_layer.vector_data:
            for obj in current_layer.vector_data.objects:
                points = obj.get_points()
                for px, py in points:
                    sx, sy = self.screen_coords(px, py)
                    self.canvas.create_rectangle(sx - 3, sy - 3, sx + 3, sy + 3,
                                               outline="cyan", fill="cyan", width=1,
                                               tags=("overlay",))


if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")   # maximized on Windows/macOS
    try:
        root.attributes("-zoomed", True)  # maximized on Linux
    except tk.TclError:
        pass
    PaintApp(root)
    root.mainloop()
