"""
globe_view.py
=============

Secondary globe view for PyPaint.

Responsibilities
----------------
* Display an equirectangular image wrapped onto a sphere.
* Allow globe rotation.
* Convert mouse interaction into image coordinates.
* Notify the main application of paint events.

Does NOT:
----------
* Paint pixels directly.
* Modify layers.
* Handle undo.
* Know about tools.

Those remain entirely inside Main.py.
"""

from __future__ import annotations

import tkinter as tk

from PIL import Image
from PIL import ImageTk

import numpy as np

from sphere_math import (
    Vec3,
    make_camera_ray,
    ray_sphere_intersection,
    vec_to_uv,
    uv_to_vec,
    spherical_brush_uv,
    apply_globe_rotation,
    remove_globe_rotation,
)


class GlobeWindow(tk.Toplevel):

    DEFAULT_SIZE = 700

    def __init__(self, app):

        super().__init__(app.root)

        self.app = app

        self.title("Globe View")
        self.geometry(
            f"{self.DEFAULT_SIZE}x{self.DEFAULT_SIZE}"
        )

        self.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

        #
        # Camera
        #

        self.yaw = 0.0
        self.pitch = 0.0

        self.camera_distance = 3.0
        self.fov = 45.0

        #
        # Globe
        #

        self.radius = 0.98
        self.display_scale = 0.45

        #
        # Cached image
        #

        self.texture = None
        self.texture_array = None

        self.photo = None

        #
        # Cached render
        #

        self.render_image = None

        #
        # Mouse state
        #

        self.rotating = False
        self.painting = False
        self.paint_button = 1

        self.last_mouse = None
        self.last_uv = None
        self.vector_start_screen = None
        self.document_refresh_pending = False

        #
        # Build widgets
        #

        self.build_ui()
        self.on_resize()

        self.bind_events()

        self.update_texture()

        self.after(
            1,
            self.redraw
        )

    # --------------------------------------------------

    def build_ui(self):

        self.canvas = tk.Canvas(
            self,
            bg="#303030",
            highlightthickness=0
        )

        self.canvas.pack(
            fill="both",
            expand=True
        )

    # --------------------------------------------------

    def bind_events(self):

        self.canvas.bind(
            "<Configure>",
            self.on_resize
        )

        self.canvas.bind(
            "<ButtonPress-1>",
            self.on_left_press
        )

        self.canvas.bind(
            "<B1-Motion>",
            self.on_left_drag
        )

        self.canvas.bind(
            "<ButtonRelease-1>",
            self.on_left_release
        )

        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)

        self.canvas.bind(
            "<ButtonPress-2>",
            self.on_middle_press
        )

        self.canvas.bind(
            "<B2-Motion>",
            self.on_middle_drag
        )

        self.canvas.bind(
            "<ButtonRelease-2>",
            self.on_middle_release
        )

        #
        # Linux wheel
        #

        self.canvas.bind(
            "<Button-4>",
            self.zoom_in
        )

        self.canvas.bind(
            "<Button-5>",
            self.zoom_out
        )

        #
        # Windows wheel
        #

        self.canvas.bind(
            "<MouseWheel>",
            self.on_mousewheel
        )

    # --------------------------------------------------

    def on_close(self):
        if getattr(self.app, "globe_window", None) is self:
            self.app.globe_window = None
        self.destroy()

    # --------------------------------------------------

    def update_texture(self):
        """
        Ask the main application for the current
        composited image.
        """

        image = self.app.composite_image()

        self.texture = image

        self.texture_array = np.asarray(image)

    # --------------------------------------------------

    def notify_document_changed(self):
        """
        Main.py should call this whenever the document
        changes.
        """

        self.request_document_refresh()

    def request_document_refresh(self):
        """Coalesce document updates so painting does not redraw twice per mouse event."""
        if self.document_refresh_pending:
            return
        self.document_refresh_pending = True
        self.after(33, self._refresh_document)

    def _refresh_document(self):
        self.document_refresh_pending = False
        if self.winfo_exists():
            self.update_texture()
            self.redraw()

    # --------------------------------------------------

    def redraw(self):

        if self.texture is None:
            return

        if not hasattr(self, "lookup_normals"):
            self.build_lookup()
        if not hasattr(self, "lookup_normals"):
            return

        rgb = self.render_numpy()

        image = Image.fromarray(rgb)

        self.photo = ImageTk.PhotoImage(image)

        self.canvas.delete("all")

        self.canvas.create_image(
            0,
            0,
            image=self.photo,
            anchor="nw"
        )

    # --------------------------------------------------

    def render_numpy(self):

        tex = self.texture_array

        h, w = self.lookup_mask.shape

        out = np.zeros(
            (h, w, 3),
            dtype=np.uint8
        )

        # This array is only read below.  Copying the full screen-sized normal
        # map for every frame was a substantial allocation during painting.
        normals = self.lookup_normals

        #
        # Rotate globe
        #

        yaw = self.yaw
        pitch = self.pitch

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        cp = np.cos(pitch)
        sp = np.sin(pitch)

        x = normals[...,0]
        y = normals[...,1]
        z = normals[...,2]

        # Map screen normals back into texture space using the same inverse
        # rotation as screen_to_uv(), so rendered and painted coordinates
        # remain aligned after rotation.
        yy = cp*y + sp*z
        zz = -sp*y + cp*z
        xr = cy*x - sy*zz
        zr = sy*x + cy*zz

        #
        # Longitude
        #

        lon = np.arctan2(
            zr,
            xr
        )

        #
        # Latitude
        #

        lat = np.arcsin(
            np.clip(
                yy,
                -1.0,
                1.0
            )
        )

        # The equirectangular image is mirrored horizontally relative to the
        # viewer's physical X axis.  Mirror the texture coordinate, not the
        # sphere geometry itself; that preserves a proper yaw rotation.
        u = (1.0 - (lon + np.pi) / (2*np.pi)) % 1.0

        v = 0.5 - lat / np.pi

        tw = tex.shape[1]
        th = tex.shape[0]

        tx = (u * tw).astype(np.int32)
        ty = (v * (th-1)).astype(np.int32)

        tx %= tw

        ty = np.clip(
            ty,
            0,
            th-1
        )

        mask = self.lookup_mask

        out[mask] = tex[
            ty[mask],
            tx[mask],
            :3
        ]

        return out

    # --------------------------------------------------

    def screen_to_uv(self, x, y):
        """
        Convert a canvas coordinate into texture UV.

        Returns

            (u,v)

        or

            None
        """

        if not hasattr(self, "lookup_mask"):
            return None

        h, w = self.lookup_mask.shape

        if x < 0 or y < 0:
            return None

        if x >= w or y >= h:
            return None

        if not self.lookup_mask[y, x]:
            return None

        #
        # Normal in viewer space.
        #

        n = self.lookup_normals[y, x]

        normal = Vec3(
            float(n[0]),
            float(n[1]),
            float(n[2])
        )

        #
        # Undo globe rotation.
        #

        normal = remove_globe_rotation(
            normal,
            self.yaw,
            self.pitch
        )

        u, v = vec_to_uv(normal)
        return ((1.0 - u) % 1.0, v)


    # --------------------------------------------------

    def uv_to_image(self, u, v):

        if self.texture is None:
            return None

        w = self.texture.width
        h = self.texture.height

        return (
            int((u % 1.0) * w),
            max(0, min(h - 1, int(v * h)))
        )

    # --------------------------------------------------

    def build_lookup(self):
        """
        Precompute a sphere normal for every screen pixel.

        This only depends on window size, not camera rotation,
        so it can be reused indefinitely.
        """

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if w < 2 or h < 2:
            return

        cx = w * 0.5
        cy = h * 0.5

        radius = min(w, h) * self.display_scale

        yy, xx = np.mgrid[0:h, 0:w]

        dx = (xx - cx) / radius
        dy = (yy - cy) / radius

        r2 = dx * dx + dy * dy

        mask = r2 <= 1.0

        dz = np.zeros_like(dx)
        dz[mask] = np.sqrt(1.0 - r2[mask])

        self.lookup_mask = mask

        #
        # Unit sphere normals
        #

        normals = np.zeros((h, w, 3), dtype=np.float32)

        normals[..., 0] = dx
        normals[..., 1] = -dy
        normals[..., 2] = dz

        length = np.linalg.norm(
            normals,
            axis=2,
            keepdims=True
        )

        length[length == 0] = 1.0

        normals /= length

        self.lookup_normals = normals

    # --------------------------------------------------
    # Event handlers
    # --------------------------------------------------

    def on_resize(self, event=None):

        self.build_lookup()
        self.redraw()

    def on_left_press(self, event):
        self.start_paint(event, button=1)

    def on_left_drag(self, event):

        if self.painting:

            if self.vector_start_screen is not None:
                self.last_mouse = (event.x, event.y)
            else:
                self.paint_from_mouse(event.x, event.y)

    def on_left_release(self, event):
        self.end_paint()

    def on_right_press(self, event):
        self.start_paint(event, button=3)

    def on_right_drag(self, event):
        if self.painting:
            if self.vector_start_screen is not None:
                self.last_mouse = (event.x, event.y)
            else:
                self.paint_from_mouse(event.x, event.y)

    def on_right_release(self, event):
        self.end_paint()

    def start_paint(self, event, button):
        if not self.app.can_paint_from_globe():
            self.bell()
            return
        if self.app.can_draw_vector_from_globe():
            if self.screen_to_uv(event.x, event.y) is None:
                return
            self.vector_start_screen = (event.x, event.y)
            self.painting = True
            self.paint_button = button
            self.last_mouse = self.vector_start_screen
            return
        self.painting = True
        self.paint_button = button
        self.paint_from_mouse(event.x, event.y, first=True)

    def end_paint(self):
        if self.vector_start_screen is not None:
            self.finish_globe_vector(self.vector_start_screen,
                                     self.last_mouse or self.vector_start_screen)
            self.vector_start_screen = None
            self.painting = False
            self.last_mouse = None
            return
        if self.painting:
            self.app.end_external_raster_draw()
        self.painting = False
        self.last_uv = None

    def finish_globe_vector(self, start, end):
        """Store a globe gesture as sphere-relative line primitives."""
        x1, y1 = start
        x2, y2 = end
        if self.app.tool == "line":
            samples = [start, end]
        elif self.app.tool == "rect":
            samples = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            rx, ry = abs(x2-x1)/2, abs(y2-y1)/2
            samples = [(cx + rx*np.cos(t), cy + ry*np.sin(t))
                       for t in np.linspace(0, 2*np.pi, 32, endpoint=False)]
        image_points = []
        for sx, sy in samples:
            uv = self.screen_to_uv(int(sx), int(sy))
            if uv is None:
                return
            image_points.append(self.uv_to_image(*uv))
        self.app.create_globe_vector(self.app.tool, image_points)

    def on_middle_press(self, event):

        self.rotating = True

        self.last_mouse = (
            event.x,
            event.y
        )

    def on_middle_drag(self, event):

        if not self.rotating:
            return

        lx, ly = self.last_mouse

        dx = event.x - lx
        dy = event.y - ly

        self.last_mouse = (
            event.x,
            event.y
        )

        #
        # Rotation speed
        #

        self.yaw += dx * 0.01

        self.pitch += dy * 0.01

        #
        # Prevent flipping
        #

        limit = np.pi / 2.0 - 0.02

        self.pitch = max(
            -limit,
            min(limit, self.pitch)
        )

        self.redraw()

    def on_middle_release(self, event):

        self.rotating = False

    def zoom_in(self, event=None):
        self.display_scale = min(0.49, self.display_scale * 1.08)
        self.build_lookup()
        self.redraw()

    def zoom_out(self, event=None):
        self.display_scale = max(0.15, self.display_scale / 1.08)
        self.build_lookup()
        self.redraw()

    def on_mousewheel(self, event):

        if event.delta > 0:
            self.zoom_in()

        else:
            self.zoom_out()

    def begin_external_raster_draw(
        self,
        x,
        y
    ):
        self.snapshot()
        self.last_x = x
        self.last_y = y

    def end_external_raster_draw(self):

        self.last_x = None
        self.last_y = None

    # --------------------------------------------------

    def stamp_spherical_brush(self, uv, refresh=False):
        """Paint a geodesic circular brush footprint at a texture UV point."""
        if self.texture is None:
            return

        radius = max(0.5, int(self.app.size_var.get()) / 2)
        angular_radius = (2 * np.pi * radius) / self.texture.width

        # sphere_math uses the physical longitude direction; the displayed
        # texture is mirrored horizontally, just as it is in render_numpy().
        center = uv_to_vec((1.0 - uv[0]) % 1.0, uv[1])
        boundary = spherical_brush_uv(
            center,
            angular_radius,
            rings=1,
            segments=32,
        )[1:]

        # Keep the boundary continuous around the brush centre.  Coordinates
        # are deliberately allowed outside [0, 1] for seam-safe rasterizing.
        footprint_uv = []
        for point_u, point_v in boundary:
            texture_u = (1.0 - point_u) % 1.0
            texture_u = uv[0] + ((texture_u - uv[0] + 0.5) % 1.0 - 0.5)
            footprint_uv.append((texture_u, point_v))

        center_x, center_y = self.uv_to_image(*uv)
        self.app.stamp_external_spherical_raster(
            footprint_uv,
            center_x,
            center_y,
            refresh=refresh,
        )

    # --------------------------------------------------

    def paint_from_mouse(
        self,
        x,
        y,
        first=False
    ):
        """
        Convert a mouse position into image coordinates
        and forward them to Main.py's existing raster
        painting engine.
        """

        uv = self.screen_to_uv(x, y)

        if uv is None:
            self.last_uv = None
            return

        #
        # Convert to image coordinates.
        #

        ix, iy = self.uv_to_image(*uv)

        #
        # Beginning of a stroke.
        #

        if first or self.last_uv is None:
            self.app.begin_external_raster_draw(ix, iy, self.paint_button)
            self.stamp_spherical_brush(uv, refresh=False)
            self.last_uv = uv
            self.app.request_redraw()
            self.request_document_refresh()
            return

        #
        # Continue the stroke along the sphere.
        #

        from sphere_math import arc_to_uv

        # last_uv / uv are texture coordinates, whose U axis is mirrored
        # relative to sphere_math's physical longitude coordinate.
        start = uv_to_vec((1.0 - self.last_uv[0]) % 1.0, self.last_uv[1])
        end = uv_to_vec((1.0 - uv[0]) % 1.0, uv[1])

        # Match the sampling distance to the current brush.  The previous
        # fixed quarter-degree step oversampled ordinary brushes heavily,
        # creating many redundant PIL stamps for a single mouse event.
        radius = max(0.5, int(self.app.size_var.get()) / 2)
        step_radians = (2 * np.pi * (radius * 0.25)) / self.texture.width
        samples = arc_to_uv(start, end, step_radians=step_radians)

        # The first sample is the previous event's endpoint, which has already
        # been painted.  Skipping it avoids an extra seam-wrapped stamp.
        for su, sv in samples[1:]:

            su = (1.0 - su) % 1.0

            # Stamp each spherical sample independently.  Connecting their
            # flat-map X coordinates would draw a line across the entire map
            # when the stroke crosses the longitude seam.
            self.stamp_spherical_brush((su, sv), refresh=False)

        self.last_uv = uv

        #
        # Update the globe after Main.py has modified
        # the document.
        #

        # Refresh each view once after all stamps from this mouse event have
        # been applied, rather than once per spherical sample.
        self.app.request_redraw()
        self.request_document_refresh()
