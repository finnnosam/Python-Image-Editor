import base64
import colorsys
import copy
import io
import json
import math
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageTk

from pypaint.display import DisplaySurface
from pypaint.effect_dialog import show as show_effect
from pypaint.extensions.blend_modes import composite as blend_composite
from pypaint.extensions.blend_modes import mode_labels as blend_mode_labels
from pypaint.extensions.blend_modes import normalize_mode
from pypaint.extensions.registry import builtins as builtin_operations
from pypaint.history import DocumentSnapshot, LayerRecord, history_for
from pypaint.jobs import Job, JobQueue
from pypaint.layer import Layer
from pypaint.layer import combine as combine_masks
from pypaint.layer import outline as selection_outline
from pypaint.native import read_native, save_native
from pypaint.pdn import PDNError, RasterLayer, read_pdn, write_pdn
from pypaint.platform import copy_image, paste_image, read_shortcuts, shortcut_label
from pypaint.rendering import (
    Renderer,
    _apply_layer_masks,
    _compositing_layer_indices,
    build_pyramids,
)
from pypaint.state import ChangeKind, DocumentContext, bind_state, context_for
from pypaint.surface import TiledSurface
from pypaint.tools import RasterGesture, flood_region
from pypaint.vectors import VectorSnapshot


def mask_lighter(a, b):
    return combine_masks(a, b, "lighter")


def mask_multiply(a, b):
    return combine_masks(a, b, "multiply")


from pypaint.platform import resource, shortcuts_path
from pypaint.rendering import render_vector_object
from pypaint.tools import (
    _apply_hardness_to_alpha,
    _accumulate_build_up_mask,
    _brush_ellipse_box,
    _brush_shape_mask,
    _composite_brush_shape,
    _connected_region_mask,
    _pencil_path,
)
from pypaint.vectors import *


class DelayedToolTip:
    """A small pointer-adjacent tooltip displayed after a hover delay."""

    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.pending = None
        self.window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self.pending = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self.pending is not None:
            self.widget.after_cancel(self.pending)
            self.pending = None

    def _show(self):
        self.pending = None
        if self.window is not None or not self.widget.winfo_containing(
            self.widget.winfo_pointerx(), self.widget.winfo_pointery()
        ):
            return
        pointer_x = self.widget.winfo_pointerx()
        pointer_y = self.widget.winfo_pointery()
        app = self.widget.winfo_toplevel()
        app.update_idletasks()
        app_left = app.winfo_rootx()
        app_top = app.winfo_rooty()
        app_right = app_left + app.winfo_width()
        app_bottom = app_top + app.winfo_height()
        margin = 4
        offset_x = 12
        offset_y = 16

        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.withdraw()
        tk.Label(
            self.window,
            text=self.text,
            padx=6,
            pady=3,
            relief="solid",
            borderwidth=1,
            background="#ffffe0",
            wraplength=max(1, app_right - app_left - 2 * margin - 14),
        ).pack()
        self.window.update_idletasks()
        width = self.window.winfo_reqwidth()
        height = self.window.winfo_reqheight()

        x = pointer_x + offset_x
        if x + width > app_right - margin:
            x = pointer_x - width - offset_x
        y = pointer_y + offset_y
        if y + height > app_bottom - margin:
            y = pointer_y - height - offset_y
        x = max(app_left + margin, min(x, app_right - width - margin))
        y = max(app_top + margin, min(y, app_bottom - height - margin))

        self.window.wm_geometry(f"+{x}+{y}")
        self.window.deiconify()
        self.window.lift()

    def _hide(self, _event=None):
        self._cancel()
        if self.window is not None:
            self.window.destroy()
            self.window = None


class PaintApp:
    GLOBE_BLOCKED_TOOLS = {
        "clone",
        "selection",
        "brush selection",
        "move",
        "move selection",
        "magic wand",
    }

    def __init__(self, root):
        self.root = root
        self.renderer = Renderer()
        self.jobs = JobQueue(max_bytes=1536 * 1024**2)
        self.job_callbacks = {}
        self.root.after(16, self._poll_jobs)
        self.root.title("PyPaint")

        self.doc_w = 1024  # this is what it launches with
        self.doc_h = 512
        self.current_file = None

        self.redraw_pending = False
        self.last_redraw = 0.0
        self.redraw_after_id = None
        self.zoom_preview_after_id = None
        self.zoom_redraw_after_id = None
        self.last_zoom_preview = 0.0
        self.overlay_after_id = None
        self.pending_redraw_box = None
        self.mipmap_after_id = None
        self.pending_mipmap_level = None
        self.mipmap_future = None
        self.mipmap_build_level = None
        self.main_view_dirty = False
        self.target_frame_time = 1 / 60.0  # FPS

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
        self._build_up_base_image = None
        self._build_up_coverage = None
        self.active_color_slot = "primary"
        self.picker_hue = 0.0
        self.picker_saturation = 0.0
        self.picker_value = 0.0
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
        self._checker_pil = None  # cached full-viewport tiled PIL image
        self._checker_pil_dims = None  # (cw, ch) it was built for
        self._canvas_image_id = None

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
        self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 0)
        self._selection_mask_bounds = None
        self.selection_operation = None
        self.selection_base_mask = None
        self.selection_edges = []
        self.selection_dash_offset = 0
        self.selection_animation_id = None
        self.move_start = None
        self.move_source_box = None
        self.move_pixels = None
        self.move_is_paste = False
        self.move_mask = None
        self.move_base_image = None
        self.move_selection_bounds = None
        self.move_selection_edges = None
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
        self.wand_pending = None
        self.layer_preview_after_id = None
        self.layer_preview_dirty = False
        self.raster_stroke_active = False
        self.layer_preview_cache = {}

        # Drag and drop variables
        self.drag_start_index = None
        self.drag_start_y = None

        # UI scale (1.5 = launch default)
        self.ui_scale = 1.5
        self.canvas_resize_anchor = "center"

        self.globe_documents = {}
        self.build_ui()
        self._initialize_documents()
        # Tk does not normally move keyboard focus when a label, frame, or
        # canvas background is clicked. Treat those clicks as "click away"
        # so entries commit through their existing <FocusOut> handlers.
        self.root.bind_all("<Button-1>", self._commit_active_entry, add="+")
        self.root.bind_all("<Button-1>", self._commit_pending_bucket_on_click, add="+")
        self.root.bind_all("<Button-3>", self._commit_pending_bucket_on_click, add="+")
        self.apply_ui_scale(self.ui_scale)  # A: apply 1.5× on launch
        self.refresh_layers()
        self.redraw()
        self.update_title()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _commit_active_entry(self, event):
        """Commit and unfocus an entry when the user clicks outside it."""
        try:
            focused = self.root.focus_get()
        except (KeyError, tk.TclError):
            # ttk.Combobox uses a temporary Tcl popdown window that is not in
            # Tkinter's Python widget tree. Clicking its list can briefly put
            # focus there, so there is no Entry widget for us to commit.
            return
        if focused is None or focused.winfo_class() not in {
            "Entry",
            "TEntry",
            "Spinbox",
            "TSpinbox",
        }:
            return

        clicked = event.widget
        while clicked is not None:
            if clicked == focused:
                return
            clicked = getattr(clicked, "master", None)

        # Attribute-table whitespace and an inactive scrollbar are neutral
        # surfaces. Keep the current cell focused instead of treating those
        # clicks as a request to return keyboard focus to the main editor.
        focused_window = focused.winfo_toplevel()
        clicked_window = event.widget.winfo_toplevel()
        if (
            focused_window == clicked_window
            and getattr(focused_window, "_neutral_background_clicks", False)
            and event.widget.winfo_class() not in {"Entry", "TEntry", "Spinbox", "TSpinbox"}
        ):
            return

        # Moving focus fires the field's FocusOut callback synchronously,
        # which validates and applies the edited value.  Put focus back on
        # the canvas, where plain-letter tool shortcuts are bound; focusing
        # only the root leaves those shortcuts inactive after editing a field.
        if clicked_window == self.root:
            self.canvas.focus_set()
        else:
            clicked_window.focus_set()

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
        menu_bar = tk.Frame(self.root, bd=1, relief="raised")
        menu_bar.pack(fill="x", side="top")

        image_button = tk.Menubutton(menu_bar, text="Image", padx=8, relief="flat")
        image_menu = tk.Menu(image_button, tearoff=False)
        image_menu.add_command(label="Canvas Size…", command=self.open_canvas_size)
        image_menu.add_command(label="Resize…", command=self.open_resize)
        image_button.configure(menu=image_menu)
        image_button.pack(side="left")

        edit_button = tk.Menubutton(menu_bar, text="Edit", padx=8, relief="flat")
        edit_menu = tk.Menu(edit_button, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_button.configure(menu=edit_menu)
        edit_button.pack(side="left")
        registry = builtin_operations()
        for label in ("Adjustments", "Effects"):
            button = tk.Menubutton(menu_bar, text=label, padx=8, relief="flat")
            menu = tk.Menu(button, tearoff=False)
            kind = "adjustment" if label == "Adjustments" else "effect"
            for operation in registry.operations.values():
                if operation.kind == kind:
                    menu.add_command(
                        label=operation.name + "…",
                        command=lambda operation=operation: show_effect(self, operation),
                    )
            for name, error in registry.errors.items():
                menu.add_command(label=f"{name}: {error}", state="disabled")
            button.configure(menu=menu)
            button.pack(side="left")
        # ── Top bar: file & edit actions ──────────────────────────────────
        top = tk.Frame(self.root, bd=1, relief="raised")
        top.pack(fill="x", side="top")

        toolbar_actions = (
            ("New", "new", self.new_project),
            ("Open", "open", self.open_project),
            ("Copy", "copy", self.copy_to_clipboard),
            ("Paste", "paste", self.paste_from_clipboard),
            ("Save", "save", self.save_project),
            ("Save As", "save-as", self.save_project_as),
            ("Export PNG", "export", self.save_image),
            ("Undo", "undo", self.undo),
            ("Globe View", "globe", self.open_globe_view),
            ("Settings", "settings", self.open_settings),
        )
        self.toolbar_icons = {}
        for label, name, command in toolbar_actions:
            with Image.open(resource("icons") / f"toolbar-{name}.png") as source:
                icon = ImageTk.PhotoImage(
                    source.convert("RGBA").resize((24, 24), Image.Resampling.LANCZOS)
                )
            self.toolbar_icons[name] = icon
            tk.Button(top, text=label, image=icon, compound="left", padx=5, command=command).pack(
                side="right" if name == "settings" else "left", padx=2, pady=2
            )

        self.left_panel_collapsed = False
        self.right_panel_collapsed = False
        self.top_panel_collapsed = False

        # ── Main area: tools | canvas | layers ────────────────────────────
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # ── Left panel: tools ─────────────────────────────────────────────
        left = tk.Frame(main, width=190, bd=1, relief="sunken")
        self.left_panel = left
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        left_header = tk.Frame(left)
        left_header.pack(fill="x", pady=(3, 2))
        self.left_panel_title = tk.Label(
            left_header, text="Tools", font=("TkDefaultFont", 9, "bold")
        )
        self.left_panel_title.pack(side="left", padx=(8, 2))
        self.left_panel_toggle = tk.Button(
            left_header, text="◀", width=2, command=self.toggle_left_panel
        )
        self.left_panel_toggle.pack(side="right", padx=3)

        self.left_panel_content = tk.Frame(left)
        self.left_panel_content.pack(fill="both", expand=True)

        tool_frame = tk.Frame(self.left_panel_content)
        tool_frame.pack(fill="x", padx=4)

        tools = [
            ("Selection", "selection", "selection.png"),
            ("Brush Selection", "brush selection", "brush-selection.png"),
            ("Move", "move", "move.png"),
            ("Move Selection", "move selection", "move-selection.png"),
            ("Pan", "pan", "pan.png"),
            ("Color Picker", "color picker", "color-picker.png"),
            ("Brush", "brush", "brush.png"),
            ("Eraser", "eraser", "eraser.png"),
            ("Clone", "clone", "clone.png"),
            ("Paint Bucket", "paint bucket", "paint-bucket.png"),
            ("Vector Select", "vector select", "selection.png"),
            ("Vector Edit", "vector edit", "vector-edit.png"),
            ("Line", "line", "line.png"),
            ("Rectangle", "rect", "rect.png"),
            ("Ellipse", "ellipse", "ellipse.png"),
            ("Point", "point", "point.png"),
            ("Magic Wand", "magic wand", "magic-wand.png"),
            ("Pencil", "pencil", "pencil.png"),
        ]
        icon_dir = resource("icons")
        self.tool_icons = {}
        self.tool_blocked_overlay = ImageTk.PhotoImage(
            Image.new("RGBA", (32, 32), (28, 28, 28, 255))
        )
        self.tool_buttons = {}
        self.tool_button_labels = {}
        self.tool_tooltips = []
        self.tools_by_layer_type = {
            "raster": (
                "selection",
                "move",
                "move selection",
                "brush selection",
                "pan",
                "color picker",
                "brush",
                "eraser",
                "clone",
                "paint bucket",
                "magic wand",
                "pencil",
            ),
            "vector": (
                "pan",
                "color picker",
                "vector select",
                "vector edit",
                "line",
                "rect",
                "ellipse",
                "point",
            ),
        }
        for index, (label, tool, filename) in enumerate(tools):
            with Image.open(icon_dir / filename) as source_image:
                icon_image = source_image.convert("RGBA").resize((32, 32), Image.Resampling.LANCZOS)
            icon = ImageTk.PhotoImage(icon_image)
            self.tool_icons[tool] = icon
            button = tk.Button(
                tool_frame,
                image=icon,
                width=42,
                height=42,
                command=lambda t=tool: self.set_tool(t),
                relief="sunken" if tool == self.tool else "raised",
                takefocus=True,
            )
            if tool == "selection":
                button.grid(row=0, column=1, padx=2, pady=2, sticky="w")
            elif tool == "move":
                button.grid(row=3, column=1, padx=2, pady=2, sticky="w")
            elif tool == "move selection":
                button.grid(row=4, column=1, padx=2, pady=2, sticky="w")
            elif tool == "brush selection":
                button.grid(row=1, column=1, padx=2, pady=2, sticky="w")
            elif tool == "magic wand":
                button.grid(row=2, column=1, padx=2, pady=2, sticky="w")
            elif tool == "pencil":
                button.grid(row=5, column=1, padx=2, pady=2, sticky="w")
            else:
                button.grid(row=index - 4, column=0, padx=2, pady=2, sticky="w")
            self.tool_buttons[tool] = button
            self.tool_button_labels[tool] = label
        self.tool_button_background = next(iter(self.tool_buttons.values())).cget("background")
        self.brush_build_up_var = tk.BooleanVar(value=False)
        self.brush_antialias_var = tk.BooleanVar(value=True)
        self.brush_hardness_var = tk.IntVar(value=75)
        self.brush_flow_var = tk.IntVar(value=25)
        self.brush_spacing_var = tk.StringVar(value="12.5")
        # Same-named tool options are universal. Each tool presents its own
        # relevant controls, but all of those controls reference these shared
        # variables and therefore stay synchronized automatically.
        self.clone_antialias_var = self.brush_antialias_var
        self.clone_build_up_var = self.brush_build_up_var
        self.clone_hardness_var = self.brush_hardness_var
        self.clone_flow_var = self.brush_flow_var
        self.clone_spacing_var = self.brush_spacing_var
        self.bucket_antialias_var = self.brush_antialias_var
        self.bucket_hardness_var = self.brush_hardness_var
        self.bucket_tolerance_var = tk.IntVar(value=0)
        self.picker_sample_area_var = tk.BooleanVar(value=False)
        self.vector_antialias_var = self.brush_antialias_var
        self.vector_hardness_var = self.brush_hardness_var
        # ── Color Selector ─────────────────────────────────────────────
        # Keep the whole group anchored to the bottom of the sidebar. The
        # unused height between Tools and Colors expands with the window.
        color_panel = tk.Frame(self.left_panel_content)
        color_panel.pack(side="bottom", fill="x", pady=(0, 6))

        ttk.Separator(color_panel, orient="horizontal").pack(fill="x", padx=4, pady=(0, 12))
        tk.Label(color_panel, text="Colors", font=("TkDefaultFont", 9, "bold")).pack(pady=(0, 2))

        color_frame = tk.Frame(color_panel)
        color_frame.pack(pady=4)

        # Primary color square (left)
        self.primary_square = tk.Canvas(
            color_frame,
            width=30,
            height=30,
            bg=self.primary_color,
            highlightthickness=3,
            highlightbackground="#2878d7",
        )
        self.primary_square.pack(side="left", padx=2)
        self.primary_square.bind("<Button-1>", lambda e: self.choose_primary_color())

        # Secondary color square (right)
        self.secondary_square = tk.Canvas(
            color_frame,
            width=30,
            height=30,
            bg=self.secondary_color,
            highlightthickness=3,
            highlightbackground="black",
        )
        self.secondary_square.pack(side="left", padx=2)
        self.secondary_square.bind("<Button-1>", lambda e: self.choose_secondary_color())

        # Swap colors button
        tk.Button(color_panel, text="↔", width=3, command=self.swap_colors).pack(pady=(0, 6))

        controls = tk.Frame(color_panel)
        controls.pack(fill="x", padx=8, pady=(4, 0))
        self.rgb_vars = [tk.IntVar(value=0) for _ in range(3)]
        self.hsv_vars = [tk.IntVar(value=0), tk.IntVar(value=0), tk.IntVar(value=0)]
        self.opacity_var = tk.IntVar(value=255)
        self.hex_var = tk.StringVar(value="000000")
        self.color_sliders = []

        tk.Label(controls, text="RGB", anchor="w").pack(fill="x")
        for label, variable in zip(("R", "G", "B"), self.rgb_vars):
            self._add_color_slider(controls, label, variable, 0, 255, self._rgb_controls_changed)

        hex_row = tk.Frame(controls)
        hex_row.pack(fill="x", pady=(2, 3))
        tk.Label(hex_row, text="Hex:", width=4, anchor="w").pack(side="left")
        self.hex_entry = tk.Entry(hex_row, textvariable=self.hex_var, width=8, justify="right")
        self.hex_entry.pack(side="right")
        self.hex_entry.bind("<Return>", self._hex_control_changed)
        self.hex_entry.bind("<FocusOut>", self._hex_control_changed)

        tk.Label(controls, text="HSV", anchor="w").pack(fill="x")
        for index, (label, variable, maximum) in enumerate(
            zip(("H", "S", "V"), self.hsv_vars, (359, 100, 100))
        ):
            self._add_color_slider(
                controls,
                label,
                variable,
                0,
                maximum,
                lambda component=index: self._hsv_controls_changed(component),
            )

        tk.Label(controls, text="Opacity", anchor="w").pack(fill="x", pady=(3, 0))
        self._add_color_slider(
            controls, "A", self.opacity_var, 0, 255, self._opacity_control_changed
        )

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
            self._apply_selected_vector_size()

        def adjust_size(delta):
            """Adjust tool size by delta while preserving its valid range."""
            try:
                current = int(self.size_var.get())
            except ValueError:
                current = 2
            self.size_var.set(str(max(1, min(999, current + delta))))
            self._apply_selected_vector_size()
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

        self.top_workspace_panel = tk.Frame(self.view_workspace)
        self.top_workspace_panel.pack(side="top", fill="x")

        self.view_tabs = tk.Frame(self.top_workspace_panel, bd=1, relief="raised")
        self.view_tabs.pack(side="top", fill="x")
        self.top_panel_toggle = tk.Button(
            self.view_tabs, text="▲", width=2, command=self.toggle_top_panel
        )
        self.top_panel_toggle.pack(side="right", padx=3, pady=2)

        # Keep tool settings in a stable horizontal strip below the view tabs.
        # The fixed-height outer frame remains visible even for tools without
        # options, preventing the canvas from changing size as tools change.
        self.tool_settings_bar = tk.Frame(
            self.top_workspace_panel, height=44, bd=1, relief="groove"
        )
        self.tool_settings_bar.pack(side="top", fill="x")
        self.tool_settings_bar.pack_propagate(False)
        tk.Label(
            self.tool_settings_bar, text="Tool settings:", font=("TkDefaultFont", 9, "bold")
        ).pack(side="left", padx=(8, 10))

        self.size_frame = tk.Frame(self.tool_settings_bar)
        self.size_label = tk.Label(self.size_frame, text="Size:")
        self.size_label.pack(side="left")
        self.size_entry = tk.Entry(self.size_frame, width=6, textvariable=self.size_var)
        self.size_entry.pack(side="left", padx=(4, 3))
        self.size_entry.bind("<Return>", update_size)
        self.size_entry.bind("<FocusOut>", update_size)
        self.size_minus_button = tk.Button(
            self.size_frame, text="−", width=2, command=lambda: adjust_size(-1)
        )
        self.size_minus_button.pack(side="left")
        self.size_minus_button.bind("<Control-Button-1>", adjust_size_with_control(-1))
        self.size_plus_button = tk.Button(
            self.size_frame, text="+", width=2, command=lambda: adjust_size(1)
        )
        self.size_plus_button.pack(side="left", padx=(2, 10))
        self.size_plus_button.bind("<Control-Button-1>", adjust_size_with_control(1))

        def scroll_size(event):
            if event.delta:
                adjust_size(1 if event.delta > 0 else -1)
            return "break"

        for widget in (
            self.size_frame,
            self.size_label,
            self.size_entry,
            self.size_minus_button,
            self.size_plus_button,
        ):
            widget.bind("<MouseWheel>", scroll_size)

        self.tool_setting_tooltips = []

        def add_percentage_control(frame, label, variable, default):
            tk.Label(frame, text=f"{label}:").pack(side="left", padx=(10, 3))
            tk.Scale(
                frame,
                from_=0,
                to=100,
                orient="horizontal",
                length=90,
                showvalue=False,
                variable=variable,
            ).pack(side="left")
            entry = tk.Entry(frame, width=4, justify="right")
            entry.insert(0, str(variable.get()))
            entry.pack(side="left", padx=(3, 2))

            def sync_entry(*_):
                try:
                    value = int(variable.get())
                except (tk.TclError, ValueError):
                    return
                if entry.get() != str(value):
                    entry.delete(0, "end")
                    entry.insert(0, str(value))

            def commit_entry(_event=None):
                try:
                    value = int(entry.get())
                except ValueError:
                    value = default
                variable.set(max(0, min(100, value)))
                sync_entry()

            variable.trace_add("write", sync_entry)
            entry.bind("<Return>", commit_entry)
            entry.bind("<FocusOut>", commit_entry)
            reset = tk.Button(frame, text="⤺", width=2, command=lambda: variable.set(default))
            reset.pack(side="left")
            self.tool_setting_tooltips.append(DelayedToolTip(reset, "reset to default"))

        self.brush_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.brush_settings_frame, text="Build up", variable=self.brush_build_up_var
        ).pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            self.brush_settings_frame, text="Anti-alias", variable=self.brush_antialias_var
        ).pack(side="left")
        add_percentage_control(
            self.brush_settings_frame, "Hardness", self.brush_hardness_var, 75
        )
        add_percentage_control(self.brush_settings_frame, "Flow", self.brush_flow_var, 25)

        self.clone_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.clone_settings_frame, text="Build up", variable=self.clone_build_up_var
        ).pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            self.clone_settings_frame, text="Anti-alias", variable=self.clone_antialias_var
        ).pack(side="left")
        add_percentage_control(
            self.clone_settings_frame, "Hardness", self.clone_hardness_var, 75
        )
        add_percentage_control(self.clone_settings_frame, "Flow", self.clone_flow_var, 25)

        self.bucket_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.bucket_settings_frame,
            text="Anti-alias",
            variable=self.bucket_antialias_var,
            command=self._refresh_bucket_preview,
        ).pack(side="left")
        tk.Label(self.bucket_settings_frame, text="Hardness:").pack(side="left", padx=(10, 3))
        tk.Scale(
            self.bucket_settings_frame,
            from_=0,
            to=100,
            orient="horizontal",
            length=110,
            variable=self.bucket_hardness_var,
            command=lambda value: self._refresh_bucket_preview(),
        ).pack(side="left")
        tk.Button(
            self.bucket_settings_frame, text="Reset", command=self._reset_bucket_hardness
        ).pack(side="left", padx=(3, 0))
        tk.Label(self.bucket_settings_frame, text="Tolerance:").pack(side="left", padx=(10, 3))
        tk.Scale(
            self.bucket_settings_frame,
            from_=0,
            to=100,
            orient="horizontal",
            length=110,
            variable=self.bucket_tolerance_var,
            command=lambda value: self._refresh_bucket_preview(),
        ).pack(side="left")
        tk.Button(
            self.bucket_settings_frame, text="Reset", command=self._reset_bucket_tolerance
        ).pack(side="left", padx=(3, 0))
        self.wand_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Label(self.wand_settings_frame, text="Tolerance:").pack(side="left")
        tk.Scale(
            self.wand_settings_frame,
            from_=0,
            to=100,
            orient="horizontal",
            length=110,
            variable=self.bucket_tolerance_var,
        ).pack(side="left")
        tk.Button(
            self.wand_settings_frame, text="Reset", command=lambda: self.bucket_tolerance_var.set(0)
        ).pack(side="left")
        self.bucket_tolerance_var.trace_add("write", lambda *_: self._refresh_wand_selection())
        tk.Label(self.clone_settings_frame, text="Spacing:").pack(side="left", padx=(10, 3))
        self.clone_spacing_entry = tk.Entry(
            self.clone_settings_frame, width=5, textvariable=self.clone_spacing_var
        )
        self.clone_spacing_entry.pack(side="left")
        tk.Label(self.clone_settings_frame, text="%").pack(side="left")
        tk.Label(self.brush_settings_frame, text="Spacing:").pack(side="left", padx=(10, 3))
        self.brush_spacing_entry = tk.Entry(
            self.brush_settings_frame, width=5, textvariable=self.brush_spacing_var
        )
        self.brush_spacing_entry.pack(side="left")
        tk.Label(self.brush_settings_frame, text="%").pack(side="left")

        self.picker_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.picker_settings_frame,
            text="Sample area",
            variable=self.picker_sample_area_var,
            command=self.update_tool_settings_visibility,
        ).pack(side="left")

        self.vector_settings_frame = tk.Frame(self.tool_settings_bar)
        tk.Checkbutton(
            self.vector_settings_frame, text="Anti-alias", variable=self.vector_antialias_var
        ).pack(side="left")
        tk.Label(self.vector_settings_frame, text="Hardness:").pack(side="left", padx=(10, 3))
        tk.Scale(
            self.vector_settings_frame,
            from_=0,
            to=100,
            orient="horizontal",
            length=110,
            variable=self.vector_hardness_var,
        ).pack(side="left")
        tk.Button(
            self.vector_settings_frame,
            text="Reset",
            command=lambda: self.vector_hardness_var.set(75),
        ).pack(side="left", padx=(3, 0))

        self.vector_select_settings_frame = tk.Frame(self.tool_settings_bar)
        self.vector_points_var = tk.StringVar(value="Points: —")
        tk.Label(
            self.vector_select_settings_frame,
            textvariable=self.vector_points_var,
            anchor="w",
            justify="left",
        ).pack(side="left", padx=(8, 0))

        def validate_spacing(variable):
            try:
                value = max(0.1, min(1000, float(variable.get())))
            except ValueError:
                value = 12.5
            variable.set(f"{value:g}")

        for entry, variable in (
            (self.brush_spacing_entry, self.brush_spacing_var),
            (self.clone_spacing_entry, self.clone_spacing_var),
        ):
            entry.bind("<Return>", lambda event, var=variable: validate_spacing(var))
            entry.bind("<FocusOut>", lambda event, var=variable: validate_spacing(var))

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

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.mouse_move)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<Button-3>", self.on_mouse_down)
        self.canvas.bind("<B3-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Plain scroll for panning
        self.canvas.bind("<Control-MouseWheel>", self.zoom_mouse)  # Ctrl+scroll for zoom
        self._load_keyboard_shortcuts(adjust_size)
        self.canvas.focus_set()
        self.update_tool_settings_visibility()
        self.switch_view("main")

        # ── Right panel: layers ───────────────────────────────────────────
        right = tk.Frame(main, width=250, bd=1, relief="sunken")
        self.right_panel = right
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        right_header = tk.Frame(right)
        right_header.pack(fill="x", pady=(3, 2))
        self.right_panel_toggle = tk.Button(
            right_header, text="▶", width=2, command=self.toggle_right_panel
        )
        self.right_panel_toggle.pack(side="left", padx=3)
        self.right_panel_title = tk.Label(
            right_header, text="Layers", font=("TkDefaultFont", 9, "bold")
        )
        self.right_panel_title.pack(side="right", padx=(2, 8))

        self.right_panel_content = tk.Frame(right)
        self.right_panel_content.pack(fill="both", expand=True)

        self.layer_style = ttk.Style()
        self.layer_style.configure("Layer.Treeview", rowheight=44)
        self.layer_selected_vector_color = (
            self.layer_style.lookup("Treeview", "background", ("selected",)) or "#4a6984"
        )
        self.layer_selected_text_color = (
            self.layer_style.lookup("Treeview", "foreground", ("selected",)) or "#ffffff"
        )
        self.layer_selected_raster_color = "#c94f4f"
        self.layer_list = ttk.Treeview(
            self.right_panel_content, show="tree", selectmode="browse", style="Layer.Treeview"
        )
        self.layer_list.pack(fill="both", expand=True, padx=4)
        self.layer_list.column("#0", stretch=True, width=220)
        self.layer_list.tag_configure("raster", background="#f7dddd")
        self.layer_list.tag_configure("vector", background="#dcecff")
        self.layer_list.bind("<<TreeviewSelect>>", self.select_layer)
        self.layer_list.bind("<Button-1>", self.on_layer_pointer_down)
        self.layer_list.bind("<Button-3>", self.show_layer_properties)
        self.layer_list.bind("<B1-Motion>", self.on_layer_drag)
        self.layer_list.bind("<ButtonRelease-1>", self.on_layer_drag_end)
        self.layer_control_tooltip = DelayedToolTip(self.layer_list, "")
        self.layer_list.bind("<Motion>", self._update_layer_control_tooltip, add="+")

        self.layer_row_icons = {}
        row_sources = {}
        for name in ("layer-visible", "layer-hidden", "layer-raster", "layer-vector"):
            with Image.open(icon_dir / f"{name}.png") as row_icon:
                row_sources[name] = row_icon.convert("RGBA").resize(
                    (24, 24), Image.Resampling.LANCZOS
                )
        for visible in (True, False):
            for layer_type in ("raster", "vector"):
                row_image = Image.new("RGBA", (88, 28), (0, 0, 0, 0))
                row_draw = ImageDraw.Draw(row_image)
                row_draw.rounded_rectangle(
                    (31, 0, 58, 27),
                    radius=4,
                    fill=(232, 232, 232, 255),
                    outline=(135, 135, 135, 255),
                )
                visibility_name = "layer-visible" if visible else "layer-hidden"
                # Visibility always occupies the second control slot so raster
                # and vector eye icons line up vertically.
                visibility_x = 33
                row_image.alpha_composite(row_sources[visibility_name], (visibility_x, 2))
                if layer_type == "vector":
                    row_draw.rounded_rectangle(
                        (0, 0, 27, 27),
                        radius=4,
                        fill=(232, 232, 232, 255),
                        outline=(135, 135, 135, 255),
                    )
                    # A tiny attribute-table glyph drawn at native row size.
                    for table_x in (5, 13, 21):
                        row_draw.line((table_x, 5, table_x, 22), fill=(55, 85, 105, 255), width=1)
                    for table_y in (5, 11, 17, 22):
                        row_draw.line((5, table_y, 22, table_y), fill=(55, 85, 105, 255), width=1)
                row_image.alpha_composite(row_sources[f"layer-{layer_type}"], (64, 2))
                self.layer_row_icons[(visible, layer_type)] = row_image

        layer_actions = [
            ("Add Raster Layer", "add-raster-layer.png", lambda: self.add_layer("raster")),
            ("Add Vector Layer", "add-vector-layer.png", lambda: self.add_layer("vector")),
            ("Delete Layer", "delete-layer.png", self.delete_layer),
            ("Duplicate Layer", "toolbar-copy.png", self.duplicate_layer),
            ("Toggle Visibility", "toggle-visibility.png", self.toggle_visibility),
            ("Move Layer Up", "move-layer-up.png", self.move_layer_up),
            ("Move Layer Down", "move-layer-down.png", self.move_layer_down),
        ]
        action_frame = tk.Frame(self.right_panel_content)
        action_frame.pack(padx=4, pady=4)
        self.layer_action_icons = {}
        self.layer_action_tooltips = []
        for index, (label, filename, command) in enumerate(layer_actions):
            with Image.open(icon_dir / filename) as icon_image:
                icon_image = icon_image.convert("RGBA").resize((24, 24), Image.Resampling.LANCZOS)
            icon = ImageTk.PhotoImage(icon_image)
            self.layer_action_icons[filename] = icon
            button = tk.Button(
                action_frame, image=icon, width=26, height=26, command=command, takefocus=True
            )
            button.grid(row=0, column=index, padx=1, pady=1)
            self.layer_action_tooltips.append(DelayedToolTip(button, label))

    @staticmethod
    def _union_boxes(first, second):
        if first is None:
            return second
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    def request_redraw(self, defer=False, dirty_box=None, *, navigation=False):
        self._schedule_layer_previews()
        if hasattr(self, "active_view") and self.active_view != "main":
            if getattr(self, "_display_surface", None) is not None:
                self._display_surface.prefetch.cancel()
            self.main_view_dirty = True
            return

        if self.redraw_after_id is not None:
            # None means that a full refresh is already pending.
            if self.pending_redraw_box is not None and dirty_box is not None:
                self.pending_redraw_box = self._union_boxes(self.pending_redraw_box, dirty_box)
            elif dirty_box is None:
                self.pending_redraw_box = None
            return

        now = time.perf_counter()
        frame_time = 1 / 30.0 if defer and not navigation else self.target_frame_time
        delay = max(0.0, frame_time - (now - self.last_redraw))
        if navigation:
            # Keep pan/wheel callbacks short, coalescing at the normal frame
            # interval without the slower stroke-preview cadence.
            delay = max(delay, 0.001)
        elif defer:
            delay = max(delay, 0.008)
        if delay == 0:
            self.last_redraw = now
            self.redraw(dirty_box)
        else:
            # Keep a trailing redraw.  Merely discarding requests received
            # inside the frame interval makes fast drags look jerky and can
            # leave the canvas one event behind the document.
            self.pending_redraw_box = dirty_box
            self.redraw_after_id = self.root.after(
                max(1, math.ceil(delay * 1000)), self._scheduled_redraw
            )

    def _scheduled_redraw(self):
        self.redraw_after_id = None
        self.last_redraw = time.perf_counter()
        dirty_box = self.pending_redraw_box
        self.pending_redraw_box = None
        self.redraw(dirty_box)

    def request_overlay_redraw(self):
        """Pointer movement changes overlays, not document pixels."""
        if self.active_view != "main" or self.overlay_after_id is not None:
            return
        self.overlay_after_id = self.root.after(16, self._scheduled_overlay_redraw)

    def _scheduled_overlay_redraw(self):
        self.overlay_after_id = None
        if self.active_view == "main" and self.redraw_after_id is None:
            self._draw_overlays()

    def request_mipmap_level(self, level):
        """Build a missing zoom level after wheel input has settled.

        Constructing a large level in the wheel callback makes zooming hitch.
        Until this callback runs, compositing uses the nearest cached level.
        """
        if self.mipmap_future is not None:
            self.pending_mipmap_level = max(level, self.pending_mipmap_level or 0)
            return
        if self.mipmap_after_id is not None and self.pending_mipmap_level == level:
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
        document = context_for(self).document
        snapshot = DocumentSnapshot.capture(document)
        inputs = []
        originals = {}
        for index in self._compositing_layer_indices(self.layers):
            layer = self.layers[index]
            if len(layer._mipmaps) > level:
                continue
            originals[layer.id] = tuple(layer._mipmaps)
            cached = tuple(
                image.snapshot()
                if isinstance(image, TiledSurface)
                else TiledSurface.from_image(image).snapshot()
                for image in layer._mipmaps[1:]
            )
            inputs.append(
                (layer.id, layer._mipmap_revision, snapshot.size, snapshot.layers[index], cached)
            )
        if not inputs:
            return
        job = Job(
            document.id, document.generation, document.state_id, preview_key="mipmaps", priority=20
        )
        renderer = self._renderer()

        def publish(result):
            self.mipmap_future = None
            if result.error:
                messagebox.showerror("Preview failed", str(result.error))
                return
            layers = {layer.id: layer for layer in document.layers}
            for layer_id, revision, pyramid in result.value:
                layer = layers.get(layer_id)
                if layer is None or layer._mipmap_revision != revision:
                    continue
                # Reuse verified existing cache objects; new levels are owned
                # writable copies, so subsequent incremental updates are local.
                old = originals[layer_id]
                layer._mipmaps = list(old) + [image.copy() for image in pyramid[len(old) :]]
                if not old and layer.vector_data is not None:
                    layer._image = layer._mipmaps[0]
            if context_for(self).document is document:
                self.request_redraw(defer=True)
            if self.pending_mipmap_level is not None:
                pending = self.pending_mipmap_level
                self.pending_mipmap_level = None
                self.request_mipmap_level(pending)

        try:
            self.jobs.submit(
                job, lambda token: build_pyramids(inputs, level, token, renderer), 48 * 1024**2
            )
            self.mipmap_future = job.operation_id
            self.job_callbacks[job.operation_id] = publish
        except RuntimeError:
            self.pending_mipmap_level = level
            self.mipmap_after_id = self.root.after(140, self._build_pending_mipmaps)

    def _poll_mipmap_build(self):
        # Compatibility entry point: the single UI-owned job poller publishes all work.
        self._poll_jobs()

    def choose_primary_color(self):
        self.active_color_slot = "primary"
        self._sync_picker_to_active_color()

    def choose_secondary_color(self):
        self.active_color_slot = "secondary"
        self._sync_picker_to_active_color()

    def toggle_color_focus(self):
        """Switch which color swatch is edited without swapping the colors."""
        self.active_color_slot = (
            "secondary" if self.active_color_slot == "primary" else "primary"
        )
        self._sync_picker_to_active_color()

    @staticmethod
    def _hex_to_rgb(color):
        color = color.lstrip("#")
        return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _add_color_slider(self, parent, label, variable, minimum, maximum, callback):
        row = tk.Frame(parent)
        row.pack(fill="x")
        tk.Label(row, text=f"{label}:", width=2, anchor="w").pack(side="left")
        if label == "H":
            slider = tk.Canvas(
                row, width=92, height=14, bd=0, highlightthickness=0, cursor="sb_h_double_arrow"
            )
            slider.pack(side="left", fill="x", expand=True)
            slider.bind(
                "<Button-1>",
                lambda event, control=slider, value=variable, changed=callback: (
                    self._set_hue_slider_from_pointer(control, value, event, changed)
                ),
            )
            slider.bind(
                "<B1-Motion>",
                lambda event, control=slider, value=variable, changed=callback: (
                    self._set_hue_slider_from_pointer(control, value, event, changed)
                ),
            )
            variable.trace_add(
                "write",
                lambda *args, control=slider, value=variable: self._render_hue_slider(
                    control, value
                ),
            )
            slider.bind(
                "<Configure>",
                lambda event, control=slider, value=variable: self._render_hue_slider(
                    control, value
                ),
            )
            slider.after_idle(lambda: self._render_hue_slider(slider, variable))
        else:
            slider = tk.Canvas(
                row, width=92, height=14, bd=0, highlightthickness=0, cursor="sb_h_double_arrow"
            )
            slider._color_component = label
            slider._minimum = minimum
            slider._maximum = maximum
            slider._variable = variable
            slider.pack(side="left", fill="x", expand=True)
            slider.bind(
                "<Button-1>",
                lambda event, control=slider, value=variable, changed=callback: (
                    self._set_color_slider_from_pointer(control, value, event, changed)
                ),
            )
            slider.bind(
                "<B1-Motion>",
                lambda event, control=slider, value=variable, changed=callback: (
                    self._set_color_slider_from_pointer(control, value, event, changed)
                ),
            )
            variable.trace_add(
                "write", lambda *args, control=slider: self._render_color_slider(control)
            )
            slider.bind(
                "<Configure>", lambda event, control=slider: self._render_color_slider(control)
            )
            slider.after_idle(lambda control=slider: self._render_color_slider(control))
        spinner = tk.Spinbox(
            row,
            textvariable=variable,
            from_=minimum,
            to=maximum,
            width=4,
            justify="right",
            command=callback,
        )
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
        slider.create_line(marker_x, 0, marker_x, height, fill="white", width=3)
        slider.create_line(marker_x, 0, marker_x, height, fill="black")

    def _set_hue_slider_from_pointer(self, slider, variable, event, callback):
        width = max(2, slider.winfo_width())
        fraction = min(1.0, max(0.0, event.x / (width - 1)))
        variable.set(round(fraction * 359))
        callback()
        return "break"

    def _render_color_slider(self, slider):
        """Draw a component's true color range behind its position marker."""
        width = max(2, slider.winfo_width())
        height = max(2, slider.winfo_height())
        component = slider._color_component
        current_rgb = self._hex_to_rgb(
            self.primary_color if self.active_color_slot == "primary" else self.secondary_color
        )
        image = Image.new("RGB", (width, height))
        pixels = image.load()
        checker = ((235, 235, 235), (195, 195, 195))

        for x in range(width):
            fraction = x / (width - 1)
            if component in ("R", "G", "B"):
                color = list(current_rgb)
                color[("R", "G", "B").index(component)] = round(fraction * 255)
                color = tuple(color)
            elif component == "S":
                hsv = (self.picker_hue, fraction, getattr(self, "picker_value", 0.0))
                color = tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(*hsv))
            elif component == "V":
                hsv = (self.picker_hue, getattr(self, "picker_saturation", 0.0), fraction)
                color = tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(*hsv))
            else:  # Alpha previews transparency over a checkerboard.
                color = current_rgb
            for y in range(height):
                if component == "A":
                    background = checker[((x // 5) + (y // 5)) % 2]
                    pixels[x, y] = tuple(
                        round(foreground * fraction + backdrop * (1 - fraction))
                        for foreground, backdrop in zip(color, background)
                    )
                else:
                    pixels[x, y] = color

        slider._gradient_image = ImageTk.PhotoImage(image)
        slider.delete("all")
        slider.create_image(0, 0, image=slider._gradient_image, anchor="nw")
        try:
            value = int(slider._variable.get())
        except (tk.TclError, ValueError):
            value = slider._minimum
        span = max(1, slider._maximum - slider._minimum)
        marker_x = ((value - slider._minimum) / span) * (width - 1)
        slider.create_line(marker_x, 0, marker_x, height, fill="white", width=3)
        slider.create_line(marker_x, 0, marker_x, height, fill="black")

    def _refresh_color_slider_previews(self):
        for slider in self.color_sliders:
            if getattr(slider, "_color_component", None):
                self._render_color_slider(slider)

    def _set_color_slider_from_pointer(self, slider, variable, event, callback):
        """Set a gradient slider directly from its pointer position."""
        fraction = event.x / max(1.0, slider.winfo_width() - 1)
        fraction = min(1.0, max(0.0, fraction))
        variable.set(round(slider._minimum + fraction * (slider._maximum - slider._minimum)))
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
        rgb = tuple(self._clamp_control(variable, 0, 255) for variable in self.rgb_vars)
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
            saturation = self._clamp_control(self.hsv_vars[1], 0, 100) / 100
        if component in (None, 2):
            value = self._clamp_control(self.hsv_vars[2], 0, 100) / 100
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        self.picker_hue = hue
        self.picker_saturation = saturation
        self.picker_value = value
        self._set_selected_color(
            self._rgb_to_hex(tuple(round(channel * 255) for channel in rgb)), preserve_hsv=True
        )

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
        self._apply_selected_vector_color(self.active_color_slot)
        self.request_redraw()

    def _set_selected_color(self, color, preserve_hsv=False):
        if self.active_color_slot == "primary":
            self.primary_color = color
            self.color = color
            self.primary_square.config(bg=color)
        else:
            self.secondary_color = color
            self.secondary_square.config(bg=color)
        self._sync_picker_to_active_color(preserve_hsv=preserve_hsv)
        self._apply_selected_vector_color(self.active_color_slot)
        self.request_redraw()

    def _color_with_opacity(self, slot):
        if slot == "primary":
            color, opacity = self.primary_color, self.primary_opacity
        else:
            color, opacity = self.secondary_color, self.secondary_opacity
        return color if opacity == 255 else f"{color}{opacity:02x}"

    def _sync_picker_to_active_color(self, preserve_hsv=False):
        color = self.primary_color if self.active_color_slot == "primary" else self.secondary_color
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
            highlightbackground="#2878d7" if self.active_color_slot == "primary" else "black"
        )
        self.secondary_square.config(
            highlightbackground="#2878d7" if self.active_color_slot == "secondary" else "black"
        )
        opacity = (
            self.primary_opacity if self.active_color_slot == "primary" else self.secondary_opacity
        )
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
        self._refresh_color_slider_previews()

    def swap_colors(self):
        self.primary_color, self.secondary_color = self.secondary_color, self.primary_color
        self.primary_opacity, self.secondary_opacity = (
            self.secondary_opacity,
            self.primary_opacity,
        )
        self.color = self.primary_color
        self.primary_square.config(bg=self.primary_color)
        self.secondary_square.config(bg=self.secondary_color)
        self._sync_picker_to_active_color()
        self._apply_selected_vector_color("primary")
        self._apply_selected_vector_color("secondary")
        self.request_redraw()

    def _panel_layout_changed(self):
        """Refresh the active view after a panel changes the available space."""
        self.root.update_idletasks()
        self.request_redraw()
        globe = getattr(self, "globe_window", None)
        if globe is not None and self.active_view in self.globe_documents:
            try:
                globe.notify_document_changed()
            except tk.TclError:
                self.globe_window = None

    def toggle_left_panel(self):
        """Collapse or restore the shared tools and colors column."""
        self.left_panel_collapsed = not self.left_panel_collapsed
        if self.left_panel_collapsed:
            self.left_panel_content.pack_forget()
            self.left_panel_title.pack_forget()
            self.left_panel.configure(width=42)
            self.left_panel_toggle.configure(text="▶")
        else:
            self.left_panel.configure(width=190)
            self.left_panel_title.pack(side="left", padx=(8, 2))
            self.left_panel_content.pack(fill="both", expand=True)
            self.left_panel_toggle.configure(text="◀")
        self._panel_layout_changed()

    def toggle_right_panel(self):
        """Collapse or restore the shared layers column."""
        self.right_panel_collapsed = not self.right_panel_collapsed
        if self.right_panel_collapsed:
            self.right_panel_content.pack_forget()
            self.right_panel_title.pack_forget()
            self.right_panel.configure(width=42)
            self.right_panel_toggle.configure(text="◀")
        else:
            self.right_panel.configure(width=250)
            self.right_panel_title.pack(side="right", padx=(2, 8))
            self.right_panel_content.pack(fill="both", expand=True)
            self.right_panel_toggle.configure(text="▶")
        self._panel_layout_changed()

    def toggle_top_panel(self):
        """Collapse or restore the view tabs and tool-settings row."""
        self.top_panel_collapsed = not self.top_panel_collapsed
        if self.top_panel_collapsed:
            for tab in self.view_tab_widgets.values():
                tab.pack_forget()
            self.tool_settings_bar.pack_forget()
            self.top_panel_toggle.configure(text="▼")
        else:
            for tab in self.view_tab_widgets.values():
                tab.pack(side="left", padx=2, pady=2)
            self.tool_settings_bar.pack(side="top", fill="x")
            self.top_panel_toggle.configure(text="▲")
        self._panel_layout_changed()

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.resizable(False, False)
        win.grab_set()  # modal

        tk.Label(win, text="UI Scale", font=("TkDefaultFont", 9, "bold")).pack(pady=(12, 2))
        tk.Label(
            win,
            text="Adjusts the size of all text and widgets.\nTakes effect immediately.",
            justify="center",
        ).pack(padx=16)

        scale_var = tk.DoubleVar(value=self.ui_scale)
        slider = tk.Scale(
            win,
            variable=scale_var,
            from_=0.5,
            to=2.5,
            resolution=0.05,
            orient="horizontal",
            length=260,
            label="Scale factor",
        )
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

        tk.Button(btn_row, text="Apply", command=apply).pack(side="left", padx=4)
        tk.Button(btn_row, text="OK", command=ok).pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="left", padx=4)

    def register_view(self, view_id, label, widget, closable=True):
        """Add an embedded workspace view and its compact tab."""
        self.views[view_id] = widget
        if view_id == "main":
            return
        tab = tk.Frame(self.view_tabs)
        tk.Button(tab, text=label, bd=0, command=lambda key=view_id: self.switch_view(key)).pack(
            side="left"
        )
        if closable:
            tk.Button(
                tab, text="×", bd=0, padx=4, command=lambda key=view_id: self.close_view(key)
            ).pack(side="left")
        if not self.top_panel_collapsed:
            tab.pack(side="left", padx=2, pady=2)
        self.view_tab_widgets[view_id] = tab

    def _load_keyboard_shortcuts(self, adjust_size):
        actions = {
            "cycle_selection": self.select_selection_tool,
            "cycle_move": self.select_move_tool,
            "cycle_vector_shapes": self.select_vector_shape_tool,
            "increase_size": lambda: adjust_size(1),
            "decrease_size": lambda: adjust_size(-1),
            "undo": self.undo,
            "redo": self.redo,
            "copy": self.copy_to_clipboard,
            "paste": self.paste_from_clipboard,
            "select_all": self.select_all,
            "zoom_to_selection": self.zoom_to_selection,
            "save": self.save_project,
            "new": self.new_project,
            "open": self.open_project,
            "commit_fill": self._finish_bucket_preview,
            "close_image": lambda: self.close_document(self.active_document),
            "swap_colors": self.swap_colors,
            "focus_colors": self.toggle_color_focus,
            "zoom_in": lambda: self.zoom_keyboard(1),
            "zoom_out": lambda: self.zoom_keyboard(-1),
        }
        for name, tool in {
            "pan": "pan",
            "color_picker": "color picker",
            "brush": "brush",
            "pencil": "pencil",
            "eraser": "eraser",
            "clone": "clone",
            "paint_bucket": "paint bucket",
            "vector_edit": "vector edit",
            "line": "line",
            "rectangle": "rect",
            "ellipse": "ellipse",
            "selection": "selection",
            "brush_selection": "brush selection",
            "magic_wand": "magic wand",
            "move": "move",
            "move_selection": "move selection",
        }.items():
            actions[name] = lambda tool=tool: self.set_tool(tool)
        self.canvas.bind("<Control-y>", lambda event: self.redo())
        self.canvas.bind("<Escape>", self.cancel_gesture)
        path = shortcuts_path()
        try:
            bindings, errors = read_shortcuts(path, actions)
        except (OSError, UnicodeError) as error:
            bindings, errors = [], [str(error)]
        # Carry installations with the previous untouched defaults forward:
        # C used to select Clone and L used to select Line. Explicit custom
        # configurations, including any file that names focus_colors, remain
        # authoritative.
        configured_actions = {action for action, _sequence in bindings}
        legacy_defaults = {
            action: sequence for action, sequence in bindings if action in {"clone", "line"}
        }
        if (
            "focus_colors" not in configured_actions
            and legacy_defaults.get("clone") == "<KeyPress-c>"
            and legacy_defaults.get("line") == "<KeyPress-l>"
        ):
            bindings = [
                (action, sequence)
                for action, sequence in bindings
                if action not in {"clone", "line"}
            ]
            bindings.extend(
                (("clone", "<KeyPress-l>"), ("focus_colors", "<KeyPress-c>"))
            )
        configured_actions = {action for action, _sequence in bindings}
        legacy_shapes = {
            action: sequence
            for action, sequence in bindings
            if action in {"line", "rectangle", "ellipse"}
        }
        if (
            "cycle_vector_shapes" not in configured_actions
            and "line" not in legacy_shapes
            and legacy_shapes.get("rectangle") == "<KeyPress-r>"
            and legacy_shapes.get("ellipse") == "<KeyPress-o>"
        ):
            bindings = [
                (action, sequence)
                for action, sequence in bindings
                if action not in {"rectangle", "ellipse"}
            ]
            bindings.append(("cycle_vector_shapes", "<KeyPress-o>"))
        window_actions = {
            "copy",
            "paste",
            "increase_size",
            "decrease_size",
            "commit_fill",
            "close_image",
            "swap_colors",
            "focus_colors",
            "zoom_in",
            "zoom_out",
        }
        for action, sequence in bindings:

            def invoke(event, callback=actions[action]):
                if self._clipboard_text_focus(event):
                    return
                callback()
                return "break"

            try:
                # Canvas bindings take precedence over plain letter tools;
                # selected actions also work when toolbar controls have focus.
                self.canvas.bind(sequence, invoke)
                if action in window_actions:
                    self.root.bind(sequence, invoke)
            except tk.TclError as error:
                errors.append(f"{action}: {error}")
        tool_actions = {
            "pan": "pan",
            "color_picker": "color picker",
            "brush": "brush",
            "pencil": "pencil",
            "eraser": "eraser",
            "clone": "clone",
            "paint_bucket": "paint bucket",
            "vector_edit": "vector edit",
            "vector_select": "vector select",
            "line": "line",
            "rectangle": "rect",
            "ellipse": "ellipse",
            "selection": "selection",
            "brush_selection": "brush selection",
            "magic_wand": "magic wand",
            "move": "move",
            "move_selection": "move selection",
        }
        shortcuts_by_action = {}
        for action, sequence in bindings:
            shortcuts_by_action.setdefault(action, []).append(shortcut_label(sequence))
        for action, tool in tool_actions.items():
            label = self.tool_button_labels[tool]
            shortcuts = shortcuts_by_action.get(action, [])
            if not shortcuts and tool in {"selection", "brush selection", "magic wand"}:
                shortcuts = shortcuts_by_action.get("cycle_selection", [])
            elif not shortcuts and tool in {"move", "move selection"}:
                shortcuts = shortcuts_by_action.get("cycle_move", [])
            elif not shortcuts and tool in {"line", "rect", "ellipse"}:
                shortcuts = shortcuts_by_action.get("cycle_vector_shapes", [])
            text = f"{label} ({', '.join(shortcuts)})" if shortcuts else label
            self.tool_tooltips.append(DelayedToolTip(self.tool_buttons[tool], text))
        if errors:
            message = f"Check {path}:\n\n" + "\n".join(errors)
            self.root.after_idle(lambda: messagebox.showwarning("Keyboard shortcuts", message))

    def _initialize_documents(self):
        self.document_defaults = {"doc_w": 1024, "doc_h": 512, "bg_color": (255, 255, 255, 0)}
        self.documents = {}
        self.active_document = None
        self.document_counter = 0
        self._add_document_tab()
        self.startup_document = self.active_document

    def _capture_document(self):
        return context_for(self)

    def _store_document(self):
        if self.active_document is None:
            return
        self.wand_pending = None
        self._finish_bucket_preview()
        self._finish_raster_stroke()
        self._finish_clone_stroke()
        self._release_selection_move()
        self._finish_selection_boundary_move()
        self.last_x = self.last_y = None
        self.is_dragging_point = False
        self.selection_brush_last = None
        if self.selection_animation_id is not None:
            self.root.after_cancel(self.selection_animation_id)
            self.selection_animation_id = None
        self.documents[self.active_document]["state"] = self._capture_document()
        self._update_document_preview()

    def _add_document_tab(self, name=None):
        self.document_counter += 1
        key = context_for(self).document.id
        self.active_document = key
        self.documents[key] = {
            "name": name or f"Untitled {self.document_counter}",
            "state": self._capture_document(),
            "modified": True,
        }
        tab = tk.Frame(self.view_tabs)
        tk.Button(
            tab,
            text=self.documents[key]["name"],
            bd=0,
            compound="left",
            padx=5,
            command=lambda: self.switch_document(key),
        ).pack(side="left")
        tk.Button(tab, text="×", bd=0, padx=4, command=lambda: self.close_document(key)).pack(
            side="left"
        )
        self.view_tab_widgets[key] = tab
        if not self.top_panel_collapsed:
            tab.pack(side="left", padx=2, pady=2)
        self._highlight_document_tabs()
        self._update_document_preview()

    @staticmethod
    def _document_preview_key(state):
        return (
            state["doc_w"],
            state["doc_h"],
            state["bg_color"],
            tuple(
                (
                    id(layer),
                    layer.visible,
                    layer.opacity,
                    layer.blend_mode,
                    layer.masked,
                    layer.anti_mask,
                    layer.mask_mode,
                    layer.mask_visibility,
                    layer.vector_data.revision
                    if layer.vector_data is not None
                    else (id(layer.image), layer._mipmap_revision),
                )
                for layer in state["layers"]
            ),
        )

    def _render_document_thumbnail(self, state):
        snapshot = DocumentSnapshot.capture(state.document)
        ratio = min(46 / state.document.doc_w, 32 / state.document.doc_h)
        size = (
            max(1, round(state.document.doc_w * ratio)),
            max(1, round(state.document.doc_h * ratio)),
        )
        result = self._renderer().render(snapshot, output_size=size, quality="interactive")
        layer = Layer(*size, "Preview")
        layer.image = result
        return self._render_layer_thumbnail(layer)

    def _update_document_preview(self):
        if getattr(self, "active_document", None) is None:
            return
        document = self.documents[self.active_document]
        state = self._capture_document()
        key = self._document_preview_key(state)
        if document.get("preview_key") == key:
            return
        if document.get("preview_job") in self.jobs.pending:
            return
        from pypaint.rendering import render_thumbnail

        snapshot = DocumentSnapshot.capture(state.document)
        job = Job(
            snapshot.document_id,
            snapshot.generation,
            snapshot.state_id,
            preview_key="tab-thumbnail",
            priority=20,
        )

        def publish(result):
            document.pop("preview_job", None)
            if result.error or self._document_preview_key(state) != key:
                return
            photo = ImageTk.PhotoImage(result.value)
            document["preview"], document["preview_key"] = photo, key
            tab = self.view_tab_widgets.get(snapshot.document_id)
            if tab is not None:
                tab.winfo_children()[0].configure(image=photo)

        try:
            self.jobs.submit(job, lambda token: render_thumbnail(snapshot), 16 * 1024**2)
            document["preview_job"] = job.operation_id
            self.job_callbacks[job.operation_id] = publish
        except RuntimeError:
            pass  # A later document/preview refresh retries after interaction.

    def _unchanged_startup_document(self):
        """Return the disposable startup tab, ignoring view/tool changes."""
        if (
            len(self.documents) != 1
            or self.active_document != self.startup_document
            or self.current_file is not None
            or self.undo_stack
            or self.move_pixels is not None
            or self.bucket_pending is not None
            or len(self.layers) != 1
        ):
            return None
        if (self.doc_w, self.doc_h, self.bg_color) != (
            self.document_defaults["doc_w"],
            self.document_defaults["doc_h"],
            self.document_defaults["bg_color"],
        ):
            return None
        layer = self.layers[0]
        if (
            not layer.is_raster
            or layer.name != "Background"
            or not layer.visible
            or layer.opacity != 100
            or layer.blend_mode != "normal"
            or layer.masked
            or layer.anti_mask
            or layer.mask_mode != Layer.MASK_LAYERS_UNDERNEATH
            or layer.mask_visibility != Layer.MASK_VISIBLE_ONLY
            or layer.image.size != (self.doc_w, self.doc_h)
            or any(band.getbbox() is not None for band in layer.image.split())
        ):
            return None
        return self.startup_document

    def _begin_document(self, name=None):
        self._store_document()
        self.globe_window = None
        self._context = DocumentContext()
        self.layers = [Layer(self.doc_w, self.doc_h, "Background")]
        self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 0)
        self._add_document_tab(name)

    def _highlight_document_tabs(self):
        for key in list(self.documents) + list(self.globe_documents):
            selected = (
                key == self.active_view
                if key in self.globe_documents
                else key == self.active_document and self.active_view == "main"
            )
            tab = self.view_tab_widgets[key]
            document_key = self.globe_documents.get(key, key)
            entry = self.documents.get(document_key)
            modified = entry["state"].document.modified if entry else False
            if modified:
                border = "#c62828" if selected else "#e05252"
                background = "#ffd9d9" if selected else "#ffe8e8"
                active_background = "#ffc4c4"
            else:
                border = "#2878d7" if selected else "#d9d9d9"
                background = "#dcecff" if selected else "#f0f0f0"
                active_background = "#c6dfff" if selected else "#e5e5e5"
            # Reserve the same border width on every tab to avoid layout jumps.
            tab.configure(
                background=border,
                highlightthickness=3,
                highlightbackground=border,
                highlightcolor=border,
            )
            for button in tab.winfo_children():
                button.configure(
                    background=background, activebackground=active_background, relief="flat"
                )

    def switch_document(self, key):
        if key not in self.documents:
            return
        if key != self.active_document:
            self._store_document()
            self.active_document = key
            self._context = self.documents[key]["state"]
            self.globe_window = self.views.get(f"globe-{key}")
            self.refresh_layers()
            self._ensure_selection_animation()
            self.notify_globe_document_changed()
        self.main_view_dirty = True
        self.switch_view("main")
        self.canvas.configure(
            cursor=("fleur" if self.tool in ("pan", "move", "move selection") else "crosshair")
        )
        for name, button in getattr(self, "tool_buttons", {}).items():
            button.configure(relief="sunken" if name == self.tool else "raised")
        self.update_title()
        self._highlight_document_tabs()
        self._update_document_preview()

    def close_document(self, key):
        if key not in self.documents:
            return
        state = (
            self._capture_document()
            if key == self.active_document
            else self.documents[key]["state"]
        )
        if state.document.modified and not messagebox.askyesno(
            "Unsaved Changes", f"Close {self.documents[key]['name']} without saving?"
        ):
            return
        if key == self.active_document:
            other = next((item for item in self.documents if item != key), None)
            if other is None:
                self.new_project()
            else:
                self.switch_document(other)
        for view_id, owner in list(self.globe_documents.items()):
            if owner == key:
                self.close_view(view_id)
        self.jobs.cancel_document(state.document.id)
        state.document.closed = True
        state.document.undo_stack.clear()
        del self.documents[key]
        self.view_tab_widgets.pop(key).destroy()

    def switch_view(self, view_id):
        """Show one view without disturbing the shared tools or layers."""
        if view_id not in self.views:
            return
        old_is_globe = self.active_view in self.globe_documents
        new_is_globe = view_id in self.globe_documents
        clone_was_selected = self.tool == "clone"
        if new_is_globe and self.tool in self.GLOBE_BLOCKED_TOOLS:
            self.set_tool("brush")
        if old_is_globe != new_is_globe and clone_was_selected:
            self.clone_source_center = None
            self.clone_offset = None
            globe = self.views.get(self.active_view) if old_is_globe else self.views.get(view_id)
            if globe is not None:
                globe.clone_source_vector = None
                globe.clone_rotation = None
        owner = self.globe_documents.get(view_id)
        if owner is not None and owner != self.active_document:
            self.switch_document(owner)
        if self.active_view in self.views:
            old_view = self.views[self.active_view]
            if hasattr(old_view, "on_hidden"):
                old_view.on_hidden()
            old_view.pack_forget()
        self.active_view = view_id
        self._update_globe_tool_availability()
        new_view = self.views[view_id]
        new_view.pack(fill="both", expand=True)
        if hasattr(new_view, "on_shown"):
            new_view.on_shown()
        if view_id == "main":
            if self.main_view_dirty:
                self.main_view_dirty = False
                self.redraw()
            self.canvas.focus_set()
        if hasattr(self, "documents"):
            self._highlight_document_tabs()

    def _update_globe_tool_availability(self):
        """Visually and functionally disable unsupported globe tools."""
        if not hasattr(self, "tool_buttons"):
            return
        globe_active = self.active_view in self.globe_documents
        normal_background = self.tool_button_background
        for name, button in self.tool_buttons.items():
            blocked = globe_active and name in self.GLOBE_BLOCKED_TOOLS
            button.configure(
                state="disabled" if blocked else "normal",
                image=(self.tool_blocked_overlay if blocked else self.tool_icons[name]),
                background="#3b3b3b" if blocked else normal_background,
                activebackground="#3b3b3b" if blocked else normal_background,
                relief="flat" if blocked else ("sunken" if name == self.tool else "raised"),
            )

    def close_view(self, view_id):
        """Remove an optional view and return to the main canvas."""
        if view_id == "main" or view_id not in self.views:
            return
        widget = self.views.pop(view_id)
        tab = self.view_tab_widgets.pop(view_id)
        self.globe_documents.pop(view_id, None)
        if self.active_view == view_id:
            self.active_view = None
            self.switch_view("main")
        tab.destroy()
        widget.destroy()
        if getattr(self, "globe_window", None) is widget:
            self.globe_window = None

    def apply_ui_scale(self, scale):
        self.ui_scale = scale
        # Compute an absolute font size from the scale (base size = 9pt)
        size = max(7, round(9 * scale))

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
        if any(item["state"].document.modified for item in self.documents.values()):
            if not messagebox.askyesno("Unsaved Changes", "You have unsaved changes. Quit anyway?"):
                return
        if self.selection_animation_id is not None:
            self.root.after_cancel(self.selection_animation_id)
            self.selection_animation_id = None
        self.jobs.shutdown()
        self.root.destroy()

    def update_title(self):
        if getattr(self, "active_document", None) is not None:
            document = self.documents[self.active_document]
            if self.current_file:
                document["name"] = Path(self.current_file).name
            self.view_tab_widgets[self.active_document].winfo_children()[0].configure(
                text=document["name"]
            )
            for view_id, owner in self.globe_documents.items():
                if owner == self.active_document:
                    self.view_tab_widgets[view_id].winfo_children()[0].configure(
                        text=f"{document['name']} · Globe"
                    )
        if self.current_file:
            self.root.title(f"PyPaint - {self.current_file}")
        else:
            self.root.title("PyPaint - Untitled")

    def notify_globe_document_changed(self):
        """Refresh an open globe view after document content changes."""
        document = context_for(self).document
        signature = tuple(
            (layer.id, layer.vector_data.revision)
            for layer in document.layers
            if layer.vector_data is not None
        )
        old_signature = getattr(document, "_notified_vectors", None)
        if signature != old_signature:
            document._notified_vectors = signature
            if old_signature is not None:
                # Drag updates can precede transaction commit. Reject workers
                # captured before any intermediate geometry/style mutation.
                document.change(ChangeKind.VECTOR)
        self._schedule_layer_previews()
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
        return (layer.is_raster and self.tool in ("brush", "eraser")) or (
            layer.layer_type == "vector" and self.tool in ("line", "rect", "ellipse")
        )

    def can_draw_vector_from_globe(self):
        return self.layers[self.active_layer].layer_type == "vector" and self.tool in (
            "line",
            "rect",
            "ellipse",
        )

    def snapshot(self, name=None):
        document = context_for(self).document
        history_for(document).begin(document, name or self.tool.title())
        if self.active_document in self.documents:
            self.documents[self.active_document]["modified"] = document.modified
            self._highlight_document_tabs()

    def undo(self):
        self._finish_bucket_preview()
        self._finish_raster_stroke()
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        if history_for(context_for(self).document).undo(context_for(self).document):
            self._history_changed()

    def redo(self):
        self._finish_clipboard_edit()
        if history_for(context_for(self).document).redo(context_for(self).document):
            self._history_changed()

    def _history_changed(self):
        # Deliberately retain the original clear-selection-on-undo behavior.
        self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 0)
        self.selected_vector_obj = None
        self._update_selection_geometry()
        document = context_for(self).document
        if self.active_document in self.documents:
            self.documents[self.active_document]["modified"] = document.modified
            self._highlight_document_tabs()
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def cancel_gesture(self, event=None):
        if self.bucket_pending is not None:
            job = self.bucket_pending.get("job")
            if job:
                job.cancellation.cancel()
            self.bucket_pending = None
        self.wand_pending = None
        session = context_for(self).session
        if session.stroke is not None:
            session.stroke.cancel()
            session.stroke = None
        elif session.clone_gesture is not None:
            session.clone_gesture.cancel()
            session.clone_gesture = None
        else:
            history_for(context_for(self).document).cancel(context_for(self).document)
        self._stroke_base_image = self._stroke_coverage = None
        self._build_up_base_image = self._build_up_coverage = None
        self.clone_stroke_source = self.clone_stroke_base = self.clone_stroke_coverage = None
        self.move_pixels = self.move_mask = self.move_base_image = self.move_start = None
        self.last_x = self.last_y = None
        self.is_dragging_point = False
        self._history_changed()
        return "break"

    def new_project(self):
        self._begin_document()
        self._finish_open()

    def open_canvas_size(self):
        """Show a modal editor for the active document dimensions."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Canvas Size")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill="both", expand=True)
        width_var = tk.StringVar(value=str(self.doc_w))
        height_var = tk.StringVar(value=str(self.doc_h))

        ttk.Label(body, text="Width:").grid(row=0, column=0, sticky="w", pady=3)
        width_entry = ttk.Entry(body, textvariable=width_var, width=12)
        width_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Label(body, text="Height:").grid(row=1, column=0, sticky="w", pady=3)
        height_entry = ttk.Entry(body, textvariable=height_var, width=12)
        height_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3)

        maintain_aspect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Maintain aspect ratio", variable=maintain_aspect_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 3)
        )

        updating_dimensions = False

        def keep_aspect(changed):
            nonlocal updating_dimensions
            if updating_dimensions or not maintain_aspect_var.get():
                return
            try:
                updating_dimensions = True
                if changed == "width":
                    width = int(width_var.get())
                    if width > 0:
                        height_var.set(str(max(1, round(width * self.doc_h / self.doc_w))))
                else:
                    height = int(height_var.get())
                    if height > 0:
                        width_var.set(str(max(1, round(height * self.doc_w / self.doc_h))))
            except ValueError:
                # Intermediate entry states such as an empty field are valid
                # while the user is typing and are checked again on OK.
                pass
            finally:
                updating_dimensions = False

        width_var.trace_add("write", lambda *_: keep_aspect("width"))
        height_var.trace_add("write", lambda *_: keep_aspect("height"))

        ttk.Label(body, text="Anchor:").grid(row=3, column=0, sticky="nw", pady=(8, 3))
        anchor_var = tk.StringVar(value=self.canvas_resize_anchor)
        anchor_frame = ttk.Frame(body)
        anchor_frame.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 3))
        anchors = (
            ("top-left", "top", "top-right"),
            ("left", "center", "right"),
            ("bottom-left", "bottom", "bottom-right"),
        )
        for row, names in enumerate(anchors):
            for column, name in enumerate(names):
                ttk.Radiobutton(
                    anchor_frame,
                    text="●",
                    value=name,
                    variable=anchor_var,
                    style="Toolbutton",
                    width=2,
                ).grid(row=row, column=column, padx=1, pady=1)

        ttk.Label(
            body, text="Choose where the existing image stays anchored.", foreground="#555555"
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(7, 10))

        def apply_size(event=None):
            try:
                width = int(width_var.get())
                height = int(height_var.get())
            except ValueError:
                messagebox.showerror(
                    "Canvas Size", "Width and height must be whole numbers.", parent=dialog
                )
                return
            if width < 1 or height < 1:
                messagebox.showerror(
                    "Canvas Size", "Width and height must be at least 1 pixel.", parent=dialog
                )
                return
            if width > 32768 or height > 32768:
                messagebox.showerror(
                    "Canvas Size", "Width and height cannot exceed 32,768 pixels.", parent=dialog
                )
                return
            self.canvas_resize_anchor = anchor_var.get()
            if (width, height) != (self.doc_w, self.doc_h):
                self.resize_canvas(width, height, self.canvas_resize_anchor)
            dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="OK", command=apply_size).pack(side="right")
        dialog.bind("<Return>", apply_size)
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        width_entry.focus_set()
        width_entry.selection_range(0, "end")

    def resize_canvas(self, width, height, anchor="center"):
        """Resize the document without scaling its existing layer content."""
        self.wand_pending = None
        self._finish_bucket_preview()
        self._finish_raster_stroke()
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        old_width, old_height = self.doc_w, self.doc_h
        horizontal, vertical = {
            "top-left": ("left", "top"),
            "top": ("center", "top"),
            "top-right": ("right", "top"),
            "left": ("left", "center"),
            "center": ("center", "center"),
            "right": ("right", "center"),
            "bottom-left": ("left", "bottom"),
            "bottom": ("center", "bottom"),
            "bottom-right": ("right", "bottom"),
        }.get(anchor, ("center", "center"))

        x = {"left": 0, "center": (width - old_width) // 2, "right": width - old_width}[horizontal]
        y = {"top": 0, "center": (height - old_height) // 2, "bottom": height - old_height}[
            vertical
        ]

        if hasattr(self, "jobs"):
            from pypaint.editor import apply_snapshot, canvas_resized

            document = context_for(self).document
            history_for(document).commit(document)
            snapshot = DocumentSnapshot.capture(document)
            selection = (
                self.selection_mask.snapshot()
                if isinstance(self.selection_mask, TiledSurface)
                else TiledSurface.from_image(self.selection_mask).snapshot()
            )
            job = Job(document.id, document.generation, document.state_id, preview_key="resize")

            def publish(result):
                if result.error:
                    messagebox.showerror("Canvas resize failed", str(result.error))
                    return
                snapshot, mask = result.value
                apply_snapshot(document, job.generation, snapshot, "Resize canvas")
                context = self.documents[document.id]["state"]
                context.session.selection_mask = mask.copy()
                if context_for(self).document is document:
                    self._history_changed()
                    self.selection_mask = mask.copy()
                    self._update_selection_geometry()

            self._submit_job(
                job,
                lambda token: canvas_resized(snapshot, (width, height), (x, y), selection, token),
                48 * 1024**2,
                publish,
            )
            return

        self.snapshot("Resize canvas")

        old_selection = self.selection_mask
        self.doc_w, self.doc_h = width, height
        for layer in self.layers:
            layer.width, layer.height = width, height
            if layer.vector_data is not None:
                for obj in layer.vector_data.objects:
                    self._translate_vector_object(obj, x, y)
                layer.vector_data.width, layer.vector_data.height = width, height
                layer.render_vector(regional=True)
            else:
                resized = TiledSurface("RGBA", (width, height), (0, 0, 0, 0))
                resized.paste(layer.image, (x, y))
                layer.image = resized
                layer.draw = None
                layer.reset_mipmaps()

        self.selection_mask = TiledSurface("L", (width, height), 0)
        self.selection_mask.paste(old_selection, (x, y))
        self._update_selection_geometry()
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def open_resize(self):
        """Show the image-resampling dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Resize")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill="both", expand=True)
        width_var = tk.StringVar(value=str(self.doc_w))
        height_var = tk.StringVar(value=str(self.doc_h))
        maintain_aspect_var = tk.BooleanVar(value=True)
        resampling_var = tk.StringVar(value="Nearest Neighbor")

        ttk.Label(body, text="Width:").grid(row=0, column=0, sticky="w", pady=3)
        width_entry = ttk.Entry(body, textvariable=width_var, width=15)
        width_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Label(body, text="Height:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(body, textvariable=height_var, width=15).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=3
        )
        ttk.Checkbutton(body, text="Maintain aspect ratio", variable=maintain_aspect_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 7)
        )
        ttk.Label(body, text="Resampling:").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(
            body,
            textvariable=resampling_var,
            values=("Nearest Neighbor",),
            state="readonly",
            width=17,
        ).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=3)

        updating_dimensions = False

        def keep_aspect(changed):
            nonlocal updating_dimensions
            if updating_dimensions or not maintain_aspect_var.get():
                return
            try:
                updating_dimensions = True
                if changed == "width":
                    width = int(width_var.get())
                    if width > 0:
                        height_var.set(str(max(1, round(width * self.doc_h / self.doc_w))))
                else:
                    height = int(height_var.get())
                    if height > 0:
                        width_var.set(str(max(1, round(height * self.doc_w / self.doc_h))))
            except ValueError:
                pass
            finally:
                updating_dimensions = False

        width_var.trace_add("write", lambda *_: keep_aspect("width"))
        height_var.trace_add("write", lambda *_: keep_aspect("height"))

        def apply_size(event=None):
            try:
                width = int(width_var.get())
                height = int(height_var.get())
            except ValueError:
                messagebox.showerror(
                    "Resize", "Width and height must be whole numbers.", parent=dialog
                )
                return
            if width < 1 or height < 1:
                messagebox.showerror(
                    "Resize", "Width and height must be at least 1 pixel.", parent=dialog
                )
                return
            if width > 32768 or height > 32768:
                messagebox.showerror(
                    "Resize", "Width and height cannot exceed 32,768 pixels.", parent=dialog
                )
                return
            if (width, height) != (self.doc_w, self.doc_h):
                self.resize_image(width, height, resampling_var.get())
            dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="OK", command=apply_size).pack(side="right")
        dialog.bind("<Return>", apply_size)
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        width_entry.focus_set()
        width_entry.selection_range(0, "end")

    def resize_image(self, width, height, resampling="Nearest Neighbor"):
        from pypaint.editor import apply_snapshot, resized

        self._finish_clipboard_edit()
        document = context_for(self).document
        history_for(document).commit(document)
        snapshot = DocumentSnapshot.capture(document)
        selection = self.selection_mask
        if not isinstance(selection, TiledSurface):
            selection = TiledSurface.from_image(selection)
        selection = selection.snapshot()
        job = Job(document.id, document.generation, document.state_id, preview_key="resize")

        def publish(result):
            if result.error:
                messagebox.showerror("Resize failed", str(result.error))
                return
            snapshot, mask = result.value
            apply_snapshot(document, job.generation, snapshot, "Resize image")
            self.documents[document.id]["state"].session.selection_mask = mask.copy()
            if context_for(self).document is document:
                self._history_changed()
                self.selection_mask = mask.copy()
                self._update_selection_geometry()

        self._submit_job(
            job,
            lambda token: resized(snapshot, (width, height), selection, token),
            64 * 1024**2,
            publish,
        )

    @staticmethod
    def _scale_vector_object(obj, scale_x, scale_y):
        from pypaint.vectors import scale_vector

        return scale_vector(obj, scale_x, scale_y)

    @staticmethod
    def _translate_vector_object(obj, x, y):
        from pypaint.vectors import translate_vector

        return translate_vector(obj, x, y)

    def save_project(self):
        self._finish_bucket_preview()
        if self.current_file:
            if self._save_to_file(self.current_file):
                self.update_title()
        else:
            self.save_project_as()

    def save_project_as(self):
        self._finish_bucket_preview()
        filetypes = [("PyPaint files", "*.pypaint")]
        if self._pdn_save_supported():
            filetypes.append(("Paint.NET files", "*.pdn"))
        filetypes.append(("All files", "*.*"))
        filename = filedialog.asksaveasfilename(defaultextension=".pypaint", filetypes=filetypes)
        if filename:
            if self._save_to_file(filename):
                self.current_file = filename
                self.update_title()
                messagebox.showinfo("Success", f"Project saved to {filename}")

    def _save_to_file(self, filename):
        self._finish_clipboard_edit()
        self._finish_raster_stroke()
        document = context_for(self).document
        history_for(document).commit(document)
        snapshot = DocumentSnapshot.capture(document)
        if Path(filename).suffix.lower() == ".pdn" and not self._pdn_save_supported():
            messagebox.showerror("Save", "PDN requires Normal raster layers without masks.")
            return False
        if (
            Path(filename).suffix.lower() == ".pdn"
            and snapshot.size[0] * snapshot.size[1] * 4 * len(snapshot.layers) > 512 * 1024**2
        ):
            messagebox.showerror(
                "Save", "PDN layers exceed the 512 MiB decoded limit; use native tiled storage."
            )
            return False
        job = Job(
            document.id,
            document.generation,
            snapshot.state_id,
            publication="saved",
            resource_key=str(Path(filename).resolve()).casefold(),
        )

        def save(token):
            if Path(filename).suffix.lower() == ".pdn":
                token.check()
                write_pdn(
                    filename,
                    *snapshot.size,
                    [
                        RasterLayer(
                            dict(
                                zip(
                                    __import__(
                                        "pypaint.history", fromlist=["LAYER_FIELDS"]
                                    ).LAYER_FIELDS,
                                    r.metadata,
                                )
                            )["name"],
                            r.surface.materialize(),
                            r.metadata[2],
                            round(r.metadata[3] * 255 / 100),
                        )
                        for r in snapshot.layers
                    ],
                    cancellation=token,
                )
                return snapshot.state_id
            return save_native(snapshot, filename, token)

        def publish(result):
            if result.error:
                messagebox.showerror("Save failed", str(result.error))
                return
            document.saved_state_id = result.value
            document.current_file = filename
            entry = self.documents.get(document.id)
            if entry is not None:
                entry["modified"] = document.modified
                entry["name"] = Path(filename).name
            if context_for(self).document is document:
                self.update_title()
            self._highlight_document_tabs()

        try:
            reserve = 1536 * 1024**2 if Path(filename).suffix.lower() == ".pdn" else 32 * 1024**2
            self.jobs.submit(job, save, reserve)
            self.job_callbacks[job.operation_id] = publish
        except Exception as error:
            messagebox.showerror("Save failed", str(error))
        return False  # completion is published asynchronously by the UI poller

    def _poll_jobs(self):
        documents = {
            key: value["state"].document for key, value in getattr(self, "documents", {}).items()
        }
        for result in self.jobs.drain(documents):
            callback = self.job_callbacks.pop(result.job.operation_id, None)
            if callback:
                callback(result)
        for operation_id in self.jobs.discarded:
            self.job_callbacks.pop(operation_id, None)
        self.jobs.discarded.clear()
        if self.mipmap_future is not None and self.mipmap_future not in self.jobs.pending:
            self.job_callbacks.pop(self.mipmap_future, None)
            self.mipmap_future = None
        if not self.jobs.closed:
            self.root.after(16, self._poll_jobs)

    def _pdn_save_supported(self):
        return bool(self.layers) and all(
            layer.layer_type == "raster"
            and not layer.masked
            and not layer.anti_mask
            and layer.blend_mode == "normal"
            and not (layer.vector_data and layer.vector_data.objects)
            for layer in self.layers
        )

    def _mark_project_saved(self, state_id=None):
        document = context_for(self).document
        history_for(document).commit(document)
        document.saved_state_id = state_id or document.state_id
        if self.active_document in self.documents:
            self.documents[self.active_document]["modified"] = document.modified
            self._highlight_document_tabs()

    def open_project(self):
        from pypaint.files import read_document

        filenames = filedialog.askopenfilenames(
            filetypes=[("All files", "*.*"), ("PyPaint", "*.pypaint"), ("Paint.NET", "*.pdn")]
        )
        owner = context_for(self).document
        for filename in filenames:
            job = Job(owner.id, owner.generation, owner.state_id, publication="saved")

            def publish(result, filename=filename):
                if result.error:
                    messagebox.showerror("Open failed", str(result.error))
                    return
                self._accept_open_document(result.value, filename)

            # Legacy PDN decoding has a larger bounded contiguous working set.
            reserve = 1536 * 1024**2 if Path(filename).suffix.lower() == ".pdn" else 512 * 1024**2
            self._submit_job(
                job,
                lambda token, filename=filename: read_document(filename, token),
                reserve,
                publish,
            )

    def _accept_open_document(self, document, filename):
        startup = self._unchanged_startup_document()
        self._begin_document(Path(filename).name)
        current = context_for(self).document
        current.doc_w, current.doc_h = document.doc_w, document.doc_h
        current.layers, current.active_layer = document.layers, document.active_layer
        current.bg_color, current.state_id = document.bg_color, document.state_id
        current.persistent_id = document.persistent_id
        current.saved_state_id, current.current_file = (
            document.saved_state_id,
            document.current_file,
        )
        self._finish_open(startup)
        self.documents[self.active_document]["modified"] = current.modified
        self._highlight_document_tabs()

    def _submit_job(self, job, function, reserve, callback):
        try:
            self.jobs.submit(job, function, reserve)
            self.job_callbacks[job.operation_id] = callback
            return True
        except (RuntimeError, OSError, MemoryError) as error:
            messagebox.showerror("Operation could not start", str(error))
            return False

    def _open_pdn_file(self, filename):
        """Synchronous compatibility hook; the Open command uses immutable jobs."""
        from pypaint.files import read_document
        from pypaint.jobs import Cancellation

        self._accept_open_document(read_document(filename, Cancellation()), filename)

    def _open_pypaint_file(self, filename):
        """Synchronous compatibility hook; the Open command uses immutable jobs."""
        from pypaint.files import read_document
        from pypaint.jobs import Cancellation

        self._accept_open_document(read_document(filename, Cancellation()), filename)

    def _open_image_file(self, filename):
        """Synchronous compatibility hook; the Open command uses immutable jobs."""
        from pypaint.files import read_document
        from pypaint.jobs import Cancellation

        self._accept_open_document(read_document(filename, Cancellation()), filename)

    def _finish_open(self, startup_to_replace=None):
        """Refresh shared UI state after either kind of file is opened."""
        self.undo_stack = []
        self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 0)
        self._update_selection_geometry()
        self.clone_source_center = None
        self.clone_offset = None
        self.refresh_layers()
        self.redraw()
        self.update_title()
        self.notify_globe_document_changed()
        self.switch_document(self.active_document)
        if startup_to_replace is not None:
            for view_id, owner in list(self.globe_documents.items()):
                if owner == startup_to_replace:
                    self.close_view(view_id)
            del self.documents[startup_to_replace]
            self.view_tab_widgets.pop(startup_to_replace).destroy()

    def set_tool(self, tool):
        if (
            self.active_view in getattr(self, "globe_documents", {})
            and tool in self.GLOBE_BLOCKED_TOOLS
        ):
            return
        if (
            hasattr(self, "tools_by_layer_type")
            and self.layers
            and tool not in self.tools_by_layer_type[self.layers[self.active_layer].layer_type]
        ):
            return
        self._finish_raster_stroke()
        if tool != self.tool:
            self.wand_pending = None
            self._finish_bucket_preview()
            self._finish_clone_stroke()
            self._finish_selection_move()
            self._finish_selection_boundary_move()
            self.selection_brush_last = None
            self.selection_brush_remove = False
        self.tool = tool
        if hasattr(self, "canvas"):
            cursor = "fleur" if tool in ("pan", "move", "move selection") else "crosshair"
            try:
                self.canvas.configure(cursor=cursor)
            except tk.TclError:
                self.canvas.configure(cursor="crosshair")
        if hasattr(self, "tool_buttons"):
            for name, button in self.tool_buttons.items():
                button.configure(relief="sunken" if name == tool else "raised")
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
        """Cycle rectangle, brush selection, and magic wand with S."""
        tools = ("selection", "brush selection", "magic wand")
        next_tool = (
            tools[(tools.index(self.tool) + 1) % len(tools)] if self.tool in tools else tools[0]
        )
        self.set_tool(next_tool)

    def select_vector_shape_tool(self, event=None):
        """Cycle line, rectangle, and ellipse vector tools with O."""
        tools = ("line", "rect", "ellipse")
        next_tool = (
            tools[(tools.index(self.tool) + 1) % len(tools)] if self.tool in tools else tools[0]
        )
        self.set_tool(next_tool)

    def select_all(self, event=None):
        """Select every pixel in the document."""
        self.wand_pending = None
        self._finish_bucket_preview()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        self.selection_brush_last = None
        self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 255)
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
        self.zoom = max(0.01, min(20, usable_width / width, usable_height / height))
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

        for frame in (
            self.size_frame,
            self.picker_settings_frame,
            self.brush_settings_frame,
            self.clone_settings_frame,
            self.bucket_settings_frame,
            self.wand_settings_frame,
            self.vector_settings_frame,
            self.vector_select_settings_frame,
        ):
            frame.pack_forget()

        uses_size = self.tool not in (
            "pan",
            "selection",
            "move",
            "move selection",
            "paint bucket",
            "magic wand",
            "pencil",
        ) and (self.tool != "color picker" or self.picker_sample_area_var.get())
        if uses_size:
            self.size_frame.pack(side="left")

        if self.tool == "color picker":
            self.picker_settings_frame.pack(side="left")
        elif (
            self.tool in ("brush", "eraser")
            and self.layers
            and self.layers[self.active_layer].is_raster
        ):
            self.brush_settings_frame.pack(side="left")
        elif self.tool == "clone" and self.layers and self.layers[self.active_layer].is_raster:
            self.clone_settings_frame.pack(side="left")
        elif (
            self.tool == "paint bucket" and self.layers and self.layers[self.active_layer].is_raster
        ):
            self.bucket_settings_frame.pack(side="left")
        elif self.tool == "magic wand" and self.layers and self.layers[self.active_layer].is_raster:
            self.wand_settings_frame.pack(side="left")
        elif self.tool in ("line", "rect", "ellipse", "point"):
            self.vector_settings_frame.pack(side="left")
        elif self.tool == "vector select":
            self.vector_select_settings_frame.pack(side="left", fill="x", expand=True)
        self.request_redraw()

    def add_layer(self, layer_type="raster"):
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        self.snapshot("Add layer")
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
        self.snapshot("Delete layer")
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

    def duplicate_layer(self):
        """Duplicate the active layer immediately above the source layer."""
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        self.snapshot("Duplicate layer")

        source = self.layers[self.active_layer]
        duplicate = Layer(self.doc_w, self.doc_h, f"{source.name} copy", source.layer_type)
        duplicate.visible = source.visible
        duplicate.opacity = source.opacity
        duplicate.blend_mode = source.blend_mode
        duplicate.masked = source.masked
        duplicate.anti_mask = source.anti_mask
        duplicate.mask_mode = source.mask_mode
        duplicate.mask_visibility = source.mask_visibility
        if source.vector_data is not None:
            duplicate.vector_data = copy.deepcopy(source.vector_data)
            duplicate.vector_data.name = duplicate.name
            from pypaint.state import identity

            for obj in duplicate.vector_data.objects:
                obj.id = identity()
                for line in getattr(obj, "lines", ()):
                    line.id = identity()
        else:
            duplicate.image = source.image.copy()
        duplicate.draw = None
        duplicate.reset_mipmaps()

        self.active_layer += 1
        self.layers.insert(self.active_layer, duplicate)
        self.refresh_layers()
        self.update_tools_for_active_layer()
        self.request_redraw()
        self.notify_globe_document_changed()

    def _bake_layer_mask(self, layer_index):
        """Replace a layer with its current masked, opacity-adjusted pixels."""
        layer = self.layers[layer_index]
        if not layer.masked:
            return False

        if layer.layer_type == "vector" and layer.vector_data:
            layer.render_vector(regional=True)

        from pypaint.editor import baked_mask

        pixels = baked_mask(
            DocumentSnapshot.capture(context_for(self).document), layer_index, self._renderer()
        )
        layer.layer_type = "raster"
        layer.image = pixels.copy()
        layer.opacity = 100
        layer.masked = False
        layer.anti_mask = False
        layer.mask_visibility = Layer.MASK_VISIBLE_ONLY
        layer.layer_type = "raster"
        layer.vector_data = None
        layer.draw = None
        layer.reset_mipmaps()
        return True

    def apply_layer_mask(self):
        """Bake the active layer's visible masked output into independent pixels."""
        if not self.layers[self.active_layer].masked:
            return
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()
        self.snapshot("Apply layer mask")
        self._bake_layer_mask(self.active_layer)
        self.refresh_layers()
        self.update_tools_for_active_layer()
        self.request_redraw()
        self.notify_globe_document_changed()

    def move_layer_up(self):
        if self.active_layer >= len(self.layers) - 1:
            return
        self.snapshot("Move layer up")
        self.layers[self.active_layer], self.layers[self.active_layer + 1] = (
            self.layers[self.active_layer + 1],
            self.layers[self.active_layer],
        )
        self.active_layer += 1
        self.refresh_layers()
        self.request_redraw()
        self.notify_globe_document_changed()

    def move_layer_down(self):
        if self.active_layer <= 0:
            return
        self.snapshot("Move layer down")
        self.layers[self.active_layer], self.layers[self.active_layer - 1] = (
            self.layers[self.active_layer - 1],
            self.layers[self.active_layer],
        )
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
            foreground=[("selected", self.layer_selected_text_color)],
        )

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
        original_blend_mode = layer.blend_mode
        original_masked = layer.masked
        original_anti_mask = layer.anti_mask
        original_mask_mode = layer.mask_mode
        original_mask_visibility = layer.mask_visibility
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

        ttk.Label(body, text="Name:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 10))
        name_var = tk.StringVar(value=layer.name)
        name_entry = ttk.Entry(body, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 10))

        masked_var = tk.BooleanVar(value=layer.masked)
        masked_check = ttk.Checkbutton(body, text="Enable masking", variable=masked_var)
        masked_check.grid(row=1, column=1, columnspan=3, sticky="w")

        anti_mask_var = tk.BooleanVar(value=layer.anti_mask)
        anti_mask_check = ttk.Checkbutton(body, text="Anti-mask", variable=anti_mask_var)
        anti_mask_check.grid(row=2, column=1, columnspan=3, sticky="w", padx=(18, 0), pady=(0, 10))

        visibility_labels = {
            Layer.MASK_VISIBLE_ONLY: "Only Visible",
            Layer.MASK_ALL_BELOW: "Visible or Hidden",
        }
        visibility_values = {label: value for value, label in visibility_labels.items()}
        mask_visibility_var = tk.StringVar(value=visibility_labels[layer.mask_visibility])
        ttk.Label(body, text="Visibility:").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=(0, 6)
        )
        visibility_combo = ttk.Combobox(
            body,
            textvariable=mask_visibility_var,
            values=tuple(visibility_labels.values()),
            state="readonly",
            width=18,
        )
        visibility_combo.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(0, 6))

        mode_labels = {
            Layer.MASK_LAYERS_UNDERNEATH: "All Below",
            Layer.MASK_LAYER_BELOW: "One Below",
        }
        mode_values = {label: value for value, label in mode_labels.items()}
        mask_mode_var = tk.StringVar(value=mode_labels[layer.mask_mode])
        ttk.Label(body, text="Considered Layers:").grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=(0, 10)
        )
        mode_combo = ttk.Combobox(
            body,
            textvariable=mask_mode_var,
            values=tuple(mode_labels.values()),
            state="readonly",
            width=18,
        )
        mode_combo.grid(row=4, column=1, columnspan=3, sticky="ew", pady=(0, 10))

        apply_mask_button = ttk.Button(body, text="Bake Mask")
        apply_mask_button.grid(row=5, column=1, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(body, text="Opacity:").grid(row=6, column=0, sticky="w", padx=(0, 8))
        opacity_var = tk.IntVar(value=round(layer.opacity))
        opacity_scale = ttk.Scale(body, from_=0, to=100, orient="horizontal", length=190)
        opacity_scale.set(layer.opacity)
        opacity_scale.grid(row=6, column=1, sticky="ew")

        opacity_spinbox = ttk.Spinbox(
            body, from_=0, to=100, textvariable=opacity_var, width=5, justify="right"
        )
        opacity_spinbox.grid(row=6, column=2, padx=(8, 0))
        ttk.Label(body, text="%").grid(row=6, column=3, sticky="w", padx=(3, 0))

        blend_labels = blend_mode_labels()
        blend_values = {label: mode for mode, label in blend_labels.items()}
        blend_var = tk.StringVar(value=blend_labels[layer.blend_mode])
        ttk.Label(body, text="Blend Mode:").grid(
            row=7, column=0, sticky="w", padx=(0, 8), pady=(10, 0)
        )
        ttk.Combobox(
            body, textvariable=blend_var, values=tuple(blend_values), state="readonly", width=18
        ).grid(row=7, column=1, columnspan=3, sticky="ew", pady=(10, 0))

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

        def blend_changed(*_args):
            layer.blend_mode = blend_values[blend_var.get()]
            schedule_preview()

        blend_var.trace_add("write", blend_changed)

        def masked_changed():
            layer.masked = masked_var.get()
            control_state = "readonly" if layer.masked else "disabled"
            anti_mask_check.configure(state="normal" if layer.masked else "disabled")
            visibility_combo.configure(state=control_state)
            mode_combo.configure(state=control_state)
            apply_mask_button.configure(state="normal" if layer.masked else "disabled")
            self.refresh_layers()
            schedule_preview()

        masked_check.configure(command=masked_changed)

        def anti_mask_changed():
            layer.anti_mask = anti_mask_var.get()
            schedule_preview()

        anti_mask_check.configure(command=anti_mask_changed)

        def mask_mode_changed(*_args):
            layer.mask_mode = mode_values[mask_mode_var.get()]
            schedule_preview()

        mask_mode_var.trace_add("write", mask_mode_changed)

        def mask_visibility_changed(*_args):
            layer.mask_visibility = visibility_values[mask_visibility_var.get()]
            schedule_preview()

        mask_visibility_var.trace_add("write", mask_visibility_changed)
        masked_changed()

        def apply_mask():
            nonlocal preview_after_id
            if preview_after_id is not None:
                self.root.after_cancel(preview_after_id)
                preview_after_id = None

            # Preview controls mutate the live layer. Restore the state from
            # before the dialog so the bake remains a single undoable action.
            current_name = name_var.get().strip() or original_name
            current_opacity = layer.opacity
            current_blend_mode = layer.blend_mode
            current_masked = masked_var.get()
            current_anti_mask = anti_mask_var.get()
            current_mask_mode = mode_values[mask_mode_var.get()]
            current_mask_visibility = visibility_values[mask_visibility_var.get()]
            layer.name = original_name
            layer.opacity = original_opacity
            layer.blend_mode = original_blend_mode
            layer.masked = original_masked
            layer.anti_mask = original_anti_mask
            layer.mask_mode = original_mask_mode
            layer.mask_visibility = original_mask_visibility
            if layer.vector_data:
                layer.vector_data.name = original_name
            self.snapshot()

            layer.name = current_name
            layer.opacity = current_opacity
            layer.blend_mode = current_blend_mode
            layer.masked = current_masked
            layer.anti_mask = current_anti_mask
            layer.mask_mode = current_mask_mode
            layer.mask_visibility = current_mask_visibility
            if layer.vector_data:
                layer.vector_data.name = current_name
            self._bake_layer_mask(layer_index)
            self.refresh_layers()
            self.update_tools_for_active_layer()
            self.request_redraw()
            self.notify_globe_document_changed()
            dialog.destroy()

        apply_mask_button.configure(command=apply_mask)

        def accept(_event=None):
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning(
                    "Layer Properties", "Layer name cannot be empty.", parent=dialog
                )
                name_entry.focus_set()
                return
            try:
                opacity = max(0, min(100, int(opacity_var.get())))
            except (tk.TclError, ValueError):
                messagebox.showwarning(
                    "Layer Properties", "Opacity must be a number from 0 to 100.", parent=dialog
                )
                opacity_spinbox.focus_set()
                return

            # The controls have already previewed their values. Temporarily
            # restore the originals so Undo records the pre-dialog state.
            new_masked = masked_var.get()
            new_blend_mode = blend_values[blend_var.get()]
            new_anti_mask = anti_mask_var.get()
            new_mask_mode = mode_values[mask_mode_var.get()]
            new_mask_visibility = visibility_values[mask_visibility_var.get()]
            layer.name = original_name
            layer.opacity = original_opacity
            layer.blend_mode = original_blend_mode
            layer.masked = original_masked
            layer.anti_mask = original_anti_mask
            layer.mask_mode = original_mask_mode
            layer.mask_visibility = original_mask_visibility
            if layer.vector_data:
                layer.vector_data.name = original_name
            self.snapshot()
            layer.name = name
            layer.opacity = opacity
            layer.blend_mode = new_blend_mode
            layer.masked = new_masked
            layer.anti_mask = new_anti_mask
            layer.mask_mode = new_mask_mode
            layer.mask_visibility = new_mask_visibility
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
            layer.blend_mode = original_blend_mode
            layer.masked = original_masked
            layer.anti_mask = original_anti_mask
            layer.mask_mode = original_mask_mode
            layer.mask_visibility = original_mask_visibility
            if layer.vector_data:
                layer.vector_data.name = original_name
            self.refresh_layers()
            self.request_redraw()
            self.notify_globe_document_changed()
            dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=8, column=0, columnspan=4, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(6, 0))
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
        self.layer_preview_cache = {
            layer: cached
            for layer, cached in self.layer_preview_cache.items()
            if layer in self.layers
        }
        self.layer_list.delete(*self.layer_list.get_children())
        for i in range(len(self.layers) - 1, -1, -1):
            l = self.layers[i]
            self.layer_list.insert(
                "",
                "end",
                text=f"{l.name}{' [Masked]' if l.masked else ''}",
                image=self._layer_row_preview(l),
                tags=(l.layer_type,),
            )

        self.update_layer_selection_style()
        self.update_tools_for_active_layer()

        display_index = len(self.layers) - 1 - self.active_layer
        rows = self.layer_list.get_children()
        if 0 <= display_index < len(rows):
            self.layer_list.selection_set(rows[display_index])
            self.layer_list.focus(rows[display_index])

    @staticmethod
    def _render_layer_thumbnail(layer, size=(48, 34)):
        """Show the whole layer with its aspect ratio and transparency intact."""
        width, height = size
        preview = Image.new("RGBA", size, "#eeeeee")
        draw = ImageDraw.Draw(preview)
        for y in range(0, height, 5):
            for x in range(0, width, 5):
                if (x // 5 + y // 5) % 2:
                    draw.rectangle((x, y, x + 4, y + 4), fill="#cccccc")
        ratio = min((width - 2) / layer.width, (height - 2) / layer.height)
        target = (max(1, round(layer.width * ratio)), max(1, round(layer.height * ratio)))
        from pypaint.state import Document

        thumbnail_layer = copy.copy(layer)
        thumbnail_layer.visible = True
        thumbnail_layer.masked = False
        thumbnail_layer.blend_mode = "normal"
        document = Document(doc_w=layer.width, doc_h=layer.height, layers=[thumbnail_layer])
        thumb = Renderer(cache_bytes=1024 * 1024).render(
            DocumentSnapshot.capture(document), output_size=target, quality="interactive"
        )
        preview.alpha_composite(thumb, ((width - target[0]) // 2, (height - target[1]) // 2))
        ImageDraw.Draw(preview).rectangle((0, 0, width - 1, height - 1), outline="#888888")
        return preview

    def _layer_row_preview(self, layer):
        content = (
            layer.vector_data.revision
            if layer.vector_data is not None
            else (id(layer.image), layer._mipmap_revision)
        )
        key = (content, layer.visible, layer.opacity, layer.blend_mode, layer.layer_type)
        cached = self.layer_preview_cache.get(layer)
        if cached is not None and cached[0] == key:
            return cached[1]
        if hasattr(self, "jobs"):
            return self._queue_layer_thumbnail(layer, key, cached)
        if layer.vector_data is not None:
            layer.render_vector(regional=True)
        row = Image.new("RGBA", (142, 40))
        row.alpha_composite(self.layer_row_icons[(layer.visible, layer.layer_type)], (0, 6))
        row.alpha_composite(self._render_layer_thumbnail(layer), (92, 3))
        photo = ImageTk.PhotoImage(row)
        self.layer_preview_cache[layer] = (key, photo)
        return photo

    def _queue_layer_thumbnail(self, layer, key, cached):
        from pypaint.rendering import render_thumbnail

        pending = getattr(self, "_layer_thumbnail_jobs", None)
        if pending is None:
            pending = self._layer_thumbnail_jobs = {}
        if pending.get(layer.id) in self.jobs.pending and cached is not None:
            return cached[1]
        document = context_for(self).document
        record = LayerRecord.capture(layer)
        metadata = list(record.metadata)
        metadata[2], metadata[4], metadata[6], metadata[7] = True, "normal", False, False
        record = replace(record, metadata=tuple(metadata))
        snapshot = replace(
            DocumentSnapshot.capture(document),
            layers=(record,),
            active_layer=0,
            background=(0, 0, 0, 0),
        )
        row = Image.new("RGBA", (142, 40))
        row.alpha_composite(self.layer_row_icons[(layer.visible, layer.layer_type)], (0, 6))
        placeholder = cached[1] if cached else ImageTk.PhotoImage(row)
        self.layer_preview_cache[layer] = (None, placeholder)
        job = Job(
            document.id,
            document.generation,
            document.state_id,
            preview_key=f"layer-thumbnail:{layer.id}",
            priority=20,
        )

        def publish(result):
            pending.pop(layer.id, None)
            if result.error or layer not in document.layers:
                return
            row.alpha_composite(result.value, (92, 3))
            photo = ImageTk.PhotoImage(row)
            self.layer_preview_cache[layer] = (key, photo)
            if context_for(self).document is document:
                rows = self.layer_list.get_children()
                index = len(document.layers) - 1 - document.layers.index(layer)
                if index < len(rows):
                    self.layer_list.item(rows[index], image=photo)

        try:
            self.jobs.submit(job, lambda token: render_thumbnail(snapshot), 16 * 1024**2)
            pending[layer.id] = job.operation_id
            self.job_callbacks[job.operation_id] = publish
        except RuntimeError:
            self._schedule_layer_previews()
        return placeholder

    def _schedule_layer_previews(self):
        """Debounce previews, and never build them while a stroke is active."""
        self.layer_preview_dirty = True
        if not hasattr(self, "layer_list") or self.raster_stroke_active:
            return
        if self.layer_preview_after_id is not None:
            self.root.after_cancel(self.layer_preview_after_id)
        self.layer_preview_after_id = self.root.after(400, self._update_layer_previews)

    def _update_layer_previews(self):
        self.layer_preview_after_id = None
        if self.raster_stroke_active:
            return
        self.layer_preview_dirty = False
        self._update_document_preview()
        rows = self.layer_list.get_children()
        if len(rows) != len(self.layers):
            return
        for row, layer in zip(rows, reversed(self.layers)):
            self.layer_list.item(row, image=self._layer_row_preview(layer))

    def on_layer_pointer_down(self, event):
        row = self.layer_list.identify_row(event.y)
        if not row:
            return
        display_index = self.layer_list.index(row)

        row_box = self.layer_list.bbox(row, "#0")
        layer_index = len(self.layers) - 1 - display_index
        layer = self.layers[layer_index]

        # Vector rows put their table button before the visibility button.
        if row_box and layer.layer_type == "vector" and event.x < row_box[0] + 50:
            self.layer_list.selection_set(row)
            self.layer_list.focus(row)
            self.active_layer = layer_index
            self.update_layer_selection_style()
            self.update_tools_for_active_layer()
            self.show_vector_object_table(layer)
            return "break"

        visibility_start = row_box[0] + 50
        visibility_limit = row_box[0] + 81
        if row_box and visibility_start <= event.x < visibility_limit:
            layer.visible = not layer.visible
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

    def _update_layer_control_tooltip(self, event):
        """Describe the layer-row control currently beneath the pointer."""
        tooltip = self.layer_control_tooltip
        row = self.layer_list.identify_row(event.y)
        text = ""
        if row:
            row_box = self.layer_list.bbox(row, "#0")
            if row_box:
                display_index = self.layer_list.index(row)
                layer = self.layers[len(self.layers) - 1 - display_index]
                relative_x = event.x - row_box[0]
                if layer.layer_type == "vector" and relative_x < 50:
                    text = "Open vector object table"
                elif 50 <= relative_x < 81:
                    text = "Toggle layer visibility"
        if text == tooltip.text:
            return
        tooltip._hide()
        tooltip.text = text
        if text:
            tooltip._schedule()

    def show_vector_object_table(self, layer):
        """Open a compact QGIS-inspired attribute table for a vector layer."""
        if layer.layer_type != "vector" or layer.vector_data is None:
            return

        dialog = tk.Toplevel(self.root)
        dialog._neutral_background_clicks = True
        dialog.title(f"{layer.name} — Objects: {len(layer.vector_data.objects)}")
        dialog.geometry("760x360")
        dialog.minsize(560, 220)

        body = ttk.Frame(dialog, padding=8)
        body.pack(fill="both", expand=True)
        canvas = tk.Canvas(body, highlightthickness=0, background="#ffffff")
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        grid = tk.Frame(canvas, background="#9b9b9b")
        window = canvas.create_window((0, 0), window=grid, anchor="nw")
        grid.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))

        headings = ("#", "Type", "Name", "Left Color", "Right Color", "Size")
        weights = (0, 2, 3, 2, 2, 1)
        for column, (heading, weight) in enumerate(zip(headings, weights)):
            grid.grid_columnconfigure(column, weight=weight, minsize=42 if column == 0 else 90)
            tk.Label(
                grid,
                text=heading,
                font=("TkDefaultFont", 9, "bold"),
                background="#e5e5e5",
                relief="solid",
                borderwidth=1,
                anchor="w",
                padx=5,
                pady=4,
            ).grid(row=0, column=column, sticky="nsew")

        rows = []

        def select_object(obj, row_widgets):
            self.active_layer = self.layers.index(layer)
            self.set_tool("vector select")
            self.selected_vector_obj = obj
            self.selected_point_index = None
            self._load_selected_vector_attributes()
            for widgets in rows:
                for widget in widgets:
                    if str(widget.cget("state")) != "disabled":
                        widget.configure(background="white")
            for widget in row_widgets:
                if str(widget.cget("state")) != "disabled":
                    widget.configure(background="#cce8ff")
            self.request_redraw()

        def commit(obj, field, variable):
            value = variable.get().strip()
            try:
                if field == "name":
                    new_value = None if not value or value.upper() == "NULL" else value
                elif field == "size":
                    new_value = max(1, min(999, int(value)))
                else:
                    normalized = value.lstrip("#")
                    if len(normalized) not in (6, 8):
                        raise ValueError
                    int(normalized, 16)
                    new_value = "#" + normalized.lower()
            except ValueError:
                current = (
                    obj.name
                    if field == "name"
                    else obj.width
                    if field == "size"
                    else obj.color
                    if field == "left_color"
                    else obj.fill
                )
                variable.set("NULL" if current is None else str(current))
                return

            self.snapshot()
            if field == "name":
                obj.name = new_value
                variable.set("NULL" if new_value is None else new_value)
            elif field == "size":
                obj.width = new_value
                if isinstance(obj, Shape):
                    for line in obj.lines:
                        line.width = new_value
                variable.set(str(new_value))
            elif field == "left_color":
                obj.color = new_value
                if isinstance(obj, Shape):
                    for line in obj.lines:
                        line.color = new_value
            elif field == "right_color" and isinstance(obj, Shape):
                obj.fill = new_value
                obj._spherical_fill_cache = None
            layer.render_vector(regional=True)
            if self.selected_vector_obj is obj:
                self._load_selected_vector_attributes()
            self._schedule_layer_previews()
            self.request_redraw()
            self.notify_globe_document_changed()

        for index, obj in enumerate(layer.vector_data.objects, start=1):
            object_type = (
                {"rect": "Rectangle", "ellipse": "Ellipse"}.get(obj.preset, "Shape")
                if isinstance(obj, Shape)
                else obj.__class__.__name__
            )
            right_color = obj.fill if isinstance(obj, Shape) else None
            values = (
                str(index),
                object_type,
                obj.name if obj.name is not None else "NULL",
                obj.color,
                right_color if right_color is not None else "NULL",
                str(obj.width),
            )
            row_widgets = []
            for column, value in enumerate(values):
                variable = tk.StringVar(value=value)
                editable = column in (2, 3, 5) or (column == 4 and isinstance(obj, Shape))
                widget = tk.Entry(
                    grid,
                    textvariable=variable,
                    relief="solid",
                    borderwidth=1,
                    readonlybackground="white",
                    disabledbackground="#eeeeee",
                    disabledforeground="#777777",
                )
                widget.grid(row=index, column=column, sticky="nsew", ipady=4)
                if not editable:
                    widget.configure(state="disabled")
                else:
                    field = {2: "name", 3: "left_color", 4: "right_color", 5: "size"}[column]
                    widget.bind(
                        "<Return>", lambda _event, o=obj, f=field, v=variable: commit(o, f, v)
                    )
                    widget.bind(
                        "<FocusOut>", lambda _event, o=obj, f=field, v=variable: commit(o, f, v)
                    )
                    widget.bind(
                        "<Button-1>",
                        lambda _event, o=obj, w=row_widgets: select_object(o, w),
                        add="+",
                    )
                row_widgets.append(widget)
            rows.append(row_widgets)

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
        return ((sx - self.offset_x) / self.zoom, (sy - self.offset_y) / self.zoom)

    def raster_image_coords(self, sx, sy):
        """Map a screen point to Pillow's pixel-centre coordinate system."""
        x, y = self.image_coords(sx, sy)
        # image_coords() maps to pixel boundaries (pixel 0 spans 0..1), while
        # Pillow's raster primitives address pixel centres at integer values.
        # Without this conversion every dab is biased half a pixel right/down.
        return x - 0.5, y - 0.5

    def screen_coords(self, ix, iy):
        return (ix * self.zoom + self.offset_x, iy * self.zoom + self.offset_y)

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
        if self.tool == "magic wand":
            if event.num == 1:
                self._select_magic_wand(math.floor(x), math.floor(y), add=bool(event.state & 0x4))
            return
        if self.tool == "selection":
            control_down = bool(event.state & 0x4)
            operation = (
                "subtract"
                if control_down and event.num == 3
                else "add"
                if control_down and event.num == 1
                else "replace"
                if event.num == 1
                else None
            )
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
                self.selection_brush_last = self.raster_image_coords(event.x, event.y)
                self._paint_selection_brush(*self.selection_brush_last)
            return
        if self.tool == "clone":
            clone_x, clone_y = self.raster_image_coords(event.x, event.y)
            if event.state & 0x4:
                self.clone_source_center = (clone_x, clone_y)
                self.clone_offset = None
                self.request_redraw()
            elif event.num in (1, 3) and self.clone_source_center is not None:
                self.begin_external_clone(clone_x, clone_y, event.num)
            return
        if self.tool == "paint bucket":
            if self.bucket_pending is not None:
                self._finish_bucket_preview()
                return
            self._begin_bucket_preview(math.floor(x), math.floor(y), event.num, event.serial)
            return
        current_layer = self.layers[self.active_layer]

        if current_layer.is_raster:
            self.start_raster_draw(event)
        else:  # vector layer
            self.start_vector_operation(event, x, y)

    def start_raster_draw(self, event):
        self.snapshot()
        self.raster_stroke_active = True
        if self.layer_preview_after_id is not None:
            self.root.after_cancel(self.layer_preview_after_id)
            self.layer_preview_after_id = None
        self._prepare_raster_stroke()
        self.last_x, self.last_y = self.raster_image_coords(event.x, event.y)
        # Stamp the initial point immediately so a click/tap without any
        # motion produces a dot, just as it does in the globe view.
        self.raster_paint_image(self.last_x, self.last_y)

    def start_vector_operation(self, event, x, y):
        if self.tool in ("vector select", "vector edit"):
            data = self.layers[self.active_layer].vector_data
            tolerance = 8 / self.zoom
            obj, point_idx = None, None
            selected = self.selected_vector_obj
            if selected is not None and not isinstance(selected, Point):
                rx, ry = rotation_handle(selected, self.zoom)
                if math.hypot(x - rx, y - ry) <= tolerance:
                    obj, point_idx = selected, "rotate"
            candidates = ([selected] if selected is not None else []) + [
                item
                for item in data.objects_in_region(
                    (x - tolerance, y - tolerance, x + tolerance, y + tolerance), reverse=True
                )
                if item is not selected
            ]
            if obj is None:
                for candidate in candidates:
                    for i, (px, py) in enumerate(candidate.get_points()):
                        if math.hypot(x - px, y - py) <= tolerance:
                            obj, point_idx = candidate, i
                            break
                    if obj is None:
                        cx, cy = vector_center(candidate)
                        if math.hypot(x - cx, y - cy) <= tolerance:
                            obj, point_idx = candidate, "move"
                    if obj is not None:
                        break
            if obj is None:
                obj = data.get_object_near(x, y, tolerance)
                point_idx = "move" if obj else None
            self.selected_vector_obj = obj
            self.selected_point_index = point_idx
            self.is_dragging_point = obj is not None
            if obj is not None:
                self.snapshot()
                self._vector_drag_original = copy.deepcopy(obj)
                self._vector_drag_start = (x, y)
            self._load_selected_vector_attributes()
            self.request_redraw()
        elif self.tool in ["line", "rect", "ellipse", "point"]:
            # Start drawing a new vector object
            self.vector_start_x = x
            self.vector_start_y = y
            self.snapshot()

    @staticmethod
    def _split_vector_color(color):
        """Return a UI hex color and alpha from a stored Pillow color."""
        if isinstance(color, str) and color.startswith("#"):
            value = color[1:]
            if len(value) == 8:
                return "#" + value[:6].lower(), int(value[6:], 16)
            if len(value) == 6:
                return "#" + value.lower(), 255
        return "#000000", 255

    def _load_selected_vector_attributes(self):
        """Load a selected vector object's properties into the editor controls."""
        obj = self.selected_vector_obj
        if obj is None:
            self.vector_points_var.set("Points: —")
            return

        self.size_var.set(str(max(1, min(999, int(round(obj.width))))))
        primary, primary_alpha = self._split_vector_color(obj.color)
        secondary, secondary_alpha = self._split_vector_color(
            obj.fill if isinstance(obj, Shape) and obj.fill else obj.color
        )
        self.primary_color, self.primary_opacity = primary, primary_alpha
        self.secondary_color, self.secondary_opacity = secondary, secondary_alpha
        self.color = primary
        self.primary_square.config(bg=primary)
        self.secondary_square.config(bg=secondary)
        self._sync_picker_to_active_color()

        self._refresh_selected_vector_points()

    def _refresh_selected_vector_points(self):
        obj = self.selected_vector_obj
        if obj is None:
            self.vector_points_var.set("Points: —")
            return

        def number(value):
            return str(int(value)) if float(value).is_integer() else f"{value:.2f}"

        point_text = ", ".join(f"({number(px)}, {number(py)})" for px, py in obj.get_points())
        self.vector_points_var.set(f"Points: {point_text}")

    def _apply_selected_vector_size(self):
        """Apply the Size control while Vector Select owns an object."""
        obj = getattr(self, "selected_vector_obj", None)
        if self.tool != "vector select" or obj is None:
            return
        self.snapshot("Vector style")
        obj.width = self.vector_line_width()
        if isinstance(obj, Shape):
            for line in obj.lines:
                line.width = obj.width
        self._render_selected_vector_change()

    def _apply_selected_vector_color(self, slot):
        """Apply the left stroke or right fill color to the selected object."""
        obj = getattr(self, "selected_vector_obj", None)
        if self.tool != "vector select" or obj is None:
            return
        self.snapshot("Vector style")
        color = self._color_with_opacity(slot)
        if slot == "primary":
            obj.color = color
            if isinstance(obj, Shape):
                for line in obj.lines:
                    line.color = color
        elif isinstance(obj, Shape):
            obj.fill = color
            obj._spherical_fill_cache = None
        elif hasattr(obj, "fill"):
            obj.fill = color
        self._render_selected_vector_change()

    def _render_selected_vector_change(self):
        layer = self.layers[self.active_layer]
        if layer.layer_type != "vector" or layer.vector_data is None:
            return
        layer.render_vector(regional=True)
        self.request_redraw()
        self.notify_globe_document_changed()

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
        if self.tool in ("paint bucket", "magic wand"):
            return
        current_layer = self.layers[self.active_layer]

        if current_layer.is_raster:
            self.raster_paint(event)
        else:  # vector layer
            self.vector_operation(event, x, y)

    def on_mousewheel(self, event):
        """Handle plain scroll wheel for panning up/down and shift+scroll for left/right"""
        # Get scroll amount (cross-platform)
        if hasattr(event, "delta"):
            delta = event.delta
        elif hasattr(event, "num"):
            # Linux mouse wheel
            if event.num == 4:
                delta = 120  # scroll up
            elif event.num == 5:
                delta = -120  # scroll down
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
        self.request_redraw(navigation=True)

    def begin_external_raster_draw(self, x, y, button=1):
        """
        Begin a brush stroke from an external input source
        (such as the globe window).
        """
        self.snapshot()
        self.raster_stroke_active = True
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
        self.raster_stroke_active = False
        # Replace incremental display patches with one authoritative frame so
        # fractional zoom rounding cannot leave hairline seams on screen.
        self.request_redraw()
        if self.layer_preview_dirty:
            self._schedule_layer_previews()

    def _prepare_raster_stroke(self):
        session = context_for(self).session
        session.stroke = RasterGesture(
            context_for(self).document, self.layers[self.active_layer], self.tool.title()
        )
        capped = self.tool == "pencil" or not self.brush_build_up_var.get()
        self._stroke_base_image = session.stroke.baseline if capped else None
        self._stroke_coverage = session.stroke.coverage if capped else None
        self._build_up_base_image = session.stroke.baseline if not capped else None
        self._build_up_coverage = session.stroke.coverage if not capped else None

    def _finish_raster_stroke(self):
        session = context_for(self).session
        if session.stroke is not None:
            session.stroke.commit()
            session.stroke = None
        self._stroke_base_image = None
        self._stroke_coverage = None
        self._build_up_base_image = None
        self._build_up_coverage = None

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

    def brush_flow(self):
        """Return build-up flow as a validated percentage."""
        try:
            return max(0, min(100, int(self.brush_flow_var.get())))
        except (tk.TclError, ValueError):
            return 25

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

    def clone_flow(self):
        try:
            return max(0, min(100, int(self.clone_flow_var.get())))
        except (tk.TclError, ValueError):
            return 25

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
        self.snapshot("Bucket fill")
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
        if self.doc_w * self.doc_h > 1024 * 1024 and hasattr(self, "jobs"):
            self._background_bucket(pending)
            return
        layer = self.layers[pending["layer_index"]]
        layer.image = pending["original"].copy()
        layer.reset_mipmaps()
        pixel_x, pixel_y = pending["pixel_x"], pending["pixel_y"]

        fill_mask = self._bucket_region_mask(pending["original"], pixel_x, pixel_y)
        fill_mask = self._clip_raster_mask_to_selection((0, 0, self.doc_w, self.doc_h), fill_mask)
        dirty_box = fill_mask.getbbox()
        if dirty_box is None:
            self.request_redraw()
            return

        local_mask = fill_mask.crop(dirty_box)
        color = self._color_with_opacity("primary" if pending["button"] == 1 else "secondary")
        source = Image.new("RGBA", local_mask.size, color)
        source.putalpha(mask_multiply(source.getchannel("A"), local_mask))
        result = layer.image.crop(dirty_box)
        result.alpha_composite(source)
        self.apply_raster_result(layer, result, dirty_box)
        layer.update_mipmaps(dirty_box)
        self.request_redraw()

    def _background_bucket(self, pending):
        from pypaint.tools import compute

        document = context_for(self).document
        if pending.get("job"):
            pending["job"].cancellation.cancel()
        source = pending["original"].snapshot()
        selection = self.selection_mask if self._selection_pixel_box() is not None else None
        if selection is not None:
            selection = (
                selection.snapshot()
                if isinstance(selection, TiledSurface)
                else TiledSurface.from_image(selection).snapshot()
            )
        values = (
            (pending["pixel_x"], pending["pixel_y"]),
            self.bucket_tolerance(),
            self.bucket_antialias_var.get(),
            self.bucket_hardness(),
            self._color_with_opacity("primary" if pending["button"] == 1 else "secondary"),
        )
        job = Job(
            document.id, document.generation, document.state_id, preview_key="bucket", priority=0
        )
        pending["job"] = job

        def publish(result):
            if self.bucket_pending is not pending:
                return
            pending["job"] = None
            if result.error:
                history_for(document).cancel(document)
                self.bucket_pending = None
                messagebox.showerror("Fill failed", str(result.error))
                self._history_changed()
                return
            surface, bounds = result.value
            layer = document.layers[pending["layer_index"]]
            layer.image = surface.copy()
            layer.reset_mipmaps()
            document.change(ChangeKind.RASTER, layer.id, (bounds,) if bounds else ())
            self.request_redraw()
            self.notify_globe_document_changed()

        self._submit_job(
            job, lambda token: compute(source, *values, selection, token), 48 * 1024**2, publish
        )

    def _bucket_region_mask(self, image, pixel_x, pixel_y, antialias=True):
        """Find a contiguous color region for both bucket and wand."""
        fill_mask = flood_region(image, pixel_x, pixel_y, self.bucket_tolerance())

        if antialias and self.bucket_antialias_var.get():
            fill_mask = fill_mask.filter(ImageFilter.GaussianBlur(0.65))
            fill_mask = _apply_hardness_to_alpha(fill_mask, self.bucket_hardness(), 2)
        return fill_mask

    def _select_magic_wand(self, pixel_x, pixel_y, add=False):
        if not (0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h):
            return
        layer = self.layers[self.active_layer]
        if not layer.is_raster:
            return
        self._finish_clipboard_edit()
        self.wand_pending = {
            "layer": layer,
            "original": layer.image.copy(),
            "seed": (pixel_x, pixel_y),
            "base_mask": self.selection_mask.copy() if add else None,
        }
        self._refresh_wand_selection()

    def _refresh_wand_selection(self):
        """Rebuild the last wand click from its fixed source and prior selection."""
        pending = self.wand_pending
        if pending is None or self.tool != "magic wand":
            return
        if self.layers[self.active_layer] is not pending["layer"]:
            self.wand_pending = None
            return
        if self.doc_w * self.doc_h > 1024 * 1024 and hasattr(self, "jobs"):
            document = context_for(self).document
            source = pending["original"].snapshot()
            tolerance = self.bucket_tolerance()
            seed = pending["seed"]
            base = pending["base_mask"]
            if base is not None:
                base = (
                    base.snapshot()
                    if isinstance(base, TiledSurface)
                    else TiledSurface.from_image(base).snapshot()
                )
            job = Job(
                document.id, document.generation, document.state_id, preview_key="wand", priority=0
            )

            def compute(token):
                region = flood_region(source, *seed, tolerance, token)
                return combine_masks(base, region, "lighter", token) if base is not None else region

            def publish(result):
                if self.wand_pending is not pending:
                    return
                if result.error:
                    messagebox.showerror("Selection failed", str(result.error))
                    return
                self.selection_mask = result.value
                self._update_selection_geometry()
                self._ensure_selection_animation()
                self.request_redraw()

            self._submit_job(job, compute, 48 * 1024**2, publish)
            return
        region = self._bucket_region_mask(pending["original"], *pending["seed"], antialias=False)
        base = pending["base_mask"]
        self.selection_mask = mask_lighter(base, region) if base is not None else region
        self._update_selection_geometry()
        self._ensure_selection_animation()
        self.request_redraw()

    def _finish_bucket_preview(self, event=None):
        """Commit the currently displayed bucket preview."""
        if self.bucket_pending is None:
            return
        job = self.bucket_pending.get("job")
        if job is not None:
            job.cancellation.cancel()
            history_for(context_for(self).document).cancel(context_for(self).document)
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
                (
                    (pixel_x - left) * scale,
                    (pixel_y - top) * scale,
                    (pixel_x - left + 1) * scale - 1,
                    (pixel_y - top + 1) * scale - 1,
                ),
                fill=255,
            )
        else:
            effective_radius = radius if raster_radius == 0 else raster_radius
            bounds = (
                x - effective_radius,
                y - effective_radius,
                x + effective_radius,
                y + effective_radius,
            )
            paint_mask = lambda draw, left, top, scale: draw.ellipse(
                _brush_ellipse_box(bounds, left, top, scale), fill=255, outline=255
            )
        box, dab_mask = _brush_shape_mask(
            self.layers[self.active_layer].image,
            bounds,
            paint_mask,
            antialias=antialias,
            hardness=self.clone_hardness(),
        )
        if box is None:
            return None
        dab_mask = self._clip_raster_mask_to_selection(box, dab_mask)
        opacity = self.primary_opacity if self.last_button == 1 else self.secondary_opacity
        build_up = self.clone_build_up_var.get()
        mask_strength = opacity
        if build_up:
            flow = self.clone_flow() / 100.0
            spacing_fraction = min(1.0, self.clone_spacing() / 100.0)
            mask_strength = round((1.0 - ((1.0 - flow) ** spacing_fraction)) * 255)
            if flow > 0:
                mask_strength = max(1, mask_strength)
        if mask_strength < 255:
            dab_mask = dab_mask.point(
                lambda value: (value * mask_strength + 127) // 255
            )

        if self.clone_stroke_coverage is not None:
            previous = self.clone_stroke_coverage.crop(box)
            coverage = (
                _accumulate_build_up_mask(previous, dab_mask)
                if build_up
                else mask_lighter(previous, dab_mask)
            )
            self.clone_stroke_coverage.paste(coverage, (box[0], box[1]))
            composite_mask = (
                coverage.point(lambda value: min(value, opacity)) if build_up else coverage
            )
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
                    (clipped_left, clipped_top, clipped_right, clipped_bottom)
                ),
                (clipped_left - source_left, clipped_top - source_top),
            )
        source.putalpha(mask_multiply(source.getchannel("A"), composite_mask))
        result_base = (
            self.clone_stroke_base
            if self.clone_stroke_base is not None
            else self.layers[self.active_layer].image
        )
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
            dirty = self._stamp_clone(last_x + dx * amount, last_y + dy * amount)
            if dirty is not None:
                dirty_boxes.append(dirty)
        self.clone_last = (x, y)
        if dirty_boxes:
            dirty = (
                min(box[0] for box in dirty_boxes),
                min(box[1] for box in dirty_boxes),
                max(box[2] for box in dirty_boxes),
                max(box[3] for box in dirty_boxes),
            )
            self.layers[self.active_layer].update_mipmaps(dirty)
        self.request_redraw()

    def _finish_clone_stroke(self):
        if self.clone_stroke_source is None:
            return
        session = context_for(self).session
        if session.clone_gesture is not None:
            session.clone_gesture.commit()
            session.clone_gesture = None
        self.clone_last = None
        self.clone_stroke_source = None
        self.clone_stroke_base = None
        self.clone_stroke_coverage = None
        self.notify_globe_document_changed()

    def _ensure_selection_animation(self):
        """Start the marquee timer if it is not already running."""
        if self.selection_animation_id is None:
            self.selection_animation_id = self.root.after(70, self._animate_selection_marquee)

    def _start_selection_move(self, x, y):
        """Capture the selected raster pixels for an interactive move."""
        # A floating edit can extend beyond the document, where there is no
        # document-sized selection mask to hit-test.  Its own alpha is the
        # authoritative hit area until the edit is committed.
        if self.move_pixels is not None:
            source_left, source_top, _, _ = self.move_source_box
            local_x = math.floor(x - source_left - self.move_offset[0])
            local_y = math.floor(y - source_top - self.move_offset[1])
            if (
                0 <= local_x < self.move_pixels.width
                and 0 <= local_y < self.move_pixels.height
                and self.move_pixels.getpixel((local_x, local_y))[3] > 0
            ):
                self.move_start = (x, y)
                self.move_drag_origin_offset = self.move_offset
            return
        if self.selection_bounds is None:
            return
        if not self._point_in_selection(x, y):
            return

        box = self._selection_pixel_box()
        if box is None:
            return
        box = (max(0, box[0]), max(0, box[1]), min(self.doc_w, box[2]), min(self.doc_h, box[3]))
        if box[2] <= box[0] or box[3] <= box[1]:
            return

        layer = self.layers[self.active_layer]
        if not layer.is_raster:
            return
        self.snapshot("Move selection")
        self.move_start = (x, y)
        self.move_is_paste = False
        self.move_source_box = box
        self.move_pixels = layer.image.region_surface(box)
        if not isinstance(self.selection_mask, TiledSurface):
            self.selection_mask = TiledSurface.from_image(self.selection_mask)
        self.move_mask = self.selection_mask.region_surface(box)
        moved_alpha = mask_multiply(self.move_pixels.getchannel("A"), self.move_mask)
        self.move_pixels.putalpha(moved_alpha)
        self.move_base_image = layer.image.copy()
        self.move_base_image.paste((0, 0, 0, 0), box, self.move_mask)
        self.move_selection_bounds = self.selection_bounds
        self.move_selection_edges = [
            (x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1])
            for x0, y0, x1, y1 in self.selection_edges
        ]
        self.move_offset = (0, 0)
        self.move_drag_origin_offset = (0, 0)

    def _update_selection_move(self, x, y):
        """Preview selected pixels at the current drag position."""
        if self.move_start is None:
            return
        source_left, source_top, source_right, source_bottom = self.move_source_box
        dx = self.move_drag_origin_offset[0] + round(x - self.move_start[0])
        dy = self.move_drag_origin_offset[1] + round(y - self.move_start[1])

        layer = self.layers[self.active_layer]
        # Keep the layer at its cut-out base while the pixels float.  Drawing
        # the float as an overlay preserves pixels beyond the document bounds.
        base_changed = layer.image is not self.move_base_image
        if base_changed:
            layer.image = self.move_base_image
            layer.draw = None
            layer.reset_mipmaps()

        self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 0)
        self.selection_mask.paste(self.move_mask, (source_left + dx, source_top + dy))
        self._update_selection_geometry()
        self.move_offset = (dx, dy)
        if base_changed:
            self.request_redraw()
        else:
            self.request_overlay_redraw()

    def _release_selection_move(self):
        """End one drag while keeping the selected pixels floating."""
        if self.move_pixels is not None:
            self.move_start = None
            self.move_drag_origin_offset = self.move_offset

    def _finish_selection_move(self):
        """Finish an interactive move and release its temporary images."""
        if self.move_pixels is None:
            return
        changed = self.move_is_paste or self.move_offset != (0, 0)
        source_left, source_top, _, _ = self.move_source_box
        layer = self.layers[self.active_layer]
        layer.image = self.move_base_image.copy()
        layer.image.alpha_composite(
            self.move_pixels, (source_left + self.move_offset[0], source_top + self.move_offset[1])
        )
        layer.reset_mipmaps()
        self.move_start = None
        self.move_source_box = None
        self.move_pixels = None
        self.move_is_paste = False
        self.move_mask = None
        self.move_base_image = None
        self.move_selection_bounds = None
        self.move_selection_edges = None
        self.move_offset = (0, 0)
        self.move_drag_origin_offset = (0, 0)
        if not changed and self.undo_stack:
            # Avoid adding an undo step for a click without movement.
            discarded = self.undo_stack.pop()
            context_for(self).document.state_id = discarded.state_id
        if changed:
            self.notify_globe_document_changed()

    def _start_selection_boundary_move(self, x, y):
        """Begin moving only the selection marquee, leaving pixels untouched."""
        if self.selection_bounds is None and self.move_pixels is None:
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
        self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 0)
        self.selection_mask.paste(self.selection_move_mask, (round(dx), round(dy)))
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
            bounds = (
                center_x - raster_radius,
                center_y - raster_radius,
                center_x + raster_radius,
                center_y + raster_radius,
            )
            box, dab = _brush_shape_mask(
                self.selection_mask,
                bounds,
                lambda draw, left, top, scale, shape=bounds: draw.ellipse(
                    _brush_ellipse_box(shape, left, top, scale), fill=255, outline=255
                ),
                antialias=False,
            )
            if box is not None:
                existing = self.selection_mask.crop(box)
                if self.selection_brush_remove:
                    combined = mask_multiply(existing, ImageOps.invert(dab))
                else:
                    combined = mask_lighter(existing, dab)
                self.selection_mask.paste(combined, (box[0], box[1]))

        self.selection_brush_last = (x, y)
        self._update_selection_geometry()
        self._ensure_selection_animation()
        self.request_redraw()

    def _animate_selection_marquee(self):
        """Advance the selection dashes without rerendering the document."""
        self.selection_animation_id = None
        if self.selection_bounds is None and self.move_pixels is None:
            return
        self.selection_dash_offset = (self.selection_dash_offset + 1) % 10
        if self.active_view != "main":
            self._ensure_selection_animation()
            return
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
        if (
            (self.selection_bounds is None and self.move_pixels is None)
            or not self.layers
            or not self.layers[self.active_layer].is_raster
        ):
            return
        tags = ("overlay", "selection_marquee")
        if self.move_pixels is not None:
            source_left, source_top, _, _ = self.move_source_box
            origin_x = source_left + self.move_offset[0]
            origin_y = source_top + self.move_offset[1]
            edges = [
                (origin_x + x0, origin_y + y0, origin_x + x1, origin_y + y1)
                for x0, y0, x1, y1 in self.move_selection_edges
            ]
        else:
            edges = list(self.selection_edges)
        if self.selection_start is not None:
            left, top, right, bottom = self.selection_bounds
            left, right = sorted((left, right))
            top, bottom = sorted((top, bottom))
            left = max(0, min(self.doc_w, left))
            right = max(0, min(self.doc_w, right))
            top = max(0, min(self.doc_h, top))
            bottom = max(0, min(self.doc_h, bottom))
            left, top, right, bottom = self._pixel_box_from_bounds((left, top, right, bottom))
            edges.extend(
                (
                    (left, top, right, top),
                    (right, top, right, bottom),
                    (right, bottom, left, bottom),
                    (left, bottom, left, top),
                )
            )

        # Draw the bright portions as actual moving segments rather than a
        # Tk dash pattern. Windows Tk can cache dashed rectangles and ignore
        # dashoffset changes, whereas changing line coordinates always paints.
        dash_length = 6
        period = 10
        viewport_width = self.canvas.winfo_width()
        viewport_height = self.canvas.winfo_height()

        def draw_moving_edge(start_x, start_y, end_x, end_y):
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.hypot(dx, dy)
            if length <= 0:
                return
            unit_x, unit_y = dx / length, dy / length
            visible_start, visible_end = 0, length
            for origin, direction, limit in (
                (start_x, unit_x, viewport_width),
                (start_y, unit_y, viewport_height),
            ):
                if direction == 0:
                    if origin < -2 or origin > limit + 2:
                        return
                else:
                    low, high = sorted(
                        ((-2 - origin) / direction, (limit + 2 - origin) / direction)
                    )
                    visible_start = max(visible_start, low)
                    visible_end = min(visible_end, high)
            if visible_end <= visible_start:
                return
            self.canvas.create_line(
                start_x + unit_x * visible_start,
                start_y + unit_y * visible_start,
                start_x + unit_x * visible_end,
                start_y + unit_y * visible_end,
                fill="black",
                width=3,
                tags=tags,
            )
            distance = self.selection_dash_offset - period
            distance += (
                max(0, math.floor((visible_start - distance - dash_length) / period)) * period
            )
            while distance < visible_end:
                segment_start = max(visible_start, distance)
                segment_end = min(visible_end, distance + dash_length)
                if segment_end > segment_start:
                    self.canvas.create_line(
                        start_x + unit_x * segment_start,
                        start_y + unit_y * segment_start,
                        start_x + unit_x * segment_end,
                        start_y + unit_y * segment_end,
                        fill="white",
                        width=1,
                        tags=tags,
                    )
                distance += period

        for left, top, right, bottom in edges:
            x0, y0 = self.screen_coords(left, top)
            x1, y1 = self.screen_coords(right, bottom)
            draw_moving_edge(x0, y0, x1, y1)

    def _pixel_box_from_bounds(self, bounds):
        """Convert continuous document bounds to selected pixel boundaries."""
        left, top, right, bottom = bounds
        # A pixel belongs to the selection when its center lies within the
        # marquee. This keeps clipping aligned with the displayed boundary at
        # every zoom level, including selections made at fractional positions.
        return (
            math.ceil(left - 0.5),
            math.ceil(top - 0.5),
            math.floor(right - 0.5) + 1,
            math.floor(bottom - 0.5) + 1,
        )

    def _update_selection_geometry(self):
        self.selection_bounds = self.selection_mask.getbbox()
        self._selection_mask_bounds = self.selection_bounds
        self.selection_edges = (
            selection_outline(self.selection_mask) if self.selection_bounds else []
        )

    def _selection_pixel_box(self):
        """Return the active selection's exact nonempty pixel extent."""
        if self.selection_mask.size != (self.doc_w, self.doc_h):
            self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 0)
            self._selection_mask_bounds = None
        return self._selection_mask_bounds

    def _point_in_selection(self, x, y):
        """Return whether a document point lies in an actually selected pixel."""
        pixel_x, pixel_y = math.floor(x), math.floor(y)
        return (
            0 <= pixel_x < self.doc_w
            and 0 <= pixel_y < self.doc_h
            and self.selection_mask.getpixel((pixel_x, pixel_y)) > 0
        )

    def _selection_mask_for_box(self, box):
        """Build a selection mask local to a document-space patch box."""
        selection_box = self._selection_pixel_box()
        if selection_box is None:
            return Image.new("L", (box[2] - box[0], box[3] - box[1]), 255)
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

        write_mask = (
            self._selection_mask_for_box(box) if self._selection_pixel_box() is not None else None
        )
        if mask is not None:
            if mask.size != expected_size:
                raise ValueError("Raster edit mask must match its destination box")
            write_mask = (
                mask_multiply(write_mask, mask.convert("L"))
                if write_mask is not None
                else mask.convert("L")
            )
        session = context_for(self).session
        gesture = session.stroke or session.clone_gesture
        if gesture is not None:
            gesture.update(result, box, write_mask)
        else:
            layer.image.paste(result, (box[0], box[1]), write_mask)
            context_for(self).document.change(ChangeKind.RASTER, layer.id, (box,))
        return write_mask is None or write_mask.getbbox() is not None

    def _clip_raster_mask_to_selection(self, box, mask):
        """Restrict a raster tool mask to the active selection."""
        if self._selection_pixel_box() is None:
            return mask
        return mask_multiply(mask, self._selection_mask_for_box(box))

    def _paint_brush_shape(self, layer, bounds, color, paint_mask, softness_scale=None):
        """Paint one dab, optionally capping coverage for the current stroke."""
        antialias = self.brush_antialias_var.get()
        box, dab_mask = _brush_shape_mask(
            layer.image,
            bounds,
            paint_mask,
            antialias=antialias,
            hardness=self.brush_hardness(),
            softness_scale=softness_scale,
        )
        if box is None:
            return None
        dab_mask = self._clip_raster_mask_to_selection(box, dab_mask)

        if self._stroke_base_image is None or self._stroke_coverage is None:
            source = Image.new("RGBA", dab_mask.size, color)
            source.putalpha(mask_multiply(source.getchannel("A"), dab_mask))
            result = layer.image.crop(box)
            result.alpha_composite(source)
            self.apply_raster_result(layer, result, box)
            return box

        coverage = self._stroke_coverage.crop(box)
        coverage = mask_lighter(coverage, dab_mask)
        self._stroke_coverage.paste(coverage, (box[0], box[1]))

        result = self._stroke_base_image.crop(box)
        source = Image.new("RGBA", coverage.size, color)
        source.putalpha(mask_multiply(source.getchannel("A"), coverage))
        result.alpha_composite(source)
        self.apply_raster_result(layer, result, box)
        return box

    def _erase_brush_shape(self, layer, bounds, paint_mask, softness_scale=None):
        """Erase through a hard or anti-aliased mask with stroke buildup rules."""
        box, dab_mask = _brush_shape_mask(
            layer.image,
            bounds,
            paint_mask,
            antialias=self.brush_antialias_var.get(),
            hardness=self.brush_hardness(),
            softness_scale=softness_scale,
        )
        if box is None:
            return None
        dab_mask = self._clip_raster_mask_to_selection(box, dab_mask)

        # Eraser strength follows the opacity of the color slot associated
        # with the mouse button, just like Brush chooses its paint opacity.
        eraser_opacity = self.primary_opacity if self.last_button == 1 else self.secondary_opacity
        if eraser_opacity < 255:
            dab_mask = dab_mask.point(lambda value: (value * eraser_opacity + 127) // 255)

        if self._stroke_base_image is not None and self._stroke_coverage is not None:
            coverage = self._stroke_coverage.crop(box)
            coverage = mask_lighter(coverage, dab_mask)
            self._stroke_coverage.paste(coverage, (box[0], box[1]))
            result = self._stroke_base_image.crop(box)
            source_alpha = result.getchannel("A")
            result.putalpha(mask_multiply(source_alpha, ImageOps.invert(coverage)))
        else:
            result = layer.image.crop(box)
            source_alpha = result.getchannel("A")
            result.putalpha(mask_multiply(source_alpha, ImageOps.invert(dab_mask)))

        self.apply_raster_result(layer, result, box)
        return box

    def raster_paint_image(self, x, y):
        """
        Paint using image coordinates instead of a Tk mouse event.
        """

        if self.tool == "pencil":
            self._paint_pencil(x, y)
            return

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
            color = self._color_with_opacity("primary" if self.last_button == 1 else "secondary")

        dx = x - self.last_x
        dy = y - self.last_y

        dist = math.hypot(dx, dy)

        spacing = max(1, radius * 2 * self.brush_spacing() / 100)
        # Round upward so the distance between adjacent stamps never exceeds
        # the requested spacing. Flooring this value could leave one- or
        # two-pixel holes in thin strokes between mouse-motion events.
        steps = max(1, math.ceil(dist / spacing))

        points = []
        for i in range(steps + 1):
            t = i / steps
            points.append((self.last_x + dx * t, self.last_y + dy * t))

        # A single union box for a long diagonal stroke can cover most of a
        # large document even though only a thin path changed. Keep composite
        # patches bounded so fast pointer events do work proportional to the
        # painted path instead of its potentially enormous bounding rectangle.
        max_center_span = max(128, min(384, round(radius * 4)))
        chunks = []
        chunk = [points[0]]
        chunk_left = chunk_right = points[0][0]
        chunk_top = chunk_bottom = points[0][1]
        for point in points[1:]:
            next_left = min(chunk_left, point[0])
            next_right = max(chunk_right, point[0])
            next_top = min(chunk_top, point[1])
            next_bottom = max(chunk_bottom, point[1])
            if next_right - next_left > max_center_span or next_bottom - next_top > max_center_span:
                chunks.append(chunk)
                # Repeat the boundary sample so adjacent chunks cannot leave
                # a subpixel seam at their join.
                chunk = [chunk[-1], point]
                chunk_left = min(chunk[0][0], point[0])
                chunk_right = max(chunk[0][0], point[0])
                chunk_top = min(chunk[0][1], point[1])
                chunk_bottom = max(chunk[0][1], point[1])
            else:
                chunk.append(point)
                chunk_left, chunk_right = next_left, next_right
                chunk_top, chunk_bottom = next_top, next_bottom
        chunks.append(chunk)

        dirty_box = None
        for chunk in chunks:
            chunk_box = self._paint_brush_segment(chunk, radius, color)
            if chunk_box is not None:
                dirty_box = self._union_boxes(dirty_box, chunk_box)
        if dirty_box is not None:
            self.layers[self.active_layer].update_mipmaps(dirty_box)

        self.last_x = x
        self.last_y = y

        self.request_redraw(defer=True, dirty_box=dirty_box)
        self.notify_globe_document_changed()

    def _paint_brush_segment(self, points, radius, color):
        """Rasterize and apply all interpolated dabs from one motion event.

        Mask creation remains local to each dab so subpixel antialiasing and
        hardness are unchanged. The expensive layer crop/composite and mipmap
        refresh, however, happen only once for the complete event segment.
        """
        layer = self.layers[self.active_layer]
        radius = max(0.5, radius)
        raster_radius = max(0, radius - 0.5)
        antialias = self.brush_antialias_var.get()
        hardness = self.brush_hardness()
        build_up = self.brush_build_up_var.get()
        stroke_opacity = self.primary_opacity if self.last_button == 1 else self.secondary_opacity
        dab_opacity = stroke_opacity
        if build_up:
            dab_opacity = round(self.brush_flow() * 255 / 100)
        if build_up and 0 < dab_opacity < 255:
            # Neighboring dabs overlap heavily at normal brush spacing. If
            # every dab used the full selected flow, a single pass at the
            # default 12.5% spacing would apply it about eight times. Convert
            # the flow value to a per-dab value so one brush-width of travel
            # lands near the requested strength while repeated passes still
            # build up naturally.
            spacing_fraction = min(1.0, self.brush_spacing() / 100.0)
            normalized = 1.0 - ((1.0 - dab_opacity / 255.0) ** spacing_fraction)
            dab_opacity = max(1, round(normalized * 255))
        dabs = []

        # In capped-opacity mode, overlapping stamps are combined with MAX.
        # That is geometrically the same as one round-ended path, so render
        # the path in a single mask instead of allocating a mask per sample.
        # Build-up mode retains individual samples because overlap density is
        # intentionally visible there.
        if not build_up and len(points) > 1:
            render_points = list(points)
            dx = points[-1][0] - points[0][0]
            dy = points[-1][1] - points[0][1]
            distance = math.hypot(dx, dy)
            if distance:
                # Put the beginning of this mask slightly inside the previous
                # segment. Without this overlap, two separately area-sampled
                # half-pixels are combined with MAX and can leave a periodic
                # one-pixel seam perpendicular to the stroke direction.
                overlap = min(2.0, radius)
                render_points[0] = (
                    points[0][0] - dx / distance * overlap,
                    points[0][1] - dy / distance * overlap,
                )
            shape_radius = radius if raster_radius == 0 else raster_radius
            bounds = (
                min(point[0] for point in render_points) - shape_radius,
                min(point[1] for point in render_points) - shape_radius,
                max(point[0] for point in render_points) + shape_radius,
                max(point[1] for point in render_points) + shape_radius,
            )

            def paint_path(draw, left, top, scale):
                centers = [
                    ((px - left + 0.5) * scale, (py - top + 0.5) * scale)
                    for px, py in render_points
                ]
                width = max(1, round(radius * 2 * scale))
                draw.line(centers, fill=255, width=width)
                for px, py in (render_points[0], render_points[-1]):
                    end_bounds = (
                        px - shape_radius,
                        py - shape_radius,
                        px + shape_radius,
                        py + shape_radius,
                    )
                    draw.ellipse(
                        _brush_ellipse_box(end_bounds, left, top, scale), fill=255, outline=255
                    )

            box, mask = _brush_shape_mask(
                layer.image,
                bounds,
                paint_path,
                antialias=antialias,
                hardness=hardness,
                softness_scale=radius * 0.5,
            )
            if box is not None:
                dabs.append((box, self._clip_raster_mask_to_selection(box, mask)))

        for x, y in () if dabs else points:
            if raster_radius == 0 and not antialias:
                px, py = round(x), round(y)
                bounds = (px, py, px, py)
                painter = lambda draw, left, top, scale, px=px, py=py: draw.rectangle(
                    (
                        (px - left) * scale,
                        (py - top) * scale,
                        (px - left + 1) * scale - 1,
                        (py - top + 1) * scale - 1,
                    ),
                    fill=255,
                )
            else:
                shape_radius = radius if raster_radius == 0 else raster_radius
                bounds = (x - shape_radius, y - shape_radius, x + shape_radius, y + shape_radius)
                painter = lambda draw, left, top, scale, bounds=bounds: draw.ellipse(
                    _brush_ellipse_box(bounds, left, top, scale), fill=255, outline=255
                )
            box, mask = _brush_shape_mask(
                layer.image, bounds, painter, antialias=antialias, hardness=hardness
            )
            if box is not None:
                mask = self._clip_raster_mask_to_selection(box, mask)
                # Build-up mode applies opacity per dab; scaling before the
                # SCREEN union preserves the same overlap accumulation.
                if build_up and dab_opacity < 255:
                    mask = mask.point(
                        lambda value, opacity=dab_opacity: (value * opacity + 127) // 255
                    )
                dabs.append((box, mask))

        if not dabs:
            return None
        union = (
            min(box[0] for box, _ in dabs),
            min(box[1] for box, _ in dabs),
            max(box[2] for box, _ in dabs),
            max(box[3] for box, _ in dabs),
        )
        combined = (
            dabs[0][1]
            if len(dabs) == 1
            else Image.new("L", (union[2] - union[0], union[3] - union[1]), 0)
        )
        for box, mask in dabs if len(dabs) > 1 else ():
            offset = (box[0] - union[0], box[1] - union[1])
            existing = combined.crop(
                (offset[0], offset[1], offset[0] + mask.width, offset[1] + mask.height)
            )
            merged = (
                _accumulate_build_up_mask(existing, mask)
                if build_up
                else mask_lighter(existing, mask)
            )
            combined.paste(merged, offset)

        if self.tool == "eraser":
            if not build_up and dab_opacity < 255:
                combined = combined.point(lambda value: (value * dab_opacity + 127) // 255)
            if build_up:
                coverage = _accumulate_build_up_mask(
                    self._build_up_coverage.crop(union), combined
                )
                self._build_up_coverage.paste(coverage, union[:2])
                erase_mask = coverage.point(lambda value: min(value, stroke_opacity))
                result = self._build_up_base_image.crop(union)
            elif self._stroke_base_image is not None:
                coverage = mask_lighter(self._stroke_coverage.crop(union), combined)
                self._stroke_coverage.paste(coverage, union[:2])
                result = self._stroke_base_image.crop(union)
                erase_mask = coverage
            else:
                result = layer.image.crop(union)
                erase_mask = combined
            result.putalpha(mask_multiply(result.getchannel("A"), ImageOps.invert(erase_mask)))
        else:
            if build_up:
                coverage = _accumulate_build_up_mask(
                    self._build_up_coverage.crop(union), combined
                )
                self._build_up_coverage.paste(coverage, union[:2])
                paint_mask = coverage.point(lambda value: min(value, stroke_opacity))
                result = self._build_up_base_image.crop(union)
            elif self._stroke_base_image is not None:
                coverage = mask_lighter(self._stroke_coverage.crop(union), combined)
                self._stroke_coverage.paste(coverage, union[:2])
                result = self._stroke_base_image.crop(union)
                paint_mask = coverage
            else:
                result = layer.image.crop(union)
                paint_mask = combined
            # ``_color_with_opacity`` returns a Pillow hex color string, not
            # an RGBA tuple. Build-up mode applies opacity through the mask,
            # so use the selected RGB at full alpha here.
            source_color = color[:7] if build_up else color
            source = Image.new("RGBA", paint_mask.size, source_color)
            source.putalpha(mask_multiply(source.getchannel("A"), paint_mask))
            result.alpha_composite(source)

        self.apply_raster_result(layer, result, union)
        return union

    def _paint_pencil(self, x, y):
        # Raster coordinates place integer values at pixel centers.
        x1, y1 = math.floor(x + 0.5), math.floor(y + 0.5)
        x0 = math.floor(self.last_x + 0.5) if self.last_x is not None else x1
        y0 = math.floor(self.last_y + 0.5) if self.last_y is not None else y1
        self.last_x, self.last_y = x, y
        left, top = max(0, min(x0, x1)), max(0, min(y0, y1))
        right = min(self.doc_w, max(x0, x1) + 1)
        bottom = min(self.doc_h, max(y0, y1) + 1)
        if right <= left or bottom <= top:
            return
        box = (left, top, right, bottom)
        mask = Image.new("L", (right - left, bottom - top), 0)
        for px, py in _pencil_path(x0, y0, x1, y1):
            if left <= px < right and top <= py < bottom:
                mask.putpixel((px - left, py - top), 255)
        mask = self._clip_raster_mask_to_selection(box, mask)
        layer = self.layers[self.active_layer]
        if self._stroke_coverage is not None:
            mask = mask_lighter(self._stroke_coverage.crop(box), mask)
            self._stroke_coverage.paste(mask, (left, top))
            result = self._stroke_base_image.crop(box)
        else:
            result = layer.image.crop(box)
        source = Image.new(
            "RGBA",
            mask.size,
            self._color_with_opacity("primary" if self.last_button == 1 else "secondary"),
        )
        source.putalpha(mask_multiply(source.getchannel("A"), mask))
        result.alpha_composite(source)
        layer.image.paste(result, (left, top))
        layer.update_mipmaps(box)
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
        color = (
            (0, 0, 0, 0)
            if self.tool == "eraser"
            else self._color_with_opacity("primary" if self.last_button == 1 else "secondary")
        )
        self.draw_circle(x, y, radius, color)
        self.draw_circle(x - self.doc_w, y, radius, color)
        self.draw_circle(x + self.doc_w, y, radius, color)
        self.last_x, self.last_y = x, y
        if refresh:
            self.request_redraw()
            self.notify_globe_document_changed()

    def stamp_external_spherical_raster(
        self, footprint_uv, center_x, center_y, brush_softness_scale=None, refresh=True
    ):
        """Fill a globe-relative brush footprint on the equirectangular map.

        ``footprint_uv`` is the spherical brush boundary expressed in texture
        coordinates.  Its U values may extend beyond the map edges so the same
        shape can be drawn cleanly across the longitude seam.
        """
        if not self.can_paint_from_globe() or not footprint_uv:
            return

        color = (
            (0, 0, 0, 0)
            if self.tool == "eraser"
            else self._color_with_opacity("primary" if self.last_button == 1 else "secondary")
        )
        polygon = [(u * self.doc_w, v * self.doc_h) for u, v in footprint_uv]

        # Repeat the unwrapped polygon on both sides of the texture.  Draw all
        # copies into one mask before antialiasing/hardness is applied; treating
        # them as separate dabs would soften their artificial seam edges.
        layer = self.layers[self.active_layer]
        copies = [[(x + offset, y) for x, y in polygon] for offset in (-self.doc_w, 0, self.doc_w)]
        all_points = [point for points in copies for point in points]
        bounds = (
            min(x for x, _ in all_points),
            min(y for _, y in all_points),
            max(x for x, _ in all_points),
            max(y for _, y in all_points),
        )

        def paint_copies(draw, left, top, scale):
            for points in copies:
                draw.polygon(
                    [((x - left + 0.5) * scale, (y - top + 0.5) * scale) for x, y in points],
                    fill=255,
                )

        if self.tool == "eraser":
            dirty_box = self._erase_brush_shape(
                layer, bounds, paint_copies, softness_scale=brush_softness_scale
            )
        else:
            dirty_box = self._paint_brush_shape(
                layer, bounds, color, paint_copies, softness_scale=brush_softness_scale
            )
        if dirty_box is not None:
            layer.update_mipmaps(dirty_box)

        self.last_x, self.last_y = center_x, center_y
        if refresh:
            self.request_redraw()
            self.notify_globe_document_changed()

    def raster_paint(self, event):

        x, y = self.raster_image_coords(event.x, event.y)

        self.raster_paint_image(x, y)

    def open_globe_view(self):
        view_id = f"globe-{self.active_document}"
        if view_id in self.views:
            self.switch_view(view_id)
            return

        if self.doc_w != self.doc_h * 2:
            should_continue = messagebox.askyesno(
                "Globe View Aspect Ratio",
                "Globe View is designed for 2:1 equirectangular documents.\n\n"
                f"This document is {self.doc_w} x {self.doc_h}. Continue anyway?",
            )
            if not should_continue:
                return

        from pypaint.globe_view import GlobeView

        self.globe_window = GlobeView(self.view_host, self, view_id=view_id)
        self.globe_documents[view_id] = self.active_document
        label = f"{self.documents[self.active_document]['name']} · Globe"
        self.register_view(view_id, label, self.globe_window)
        self.switch_view(view_id)

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
                bounds = (x - radius, y - radius, x + radius, y + radius)
                paint_mask = lambda draw, left, top, scale: draw.ellipse(
                    _brush_ellipse_box(bounds, left, top, scale), fill=255, outline=255
                )
                if self.tool == "eraser":
                    dirty_box = self._erase_brush_shape(layer, bounds, paint_mask)
                else:
                    dirty_box = self._paint_brush_shape(layer, bounds, color, paint_mask)
                if dirty_box is not None:
                    layer.update_mipmaps(dirty_box)
                return

            px, py = round(x), round(y)
            if self.tool == "eraser":
                dirty_box = self._erase_brush_shape(
                    layer,
                    (px, py, px, py),
                    lambda draw, left, top, scale: draw.rectangle(
                        (
                            (px - left) * scale,
                            (py - top) * scale,
                            (px - left + 1) * scale - 1,
                            (py - top + 1) * scale - 1,
                        ),
                        fill=255,
                    ),
                )
            else:
                dirty_box = self._paint_brush_shape(
                    layer,
                    (px, py, px, py),
                    color,
                    lambda draw, left, top, scale: draw.rectangle(
                        (
                            (px - left) * scale,
                            (py - top) * scale,
                            (px - left + 1) * scale - 1,
                            (py - top + 1) * scale - 1,
                        ),
                        fill=255,
                    ),
                )
            if dirty_box is not None:
                layer.update_mipmaps(dirty_box)
            return
        bounds = (x - raster_radius, y - raster_radius, x + raster_radius, y + raster_radius)
        if self.tool == "eraser":
            dirty_box = self._erase_brush_shape(
                layer,
                bounds,
                lambda draw, left, top, scale: draw.ellipse(
                    _brush_ellipse_box(bounds, left, top, scale), fill=255, outline=255
                ),
            )
        else:
            dirty_box = self._paint_brush_shape(
                layer,
                bounds,
                color,
                lambda draw, left, top, scale: draw.ellipse(
                    _brush_ellipse_box(bounds, left, top, scale), fill=255, outline=255
                ),
            )
        if dirty_box is not None:
            layer.update_mipmaps(dirty_box)

    def vector_operation(self, event, x, y):
        if self.is_dragging_point and self.selected_vector_obj:
            obj = self.selected_vector_obj
            original = self._vector_drag_original
            for name, value in original.__dict__.items():
                if not name.startswith("_") and name not in ("id", "revision"):
                    setattr(
                        obj, name, copy.deepcopy(list(value) if isinstance(value, list) else value)
                    )
            index = self.selected_point_index
            if index == "move":
                dx, dy = x - self._vector_drag_start[0], y - self._vector_drag_start[1]
                transform_vector(obj, lambda px, py: (px + dx, py + dy))
            elif index == "rotate":
                cx, cy = vector_center(original)
                sx, sy = self._vector_drag_start
                angle = math.atan2(y - cy, x - cx) - math.atan2(sy - cy, sx - cx)
                if event.state & 1:
                    angle = round(angle / (math.pi / 12)) * (math.pi / 12)
                c, s = math.cos(angle), math.sin(angle)
                transform_vector(
                    obj,
                    lambda px, py: (
                        cx + (px - cx) * c - (py - cy) * s,
                        cy + (px - cx) * s + (py - cy) * c,
                    ),
                )
            else:
                if isinstance(obj, Line) and event.state & 1:
                    x, y = snap_line_endpoint(obj.get_points()[1 - index], (x, y))
                if isinstance(obj, Shape):
                    obj.update_point(index, x, y, square=bool(event.state & 1))
                else:
                    obj.update_point(index, x, y)
            if self.tool == "vector select":
                self._refresh_selected_vector_points()
            self.layers[self.active_layer].render_vector(regional=True)
            self.request_redraw()
            self.notify_globe_document_changed()
        elif self.tool in ["line", "rect", "ellipse", "point"] and self.vector_start_x is not None:
            # Preview the shape (by redrawing)
            self.layers[self.active_layer].render_vector(regional=True)
            self.request_redraw()
            # Draw temporary preview
            if self.tool == "line" and event.state & 1:
                x, y = snap_line_endpoint((self.vector_start_x, self.vector_start_y), (x, y))
            self.draw_vector_preview(self.vector_start_x, self.vector_start_y, x, y)

    def draw_vector_preview(self, x1, y1, x2, y2):
        obj = self.make_vector_object(self.tool, [(x1, y1), (x2, y2)], "flat")
        self._vector_preview = (self.active_layer, obj) if obj else None
        try:
            self.redraw()
        finally:
            self._vector_preview = None

    def create_vector_object(self, x1, y1, x2, y2):
        """Create a vector object and add it to the current layer"""
        layer = self.layers[self.active_layer]
        if layer.layer_type != "vector":
            return

        obj = self.make_vector_object(self.tool, [(x1, y1), (x2, y2)], "flat")

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
        if preset == "point" and points:
            return Point(
                *points[-1],
                self._color_with_opacity("primary"),
                self.vector_line_width(),
                self.vector_antialias_enabled(),
                self.vector_hardness(),
            )
        if len(points) < 2:
            return None
        if preset == "line":
            return Line(
                *points[0],
                *points[-1],
                self._color_with_opacity("primary"),
                self.vector_line_width(),
                space=space,
                antialias=self.vector_antialias_enabled(),
                hardness=self.vector_hardness(),
            )
        if preset in ("rect", "ellipse"):
            if space == "flat" and len(points) == 2:
                (x1, y1), (x2, y2) = points
                # Keep an editable frame even for a horizontal/vertical gesture.
                if abs(x2 - x1) < 1:
                    x2 = x1 + math.copysign(1, x2 - x1)
                if abs(y2 - y1) < 1:
                    y2 = y1 + math.copysign(1, y2 - y1)
                points = [(x1, y1), (x2, y2)]
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
            lines = [
                Line(
                    *vertices[i],
                    *vertices[(i + 1) % 4],
                    color,
                    width,
                    space=space,
                    antialias=antialias,
                    hardness=hardness,
                )
                for i in range(4)
            ]
        else:
            if space == "flat":
                (x1, y1), (x2, y2) = points
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rx, ry = abs(x2 - x1) / 2, abs(y2 - y1) / 2
                k = 0.5522847498
                verts = [(cx + rx, cy), (cx, cy + ry), (cx - rx, cy), (cx, cy - ry)]
                controls = [
                    (cx + rx, cy + k * ry, cx + k * rx, cy + ry),
                    (cx - k * rx, cy + ry, cx - rx, cy + k * ry),
                    (cx - rx, cy - k * ry, cx - k * rx, cy - ry),
                    (cx + k * rx, cy - ry, cx + rx, cy - k * ry),
                ]
                lines = [
                    Line(
                        *verts[i],
                        *verts[(i + 1) % 4],
                        color,
                        width,
                        curve=controls[i],
                        space=space,
                        antialias=antialias,
                        hardness=hardness,
                    )
                    for i in range(4)
                ]
            else:
                vertices = points
                lines = [
                    Line(
                        *vertices[i],
                        *vertices[(i + 1) % len(vertices)],
                        color,
                        width,
                        space=space,
                        antialias=antialias,
                        hardness=hardness,
                    )
                    for i in range(len(vertices))
                ]
        # The secondary colour represents the shape's filled (inside) side.
        return Shape(
            lines,
            color,
            width,
            self._color_with_opacity("secondary"),
            "inside",
            preset,
            antialias,
            hardness,
        )

    def create_globe_vector(self, preset, image_points):
        if not self.can_draw_vector_from_globe() or len(image_points) < 2:
            return
        self.snapshot()
        obj = self.make_vector_object(preset, image_points, "globe")
        if obj is None:
            return
        self.layers[self.active_layer].vector_data.add_object(obj)
        self.layers[self.active_layer].render_vector(regional=True)
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
                    rectangle = TiledSurface("L", (self.doc_w, self.doc_h), 0)
                    pixel_box = self._pixel_box_from_bounds((left, top, right, bottom))
                    if pixel_box[2] > pixel_box[0] and pixel_box[3] > pixel_box[1]:
                        rectangle.paste(255, pixel_box)
                    if self.selection_operation == "add":
                        self.selection_mask = mask_lighter(self.selection_base_mask, rectangle)
                    elif self.selection_operation == "subtract":
                        self.selection_mask = mask_multiply(
                            self.selection_base_mask, rectangle.point(lambda value: 255 - value)
                        )
                    else:
                        self.selection_mask = rectangle
                elif self.selection_operation == "replace":
                    self.selection_mask = TiledSurface("L", (self.doc_w, self.doc_h), 0)
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
        if self.tool in ("paint bucket", "magic wand"):
            return
        current_layer = self.layers[self.active_layer]

        if current_layer.is_raster:
            self.last_x = None
            self.last_y = None
            self._finish_raster_stroke()
            self.raster_stroke_active = False
            # Partial PhotoImage updates deliberately favor responsiveness.
            # Rebuild the full visible composite at stroke completion to
            # remove any one-pixel joins caused by patch-coordinate rounding.
            self.request_redraw()
            if self.layer_preview_dirty:
                self._schedule_layer_previews()
        else:  # vector layer
            if self.is_dragging_point:
                self.vector_operation(event, x, y)
                self.is_dragging_point = False
                self.selected_point_index = None
                if self.tool == "vector select":
                    self._refresh_selected_vector_points()
                else:
                    self.selected_vector_obj = None
            elif (
                self.tool in ["line", "rect", "ellipse", "point"]
                and self.vector_start_x is not None
            ):
                if self.tool == "line" and event.state & 1:
                    x, y = snap_line_endpoint((self.vector_start_x, self.vector_start_y), (x, y))
                if (
                    self.tool == "point"
                    or abs(x - self.vector_start_x) > 2
                    or abs(y - self.vector_start_y) > 2
                ):
                    self.create_vector_object(self.vector_start_x, self.vector_start_y, x, y)
                self.vector_start_x = None
                self.vector_start_y = None
                self.layers[self.active_layer].render_vector(regional=True)
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
        self.request_redraw(navigation=True)

    def pick_color(self, event):
        """Sample one document pixel into the left or right color slot."""
        image_x, image_y = self.image_coords(event.x, event.y)
        self.pick_color_at(image_x, image_y, event.num, event.state)

    def pick_color_at(self, image_x, image_y, button=1, state=0):
        """Sample document coordinates supplied by any view."""
        self.last_button = button
        pixel_x, pixel_y = math.floor(image_x), math.floor(image_y)
        if not (0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h):
            return

        composite = bool(state & 0x4)
        if composite:  # Ctrl: sample the final visible composite.
            for layer in self.layers:
                if layer.visible and layer.layer_type == "vector" and layer.vector_data:
                    layer.render_vector(regional=True)
        else:
            layer = self.layers[self.active_layer]
            if layer.layer_type == "vector" and layer.vector_data:
                layer.render_vector(regional=True)

        if self.picker_sample_area_var.get():
            diameter = max(1, int(self.size_var.get()))
            radius = diameter / 2
            center_x, center_y = pixel_x + 0.5, pixel_y + 0.5
            left = max(0, math.floor(center_x - radius))
            top = max(0, math.floor(center_y - radius))
            right = min(self.doc_w, math.ceil(center_x + radius))
            bottom = min(self.doc_h, math.ceil(center_y + radius))
            box = (left, top, right, bottom)
            source = self.composite_region(box) if composite else layer.image.crop(box)
            radius_squared = radius * radius
            xs = np.arange(left, right) + 0.5 - center_x
            ys = np.arange(top, bottom) + 0.5 - center_y
            inside = ys[:, None] ** 2 + xs[None, :] ** 2 <= radius_squared
            samples = np.asarray(source)[inside]
            totals = samples.sum(axis=0, dtype=np.uint64)
            rgba = tuple(round(int(total) / len(samples)) for total in totals)
        elif composite:
            rgba = self.composite_region((pixel_x, pixel_y, pixel_x + 1, pixel_y + 1)).getpixel(
                (0, 0)
            )
        else:
            rgba = layer.image.getpixel((pixel_x, pixel_y))

        self.active_color_slot = "secondary" if self.last_button == 3 else "primary"
        if self.active_color_slot == "primary":
            self.primary_opacity = rgba[3]
        else:
            self.secondary_opacity = rgba[3]
        self._set_selected_color(self._rgb_to_hex(rgba[:3]))

    def set_external_clone_source(self, x, y):
        self.clone_source_center = (x, y)
        self.clone_offset = None

    def begin_external_clone(self, x, y, button=1):
        if self.clone_source_center is None:
            return False
        self.last_button = button
        if self.clone_offset is None:
            self.clone_offset = (
                round(self.clone_source_center[0] - x),
                round(self.clone_source_center[1] - y),
            )
        from pypaint.tools import CloneGesture

        session = context_for(self).session
        session.clone_gesture = CloneGesture(
            context_for(self).document, self.layers[self.active_layer]
        )
        self.clone_stroke_source = session.clone_gesture.source
        self.clone_stroke_base = self.clone_stroke_source
        self.clone_stroke_coverage = session.clone_gesture.coverage
        self.clone_last = (x, y)
        self._paint_clone(x, y)
        return True

    def continue_external_clone(self, x, y):
        self._paint_clone(x, y)
        self.notify_globe_document_changed()

    def end_external_clone(self):
        self._finish_clone_stroke()

    def zoom_mouse(self, event):
        self._zoom_at(event.x, event.y, event.delta > 0)

    def zoom_keyboard(self, direction):
        """Zoom the active view, keeping the flat canvas center fixed."""
        if self.active_view in self.globe_documents:
            globe = getattr(self, "globe_window", None)
            if globe is not None:
                (globe.zoom_in if direction > 0 else globe.zoom_out)()
            return
        self._zoom_at(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, direction > 0)

    def _zoom_at(self, x, y, zoom_in):
        old = self.zoom
        self.zoom *= 1.1 if zoom_in else (1 / 1.1)
        self.zoom = max(0.1, min(20, self.zoom))

        # Do not rebuild an identical frame when the wheel keeps moving after
        # reaching either zoom limit.
        if self.zoom == old:
            return

        ix = (x - self.offset_x) / old
        iy = (y - self.offset_y) / old

        self.offset_x = x - ix * self.zoom
        self.offset_y = y - iy * self.zoom
        # Wheel events can arrive much faster than a resampled image can be
        # uploaded to Tk. Coalesce the burst to at most one preview per frame,
        # then do exact compositing once input settles.
        display = getattr(self, "_display_surface", None)
        if display is not None and getattr(display, "flat", None) is not None:
            if self.redraw_after_id is not None:
                self.root.after_cancel(self.redraw_after_id)
                self.redraw_after_id = None
                self.pending_redraw_box = None
            if getattr(self, "zoom_preview_after_id", None) is None:
                elapsed = time.perf_counter() - getattr(self, "last_zoom_preview", 0.0)
                delay = max(1, math.ceil((self.target_frame_time - elapsed) * 1000))
                self.zoom_preview_after_id = self.root.after(delay, self._render_zoom_preview)
            if self.zoom_redraw_after_id is not None:
                self.root.after_cancel(self.zoom_redraw_after_id)
            self.zoom_redraw_after_id = self.root.after(75, self._finish_zoom_preview)
        else:
            self.request_redraw(navigation=True)

    def _render_zoom_preview(self):
        self.zoom_preview_after_id = None
        display = getattr(self, "_display_surface", None)
        if display is None or not display.preview_zoom(
            (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())),
            self.zoom,
            (self.offset_x, self.offset_y),
        ):
            self.request_redraw(navigation=True)
            return
        self.last_zoom_preview = time.perf_counter()
        self._draw_overlays()

    def _finish_zoom_preview(self):
        self.zoom_redraw_after_id = None
        preview_after_id = getattr(self, "zoom_preview_after_id", None)
        if preview_after_id is not None:
            self.root.after_cancel(preview_after_id)
            self.zoom_preview_after_id = None
        self.last_redraw = time.perf_counter()
        inputs = getattr(self, "_display_inputs", None)
        document = context_for(self).document
        if inputs is not None:
            snapshot = inputs[0]
            reusable = (
                snapshot.document_id == document.id
                and snapshot.generation == document.generation
                and snapshot.state_id == document.state_id
                and getattr(self, "_effect_preview", None) is None
                and getattr(self, "_vector_preview", None) is None
                and not self.raster_stroke_active
                and self.bucket_pending is None
                and self.move_pixels is None
            )
        else:
            reusable = False
        self.redraw(inputs=inputs if reusable else None)

    def mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.request_overlay_redraw()

    def _clipboard_text_focus(self, event):
        return event is not None and event.widget.winfo_class() in {
            "Entry",
            "TEntry",
            "Spinbox",
            "TSpinbox",
            "Text",
            "TCombobox",
        }

    def _finish_clipboard_edit(self):
        self._finish_bucket_preview()
        self._finish_raster_stroke()
        self._finish_clone_stroke()
        self._finish_selection_move()
        self._finish_selection_boundary_move()

    def copy_to_clipboard(self, event=None):
        if self._clipboard_text_focus(event):
            return
        self._finish_clipboard_edit()
        layer = self.layers[self.active_layer]
        if not layer.is_raster:
            layer.render_vector(regional=True)
        box = self._selection_pixel_box()
        pixels = layer.image.crop(box) if box else layer.image.copy()
        if box:
            pixels.putalpha(mask_multiply(pixels.getchannel("A"), self.selection_mask.crop(box)))
        try:
            copy_image(pixels, self.root.winfo_id())
        except Exception as error:
            messagebox.showerror("Copy", str(error))
        return "break"

    def paste_from_clipboard(self, event=None):
        if self._clipboard_text_focus(event):
            return
        layer = self.layers[self.active_layer]
        if not layer.is_raster:
            messagebox.showinfo("Paste", "Select a raster layer to paste pixels.")
            return "break"
        try:
            pixels = paste_image()
        except Exception as error:
            messagebox.showerror("Paste", str(error))
            return "break"
        if pixels is None:
            messagebox.showinfo("Paste", "The Windows clipboard does not contain an image.")
            return "break"

        paste_width, paste_height = pixels.size
        expanded_width = max(self.doc_w, paste_width)
        expanded_height = max(self.doc_h, paste_height)
        if (expanded_width, expanded_height) != (self.doc_w, self.doc_h):
            affected = []
            if expanded_width != self.doc_w:
                affected.append(f"width to {expanded_width}px")
            if expanded_height != self.doc_h:
                affected.append(f"height to {expanded_height}px")
            dimension_text = (
                affected[0] if len(affected) == 1 else f"{affected[0]} and {affected[1]}"
            )
            if expanded_width <= 32768 and expanded_height <= 32768:
                should_expand = messagebox.askyesno(
                    "Expand Canvas for Paste",
                    f"The pasted image is {paste_width} × {paste_height}px, "
                    f"which is larger than the {self.doc_w} × {self.doc_h}px "
                    f"canvas.\n\nExpand the canvas {dimension_text}?\n\n"
                    "Existing artwork will remain anchored at the top-left.",
                )
                if should_expand:
                    self.resize_canvas(expanded_width, expanded_height, "top-left")
            else:
                messagebox.showwarning(
                    "Canvas Size Limit",
                    "The pasted image exceeds the maximum canvas size of "
                    "32,768px in at least one dimension. It will be pasted "
                    "without expanding the canvas.",
                )

        self._finish_clipboard_edit()
        # Tool changes finish the current float; switch before starting paste.
        self.set_tool("move")
        self.snapshot()
        box = self._selection_pixel_box()
        left, top = box[:2] if box else (0, 0)
        right = left + pixels.width
        bottom = top + pixels.height
        self.move_source_box = (left, top, right, bottom)
        self.move_pixels = pixels.copy()
        self.move_mask = Image.new("L", self.move_pixels.size, 255)
        # Paste previews preserve the complete base instead of cutting it.
        self.move_base_image = layer.image.copy()
        self.move_is_paste = True
        self.move_selection_bounds = self.move_source_box
        self.move_selection_edges = [
            (0, 0, pixels.width, 0),
            (pixels.width, 0, pixels.width, pixels.height),
            (pixels.width, pixels.height, 0, pixels.height),
            (0, pixels.height, 0, 0),
        ]
        self.move_offset = (0, 0)
        self.move_drag_origin_offset = (0, 0)
        self.move_start = (left, top)
        self._update_selection_move(left, top)
        self._release_selection_move()
        self._ensure_selection_animation()
        self.canvas.focus_set()
        self.request_redraw()
        self.notify_globe_document_changed()
        return "break"

    def save_image(self):
        from pypaint.files import RESERVATION, export_raster, validate_size

        name = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if not name:
            return
        self._finish_clipboard_edit()
        document = context_for(self).document
        snapshot = DocumentSnapshot.capture(document)
        try:
            validate_size(snapshot.size)
        except ValueError as error:
            messagebox.showerror("Export", str(error))
            return
        job = Job(
            document.id,
            document.generation,
            document.state_id,
            publication="saved",
            resource_key=str(Path(name).resolve()).casefold(),
        )

        def publish(result):
            if result.error:
                messagebox.showerror("Export failed", str(result.error))

        # Separate worker renderer prevents long exports locking the viewport renderer.
        self._submit_job(
            job,
            lambda token: export_raster(snapshot, name, Renderer(8 * 1024**2), token),
            RESERVATION,
            publish,
        )

    def _renderer(self):
        if not hasattr(self, "renderer"):
            self.renderer = Renderer()
        return self.renderer

    def composite_image(self):
        return self._renderer().render(DocumentSnapshot.capture(context_for(self).document))

    @staticmethod
    def _compositing_layer_indices(layers):
        return _compositing_layer_indices(layers)

    @staticmethod
    def _apply_layer_masks(layers, rendered_layers, visible_only=False):
        return _apply_layer_masks(layers, rendered_layers, visible_only)

    def composite_region(self, box, output_size=None, *, inputs=None, sampling_grid=None):
        size = output_size or (box[2] - box[0], box[3] - box[1])
        reduction = max((box[2] - box[0]) / size[0], (box[3] - box[1]) / size[1])
        if reduction > 1:
            desired = max(0, int(math.log2(reduction)))
            required = self._compositing_layer_indices(self.layers)
            if any(len(self.layers[i]._mipmaps) <= desired for i in required):
                self.request_mipmap_level(desired)
        snapshot, pyramids = inputs if inputs is not None else self._composite_inputs()
        return self._renderer().render(
            snapshot, box, size, "interactive", pyramids=pyramids, sampling_grid=sampling_grid
        )

    def _composite_inputs(self):
        snapshot = DocumentSnapshot.capture(context_for(self).document)
        effect = getattr(self, "_effect_preview", None)
        if effect is not None and effect[0] == snapshot.document_id:
            snapshot = replace(
                snapshot,
                layers=tuple(
                    replace(record, surface=effect[2])
                    if record.metadata[0] == effect[1]
                    else record
                    for record in snapshot.layers
                ),
            )
        preview = getattr(self, "_vector_preview", None)
        if preview is not None:
            index, obj = preview
            record = snapshot.layers[index]
            layers = list(snapshot.layers)
            layers[index] = replace(record, vector_records=record.vector_records.appended(obj))
            snapshot = replace(snapshot, layers=tuple(layers))
        pyramids = (
            tuple(tuple(layer._mipmaps) for layer in self.layers)
            if preview is None and effect is None
            else None
        )
        return snapshot, pyramids

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

        if getattr(self, "_display_surface", None) is not None:
            self._display_surface.clear()
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
            (doc_sx, doc_sy, doc_sx + sw, doc_sy + sh)
        )
        checker.alpha_composite(crop)
        self.tkimg = ImageTk.PhotoImage(checker.convert("RGB"))

        self._canvas_image_id = self.canvas.create_image(sx, sy, image=self.tkimg, anchor="nw")

    def redraw(self, dirty_box=None, *, inputs=None):
        if hasattr(self, "active_view") and self.active_view != "main":
            if getattr(self, "_display_surface", None) is not None:
                self._display_surface.prefetch.cancel()
            self.main_view_dirty = True
            return
        if getattr(self, "_display_surface", None) is None:
            self._display_surface = DisplaySurface(self.canvas)
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
            self._canvas_image_id = None
            self.tkimg = None
        viewport = (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        inputs = inputs or self._composite_inputs()
        snapshot, pyramids = inputs
        zoom = self.zoom
        required = self._compositing_layer_indices(self.layers)
        pyramid_key = None
        if pyramids is not None and self.zoom < 1:
            desired = max(0, int(math.log2(1 / self.zoom)))
            level = max(0, min([desired] + [len(pyramids[i]) - 1 for i in required]))
            pyramid_key = (level, tuple(id(pyramids[i][level]) for i in required) if level else ())
        layout = (
            snapshot.document_id,
            snapshot.size,
            self.zoom,
            tuple((i, snapshot.layers[i].metadata) for i in required),
            pyramid_key,
        )
        key = (
            layout,
            tuple(id(snapshot.layers[i]) for i in required),
        )
        if getattr(self, "_effect_preview", None) or getattr(self, "_vector_preview", None):
            # Mutable tool previews can change without a document revision.
            key = object()
            dirty_box = None
        if dirty_box is not None:
            guard = 2 / self.zoom
            dirty_box = (
                dirty_box[0] - guard,
                dirty_box[1] - guard,
                dirty_box[2] + guard,
                dirty_box[3] + guard,
            )
        self.canvas.delete("overlay")
        self._display_surface.draw(
            snapshot.size,
            viewport,
            self.zoom,
            (self.offset_x, self.offset_y),
            key,
            layout,
            dirty_box,
            self.get_checker_backdrop_pil(*viewport),
            lambda box, size: self.composite_region(
                box,
                size,
                inputs=inputs,
                sampling_grid=(zoom, round(box[0] * zoom), round(box[1] * zoom)),
            ),
            checker_period=self.checker_size * 2,
        )
        # Keep the identity keys' owners alive until the next draw completes.
        self._display_inputs = inputs
        self._draw_overlays()
        # Retain the visible extent for navigation diagnostics.
        left, top = max(0, -self.offset_x / self.zoom), max(0, -self.offset_y / self.zoom)
        right = min(self.doc_w, (viewport[0] - self.offset_x) / self.zoom)
        bottom = min(self.doc_h, (viewport[1] - self.offset_y) / self.zoom)
        self._display_geometry = (
            *viewport,
            (left, top, right, bottom),
            max(1, round((right - left) * self.zoom)),
            max(1, round((bottom - top) * self.zoom)),
            self.offset_x + left * self.zoom,
            self.offset_y + top * self.zoom,
        )

    def _redraw_legacy(self, dirty_box=None):
        """Full-viewport reference retained for navigation comparison measurements."""
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
                self._canvas_image_hidden = True
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
        geometry = (cw, ch, crop_box, sw, sh, sx, sy)
        partial = (
            dirty_box is not None
            and getattr(self, "_display_geometry", None) == geometry
            and getattr(self, "tkimg", None) is not None
            and self._canvas_image_id is not None
        )
        if partial:
            # Convert the dirty document bounds to pixels within the existing
            # viewport PhotoImage. A small guard band absorbs rounding at
            # fractional zoom levels and brush antialias edges.
            px0 = max(0, math.floor((dirty_box[0] - crop_left) * self.zoom) - 2)
            py0 = max(0, math.floor((dirty_box[1] - crop_top) * self.zoom) - 2)
            px1 = min(sw, math.ceil((dirty_box[2] - crop_left) * self.zoom) + 2)
            py1 = min(sh, math.ceil((dirty_box[3] - crop_top) * self.zoom) + 2)
            if px1 > px0 and py1 > py0:
                patch_box = (
                    crop_left + px0 / self.zoom,
                    crop_top + py0 / self.zoom,
                    crop_left + px1 / self.zoom,
                    crop_top + py1 / self.zoom,
                )
                patch = self.composite_region(patch_box, (px1 - px0, py1 - py0))
                checker = self.get_checker_backdrop_pil(cw, ch).crop(
                    (doc_sx + px0, doc_sy + py0, doc_sx + px1, doc_sy + py1)
                )
                checker.alpha_composite(patch)
                patch_tk = ImageTk.PhotoImage(checker.convert("RGB"))
                # Tk's native photo-image copy supports a destination offset;
                # Pillow's PhotoImage.paste only replaces an entire image.
                self.canvas.tk.call(str(self.tkimg), "copy", str(patch_tk), "-to", px0, py0)
        else:
            crop = self.composite_region(crop_box, (sw, sh))
            if crop.getchannel("A").getextrema() == (255, 255):
                # An opaque viewport completely covers the checkerboard.
                display = crop.convert("RGB")
            else:
                checker = self.get_checker_backdrop_pil(cw, ch).crop(
                    (doc_sx, doc_sy, doc_sx + sw, doc_sy + sh)
                )
                checker.alpha_composite(crop)
                display = checker.convert("RGB")
            reuse_image = (
                getattr(self, "tkimg", None) is not None
                and self.tkimg.width() == sw
                and self.tkimg.height() == sh
            )
            if reuse_image:
                self.tkimg.paste(display)
            else:
                self.tkimg = ImageTk.PhotoImage(display)

            if self._canvas_image_id is None:
                self._canvas_image_id = self.canvas.create_image(
                    sx, sy, image=self.tkimg, anchor="nw"
                )
            else:
                self.canvas.coords(self._canvas_image_id, sx, sy)
                if reuse_image:
                    if getattr(self, "_canvas_image_hidden", False):
                        self.canvas.itemconfigure(self._canvas_image_id, state="normal")
                else:
                    self.canvas.itemconfigure(
                        self._canvas_image_id, image=self.tkimg, state="normal"
                    )
            self._canvas_image_hidden = False
            self._display_geometry = geometry

        self._draw_overlays()

    def _draw_brush_cursor(self, x, y, diameter):
        # One constant-width outline at every zoom, without cursor bitmaps.
        radius = diameter / 2
        bounds = (x - radius, y - radius, x + radius, y + radius)
        self.canvas.create_oval(*bounds, outline="black", width=3, tags=("overlay",))
        self.canvas.create_oval(*bounds, outline="white", width=1, tags=("overlay",))

    def _draw_overlays(self):
        """Refresh interaction feedback without compositing the viewport."""
        self.canvas.delete("overlay")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        # Floating move/paste pixels are intentionally not part of the
        # document image yet.  Showing them as a canvas overlay lets the full
        # temporary object remain visible outside the document; committing the
        # edit later clips it naturally to the fixed-size raster layer.
        if self.move_pixels is not None:
            source_left, source_top, _, _ = self.move_source_box
            float_x = source_left + self.move_offset[0]
            float_y = source_top + self.move_offset[1]
            float_sx, float_sy = self.screen_coords(float_x, float_y)
            float_image = self.move_pixels
            scaled_width = max(1, round(float_image.width * self.zoom))
            scaled_height = max(1, round(float_image.height * self.zoom))
            # A large paste at high zoom can have a multi-gigabyte scaled
            # footprint. Sample only the portion that reaches the viewport.
            px0 = max(0, math.floor(-float_sx))
            py0 = max(0, math.floor(-float_sy))
            px1 = min(scaled_width, math.ceil(cw - float_sx))
            py1 = min(scaled_height, math.ceil(ch - float_sy))
            if px1 > px0 and py1 > py0:
                xscale = float_image.width / scaled_width
                yscale = float_image.height / scaled_height
                float_image = float_image.transform(
                    (px1 - px0, py1 - py0),
                    Image.Transform.EXTENT,
                    (px0 * xscale, py0 * yscale, px1 * xscale, py1 * yscale),
                    resample=Image.Resampling.NEAREST,
                )
                self._floating_move_tkimg = ImageTk.PhotoImage(float_image)
                self.canvas.create_image(
                    float_sx + px0,
                    float_sy + py0,
                    image=self._floating_move_tkimg,
                    anchor="nw",
                    tags=("overlay", "floating_move"),
                )

        # Draw brush cursor for raster layers
        current_layer = self.layers[self.active_layer]
        if self.tool == "color picker":
            if self.picker_sample_area_var.get():
                image_x, image_y = self.image_coords(self.mouse_x, self.mouse_y)
                pixel_x, pixel_y = math.floor(image_x), math.floor(image_y)
                if 0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h:
                    diameter = max(1, round(int(self.size_var.get()) * self.zoom))
                    center_x, center_y = self.screen_coords(pixel_x + 0.5, pixel_y + 0.5)
                    self._draw_brush_cursor(center_x, center_y, diameter)
            else:
                image_x, image_y = self.image_coords(self.mouse_x, self.mouse_y)
                pixel_x, pixel_y = math.floor(image_x), math.floor(image_y)
                if 0 <= pixel_x < self.doc_w and 0 <= pixel_y < self.doc_h:
                    x0, y0 = self.screen_coords(pixel_x, pixel_y)
                    x1, y1 = self.screen_coords(pixel_x + 1, pixel_y + 1)
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, outline="black", width=3, tags=("overlay",)
                    )
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, outline="white", width=1, tags=("overlay",)
                    )

        if (
            current_layer.is_raster
            and self.tool in ("brush", "eraser", "brush selection", "clone")
            and 0 <= self.mouse_x < cw
            and 0 <= self.mouse_y < ch
        ):
            diameter = max(1, round(int(self.size_var.get()) * self.zoom))
            self._draw_brush_cursor(self.mouse_x, self.mouse_y, diameter)

        if self.tool == "clone" and self.clone_source_center is not None:
            if self.clone_offset is None:
                source_x, source_y = self.clone_source_center
            else:
                hover_x, hover_y = self.raster_image_coords(self.mouse_x, self.mouse_y)
                source_x = hover_x + self.clone_offset[0]
                source_y = hover_y + self.clone_offset[1]
            center_x, center_y = self.screen_coords(source_x + 0.5, source_y + 0.5)
            radius = max(1, int(self.size_var.get()) * self.zoom / 2)
            self.canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                outline="#00ff80",
                width=2,
                tags=("overlay",),
            )
            self.canvas.create_line(
                center_x - 5,
                center_y,
                center_x + 5,
                center_y,
                fill="#00ff80",
                width=1,
                tags=("overlay",),
            )
            self.canvas.create_line(
                center_x,
                center_y - 5,
                center_x,
                center_y + 5,
                fill="#00ff80",
                width=1,
                tags=("overlay",),
            )

        # Draw vector handles if in proper mode and on vector layer
        if (
            self.tool in ("vector select", "vector edit")
            and current_layer.layer_type == "vector"
            and current_layer.vector_data
        ):
            for candidate in current_layer.vector_data.objects:
                sx, sy = self.screen_coords(*vector_center(candidate))
                self.canvas.create_oval(
                    sx - 4,
                    sy - 4,
                    sx + 4,
                    sy + 4,
                    fill="white",
                    outline="#007f99",
                    width=2,
                    tags=("overlay",),
                )
            if self.selected_vector_obj and not isinstance(self.selected_vector_obj, Point):
                cx, cy = self.screen_coords(*vector_center(self.selected_vector_obj))
                rx, ry = self.screen_coords(*rotation_handle(self.selected_vector_obj, self.zoom))
                self.canvas.create_line(cx, cy, rx, ry, fill="#00a080", tags=("overlay",))
                self.canvas.create_oval(
                    rx - 5,
                    ry - 5,
                    rx + 5,
                    ry + 5,
                    fill="#00a080",
                    outline="white",
                    tags=("overlay",),
                )
            visible_objects = (
                current_layer.vector_data.objects
                if self.tool == "vector edit"
                else ([self.selected_vector_obj] if self.selected_vector_obj else [])
            )
            for obj in visible_objects:
                points = obj.get_points()
                for px, py in points:
                    sx, sy = self.screen_coords(px, py)
                    self.canvas.create_rectangle(
                        sx - 3,
                        sy - 3,
                        sx + 3,
                        sy + 3,
                        outline="cyan",
                        fill="cyan",
                        width=1,
                        tags=("overlay",),
                    )
            if self.tool == "vector select" and self.selected_vector_obj:
                points = self.selected_vector_obj.get_points()
                screen_points = [self.screen_coords(px, py) for px, py in points]
                if len(screen_points) > 2:
                    self.canvas.create_polygon(
                        *screen_points, outline="#00ffff", fill="", width=2, tags=("overlay",)
                    )
                elif len(screen_points) == 2:
                    self.canvas.create_line(
                        *screen_points, fill="#00ffff", width=2, tags=("overlay",)
                    )

        # A committed selection is document-space state; a floating selection
        # can also occupy the surrounding workspace until it is finalized.
        if current_layer.is_raster and (
            self.selection_bounds is not None or self.move_pixels is not None
        ):
            self._draw_selection_marquee()


bind_state(PaintApp)
