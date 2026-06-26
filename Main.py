import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from PIL import Image, ImageDraw, ImageTk
import math
import copy
import io
import pickle
import base64
import json

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
        obj_type = data.pop('type')
        if obj_type == 'Line':
            return Line.from_dict(data)
        elif obj_type == 'Rectangle':
            return Rectangle.from_dict(data)
        elif obj_type == 'Ellipse':
            return Ellipse.from_dict(data)
        return None

class Line(VectorObject):
    def __init__(self, x1=0, y1=0, x2=100, y2=100, color="#000000", width=2):
        super().__init__(color, width)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def to_dict(self):
        data = super().to_dict()
        data.update({
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2
        })
        return data
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['x1'], data['y1'], data['x2'], data['y2'], data['color'], data['width'])
    
    def draw(self, draw):
        draw.line([(self.x1, self.y1), (self.x2, self.y2)], fill=self.color, width=self.width)
        
    def get_points(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]
    
    def update_point(self, index, x, y):
        if index == 0:
            self.x1, self.y1 = x, y
        elif index == 1:
            self.x2, self.y2 = x, y

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
    
    def draw(self, draw):
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
    
    def draw(self, draw):
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
            obj.draw(draw)
            
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

    def render_vector(self):
        """Render vector objects to the raster image"""
        if self.layer_type == "vector" and self.vector_data:
            # Clear the image
            self.image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            self.draw = ImageDraw.Draw(self.image)
            self.vector_data.render(self.draw)

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyPaint")

        self.doc_w = 1200
        self.doc_h = 800
        self.current_file = None

        self.layers = [Layer(self.doc_w, self.doc_h, "Background", "raster")]
        self.active_layer = 0
        self.undo_stack = []

        self.zoom = 1.0
        self.offset_x = 20
        self.offset_y = 20

        self.tool = "brush"
        self.color = "#000000"
        self.bg_color = (255, 255, 255, 255)

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

        self.build_ui()
        self.refresh_layers()
        self.redraw()
        self.update_title()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x")

        # File menu buttons
        tk.Button(top, text="New", command=self.new_project).pack(side="left")
        tk.Button(top, text="Open", command=self.open_project).pack(side="left")
        tk.Button(top, text="Save", command=self.save_project).pack(side="left")
        tk.Button(top, text="Save As", command=self.save_project_as).pack(side="left")
        
        tk.Button(top, text="Export PNG", command=self.save_image).pack(side="left")
        tk.Button(top, text="Color", command=self.choose_color).pack(side="left")
        
        # Tool buttons
        tk.Button(top, text="Brush", command=lambda: self.set_tool("brush")).pack(side="left")
        tk.Button(top, text="Eraser", command=lambda: self.set_tool("eraser")).pack(side="left")
        tk.Button(top, text="Select", command=lambda: self.set_tool("select")).pack(side="left")
        tk.Button(top, text="Line", command=lambda: self.set_tool("line")).pack(side="left")
        tk.Button(top, text="Rect", command=lambda: self.set_tool("rect")).pack(side="left")
        tk.Button(top, text="Ellipse", command=lambda: self.set_tool("ellipse")).pack(side="left")
        
        tk.Button(top, text="Undo", command=self.undo).pack(side="left")

        self.size_slider = tk.Scale(top, from_=1, to=100, orient="horizontal", label="Size")
        self.size_slider.set(20)
        self.size_slider.pack(side="left")

        main = tk.PanedWindow(self.root, sashrelief="raised")
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, width=200)
        main.add(left)

        tk.Label(left, text="Layers").pack()

        self.layer_list = tk.Listbox(left)
        self.layer_list.pack(fill="both", expand=True)
        self.layer_list.bind("<<ListboxSelect>>", self.select_layer)
        
        # Drag and drop bindings
        self.layer_list.bind("<Button-1>", self.on_layer_drag_start)
        self.layer_list.bind("<B1-Motion>", self.on_layer_drag)
        self.layer_list.bind("<ButtonRelease-1>", self.on_layer_drag_end)

        layer_buttons = tk.Frame(left)
        layer_buttons.pack(fill="x")
        
        tk.Button(layer_buttons, text="Add Raster", command=lambda: self.add_layer("raster")).pack(side="left", fill="x", expand=True)
        tk.Button(layer_buttons, text="Add Vector", command=lambda: self.add_layer("vector")).pack(side="left", fill="x", expand=True)
        
        tk.Button(left, text="Delete Layer", command=self.delete_layer).pack(fill="x")
        tk.Button(left, text="Toggle Visible", command=self.toggle_visibility).pack(fill="x")
        tk.Button(left, text="Move Up", command=self.move_layer_up).pack(fill="x")
        tk.Button(left, text="Move Down", command=self.move_layer_down).pack(fill="x")

        self.canvas = tk.Canvas(main, bg="gray25")
        main.add(self.canvas)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.canvas.bind("<Motion>", self.mouse_move)

        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<Button-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.pan)

        self.canvas.bind("<MouseWheel>", self.zoom_mouse)

    def update_title(self):
        if self.current_file:
            self.root.title(f"PyPaint - {self.current_file}")
        else:
            self.root.title("PyPaint - Untitled")

    def snapshot(self):
        snap = []
        for l in self.layers:
            n = Layer(self.doc_w, self.doc_h, l.name, l.layer_type)
            n.visible = l.visible
            n.image = l.image.copy()
            n.draw = ImageDraw.Draw(n.image)
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
        self.redraw()

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
            
            with open(filename, 'wb') as f:
                pickle.dump(project_data, f)
            
            self.undo_stack = []
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save project: {e}")

    def open_project(self):
        if self.undo_stack:
            if not messagebox.askyesno("Unsaved Changes", 
                                       "You have unsaved changes. Open project anyway?"):
                return
        
        filename = filedialog.askopenfilename(
            filetypes=[("PyPaint files", "*.pypaint"), ("All files", "*.*")]
        )
        if not filename:
            return
        
        try:
            with open(filename, 'rb') as f:
                project_data = pickle.load(f)
            
            if 'version' not in project_data or 'layers' not in project_data:
                raise ValueError("Invalid project file format")
            
            self.layers = []
            
            for layer_info in project_data['layers']:
                img_bytes = base64.b64decode(layer_info['image_data'])
                img = Image.open(io.BytesIO(img_bytes))
                
                layer = Layer(project_data['document_width'], 
                             project_data['document_height'], 
                             layer_info['name'],
                             layer_info.get('layer_type', 'raster'))
                layer.image = img
                layer.visible = layer_info['visible']
                layer.draw = ImageDraw.Draw(layer.image)
                
                if layer.layer_type == "vector" and 'vector_data' in layer_info:
                    layer.vector_data = VectorLayer.from_dict(
                        layer_info['vector_data'],
                        project_data['document_width'],
                        project_data['document_height']
                    )
                
                self.layers.append(layer)
            
            self.active_layer = project_data.get('active_layer', 0)
            if self.active_layer >= len(self.layers):
                self.active_layer = len(self.layers) - 1
            
            self.current_file = filename
            self.undo_stack = []
            self.refresh_layers()
            self.redraw()
            self.update_title()
            messagebox.showinfo("Success", f"Project loaded from {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open project: {e}")

    def set_tool(self, tool):
        self.tool = tool
        # Reset vector drawing state
        self.vector_start_x = None
        self.vector_start_y = None
        self.current_vector_obj = None

    def choose_color(self):
        c = colorchooser.askcolor()[1]
        if c:
            self.color = c

    def add_layer(self, layer_type="raster"):
        self.snapshot()
        name = f"{layer_type.capitalize()} Layer {len([l for l in self.layers if l.layer_type == layer_type]) + 1}"
        self.layers.append(Layer(self.doc_w, self.doc_h, name, layer_type))
        self.active_layer = len(self.layers) - 1
        self.refresh_layers()
        self.redraw()

    def delete_layer(self):
        if len(self.layers) == 1:
            return
        self.snapshot()
        del self.layers[self.active_layer]
        self.active_layer = max(0, self.active_layer - 1)
        self.refresh_layers()
        self.redraw()

    def toggle_visibility(self):
        self.layers[self.active_layer].visible = not self.layers[self.active_layer].visible
        self.refresh_layers()
        self.redraw()

    def move_layer_up(self):
        if self.active_layer >= len(self.layers) - 1:
            return
        self.snapshot()
        self.layers[self.active_layer], self.layers[self.active_layer + 1] = \
            self.layers[self.active_layer + 1], self.layers[self.active_layer]
        self.active_layer += 1
        self.refresh_layers()
        self.redraw()

    def move_layer_down(self):
        if self.active_layer <= 0:
            return
        self.snapshot()
        self.layers[self.active_layer], self.layers[self.active_layer - 1] = \
            self.layers[self.active_layer - 1], self.layers[self.active_layer]
        self.active_layer -= 1
        self.refresh_layers()
        self.redraw()

    def select_layer(self, event=None):
        sel = self.layer_list.curselection()
        if sel:
            self.active_layer = len(self.layers) - 1 - sel[0]
            self.redraw()

    def refresh_layers(self):
        self.layer_list.delete(0, tk.END)
        for i in range(len(self.layers) - 1, -1, -1):
            l = self.layers[i]
            prefix = "✓" if l.visible else "✗"
            type_icon = "🖌" if l.layer_type == "raster" else "✏️"
            self.layer_list.insert(tk.END, f"{prefix} {type_icon} {l.name}")
        
        display_index = len(self.layers) - 1 - self.active_layer
        if 0 <= display_index < self.layer_list.size():
            self.layer_list.selection_set(display_index)

    def on_layer_drag_start(self, event):
        self.drag_start_index = self.layer_list.nearest(event.y)
        self.drag_start_y = event.y

    def on_layer_drag(self, event):
        if self.drag_start_index is None:
            return
        
        current_index = self.layer_list.nearest(event.y)
        
        if current_index == self.drag_start_index:
            return
        
        from_idx = len(self.layers) - 1 - self.drag_start_index
        to_idx = len(self.layers) - 1 - current_index
        
        self.snapshot()
        layer = self.layers.pop(from_idx)
        self.layers.insert(to_idx, layer)
        self.active_layer = to_idx
        self.drag_start_index = current_index
        self.refresh_layers()
        self.redraw()

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
        x, y = self.image_coords(event.x, event.y)
        current_layer = self.layers[self.active_layer]
        
        if current_layer.layer_type == "raster":
            self.start_raster_draw(event)
        else:  # vector layer
            self.start_vector_operation(event, x, y)

    def start_raster_draw(self, event):
        self.snapshot()
        self.last_x, self.last_y = self.image_coords(event.x, event.y)

    def start_vector_operation(self, event, x, y):
        if self.tool == "select":
            # Try to select an object
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
        x, y = self.image_coords(event.x, event.y)
        current_layer = self.layers[self.active_layer]
        
        if current_layer.layer_type == "raster":
            self.raster_paint(event)
        else:  # vector layer
            self.vector_operation(event, x, y)

    def raster_paint(self, event):
        x, y = self.image_coords(event.x, event.y)
        radius = self.size_slider.get() / 2

        color = (0, 0, 0, 0) if self.tool == "eraser" else self.color

        dx = x - self.last_x
        dy = y - self.last_y
        dist = math.hypot(dx, dy)

        spacing = max(1, radius * 0.25)
        steps = max(1, int(dist / spacing))

        for i in range(steps + 1):
            t = i / steps
            px = self.last_x + dx * t
            py = self.last_y + dy * t
            self.draw_circle(px, py, radius, color)

        self.last_x = x
        self.last_y = y
        self.redraw()

    def draw_circle(self, x, y, radius, color):
        layer = self.layers[self.active_layer]
        layer.draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                           fill=color, outline=color)

    def vector_operation(self, event, x, y):
        if self.is_dragging_point and self.selected_vector_obj:
            # Update the point position
            self.selected_vector_obj.update_point(self.selected_point_index, x, y)
            self.layers[self.active_layer].render_vector()
            self.redraw()
        elif self.tool in ["line", "rect", "ellipse"] and self.vector_start_x is not None:
            # Preview the shape (by redrawing)
            self.layers[self.active_layer].render_vector()
            self.redraw()
            # Draw temporary preview
            self.draw_vector_preview(self.vector_start_x, self.vector_start_y, x, y)

    def draw_vector_preview(self, x1, y1, x2, y2):
        """Draw a temporary preview of the vector object being created"""
        # Get current layer
        layer = self.layers[self.active_layer]
        
        # Create a temporary image for preview
        preview_img = layer.image.copy()
        preview_draw = ImageDraw.Draw(preview_img)
        
        if self.tool == "line":
            preview_draw.line([(x1, y1), (x2, y2)], fill=self.color, width=2)
        elif self.tool == "rect":
            preview_draw.rectangle([(x1, y1), (x2, y2)], outline=self.color, width=2)
        elif self.tool == "ellipse":
            preview_draw.ellipse([(x1, y1), (x2, y2)], outline=self.color, width=2)
        
        # Display preview
        self.display_image(preview_img)

    def on_mouse_up(self, event):
        x, y = self.image_coords(event.x, event.y)
        current_layer = self.layers[self.active_layer]
        
        if current_layer.layer_type == "raster":
            self.last_x = None
            self.last_y = None
        else:  # vector layer
            if self.is_dragging_point:
                # Done dragging point
                self.is_dragging_point = False
                self.selected_vector_obj = None
                self.selected_point_index = None
            elif self.tool in ["line", "rect", "ellipse"] and self.vector_start_x is not None:
                # Finalize the vector object
                self.create_vector_object(self.vector_start_x, self.vector_start_y, x, y)
                self.vector_start_x = None
                self.vector_start_y = None
                self.layers[self.active_layer].render_vector()
                self.redraw()

    def create_vector_object(self, x1, y1, x2, y2):
        """Create a vector object and add it to the current layer"""
        layer = self.layers[self.active_layer]
        if layer.layer_type != "vector":
            return
        
        obj = None
        if self.tool == "line":
            obj = Line(x1, y1, x2, y2, self.color, 2)
        elif self.tool == "rect":
            obj = Rectangle(x1, y1, abs(x2 - x1), abs(y2 - y1), self.color, 2)
        elif self.tool == "ellipse":
            obj = Ellipse((x1 + x2) / 2, (y1 + y2) / 2, 
                         abs(x2 - x1) / 2, abs(y2 - y1) / 2, self.color, 2)
        
        if obj:
            layer.vector_data.add_object(obj)

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
                self.redraw()

    def start_pan(self, event):
        self.pan_x = event.x
        self.pan_y = event.y

    def pan(self, event):
        self.offset_x += event.x - self.pan_x
        self.offset_y += event.y - self.pan_y
        self.pan_x = event.x
        self.pan_y = event.y
        self.redraw()

    def zoom_mouse(self, event):
        old = self.zoom
        self.zoom *= 1.1 if event.delta > 0 else (1 / 1.1)
        self.zoom = max(0.1, min(20, self.zoom))

        ix = (event.x - self.offset_x) / old
        iy = (event.y - self.offset_y) / old

        self.offset_x = event.x - ix * self.zoom
        self.offset_y = event.y - iy * self.zoom
        self.redraw()

    def mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.redraw()

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

    def display_image(self, img):
        """Display an image on the canvas"""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        
        left = max(0, (-self.offset_x) / self.zoom)
        top = max(0, (-self.offset_y) / self.zoom)
        right = min(self.doc_w, (cw - self.offset_x) / self.zoom)
        bottom = min(self.doc_h, (ch - self.offset_y) / self.zoom)
        
        if right <= left or bottom <= top:
            return
            
        crop = img.crop((int(left), int(top), int(right), int(bottom)))
        sw = max(1, int((right - left) * self.zoom))
        sh = max(1, int((bottom - top) * self.zoom))
        crop = crop.resize((sw, sh), Image.Resampling.NEAREST)
        
        self.tkimg = ImageTk.PhotoImage(crop)
        
        self.canvas.delete("all")
        sx = self.offset_x + left * self.zoom
        sy = self.offset_y + top * self.zoom
        self.canvas.create_image(sx, sy, image=self.tkimg, anchor="nw")

    def redraw(self):
        self.canvas.update_idletasks()
        
        # Ensure vector layers are rendered
        for layer in self.layers:
            if layer.layer_type == "vector" and layer.vector_data:
                layer.render_vector()
        
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        left = max(0, (-self.offset_x) / self.zoom)
        top = max(0, (-self.offset_y) / self.zoom)
        right = min(self.doc_w, (cw - self.offset_x) / self.zoom)
        bottom = min(self.doc_h, (ch - self.offset_y) / self.zoom)

        self.canvas.delete("all")

        if right <= left or bottom <= top:
            return

        img = self.composite_image()
        crop = img.crop((int(left), int(top), int(right), int(bottom)))
        sw = max(1, int((right - left) * self.zoom))
        sh = max(1, int((bottom - top) * self.zoom))
        crop = crop.resize((sw, sh), Image.Resampling.NEAREST)

        self.tkimg = ImageTk.PhotoImage(crop)

        sx = self.offset_x + left * self.zoom
        sy = self.offset_y + top * self.zoom

        self.canvas.create_image(sx, sy, image=self.tkimg, anchor="nw")

        # Draw brush cursor for raster layers
        current_layer = self.layers[self.active_layer]
        if current_layer.layer_type == "raster" and 0 <= self.mouse_x < cw and 0 <= self.mouse_y < ch:
            r = self.size_slider.get() * self.zoom / 2
            self.canvas.create_oval(self.mouse_x - r, self.mouse_y - r,
                                    self.mouse_x + r, self.mouse_y + r,
                                    outline="white", width=1)
        
        # Draw vector selection handles if in select mode and on vector layer
        if self.tool == "select" and current_layer.layer_type == "vector" and current_layer.vector_data:
            for obj in current_layer.vector_data.objects:
                points = obj.get_points()
                for px, py in points:
                    sx, sy = self.screen_coords(px, py)
                    self.canvas.create_rectangle(sx - 3, sy - 3, sx + 3, sy + 3,
                                               outline="cyan", fill="cyan", width=1)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x900")
    PaintApp(root)
    root.mainloop()