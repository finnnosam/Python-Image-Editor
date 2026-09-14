"""Document, view and session state, revisions, and UI property bindings."""

from dataclasses import dataclass, field, fields
from enum import Enum
from uuid import uuid4


def identity():
    return uuid4().hex


class ChangeKind(Enum):
    RASTER = "raster"
    VECTOR = "vector"
    LAYER = "layer"
    ORDER = "order"
    SELECTION = "selection"
    DIMENSIONS = "dimensions"
    VIEW = "view"


@dataclass(frozen=True)
class Change:
    document_id: str
    entity_id: str | None
    generation: int
    kind: ChangeKind
    regions: tuple[tuple[int, int, int, int], ...] = ()


class RevisionList(list):
    """Tracked nested geometry collections, including direct legacy table edits."""

    def __init__(self, values, owner):
        super().__init__(values)
        self.owner = owner
        self._bind()

    def _bind(self):
        for value in self:
            if isinstance(value, Revisioned):
                value._parents.add(self.owner)

    def _changed(self):
        self._bind()
        self.owner._changed()

    def append(self, value):
        super().append(value)
        if isinstance(value, Revisioned):
            value._parents.add(self.owner)
        self.owner._changed()

    def extend(self, values):
        for value in values:
            super().append(value)
            if isinstance(value, Revisioned):
                value._parents.add(self.owner)
        self.owner._changed()

    def insert(self, index, value):
        super().insert(index, value)
        self._changed()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._changed()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._changed()

    def remove(self, value):
        super().remove(value)
        self._changed()

    def pop(self, index=-1):
        value = super().pop(index)
        self._changed()
        return value

    def clear(self):
        super().clear()
        self._changed()

    def reverse(self):
        super().reverse()
        self._changed()

    def sort(self, *args, **kwargs):
        super().sort(*args, **kwargs)
        self._changed()

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __imul__(self, count):
        super().__imul__(count)
        self._changed()
        return self


class Revisioned:
    def _init_revision(self):
        import weakref

        object.__setattr__(self, "_parents", weakref.WeakSet())
        object.__setattr__(self, "revision", 0)
        object.__setattr__(self, "id", identity())

    def _changed(self):
        object.__setattr__(self, "revision", self.revision + 1)
        for parent in self._parents:
            parent._changed()

    def __setattr__(self, name, value):
        if name in ("color", "fill") and isinstance(value, list):
            value = tuple(value)
        tracked = (
            not name.startswith("_")
            and name not in ("revision", "id", "selected")
            and not callable(value)
        )
        if tracked and isinstance(value, list):
            value = RevisionList(value, self)
        previous = getattr(self, name, object())
        object.__setattr__(self, name, value)
        if tracked and previous != value and hasattr(self, "_parents"):
            self._changed()

    def __deepcopy__(self, memo):
        import copy

        result = type(self).__new__(type(self))
        result._init_revision()
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "revision":
                continue
            if isinstance(value, RevisionList):
                value = list(value)
            setattr(result, name, copy.deepcopy(value, memo))
        result.revision = self.revision
        return result


# Authoritative document state. No Tk objects or widget access belong here.


@dataclass
class Document:
    doc_w: int = 1024
    doc_h: int = 512
    current_file: str | None = None
    layers: list = field(default_factory=list)
    active_layer: int = 0
    undo_stack: object = field(default_factory=list)
    bg_color: tuple = (255, 255, 255, 0)
    id: str = field(default_factory=identity)
    persistent_id: str | None = None
    generation: int = 0
    state_id: str = field(default_factory=identity)
    saved_state_id: str | None = None
    closed: bool = False
    changes: list[Change] = field(default_factory=list)

    @property
    def modified(self):
        return self.state_id != self.saved_state_id

    def change(self, kind=ChangeKind.RASTER, entity_id=None, regions=()):
        self.generation += 1
        change = Change(self.id, entity_id, self.generation, kind, tuple(regions))
        self.changes.append(change)
        # Consumers track generations; if their cursor precedes this journal,
        # they must request a full refresh. Never union seam-crossing regions.
        if len(self.changes) > 256:
            del self.changes[:128]
        return change


@dataclass
class ViewState:
    zoom: float = 1.0
    offset_x: float = 20
    offset_y: float = 20
    yaw: float = 0
    pitch: float = 0


@dataclass
class SessionState:
    tool: str = "brush"
    last_x: float | None = None
    last_y: float | None = None
    vector_start_x: float | None = None
    vector_start_y: float | None = None
    current_vector_obj: object = None
    selected_vector_obj: object = None
    selected_point_index: int | None = None
    is_dragging_point: bool = False
    selection_start: tuple | None = None
    selection_bounds: tuple | None = None
    selection_mask: object = None
    _selection_mask_bounds: tuple | None = None
    selection_operation: str | None = None
    selection_base_mask: object = None
    selection_edges: list = field(default_factory=list)
    selection_dash_offset: int = 0
    move_start: tuple | None = None
    move_source_box: tuple | None = None
    move_pixels: object = None
    move_is_paste: bool = False
    move_mask: object = None
    move_base_image: object = None
    move_selection_bounds: tuple | None = None
    move_selection_edges: list | None = None
    move_offset: tuple = (0, 0)
    move_drag_origin_offset: tuple = (0, 0)
    selection_move_start: tuple | None = None
    selection_move_bounds: tuple | None = None
    selection_move_mask: object = None
    selection_brush_last: tuple | None = None
    selection_brush_remove: bool = False
    clone_source_center: tuple | None = None
    clone_offset: tuple | None = None
    clone_last: tuple | None = None
    clone_stroke_source: object = None
    clone_stroke_base: object = None
    clone_stroke_coverage: object = None
    bucket_pending: dict | None = None
    _stroke_base_image: object = None
    _stroke_coverage: object = None
    stroke: object = None
    clone_gesture: object = None


@dataclass
class DocumentContext:
    document: Document = field(default_factory=Document)
    view: ViewState = field(default_factory=ViewState)
    session: SessionState = field(default_factory=SessionState)

    # Temporary typed adapter for existing thumbnail helpers; it contains no copy.
    def __getitem__(self, name):
        return getattr(getattr(self, STATE_OWNERS[name]), name)


STATE_OWNERS = {
    item.name: owner
    for owner, cls in (("document", Document), ("view", ViewState), ("session", SessionState))
    for item in fields(cls)
}


# Compatibility properties delegate legacy UI names to typed authoritative state.
#
# Owner: UI migration. Remove each property as its handler adopts context.document,
# context.view, or context.session explicitly. No capture/restore copying occurs.


def context_for(window):
    if "_context" not in window.__dict__:
        window.__dict__["_context"] = DocumentContext()
    return window.__dict__["_context"]


def bind_state(window_type):
    for name, owner in STATE_OWNERS.items():
        if name in (
            "id",
            "generation",
            "state_id",
            "saved_state_id",
            "closed",
            "changes",
            "yaw",
            "pitch",
        ):
            continue

        def get(window, name=name, owner=owner):
            return getattr(getattr(context_for(window), owner), name)

        def set_value(window, value, name=name, owner=owner):
            setattr(getattr(context_for(window), owner), name, value)

        setattr(window_type, name, property(get, set_value))
