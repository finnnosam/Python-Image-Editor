"""Immutable document snapshots, undo transactions and change accounting."""

import json
import os
import sys
from dataclasses import dataclass

from pypaint.layer import Layer
from pypaint.state import ChangeKind, identity
from pypaint.storage import Node, changed_coordinates
from pypaint.storage import metadata as metadata_budget
from pypaint.surface import TiledSurface
from pypaint.vectors import VectorLayer, VectorSnapshot

LAYER_FIELDS = (
    "id",
    "name",
    "visible",
    "opacity",
    "blend_mode",
    "layer_type",
    "masked",
    "anti_mask",
    "mask_mode",
    "mask_visibility",
    "width",
    "height",
)


@dataclass(frozen=True)
class LayerRecord:
    metadata: tuple
    surface: object
    vector_records: VectorSnapshot | None

    def __post_init__(self):
        metadata_budget.track(self, sys.getsizeof(self) + sys.getsizeof(self.metadata))

    @property
    def vector_json(self):
        """Explicit serialization adapter for IO; rendering uses vector_records."""
        return self.vector_records.encoded() if self.vector_records is not None else None

    @classmethod
    def capture(cls, layer):
        metadata = tuple(getattr(layer, key) for key in LAYER_FIELDS)
        vector = layer.vector_data
        if vector is not None:
            if os.environ.get("PYPAINT_VALIDATE_REVISIONS") == "1":
                fingerprint = json.dumps(vector.to_dict(), sort_keys=True, allow_nan=False)
                old = getattr(layer, "_diagnostic_fingerprint", None)
                if old and old[0] == vector.revision and old[1] != fingerprint:
                    raise AssertionError("Vector content changed without advancing its revision")
                layer._diagnostic_fingerprint = (vector.revision, fingerprint)
            key = (metadata, vector.revision)
        else:
            layer.image.publish()
            key = (metadata, id(layer.image._root))
        if getattr(layer, "_history_key", None) == key:
            return layer._history_record
        record = cls(
            metadata,
            None if vector else layer.image.snapshot(),
            VectorSnapshot.capture(vector) if vector else None,
        )
        layer._history_key, layer._history_record = key, record
        return record

    def restore(self):
        metadata = dict(zip(LAYER_FIELDS, self.metadata))
        layer = Layer(
            metadata["width"], metadata["height"], metadata["name"], metadata["layer_type"]
        )
        for key, value in metadata.items():
            setattr(layer, key, value)
        if self.vector_records is not None:
            layer.vector_data = VectorLayer.from_dict(
                self.vector_records.as_dict(), layer.width, layer.height
            )
            for obj, payload in zip(layer.vector_data.objects, self.vector_records.objects):
                obj._snapshot_payload = payload
                obj._snapshot_revision = obj.revision
        else:
            layer.image = self.surface.copy()
        layer.reset_mipmaps()
        layer._history_key = (
            self.metadata,
            layer.vector_data.revision if layer.vector_data is not None else id(layer.image._root),
        )
        layer._history_record = self
        return layer


@dataclass(frozen=True)
class DocumentSnapshot:
    document_id: str
    generation: int
    state_id: str
    size: tuple
    active_layer: int
    background: tuple
    layers: tuple[LayerRecord, ...]
    persistent_id: str | None = None

    def __post_init__(self):
        metadata_budget.track(self, sys.getsizeof(self) + sys.getsizeof(self.layers))

    @classmethod
    def capture(cls, doc):
        return cls(
            doc.id,
            doc.generation,
            doc.state_id,
            (doc.doc_w, doc.doc_h),
            doc.active_layer,
            doc.bg_color,
            tuple(LayerRecord.capture(layer) for layer in doc.layers),
            doc.persistent_id or doc.id,
        )

    def restore(self, doc):
        if doc.id != self.document_id:
            raise ValueError("Snapshot belongs to another document")
        doc.layers = [layer.restore() for layer in self.layers]
        doc.doc_w, doc.doc_h = self.size
        doc.active_layer, doc.bg_color, doc.state_id = (
            self.active_layer,
            self.background,
            self.state_id,
        )
        doc.change(ChangeKind.ORDER)

    def same_content(self, other):
        return (
            self.size == other.size
            and self.background == other.background
            and self.layers == other.layers
        )

    def __getitem__(self, key):
        # Compatibility with old external diagnostic scripts, not used by gestures.
        return ([layer.restore() for layer in self.layers], self.active_layer, *self.size)[key]


@dataclass(frozen=True)
class Operation:
    name: str
    before: DocumentSnapshot
    after: DocumentSnapshot

    def __post_init__(self):
        metadata_budget.track(self, sys.getsizeof(self))


class History:
    def __init__(self):
        self.operations = []
        self.position = 0
        self.pending = None
        self.name = None

    def begin(self, doc, name="Edit"):
        self.commit(doc)
        self.pending = DocumentSnapshot.capture(doc)
        self.name = name
        doc.state_id = identity()
        doc.change()

    def commit(self, doc):
        if self.pending is None:
            return False
        before = self.pending
        try:
            after = DocumentSnapshot.capture(doc)
            changed = not before.same_content(after)
            operation = Operation(self.name, before, after) if changed else None
            if changed:
                self.operations.append(operation)
        except (OSError, MemoryError):
            before.restore(doc)
            self.pending = None
            raise
        self.pending = None
        if not changed:
            doc.state_id = before.state_id
            return False
        del self.operations[self.position : -1]
        self.position += 1

        record_commit(doc, before, after)
        return True

    def cancel(self, doc):
        if self.pending is not None:
            self.pending.restore(doc)
            self.pending = None

    def undo(self, doc):
        self.commit(doc)
        if self.position == 0:
            return False
        self.position -= 1
        self.operations[self.position].before.restore(doc)
        return True

    def redo(self, doc):
        self.commit(doc)
        if self.position == len(self.operations):
            return False
        self.operations[self.position].after.restore(doc)
        self.position += 1
        return True

    def clear(self):
        self.operations.clear()
        self.pending = None
        self.position = 0

    def __len__(self):
        return self.position + (self.pending is not None)

    def __getitem__(self, index):
        values = [op.before for op in self.operations[: self.position]]
        if self.pending is not None:
            values.append(self.pending)
        return values[index]

    def pop(self):
        if self.pending is None:
            raise RuntimeError("Only an uncommitted compatibility gesture can be discarded")
        value, self.pending = self.pending, None
        return value


# One gesture = one history operation; rollback never depends on an undo entry.


def history_for(document):
    if not isinstance(document.undo_stack, History):
        document.undo_stack = History()
    return document.undo_stack


class Transaction:
    def __init__(self, document, name):
        self.document = document
        self.history = history_for(document)
        self.history.begin(document, name)
        self.active = True

    def write(self, layer, image, box, mask=None):
        if not self.active or layer not in self.document.layers:
            raise RuntimeError("Inactive transaction or foreign layer")
        try:
            layer.image.paste(image, box, mask)
        except (OSError, MemoryError):
            self.cancel()
            raise
        return self.document.change(ChangeKind.RASTER, layer.id, (box,))

    def commit(self):
        if self.active:
            try:
                return self.history.commit(self.document)
            finally:
                self.active = False

    def cancel(self):
        if self.active:
            self.history.cancel(self.document)
            self.active = False


# Classify committed edits without inspecting document-sized pixel buffers.


def record_commit(document, before, after):
    if before.size != after.size:
        document.change(ChangeKind.DIMENSIONS, regions=((0, 0, *after.size),))
    old = {record.metadata[0]: record for record in before.layers}
    if tuple(old) != tuple(record.metadata[0] for record in after.layers):
        document.change(ChangeKind.ORDER)
    for record in after.layers:
        previous = old.get(record.metadata[0])
        if previous is None:
            continue
        if previous.metadata != record.metadata:
            document.change(ChangeKind.LAYER, record.metadata[0])
        if record.vector_records is not None:
            if previous.vector_records is not record.vector_records:
                first, last = previous.vector_records, record.vector_records
                if first is None or first.bounds is None or last.bounds is None:
                    document.change(ChangeKind.VECTOR, record.metadata[0], ((0, 0, *after.size),))
                    continue
                regions = []
                for index in range(max(len(first.objects), len(last.objects))):
                    if (
                        index < min(len(first.objects), len(last.objects))
                        and first.objects[index] is last.objects[index]
                    ):
                        continue
                    if index < len(first.objects):
                        regions.append(first.bounds[index])
                    if index < len(last.objects):
                        regions.append(last.bounds[index])
                    if len(regions) >= 128:
                        document.change(ChangeKind.VECTOR, record.metadata[0], regions)
                        regions = []
                if regions:
                    document.change(ChangeKind.VECTOR, record.metadata[0], regions)
        elif previous.surface is not None and previous.surface.size == record.surface.size:
            regions = []
            for coordinate in changed_coordinates(previous.surface._root, record.surface._root):
                regions.append(record.surface.tile_box(coordinate))
                if len(regions) == 128:
                    document.change(ChangeKind.RASTER, record.metadata[0], regions)
                    regions = []
            if regions:
                document.change(ChangeKind.RASTER, record.metadata[0], regions)


# On-demand per-document attribution without double-counting shared history roots.


def document_usage(document):
    """Shared payloads can be attributed to more than one document; totals aren't additive."""
    seen, payloads = set(), {}
    index_bytes = 0

    def visit(node):
        nonlocal index_bytes
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Node):
            index_bytes += sys.getsizeof(node) + sys.getsizeof(node.children)
            for child in node.children:
                visit(child)
        else:
            payloads[node.payload.key] = node.payload

    records = [LayerRecord.capture(layer) for layer in document.layers]
    if isinstance(document.undo_stack, History):
        history = document.undo_stack
        for operation in history.operations:
            records.extend(operation.before.layers)
            records.extend(operation.after.layers)
        if history.pending:
            records.extend(history.pending.layers)
    record_ids = set()
    for record in records:
        if id(record) in record_ids:
            continue
        record_ids.add(id(record))
        if record.surface is not None:
            visit(record.surface._root)
        elif record.vector_records:
            for payload in record.vector_records.objects:
                payloads[payload.key] = payload
    return dict(
        unique_payload_bytes=sum(p.size for p in payloads.values()),
        resident_payload_bytes=sum(p.size for p in payloads.values() if p.data is not None),
        spilled_payload_bytes=sum(p.size for p in payloads.values() if p.path is not None),
        unique_index_bytes=index_bytes,
        layer_records=len(record_ids),
    )
