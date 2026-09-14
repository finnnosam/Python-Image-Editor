"""Version 3 native tiles, bounded decoding and atomic snapshot replacement."""

import json
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

from pypaint.extensions.blend_modes import MODES
from pypaint.history import LAYER_FIELDS, DocumentSnapshot
from pypaint.jobs import Cancellation
from pypaint.layer import Layer
from pypaint.state import Document
from pypaint.surface import TiledSurface
from pypaint.vectors import VectorLayer

VERSION = 3
MAX_MANIFEST = 32 * 1024**2
MAX_CHUNK = 16 * 1024**2
MAX_DECODED = 64 * 1024**3
MAX_MEMBERS = 65536


class NativeError(ValueError):
    pass


def validate_dimensions(width, height):
    if (
        type(width) is not int
        or type(height) is not int
        or not (1 <= width <= 1_000_000 and 1 <= height <= 1_000_000)
    ):
        raise NativeError("Invalid document dimensions")


def validate_vectors(data):
    if not isinstance(data, dict) or set(data) - {"name", "visible", "objects"}:
        raise NativeError("Unsupported vector layer record")
    if (
        not isinstance(data.get("name"), str)
        or len(data["name"]) > 4096
        or type(data.get("visible")) is not bool
    ):
        raise NativeError("Invalid vector layer metadata")
    if not isinstance(data.get("objects"), list) or len(data["objects"]) > 100_000:
        raise NativeError("Invalid vector object count")
    base = {"type", "id", "name", "color", "width", "antialias", "hardness"}
    schemas = {
        "Point": base | {"x", "y"},
        "Line": base | {"x1", "y1", "x2", "y2", "curve", "space"},
        "Shape": base | {"lines", "fill", "filled_side", "preset"},
        "Rectangle": base | {"x", "y", "w", "h", "fill"},
        "Ellipse": base | {"x", "y", "rx", "ry", "fill"},
    }
    stack, count, ids = [(data, 0)], 0, set()
    while stack:
        value, depth = stack.pop()
        count += 1
        if depth > 32 or count > 2_000_000:
            raise NativeError("Vector records exceed nesting or object limits")
        if isinstance(value, dict):
            if "type" in value and value["type"] not in (
                "Point",
                "Line",
                "Shape",
                "Rectangle",
                "Ellipse",
            ):
                raise NativeError(f"Unsupported vector type: {value['type']}")
            if "type" in value and set(value) - schemas[value["type"]]:
                raise NativeError("Unsupported vector attributes")
            for field in (
                "x",
                "y",
                "x1",
                "y1",
                "x2",
                "y2",
                "rx",
                "ry",
                "w",
                "h",
                "width",
                "hardness",
            ):
                if field in value:
                    number = value[field]
                    if (
                        type(number) not in (int, float)
                        or not -1_000_000_000 <= number <= 1_000_000_000
                    ):
                        raise NativeError(f"Invalid vector numeric field: {field}")
            if "width" in value and not 1 <= value["width"] <= 1_000_000:
                raise NativeError("Invalid vector stroke width")
            if "hardness" in value and not 0 <= value["hardness"] <= 100:
                raise NativeError("Invalid vector hardness")
            if "antialias" in value and type(value["antialias"]) is not bool:
                raise NativeError("Invalid vector antialias flag")
            if "space" in value and value["space"] not in ("flat", "globe"):
                raise NativeError("Unsupported vector coordinate space")
            curve = value.get("curve")
            if curve is not None and (
                not isinstance(curve, list)
                or len(curve) not in (2, 4)
                or any(
                    type(n) not in (int, float) or not -1_000_000_000 <= n <= 1_000_000_000
                    for n in curve
                )
            ):
                raise NativeError("Invalid curve controls")
            if value.get("type") == "Shape" and (
                not isinstance(value.get("lines"), list)
                or any(
                    not isinstance(line, dict) or line.get("type") != "Line"
                    for line in value["lines"]
                )
            ):
                raise NativeError("A shape must contain line records")
            if "id" in value:
                if not isinstance(value["id"], str) or value["id"] in ids:
                    raise NativeError("Duplicate or invalid vector ID")
                ids.add(value["id"])
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, float):
            import math

            if not math.isfinite(value):
                raise NativeError("Non-finite vector coordinate")


def validate_layer(metadata):
    if metadata.get("layer_type") not in ("raster", "vector"):
        raise NativeError("Unknown layer type")
    if metadata.get("blend_mode") not in MODES:
        raise NativeError(f"Unavailable blend mode: {metadata.get('blend_mode')}")
    if metadata.get("mask_mode") not in ("layers_underneath", "layer_below") or metadata.get(
        "mask_visibility"
    ) not in ("visible_only", "all_below"):
        raise NativeError("Unknown mask semantics")
    if not isinstance(metadata.get("opacity"), (float, int)) or not 0 <= metadata["opacity"] <= 100:
        raise NativeError("Invalid opacity")
    if not isinstance(metadata.get("name"), str) or len(metadata["name"]) > 4096:
        raise NativeError("Invalid layer name")
    for key in ("visible", "masked", "anti_mask"):
        if type(metadata.get(key)) is not bool:
            raise NativeError(f"Invalid {key} flag")


def save_native(snapshot, filename, cancellation=None, replace=os.replace):
    cancellation = cancellation or Cancellation()
    destination = Path(filename)
    manifest = dict(
        format="pypaint",
        version=VERSION,
        id=snapshot.persistent_id or snapshot.document_id,
        state_id=snapshot.state_id,
        width=snapshot.size[0],
        height=snapshot.size[1],
        pixel_format="RGBA8",
        alpha="straight",
        coordinates="top-left-pixel",
        background=snapshot.background,
        active_layer=snapshot.active_layer,
        layers=[],
    )
    validate_dimensions(*snapshot.size)
    for record in snapshot.layers:
        validate_layer(dict(zip(LAYER_FIELDS, record.metadata)))
        if record.vector_records is not None:
            validate_vectors(record.vector_records.as_dict())
    handle, temporary = tempfile.mkstemp(
        prefix="." + destination.name + "-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w+b") as raw:
            with zipfile.ZipFile(
                raw, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
            ) as archive:
                for index, record in enumerate(snapshot.layers):
                    cancellation.check()
                    metadata = dict(zip(LAYER_FIELDS, record.metadata))
                    info = dict(metadata, tiles=[])
                    manifest["layers"].append(info)
                    if record.vector_records is not None:
                        info["vectors"] = record.vector_records.as_dict()
                        continue
                    surface = record.surface
                    info["tile_size"], info["default"] = surface.tile_size, surface.fill
                    chunk = bytearray()
                    chunk_index = 0
                    name = f"pixels/{index}/{chunk_index}.bin"
                    for coordinate, box, image in surface.iter_tiles():
                        cancellation.check()
                        data = image.tobytes()
                        if chunk and len(chunk) + len(data) > MAX_CHUNK:
                            archive.writestr(name, chunk)
                            chunk.clear()
                            chunk_index += 1
                            name = f"pixels/{index}/{chunk_index}.bin"
                        info["tiles"].append([*coordinate, name, len(chunk), len(data)])
                        chunk.extend(data)
                    if chunk:
                        archive.writestr(name, chunk)
                encoded = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
                if len(encoded) > MAX_MANIFEST:
                    raise NativeError("Manifest exceeds supported size")
                archive.writestr("manifest.json", encoded)
            raw.flush()
            os.fsync(raw.fileno())
        cancellation.check()
        replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return snapshot.state_id


def read_native(filename, cancellation=None):
    cancellation = cancellation or Cancellation()
    if not zipfile.is_zipfile(filename):
        from pypaint.files import read_legacy

        return read_legacy(filename)
    with zipfile.ZipFile(filename) as archive:
        members = archive.infolist()
        names = {info.filename for info in members}
        if len(members) > MAX_MEMBERS or len(names) != len(members) or "manifest.json" not in names:
            raise NativeError("Invalid or excessive archive members")
        sizes = {}
        for info in members:
            path = info.filename
            if (
                path.startswith(("/", "\\"))
                or "\\" in path
                or ":" in path
                or any(part in ("..", "") for part in path.split("/"))
            ):
                raise NativeError("Unsafe archive path")
            limit = MAX_MANIFEST if path == "manifest.json" else MAX_CHUNK
            if info.file_size > limit or info.flag_bits & 1:
                raise NativeError("Oversized or encrypted archive member")
            sizes[path] = info.file_size
        if sum(sizes.values()) > MAX_DECODED:
            raise NativeError("Archive exceeds aggregate decoded limit")
        manifest = json.loads(archive.read("manifest.json"))
        if set(manifest) - {
            "format",
            "version",
            "id",
            "state_id",
            "width",
            "height",
            "pixel_format",
            "alpha",
            "coordinates",
            "background",
            "active_layer",
            "layers",
        }:
            raise NativeError("Unsupported manifest attributes")
        if manifest.get("format") != "pypaint" or manifest.get("version") != VERSION:
            raise NativeError("Unsupported native format/version")
        if (manifest.get("pixel_format"), manifest.get("alpha"), manifest.get("coordinates")) != (
            "RGBA8",
            "straight",
            "top-left-pixel",
        ):
            raise NativeError("Unsupported pixel/coordinate metadata")
        width, height = manifest.get("width"), manifest.get("height")
        validate_dimensions(width, height)
        for key in ("id", "state_id"):
            if not isinstance(manifest.get(key), str) or not 1 <= len(manifest[key]) <= 128:
                raise NativeError("Invalid document identity")
        background = manifest.get("background")
        if (
            not isinstance(background, list)
            or len(background) != 4
            or any(type(v) is not int or not 0 <= v <= 255 for v in background)
        ):
            raise NativeError("Invalid background RGBA value")
        records = manifest.get("layers")
        if not isinstance(records, list) or not 1 <= len(records) <= 4096:
            raise NativeError("Invalid layer count")
        ids, aggregate, count = set(), 0, 0
        # Validate every reference before allocating a layer or decoding a chunk.
        referenced = defaultdict(list)
        for layer_index, record in enumerate(records):
            if set(record) - set(LAYER_FIELDS) - {"tiles", "vectors", "tile_size", "default"}:
                raise NativeError("Unsupported layer attributes")
            validate_layer(record)
            if record.get("width") != width or record.get("height") != height:
                raise NativeError("Layer dimensions differ from document")
            if not isinstance(record.get("id"), str) or record["id"] in ids:
                raise NativeError("Duplicate/invalid layer identity")
            ids.add(record["id"])
            if record["layer_type"] == "vector":
                if record.get("tiles") or "tile_size" in record or "default" in record:
                    raise NativeError("Vector layer contains unsupported raster data")
                validate_vectors(record.get("vectors"))
                continue
            if "vectors" in record:
                raise NativeError("Raster layer contains unsupported vector data")
            tile_size = record.get("tile_size")
            if tile_size not in (128, 256, 512):
                raise NativeError("Unsupported tile size")
            seen = set()
            default = record.get("default")
            if (
                not isinstance(default, list)
                or len(default) != 4
                or any(type(v) is not int or not 0 <= v <= 255 for v in default)
            ):
                raise NativeError("Invalid default RGBA pixel")
            for tile in record.get("tiles", []):
                if not isinstance(tile, list) or len(tile) != 5:
                    raise NativeError("Invalid tile reference")
                x, y, name, offset, length = tile
                if not isinstance(name, str) or not re.fullmatch(
                    rf"pixels/{layer_index}/[0-9]+\.bin", name
                ):
                    raise NativeError("Invalid chunk path or foreign layer reference")
                if any(type(v) is not int for v in (x, y, offset, length)):
                    raise NativeError("Invalid tile coordinate/size")
                if (
                    not (0 <= x * tile_size < width and 0 <= y * tile_size < height)
                    or (x, y) in seen
                ):
                    raise NativeError("Out-of-bounds or duplicate tile")
                seen.add((x, y))
                expected = (
                    min(tile_size, width - x * tile_size)
                    * min(tile_size, height - y * tile_size)
                    * 4
                )
                if (
                    name == "manifest.json"
                    or name not in sizes
                    or length != expected
                    or offset < 0
                    or offset + length > sizes[name]
                ):
                    raise NativeError("Invalid tile payload bounds")
                aggregate += length
                referenced[name].append((offset, offset + length))
                count += 1
                if aggregate > MAX_DECODED or count > 1_000_000:
                    raise NativeError("Tile references exceed aggregate limit")
        if set(referenced) != names - {"manifest.json"}:
            raise NativeError("Unsupported or unreferenced archive content")
        for name, spans in referenced.items():
            cursor = 0
            for left, right in sorted(spans):
                if left != cursor:
                    raise NativeError("Overlapping or incomplete tile payload")
                cursor = right
            if cursor != sizes[name]:
                raise NativeError("Trailing unsupported tile data")
        doc = Document(doc_w=width, doc_h=height)
        doc.id = manifest["id"]
        doc.persistent_id = manifest["id"]
        doc.state_id = manifest["state_id"]
        doc.saved_state_id = doc.state_id
        doc.bg_color = tuple(manifest.get("background", (255, 255, 255, 0)))
        doc.active_layer = manifest.get("active_layer", 0)
        if type(doc.active_layer) is not int or not 0 <= doc.active_layer < len(records):
            raise NativeError("Invalid active layer index")
        for record in records:
            cancellation.check()
            layer = Layer(width, height, record["name"], record["layer_type"])
            for key in LAYER_FIELDS:
                setattr(layer, key, record[key])
            if layer.layer_type == "vector":
                layer.vector_data = VectorLayer.from_dict(record["vectors"], width, height)
            else:
                surface = TiledSurface(
                    "RGBA",
                    (width, height),
                    tuple(record.get("default", (0, 0, 0, 0))),
                    tile_size=record["tile_size"],
                )
                current_name, chunk = None, None
                for x, y, name, offset, length in sorted(
                    record["tiles"], key=lambda tile: (tile[2], tile[3])
                ):
                    cancellation.check()
                    if name != current_name:
                        chunk = archive.read(name)
                        current_name = name
                    box = surface.tile_box((x, y))
                    image = Image.frombytes(
                        "RGBA", (box[2] - box[0], box[3] - box[1]), chunk[offset : offset + length]
                    )
                    surface.paste(image, box)
                    # Publish one tile at a time; the shared store spills cold payloads.
                    surface.publish()
                layer.image = surface
                layer.reset_mipmaps()
            doc.layers.append(layer)
        return doc
