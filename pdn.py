"""Raster PDN3 codec; parses NRBF as data, never as executable .NET objects.

PDN3 contains an XML header, an NRBF object graph, and deferred BGRA buffers.
The chunk layout is documented by Paint.NET's MemoryBlock implementation:
https://github.com/rivy/OpenPDN/blob/master/src/Core/MemoryBlock.cs
Only normal-blend, full-canvas bitmap layers are supported by this editor.
"""

import base64
import gzip
import io
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


MAX_BYTES = 512 * 1024 * 1024
MAX_ITEMS = 100000
_METADATA_TYPE = ('System.Collections.Generic.KeyValuePair`2[['
                  'System.String, mscorlib, Version=4.0.0.0, Culture=neutral, '
                  'PublicKeyToken=b77a5c561934e089],[System.String, mscorlib, '
                  'Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089]]')
_FORMATS = {1: '?', 2: 'B', 6: 'd', 7: 'h', 8: 'i', 9: 'q',
            10: 'b', 11: 'f', 12: 'q', 13: 'q', 14: 'H', 15: 'I', 16: 'Q'}


class PDNError(ValueError):
    pass


@dataclass
class RasterLayer:
    name: str
    image: Image.Image
    visible: bool = True
    opacity: int = 255


@dataclass
class _Ref:
    id: int


@dataclass
class _EmptyMetadata:
    pass


class _Reader:
    def __init__(self, stream):
        self.stream = stream
        self.objects = {}
        self.schemas = {}
        self.blocks = []
        self.records = 0

    def take(self, count):
        if not 0 <= count <= MAX_BYTES:
            raise PDNError('PDN data exceeds the supported size limit.')
        data = self.stream.read(count)
        if len(data) != count:
            raise PDNError('Truncated PDN file.')
        return data

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def integer(self):
        return self.unpack('<i')

    def byte(self):
        return self.unpack('B')

    def count(self):
        count = self.integer()
        if not 0 <= count <= MAX_ITEMS:
            raise PDNError('Invalid or oversized PDN collection.')
        return count

    def string(self):
        size = 0
        for shift in range(0, 35, 7):
            value = self.byte()
            size |= (value & 127) << shift
            if not value & 128:
                if size > 4 * 1024 * 1024:
                    raise PDNError('Oversized PDN string.')
                return self.take(size).decode('utf-8')
        raise PDNError('Invalid PDN string length.')

    def primitive(self, kind):
        if kind == 18:
            return self.string()
        if kind not in _FORMATS:
            raise PDNError(f'Unsupported PDN primitive {kind}.')
        return self.unpack('<' + _FORMATS[kind])

    def type_info(self, kind):
        if kind in (0, 7):
            return self.byte()
        if kind == 3:
            return self.string()
        if kind == 4:
            return (self.string(), self.integer())
        if kind not in (1, 2, 5, 6):
            raise PDNError('Unsupported PDN member type.')
        return None

    def record(self, depth=0):
        self.records += 1
        if depth > 100 or self.records > MAX_ITEMS:
            raise PDNError('PDN object graph is too complex.')
        tag = self.byte()
        if tag == 12:
            self.integer()
            self.string()  # Library names are data, never imported.
            return self.record(depth + 1)
        if tag == 9:
            return _Ref(self.integer())
        if tag == 10:
            return None
        if tag == 6:
            oid = self.integer()
            result = self.string()
            self.objects[oid] = result
            return result
        if tag == 8:
            return self.primitive(self.byte())
        if tag in (1, 4, 5):
            oid = self.integer()
            if tag == 1:
                schema = self.schemas[self.integer()]
            else:
                name = self.string()
                names = [self.string() for _ in range(self.count())]
                kinds = [self.byte() for _ in names]
                infos = [self.type_info(kind) for kind in kinds]
                if tag == 5:
                    self.integer()
                schema = (name, names, kinds, infos)
                self.schemas[oid] = schema
            name, names, kinds, infos = schema
            result = {'__class__': name}
            self.objects[oid] = result
            for member, kind, info in zip(names, kinds, infos):
                result[member] = (self.primitive(info) if kind == 0
                                  else self.record(depth + 1))
            if name == 'PaintDotNet.MemoryBlock':
                self.blocks.append(result)
            return result
        if tag in (7, 15, 16, 17):
            oid = self.integer()
            if tag == 7:
                array_kind, rank = self.byte(), self.integer()
                if array_kind != 0 or rank != 1:
                    raise PDNError('Unsupported PDN array shape.')
                count = self.count()
                kind = self.byte()
                info = self.type_info(kind)
            else:
                count = self.count()
                kind = 0 if tag == 15 else 2
                info = self.byte() if tag == 15 else None
            result = []
            self.objects[oid] = result
            while len(result) < count:
                if kind == 0:
                    result.append(self.primitive(info))
                else:
                    position = self.stream.tell()
                    next_tag = self.byte()
                    if next_tag in (13, 14):
                        nulls = self.byte() if next_tag == 13 else self.count()
                        if nulls < 1 or nulls > count - len(result):
                            raise PDNError('Invalid PDN null run.')
                        result.extend([None] * nulls)
                    else:
                        self.stream.seek(position)
                        result.append(self.record(depth + 1))
            return result
        raise PDNError(f'Unsupported PDN record {tag}.')

    def resolve(self, value):
        for _ in range(100):
            if not isinstance(value, _Ref):
                return value
            value = self.objects[value.id]
        raise PDNError('Circular PDN reference.')

    def graph(self):
        if self.byte() != 0:
            raise PDNError('Invalid PDN object stream.')
        root = self.integer()
        self.integer()
        if (self.integer(), self.integer()) != (1, 0):
            raise PDNError('Unsupported PDN serialization version.')
        while True:
            position = self.stream.tell()
            if self.byte() == 11:
                break
            self.stream.seek(position)
            self.record()
        return self.objects[root]


def _inflate(data, limit):
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        result = stream.read(limit + 1)
    if len(result) > limit:
        raise PDNError('Oversized compressed PDN data.')
    return result


def read_pdn(filename):
    """Return (width, height, bottom-to-top raster layers)."""
    try:
        return _read_pdn(filename)
    except PDNError:
        raise
    except (EOFError, KeyError, TypeError, IndexError, UnicodeError,
            struct.error, OSError, OverflowError) as error:
        raise PDNError(f'Invalid or unsupported PDN file: {error}') from error


def _read_pdn(filename):
    if Path(filename).stat().st_size > MAX_BYTES:
        raise PDNError('PDN file exceeds the 512 MiB limit.')
    with open(filename, 'rb') as stream:
        reader = _Reader(stream)
        if reader.take(4) != b'PDN3':
            raise PDNError('This is not a supported PDN3 file.')
        reader.take(int.from_bytes(reader.take(3), 'little'))
        indicator = reader.take(2)
        if indicator == b'\x1f\x8b':
            reader = _Reader(io.BytesIO(_inflate(indicator + stream.read(), MAX_BYTES)))
        elif indicator != b'\x00\x01':
            raise PDNError('Unsupported PDN compression format.')
        document = reader.graph()
        resolve = reader.resolve
        if document['__class__'] != 'PaintDotNet.Document':
            raise PDNError('Missing PDN document.')
        width, height = document['width'], document['height']
        if width <= 0 or height <= 0 or width * height * 4 > MAX_BYTES:
            raise PDNError('Invalid or oversized PDN canvas.')
        layer_list = resolve(document['layers'])
        count = layer_list['ArrayList+_size']
        items = resolve(layer_list['ArrayList+_items'])
        if not 1 <= count <= min(len(items), 1024):
            raise PDNError('Invalid PDN layer count.')
        if count * width * height * 4 > MAX_BYTES:
            raise PDNError('PDN layers exceed the 512 MiB decoded limit.')

        # Deferred buffers follow serialization order, not necessarily layer order.
        buffers = {}
        total = 0
        for block in reader.blocks:
            length = block.get('length64', block.get('length', 0))
            total += length
            if length <= 0 or total > MAX_BYTES or block.get('hasParent') or not block.get('deferred'):
                raise PDNError('Unsupported PDN pixel buffer.')
            version, chunk_size = reader.byte(), reader.unpack('>I')
            if version not in (0, 1) or not 1 <= chunk_size <= MAX_BYTES:
                raise PDNError('Invalid PDN pixel compression.')
            chunk_count = (length + chunk_size - 1) // chunk_size
            if chunk_count > MAX_ITEMS:
                raise PDNError('Too many PDN pixel chunks.')
            data, seen = bytearray(length), set()
            for _ in range(chunk_count):
                index, size = reader.unpack('>I'), reader.unpack('>I')
                if index >= chunk_count or index in seen:
                    raise PDNError('Duplicate or invalid PDN pixel chunk.')
                seen.add(index)
                offset = index * chunk_size
                expected = min(chunk_size, length - offset)
                raw = reader.take(size)
                decoded = _inflate(raw, expected) if version == 0 else raw
                if len(decoded) != expected:
                    raise PDNError('Incorrect PDN pixel chunk length.')
                data[offset:offset + expected] = decoded
            buffers[id(block)] = data

        layers = []
        for item in items[:count]:
            layer = resolve(item)
            if layer['__class__'] != 'PaintDotNet.BitmapLayer':
                raise PDNError('Only bitmap PDN layers are supported.')
            props = resolve(layer['Layer+properties'])
            blend = resolve(props.get('blendMode'))
            legacy = resolve(layer.get('properties'))
            legacy_op = resolve(legacy.get('blendOp')) if legacy else None
            if ((blend is not None and blend['value__'] != 0) or
                    (legacy_op and legacy_op['__class__'] != 'PaintDotNet.UserBlendOps+NormalBlendOp')):
                raise PDNError('This PDN uses a blend mode the editor does not support. '
                               'Use Normal blending in Paint.NET before opening it.')
            surface = resolve(layer['surface'])
            if any((obj[w], obj[h]) != (width, height) for obj, w, h in
                   [(layer, 'Layer+width', 'Layer+height'), (surface, 'width', 'height')]):
                raise PDNError('PDN layer dimensions do not match the canvas.')
            stride = surface['stride']
            block = resolve(surface['scan0'])
            data = buffers[id(block)]
            if stride < width * 4 or len(data) != stride * height:
                raise PDNError('Only 32-bit BGRA PDN pixels are supported.')
            image = Image.frombytes('RGBA', (width, height), bytes(data), 'raw', 'BGRA', stride)
            opacity = props['opacity']
            if not 0 <= opacity <= 255:
                raise PDNError('Invalid PDN layer opacity.')
            layers.append(RasterLayer(str(resolve(props['name'])), image,
                                      bool(props['visible']), opacity))
        return width, height, layers


class _Writer:
    """Small typed NRBF writer. References are emitted before queued objects."""
    def __init__(self, stream):
        self.stream = stream
        self.queue = []
        self.next_id = 3  # library IDs are 1 and 2

    def pack(self, fmt, *values):
        self.stream.write(struct.pack(fmt, *values))

    def string(self, text):
        data = text.encode('utf-8')
        size = len(data)
        while size >= 128:
            self.pack('B', (size & 127) | 128)
            size >>= 7
        self.pack('B', size)
        self.stream.write(data)

    def add(self, obj):
        oid = self.next_id
        self.next_id += 1
        self.queue.append((oid, obj))
        return _Ref(oid)

    def value(self, value):
        if value is None:
            self.pack('B', 10)
        elif isinstance(value, str):
            oid = self.next_id
            self.next_id += 1
            self.pack('<Bi', 6, oid)
            self.string(value)
        else:
            ref = value if isinstance(value, _Ref) else self.add(value)
            self.pack('<Bi', 9, ref.id)

    def write(self, root):
        ref = self.add(root)
        self.pack('<Biiii', 0, ref.id, -1, 1, 0)
        for lid, library in [(1, 'Data'), (2, 'Core')]:
            self.pack('<Bi', 12, lid)
            self.string(f'PaintDotNet.{library}, Version=5.112.9563.32325, Culture=neutral, PublicKeyToken=null')
        index = 0
        while index < len(self.queue):
            oid, obj = self.queue[index]
            index += 1
            if isinstance(obj, _EmptyMetadata):
                self.pack('<BiBiiB', 7, oid, 0, 1, 0, 3)
                self.string(_METADATA_TYPE)
                continue
            if isinstance(obj, list):
                self.pack('<Bii', 16, oid, len(obj))
                for item in obj:
                    self.value(item)
                continue
            name, library, members = obj
            self.pack('<Bi', 5 if library else 4, oid)
            self.string(name)
            self.pack('<i', len(members))
            for key, _, _, _ in members:
                self.string(key)
            for _, kind, _, _ in members:
                self.pack('B', kind)
            for _, kind, info, _ in members:
                if kind == 0:
                    self.pack('B', info)
                elif kind == 3:
                    self.string(info)
                elif kind == 4:
                    self.string(info[0])
                    self.pack('<i', info[1])
            if library:
                self.pack('<i', library)
            for _, kind, info, value in members:
                if kind == 0:
                    self.pack('<' + _FORMATS[info], value)
                else:
                    self.value(value)
        self.pack('B', 11)


def _object(name, library, *members):
    return (name, library, list(members))


def _thumbnail_png(width, height, layers):
    """Return Paint.NET's embedded, flattened Explorer thumbnail as PNG."""
    composite = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    for layer in layers:
        if not layer.visible:
            continue
        image = layer.image.convert('RGBA')
        if layer.opacity != 255:
            alpha = image.getchannel('A').point(
                lambda value, opacity=layer.opacity: (value * opacity + 127) // 255)
            image.putalpha(alpha)
        composite.alpha_composite(image)

    if width > 256 or height > 256:
        scale = min(256 / width, 256 / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        composite = composite.resize(size, Image.Resampling.LANCZOS, reducing_gap=3)

    output = io.BytesIO()
    composite.save(output, format='PNG')
    return output.getvalue()


def write_pdn(filename, width, height, layers):
    """Atomically save full-canvas RasterLayers as a layered PDN3 document."""
    if not layers or len(layers) > 1024 or width <= 0 or height <= 0:
        raise PDNError('A PDN needs a valid canvas and at least one layer.')
    if width * height * 4 * len(layers) > MAX_BYTES:
        raise PDNError('PDN layers exceed the 512 MiB limit.')
    for layer in layers:
        if layer.image.size != (width, height) or not 0 <= layer.opacity <= 255:
            raise PDNError('Invalid PDN layer size or opacity.')

    def primitive(name, kind, value):
        return (name, 0, kind, value)

    def member(name, cls, library, value):
        return (name, 4, (cls, library), value)

    version = _object('System.Version', 0, *[
        primitive(key, 8, value) for key, value in
        zip(('_Major', '_Minor', '_Build', '_Revision'), (5, 112, 9563, 32325))])
    bitmaps = []
    normal = _object('PaintDotNet.UserBlendOps+NormalBlendOp', 1)
    for layer in layers:
        block = _object('PaintDotNet.MemoryBlock', 2,
                        primitive('length64', 9, width * height * 4),
                        primitive('hasParent', 1, False), primitive('deferred', 1, True))
        surface = _object('PaintDotNet.Surface', 2,
                          primitive('width', 8, width), primitive('height', 8, height),
                          primitive('stride', 8, width * 4),
                          member('scan0', 'PaintDotNet.MemoryBlock', 2, block))
        props = _object('PaintDotNet.Layer+LayerProperties', 1,
                        ('name', 1, None, layer.name),
                        ('userMetadataItems', 3, _METADATA_TYPE + '[]', _EmptyMetadata()),
                        primitive('visible', 1, layer.visible),
                        primitive('isBackground', 1, False),
                        primitive('opacity', 2, layer.opacity),
                        member('blendMode', 'PaintDotNet.LayerBlendMode', 1,
                               _object('PaintDotNet.LayerBlendMode', 1, primitive('value__', 8, 0))))
        bitmap_props = _object('PaintDotNet.BitmapLayer+BitmapLayerProperties', 1,
                               member('blendOp', normal[0], 1, normal))
        bitmaps.append(_object('PaintDotNet.BitmapLayer', 1,
                               member('properties', bitmap_props[0], 1, bitmap_props),
                               member('surface', surface[0], 2, surface),
                               primitive('Layer+isDisposed', 1, False),
                               primitive('Layer+width', 8, width),
                               primitive('Layer+height', 8, height),
                               member('Layer+properties', props[0], 1, props)))
    layer_list = _object('PaintDotNet.LayerList', 1,
                         member('parent', 'PaintDotNet.Document', 1, _Ref(3)),
                         ('ArrayList+_items', 5, None, bitmaps),
                         primitive('ArrayList+_size', 8, len(layers)),
                         primitive('ArrayList+_version', 8, len(layers)))
    document = _object('PaintDotNet.Document', 1,
                       primitive('isDisposed', 1, False),
                       member('layers', layer_list[0], 1, layer_list),
                       primitive('width', 8, width), primitive('height', 8, height),
                       ('savedWith', 3, 'System.Version', version),
                       ('userMetadataItems', 3, _METADATA_TYPE + '[]', _EmptyMetadata()))
    path = Path(filename)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pdn-', delete=False) as stream:
            temporary = stream.name
            thumbnail = base64.b64encode(
                _thumbnail_png(width, height, layers)).decode('ascii')
            header = (f'<pdnImage width="{width}" height="{height}" layers="{len(layers)}" '
                      f'savedWithVersion="5.112.9563.32325"><custom><thumb png="{thumbnail}" />'
                      '</custom></pdnImage>').encode('utf-8')
            stream.write(b'PDN3' + len(header).to_bytes(3, 'little') + header + b'\x00\x01')
            _Writer(stream).write(document)
            chunk_size = 65536
            for layer in layers:
                raw = layer.image.convert('RGBA').tobytes('raw', 'BGRA')
                stream.write(struct.pack('>BI', 0, chunk_size))
                for index, offset in enumerate(range(0, len(raw), chunk_size)):
                    chunk = gzip.compress(raw[offset:offset + chunk_size], mtime=0)
                    stream.write(struct.pack('>II', index, len(chunk)))
                    stream.write(chunk)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
