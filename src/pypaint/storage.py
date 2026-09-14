"""Pixel and metadata budgets, persistent tile indexing and scratch storage."""

import os
import sys
import tempfile
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


class Budget:
    def __init__(self, limit, name):
        self.limit, self.name = limit, name
        self.bytes = self.peak = 0
        self.lock = threading.Lock()
        self.reclaimers = weakref.WeakKeyDictionary()

    def register_reclaimer(self, callback):
        """Permit a cache to freeze working pixels on its owning thread only."""
        with self.lock:
            self.reclaimers[callback.__self__] = (
                weakref.WeakMethod(callback),
                threading.get_ident(),
            )

    def track(self, owner, amount):
        with self.lock:
            needs_room = self.bytes + amount > self.limit
            callbacks = list(self.reclaimers.values()) if needs_room else ()
        # Publishing can allocate payloads and run finalizers. Never invoke it
        # under the accounting lock or on another thread's writable cache.
        for reference, thread_id in callbacks:
            callback = reference()
            if callback is None or thread_id != threading.get_ident():
                continue
            try:
                callback()
            except (OSError, MemoryError):
                continue
            with self.lock:
                if self.bytes + amount <= self.limit:
                    break
        with self.lock:
            if self.bytes + amount > self.limit:
                raise OSError(
                    f"PyPaint {self.name} budget exhausted; close a document or increase the configured budget."
                )
            self.bytes += amount
            self.peak = max(self.peak, self.bytes)
        weakref.finalize(owner, self.release, amount)

    def release(self, amount):
        with self.lock:
            self.bytes -= amount


metadata = Budget(int(os.environ.get("PYPAINT_METADATA_MB", "256")) * 1024**2, "metadata")
working = Budget(int(os.environ.get("PYPAINT_WORKING_MB", "64")) * 1024**2, "working pixel")


# Persistent radix index: updates copy 8 short paths, never a tile dictionary.


def tile_key(x, y):
    if not (0 <= x < 65536 and 0 <= y < 65536):
        raise ValueError("Tile coordinate out of range")
    return (y << 16) | x


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Node:
    children: tuple = (None,) * 16

    def __post_init__(self):
        metadata.track(self, sys.getsizeof(self) + sys.getsizeof(self.children))


def get(root, key):
    for shift in range(28, -1, -4):
        if root is None:
            return None
        root = root.children[(key >> shift) & 15]
    return root


def set_entry(root, key, value, shift=28):
    children = list(root.children if root else (None,) * 16)
    index = (key >> shift) & 15
    children[index] = value if shift == 0 else set_entry(children[index], key, value, shift - 4)
    return Node(tuple(children)) if any(child is not None for child in children) else None


def entries(root, key=0, shift=28):
    if root is None:
        return
    for index, child in enumerate(root.children):
        if child is not None:
            child_key = key | index << shift
            if shift == 0:
                yield (child_key & 65535, child_key >> 16), child
            else:
                yield from entries(child, child_key, shift - 4)


def changed_coordinates(before, after, key=0, shift=28):
    """Visit only differing persistent branches, including removed tiles."""
    if before is after:
        return
    for index in range(16):
        left = before.children[index] if before else None
        right = after.children[index] if after else None
        if left is right:
            continue
        child_key = key | index << shift
        if shift == 0:
            yield child_key & 65535, child_key >> 16
        else:
            yield from changed_coordinates(left, right, child_key, shift - 4)


# Shared byte-budgeted storage for immutable pixel versions.
#
# Payload handles own their spill files. Releasing one document never clears another's files.
# Write failure preserves the in-memory payload and all already committed versions.


class StorageFull(OSError):
    pass


class Payload:
    __slots__ = ("store", "key", "size", "data", "path", "__weakref__")

    def __init__(self, store, data):
        self.store, self.key, self.size = store, uuid4().hex, len(data)
        self.data, self.path = data, None

    def read(self):
        return self.store.read(self)

    def __del__(self):
        self.store.release(self)


class TileStore:
    def __init__(self, ram_bytes=256 * 1024**2, disk_bytes=20 * 1024**3, directory=None):
        if ram_bytes < 1 or disk_bytes < 0:
            raise ValueError("Invalid storage budget")
        self.ram_limit, self.disk_limit = ram_bytes, disk_bytes
        self.ram_bytes = self.disk_bytes = self.spills = self.reloads = self.created_bytes = 0
        self.directory = Path(tempfile.mkdtemp(prefix="pypaint-", dir=directory))
        self._resident = OrderedDict()
        self._lock = threading.RLock()
        self.evict_caches = []

    def _make_room(self, amount, exclude=None):
        if self.ram_bytes + amount <= self.ram_limit:
            return
        for evict in self.evict_caches:
            callback = evict() if isinstance(evict, weakref.WeakMethod) else evict
            if callback is not None:
                callback()
        self.evict_caches[:] = [
            ref
            for ref in self.evict_caches
            if not isinstance(ref, weakref.WeakMethod) or ref() is not None
        ]
        for key, reference in list(self._resident.items()):
            if self.ram_bytes + amount <= self.ram_limit:
                break
            item = reference()
            if item is None or item is exclude:
                continue
            self._spill(item)

    def _spill(self, item):
        if item.path is None:
            if self.disk_bytes + item.size > self.disk_limit:
                raise StorageFull(
                    "PyPaint scratch disk budget exhausted; increase the budget or close a document."
                )
            path = self.directory / item.key
            try:
                with path.open("xb") as stream:
                    stream.write(item.data)
                    stream.flush()
            except OSError:
                path.unlink(missing_ok=True)
                raise
            item.path = path
            self.disk_bytes += item.size
        self._resident.pop(item.key, None)
        self.ram_bytes -= item.size
        item.data = None
        self.spills += 1

    def put(self, data):
        data = bytes(data)
        with self._lock:
            self._make_room(len(data))
            item = Payload(self, data)
            self.ram_bytes += item.size
            self._resident[item.key] = weakref.ref(item)
            try:
                if item.size > self.ram_limit:
                    self._spill(item)
            except OSError:
                self.release(item)
                item.data = None
                raise
            self.created_bytes += item.size
            return item

    def read(self, item):
        with self._lock:
            if item.data is not None:
                self._resident.move_to_end(item.key)
                return item.data
            data = item.path.read_bytes()
            if len(data) != item.size:
                raise OSError("Truncated PyPaint scratch payload")
            self.reloads += 1
            # A read need not evict/spill required data. The returned bounded
            # byte buffer belongs to the caller, not a hidden unbounded cache.
            return data

    def release(self, item):
        with self._lock:
            if self._resident.pop(item.key, None) is not None:
                self.ram_bytes -= item.size
            if item.path is not None:
                try:
                    item.path.unlink(missing_ok=True)
                except OSError:
                    return  # a failed unlink must not damage other handles
                self.disk_bytes -= item.size
                item.path = None

    def close(self):
        # Never remove live payloads. An empty, owned directory alone is removed.
        if self.ram_bytes or self.disk_bytes:
            raise RuntimeError("Release documents and snapshots before closing their tile store")
        self.directory.rmdir()

    def __del__(self):
        # Reference-owned payloads have already been released before their store.
        # Never recursively remove files or touch any path outside the owned directory.
        try:
            self.directory.rmdir()
        except (OSError, AttributeError):
            pass


_default = None


def default_store():
    global _default
    if _default is None:
        _default = TileStore(
            int(os.environ.get("PYPAINT_TILE_RAM_MB", "256")) * 1024**2,
            int(os.environ.get("PYPAINT_HISTORY_DISK_MB", "20480")) * 1024**2,
        )
    return _default
