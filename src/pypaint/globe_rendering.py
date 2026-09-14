"""Bounded globe textures and cached sphere projection."""

import math
import weakref

import numpy as np
from PIL import Image

from pypaint.history import DocumentSnapshot
from pypaint.storage import changed_coordinates


class GlobeTexture:
    def __init__(self, document, renderer, maximum=(2048, 1024)):
        self.document, self.renderer, self.maximum = document, renderer, maximum
        self.image = None
        self.snapshot = None
        self.last_regions = []

    def update(self):
        snapshot = DocumentSnapshot.capture(self.document)
        scale = min(1, self.maximum[0] / snapshot.size[0], self.maximum[1] / snapshot.size[1])
        size = (max(1, round(snapshot.size[0] * scale)), max(1, round(snapshot.size[1] * scale)))
        old = self.snapshot
        regions = []

        def add_region(box):
            mapped = (
                max(0, math.floor(box[0] * size[0] / snapshot.size[0]) - 1),
                max(0, math.floor(box[1] * size[1] / snapshot.size[1]) - 1),
                min(size[0], math.ceil(box[2] * size[0] / snapshot.size[0]) + 1),
                min(size[1], math.ceil(box[3] * size[1] / snapshot.size[1]) + 1),
            )
            if mapped[2] > mapped[0] and mapped[3] > mapped[1]:
                regions.append(mapped)

        if (
            self.image is None
            or self.image.size != size
            or old is None
            or old.size != snapshot.size
            or old.background != snapshot.background
            or len(old.layers) != len(snapshot.layers)
            or any(a.metadata != b.metadata for a, b in zip(old.layers, snapshot.layers))
        ):
            regions = [(0, 0, *size)]
        else:
            # Index comparison preserves edits made outside the change journal
            # by compatibility handlers. Regions remain separate at the seam.
            for a, b in zip(old.layers, snapshot.layers):
                if a.vector_records is not None:
                    first, last = a.vector_records, b.vector_records
                    if first is last:
                        continue
                    if first.bounds is None or last.bounds is None:
                        regions = [(0, 0, *size)]
                        break
                    for index in range(max(len(first.objects), len(last.objects))):
                        if (
                            index < min(len(first.objects), len(last.objects))
                            and first.objects[index] is last.objects[index]
                        ):
                            continue
                        if index < len(first.objects):
                            add_region(first.bounds[index])
                        if index < len(last.objects):
                            add_region(last.bounds[index])
                    continue
                if a.surface is None or a.surface._root is b.surface._root:
                    continue
                if a.surface.fill != b.surface.fill or a.surface.tile_size != b.surface.tile_size:
                    regions = [(0, 0, *size)]
                    break
                for coordinate in changed_coordinates(a.surface._root, b.surface._root):
                    box = b.surface.tile_box(coordinate)
                    add_region(box)
        if self.image is None or self.image.size != size:
            self.image = Image.new("RGBA", size)
        for box in dict.fromkeys(regions):
            source = (
                box[0] * snapshot.size[0] / size[0],
                box[1] * snapshot.size[1] / size[1],
                box[2] * snapshot.size[0] / size[0],
                box[3] * snapshot.size[1] / size[1],
            )
            patch = self.renderer.render(
                snapshot, source, (box[2] - box[0], box[3] - box[1]), "interactive"
            )
            self.image.paste(patch, box)
        self.snapshot = snapshot
        self.last_regions = regions
        return self.image

    def release(self):
        self.image = self.snapshot = None


# UI-independent globe sampling, with reusable projection and checker buffers.


class GlobeProjection:
    def __init__(self):
        self.key = self.checker_key = None
        self.tx = self.ty = self.checker = self.output = None
        self.normals_ref = None

    def render(self, texture, normals, mask, yaw, pitch, display_size, origin, checker_style):
        height, width = mask.shape
        key = (normals.shape, yaw, pitch, texture.shape[:2])
        if self.output is None or self.output.shape[:2] != mask.shape:
            self.output = np.empty((height, width, 3), dtype=np.uint8)
        self.output[...] = (48, 48, 48)
        # Lookup rebuilds can release an array before the next render, allowing
        # Python to reuse its id for different geometry. Track live identity
        # without keeping the old viewport-sized normals alive.
        if self.key != key or self.normals_ref is None or self.normals_ref() is not normals:
            x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
            cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
            yy, zz = cp * y + sp * z, -sp * y + cp * z
            lon = np.arctan2(sy * x + cy * zz, cy * x - sy * zz)
            lat = np.arcsin(np.clip(yy, -1, 1))
            u = (1 - (lon + np.pi) / (2 * np.pi)) % 1
            v = 0.5 - lat / np.pi
            self.tx = (u * texture.shape[1]).astype(np.int32) % texture.shape[1]
            self.ty = np.clip(
                (v * (texture.shape[0] - 1)).astype(np.int32), 0, texture.shape[0] - 1
            )
            self.key = key
            self.normals_ref = weakref.ref(normals)
        sampled = texture[self.ty[mask], self.tx[mask]]
        cell, light, dark = checker_style
        checker_key = (mask.shape, display_size, origin, cell, light, dark)
        if self.checker_key != checker_key:
            xs = origin[0] + (np.arange(width) + 0.5) * (display_size[0] / width)
            ys = origin[1] + (np.arange(height) + 0.5) * (display_size[1] / height)
            parity = (
                (ys[:, None] // cell).astype(np.int32) + (xs[None, :] // cell).astype(np.int32)
            ) & 1
            self.checker = np.where(
                parity[..., None] == 0,
                np.array(light[:3], dtype=np.uint8),
                np.array(dark[:3], dtype=np.uint8),
            )
            self.checker_key = checker_key
        if sampled.shape[1] >= 4:
            alpha = sampled[:, 3:4].astype(np.uint16)
            self.output[mask] = (
                (
                    sampled[:, :3].astype(np.uint16) * alpha
                    + self.checker[mask].astype(np.uint16) * (255 - alpha)
                    + 127
                )
                // 255
            ).astype(np.uint8)
        else:
            self.output[mask] = sampled[:, :3]
        return self.output
