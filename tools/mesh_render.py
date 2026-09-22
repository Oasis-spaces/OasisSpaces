#!/usr/bin/env python3
"""Read a glTF binary (.glb) and draw it, on the CPU with numpy.

The pipeline writes GLB files (tools/mixed_scene.py) and now reads them back:
the furniture models a generator makes (tools/object_mesh.py) have to be shown
to Claude beside the scan they might replace, and judged before anyone sees
them. Rendering them needs no GPU and no new dependency: one textured
z-buffered triangle rasteriser is enough for a review picture.

Only what our own files and a generator's exports use is read: one buffer,
triangles, POSITION, TEXCOORD_0, NORMAL, a base-colour texture or factor.

Usage:
    python3 tools/mesh_render.py model.glb out.png [--size 480x360]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import struct
from pathlib import Path

import numpy as np

COMPONENT = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class Mesh:
    """Triangles in one frame: vertices, faces, and a colour per vertex."""

    def __init__(self, vertices, faces, colours, normals=None):
        self.vertices = np.asarray(vertices, np.float64)
        self.faces = np.asarray(faces, np.int64)
        self.colours = np.asarray(colours, np.float64)      # per vertex, 0-255
        self.normals = None if normals is None else np.asarray(normals, np.float64)

    def __len__(self) -> int:
        return len(self.faces)

    @property
    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def read_glb(path: Path) -> Mesh:
    """Every primitive of a .glb as one mesh, its texture sampled per vertex.

    A review picture only needs the colour where each vertex sits, so the
    texture is sampled at the vertices instead of being interpolated across
    each triangle: a generated model carries tens of thousands of them, which
    is finer than the picture."""
    blob = Path(path).read_bytes()
    magic, _version, _length = struct.unpack_from("<4sII", blob, 0)
    if magic != b"glTF":
        raise ValueError(f"{path} is not a .glb file")
    document, buffer, at = None, b"", 12
    while at < len(blob):
        size, kind = struct.unpack_from("<I4s", blob, at)
        chunk = blob[at + 8:at + 8 + size]
        if kind == b"JSON":
            document = json.loads(chunk)
        elif kind == b"BIN\x00":
            buffer = chunk
        at += 8 + size + (-size % 4)
    if document is None:
        raise ValueError(f"{path} has no JSON chunk")

    def read(index: int) -> np.ndarray:
        accessor = document["accessors"][index]
        view = document["bufferViews"][accessor.get("bufferView", 0)]
        dtype = np.dtype(COMPONENT[accessor["componentType"]])
        wide = COUNT[accessor["type"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride") or dtype.itemsize * wide
        if stride == dtype.itemsize * wide:
            flat = np.frombuffer(buffer, dtype, accessor["count"] * wide, start)
        else:                                     # interleaved attributes
            raw = np.frombuffer(buffer, np.uint8, accessor["count"] * stride, start)
            flat = np.stack([raw[n * stride + dtype.itemsize * k:][:dtype.itemsize].view(dtype)[0]
                             for n in range(accessor["count"]) for k in range(wide)])
        return flat.reshape(accessor["count"], wide).astype(np.float64 if dtype.kind == "f" else np.int64)

    def texture_of(material: dict):
        pbr = material.get("pbrMetallicRoughness", {})
        if "baseColorTexture" not in pbr:
            factor = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1])
            return None, np.array(factor[:3]) ** (1 / 2.2) * 255
        from PIL import Image

        source = document["textures"][pbr["baseColorTexture"]["index"]]["source"]
        image = document["images"][source]
        if "bufferView" in image:
            view = document["bufferViews"][image["bufferView"]]
            start = view.get("byteOffset", 0)
            data = buffer[start:start + view["byteLength"]]
        else:                                     # a data: URI
            data = base64.b64decode(image["uri"].split(",", 1)[1])
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.float64), None

    vertices, faces, colours, normals, used = [], [], [], [], 0
    for mesh in document.get("meshes", []):
        for primitive in mesh["primitives"]:
            if primitive.get("mode", 4) != 4:      # triangles only
                continue
            attributes = primitive["attributes"]
            points = read(attributes["POSITION"])
            index = (read(primitive["indices"]).reshape(-1, 3) if "indices" in primitive
                     else np.arange(len(points)).reshape(-1, 3))
            material = document["materials"][primitive["material"]] if "material" in primitive else {}
            texture, flat = texture_of(material)
            if texture is not None and "TEXCOORD_0" in attributes:
                uv = read(attributes["TEXCOORD_0"])
                rows = np.clip((uv[:, 1] * texture.shape[0]).astype(int), 0, texture.shape[0] - 1)
                cols = np.clip((uv[:, 0] * texture.shape[1]).astype(int), 0, texture.shape[1] - 1)
                colour = texture[rows, cols]
            else:
                colour = np.broadcast_to(flat if flat is not None else np.array([200.0] * 3),
                                         (len(points), 3)).copy()
            vertices.append(points)
            faces.append(index + used)
            colours.append(colour)
            normals.append(read(attributes["NORMAL"]) if "NORMAL" in attributes
                           else np.zeros((len(points), 3)))
            used += len(points)
    if not vertices:
        raise ValueError(f"{path} holds no triangles")
    return Mesh(np.vstack(vertices), np.vstack(faces), np.vstack(colours), np.vstack(normals))


def render(mesh: Mesh, view_matrix, width: int = 480, height: int = 360, fov_y: float = 55.0,
           light=(0.4, 0.7, 0.55), background=(0, 0, 0)) -> np.ndarray:
    """HxWx3 uint8 of the mesh, lit softly from `light` so shape reads.

    The camera convention is splat_render's: a column-major world-to-camera
    matrix, looking along +z with image y pointing down."""
    V = np.array(view_matrix, float).reshape(4, 4).T
    cam = mesh.vertices @ V[:3, :3].T + V[:3, 3]
    focal = height / 2 / np.tan(np.radians(fov_y) / 2)
    z = cam[:, 2]
    ahead = z > 1e-4
    u = focal * cam[:, 0] / np.where(ahead, z, 1) + width / 2
    v = focal * cam[:, 1] / np.where(ahead, z, 1) + height / 2

    shade = np.ones(len(mesh.vertices))
    if mesh.normals is not None and np.abs(mesh.normals).sum():
        facing = (mesh.normals @ V[:3, :3].T) @ (np.array(light, float) / np.linalg.norm(light))
        shade = 0.45 + 0.55 * np.clip(np.abs(facing), 0, 1)
    tint = np.clip(mesh.colours * shade[:, None], 0, 255)

    image = np.tile(np.array(background, np.float64), (height, width, 1))
    depth = np.full((height, width), np.inf)
    keep = np.flatnonzero(ahead[mesh.faces].all(axis=1))
    corners = mesh.faces[keep]
    xs, ys, zs = u[corners], v[corners], z[corners]
    # Back to front by the far corner, so a plain painter's order needs no sorting inside a face.
    for face in keep[np.argsort(-zs.max(axis=1))]:
        a, b, c = mesh.faces[face]
        px, py, pz = u[[a, b, c]], v[[a, b, c]], z[[a, b, c]]
        lo_x, hi_x = int(max(0, np.floor(px.min()))), int(min(width - 1, np.ceil(px.max())))
        lo_y, hi_y = int(max(0, np.floor(py.min()))), int(min(height - 1, np.ceil(py.max())))
        if lo_x > hi_x or lo_y > hi_y:
            continue
        area = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0])
        if abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(lo_x, hi_x + 1) + 0.5, np.arange(lo_y, hi_y + 1) + 0.5)
        w0 = ((px[1] - gx) * (py[2] - gy) - (px[2] - gx) * (py[1] - gy)) / area
        w1 = ((px[2] - gx) * (py[0] - gy) - (px[0] - gx) * (py[2] - gy)) / area
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        here = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
        window = depth[lo_y:hi_y + 1, lo_x:hi_x + 1]
        nearer = inside & (here < window)
        if not nearer.any():
            continue
        colour = (w0[..., None] * tint[a] + w1[..., None] * tint[b] + w2[..., None] * tint[c])
        window[nearer] = here[nearer]
        image[lo_y:hi_y + 1, lo_x:hi_x + 1][nearer] = colour[nearer]
    return np.clip(image, 0, 255).astype(np.uint8)


def turntable(mesh: Mesh, around: int = 3, width: int = 480, height: int = 360,
              fov_y: float = 55.0, tilt: float = 0.45) -> list:
    """The mesh from `around` angles, each framed to fill the picture."""
    low, high = mesh.bounds
    middle = (low + high) / 2
    reach = float(np.linalg.norm(high - low)) / 2
    away = reach / np.tan(np.radians(fov_y / 2)) * 1.25
    shots = []
    for n in range(around):
        angle = 2 * np.pi * n / around + np.pi / 6
        eye = middle + away * np.array([np.sin(angle) * np.cos(tilt), -np.sin(tilt),
                                        np.cos(angle) * np.cos(tilt)])
        forward = middle - eye
        forward /= np.linalg.norm(forward)
        down = np.array([0.0, 1.0, 0.0]) - forward * (np.array([0.0, 1.0, 0.0]) @ forward)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        R = np.stack([right, down, forward])
        t = -R @ eye
        view = [R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0,
                R[0][2], R[1][2], R[2][2], 0, t[0], t[1], t[2], 1]
        shots.append(render(mesh, view, width, height, fov_y))
    return shots


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glb", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", default="480x360")
    parser.add_argument("--views", type=int, default=3)
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.split("x"))
    mesh = read_glb(args.glb)
    shots = turntable(mesh, args.views, width, height)
    strip = Image.new("RGB", (width * len(shots), height))
    for n, shot in enumerate(shots):
        strip.paste(Image.fromarray(shot), (n * width, 0))
    strip.save(args.out)
    low, high = mesh.bounds
    print(f"{len(mesh):,} triangles, {(high - low).round(2).tolist()} across -> {args.out}")


if __name__ == "__main__":
    main()
