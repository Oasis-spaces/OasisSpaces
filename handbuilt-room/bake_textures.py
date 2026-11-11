#!/usr/bin/env python3
"""Photo textures for the hand-built room, from the video's own frames.

Every surface in room.json (walls, floor, the faces of the big pieces) gets a
texture: each texel is looked up in the frames whose solved camera sees it from
the front, unblocked by the model's boxes, and the colours are averaged with a
weight that all but picks the most head-on, closest, sharpest view that shows
much of the surface at once. Texels
no frame saw are filled in from their surroundings and counted.

    python3 handbuilt-room/bake_textures.py spaces/bedroom [--px 250] [--every 2]

Needs the space's camera solve (workspace/sparse/0, workspace/images),
shapes.json and densify.json. Writes handbuilt-room/textures/<surface>.jpg and
textures/coverage.json.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SHARPEN = 12           # weight exponent: the best view wins nearly outright, so the texture stays sharp
INFLATE = 0.03         # occluders are grown by this, metres: the model's boxes are approximate
MIN_FACING = 0.4        # for walls and fronts
LEVEL_FACING = 0.15     # for tops and the floor, which a hand-held phone only ever sees at a slant
BORDER = 0.04          # fraction of the frame ignored at its edges


def read_model(folder: Path):
    """COLMAP's cameras.bin (one OPENCV camera) and images.bin: intrinsics, {name: (R, t)}."""
    raw = (folder / "cameras.bin").read_bytes()
    _, model, width, height = struct.unpack("<iiQQ", raw[8:32])
    if model != 4:
        raise SystemExit("expected one OPENCV camera")
    params = struct.unpack("<8d", raw[32:96])
    raw = (folder / "images.bin").read_bytes()
    count, at, poses = struct.unpack("<Q", raw[:8])[0], 8, {}
    for _ in range(count):
        _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack("<i7di", raw[at:at + 64]); at += 64
        end = raw.index(b"\0", at); name = raw[at:end].decode(); at = end + 1
        at += 8 + 24 * struct.unpack("<Q", raw[at:at + 8])[0]
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                      [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                      [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
        poses[name] = (R, np.array([tx, ty, tz]))
    return (width, height, params), poses


class Frame:
    """Plan coordinates (x, y up, z; metres) <-> the solve's world, for one space."""

    def __init__(self, space: Path):
        shapes = json.loads((space / "shapes.json").read_text())
        self.metre = json.loads((space / "densify.json").read_text())["colmap_units_per_metre"]
        self.world = np.array(shapes["world"], float)
        room = shapes["room"]
        self.origin = np.array([room["center"][0] - room["half_u"], room["center"][1] - room["half_v"],
                                shapes["room_level"]["floor_z"]])

    def to_solve(self, plan: np.ndarray) -> np.ndarray:
        scene = np.stack([plan[..., 0], plan[..., 2], plan[..., 1]], -1) * self.metre + self.origin
        return scene @ self.world

    def camera_in_plan(self, pose) -> np.ndarray:
        R, t = pose
        c = ((-R.T @ t) @ self.world.T - self.origin) / self.metre
        return np.array([c[0], c[2], c[1]])


def project(points_solve: np.ndarray, camera, pose):
    """Pixels and depth of world points in one frame (OPENCV model); depth <= 0 where unusable."""
    _, _, (fx, fy, cx, cy, k1, k2, p1, p2) = camera
    R, t = pose
    X = points_solve @ R.T + t
    z = X[:, 2]
    safe = np.where(z > 1e-6, z, 1.0)
    x, y = X[:, 0] / safe, X[:, 1] / safe
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    u = fx * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx
    v = fy * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy
    return u, v, np.where((z > 0.05) & (r2 < 2.0), z, -1.0)


def blocked(origin: np.ndarray, targets: np.ndarray, boxes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Whether the segment from origin to each target passes through any box (slab test)."""
    d = targets - origin
    hit = np.zeros(len(targets), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        for lo, hi in boxes:
            t0, t1 = (lo - origin) * inv, (hi - origin) * inv
            near, far = np.nanmax(np.minimum(t0, t1), axis=1), np.nanmin(np.maximum(t0, t1), axis=1)
            hit |= (near < far) & (far > 0) & (near < 0.985)
    return hit


def sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear colours of an image at float pixel positions."""
    x0 = np.clip(np.floor(x).astype(int), 0, image.shape[1] - 2); y0 = np.clip(np.floor(y).astype(int), 0, image.shape[0] - 2)
    fx, fy = np.clip(x - x0, 0, 1)[:, None], np.clip(y - y0, 0, 1)[:, None]
    a, b, c, d = image[y0, x0], image[y0, x0 + 1], image[y0 + 1, x0], image[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def bake(space: Path, px: int = 250, every: int = 2) -> dict:
    room = json.loads((HERE / "room.json").read_text())
    frame = Frame(space)
    camera, poses = read_model(space / "workspace/sparse/0")
    names = sorted(poses)[::every]
    width, height = camera[0], camera[1]
    images, sharp = {}, {}
    for name in names:
        im = cv2.imread(str(space / "workspace/images" / name))
        images[name] = im
        sharp[name] = cv2.Laplacian(cv2.cvtColor(cv2.resize(im, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    median_sharp = np.median(list(sharp.values()))
    out = HERE / "textures"; out.mkdir(exist_ok=True)
    coverage = {}
    for surface in room["surfaces"]:
        O, U, V = (np.array(surface[k], float) for k in ("origin", "u", "v"))
        normal = np.cross(U, V)
        cols, rows = max(8, round(surface["w"] * px)), max(8, round(surface["h"] * px))
        cc, rr = np.meshgrid((np.arange(cols) + 0.5) / cols * surface["w"], surface["h"] - (np.arange(rows) + 0.5) / rows * surface["h"])
        P = O + cc.reshape(-1, 1) * U + rr.reshape(-1, 1) * V
        P_solve = frame.to_solve(P)
        # Everything but the piece this surface is a face of can hide it.
        boxes = [(np.array(b["min"]) - INFLATE, np.array(b["max"]) + INFLATE) for b in room["boxes"]
                 if b["name"] != surface.get("of") and b.get("on") != surface["name"]]
        total, weight = np.zeros((len(P), 3)), np.zeros(len(P))
        for name in names:
            eye = frame.camera_in_plan(poses[name])
            ray = eye - P
            dist = np.linalg.norm(ray, axis=1)
            facing = (ray @ normal) / dist
            u, v, z = project(P_solve, camera, poses[name])
            ok = (z > 0) & (facing > (LEVEL_FACING if abs(normal[1]) > 0.9 else MIN_FACING)) & (u > BORDER * width) & (u < (1 - BORDER) * width) \
                & (v > BORDER * height) & (v < (1 - BORDER) * height)
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            idx = idx[~blocked(eye, P[idx], boxes)]
            if not len(idx):
                continue
            im = images[name]; sx = im.shape[1] / width
            colour = sample(im, u[idx] * sx, v[idx] * sx)
            # Head-on, close and sharp views win; so does a frame that shows much of the surface at once,
            # which keeps large areas from one photograph instead of a patchwork.
            share = len(idx) / len(P)
            w = (facing[idx] ** 2 / dist[idx] * share ** 0.5 * min(1.5, sharp[name] / median_sharp)) ** SHARPEN
            total[idx] += colour * w[:, None]; weight[idx] += w
        seen = weight > 0
        tex = np.zeros((len(P), 3), np.uint8)
        tex[seen] = np.clip(total[seen] / weight[seen, None], 0, 255)
        tex = tex.reshape(rows, cols, 3)
        hole = (~seen).reshape(rows, cols).astype(np.uint8)
        if hole.any() and seen.any():
            if hole.mean() < 0.12:
                tex = cv2.inpaint(tex, hole * 255, 5, cv2.INPAINT_TELEA)
            else:            # a large part unseen: its usual colour there, with what was seen feathered in
                base = np.full_like(tex, np.median(tex.reshape(-1, 3)[seen], axis=0))
                alpha = cv2.GaussianBlur(cv2.erode(1 - hole, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 6)[..., None]
                tex = (tex * alpha + base * (1 - alpha)).astype(np.uint8)
        cv2.imwrite(str(out / f"{surface['name']}.jpg"), tex, [cv2.IMWRITE_JPEG_QUALITY, 88])
        coverage[surface["name"]] = round(float(seen.mean()), 3)
        print(f"{surface['name']:16s} {cols}x{rows}  seen {seen.mean():.0%}", flush=True)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=1) + "\n")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--px", type=int, default=250, help="texels per metre")
    parser.add_argument("--every", type=int, default=2, help="use every n-th solved frame")
    args = parser.parse_args()
    bake(args.space, args.px, args.every)


if __name__ == "__main__":
    main()
