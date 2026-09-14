#!/usr/bin/env python3
"""Fill the parts of a room's surfaces that were never filmed, seamlessly.

A phone held upright at eye height never sees the floor within about 2 m of
where it stands, so a splat's floor is full of holes that show as dark gaps
or smears. For a flat surface (the floor now; walls use the same steps):

  1. photograph: project every video frame onto the surface, which stage 3
     measured, giving a sharp top-down photo of everything that was filmed.
     Each spot takes the frames that saw it closest and least obliquely,
     skipping frames where furniture blocks the view, and drops colours that
     disagree with the rest (a reflection, something moving).
  2. inpaint: an AI inpainting model (LaMa, big-lama, run locally) continues
     the flooring into the unseen parts, tile joints included, from the
     filmed floor around them. Anything standing on the floor and the space
     under furniture count as unknown, so clutter is not copied into the fill.
  3. match: the photo's colours are brought to the splat's own colours with a
     smoothly varying gain, so the fill does not change brightness at a seam.
  4. blobs: flat Gaussians carry the filled photo into the splat, only where
     the floor was never seen, fading out over the last 12 cm into the filmed
     floor instead of stopping at an edge. Stray blobs in those holes at floor
     level, guesses the training made with no view, are removed first.

Writes <surface>-photo.png, -unknown.png and -filled.png beside the space for
inspection. Used by tools/splat_edit.py fill-floor.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

LAMA_PATH = Path.home() / ".cache" / "oasisspaces" / "big-lama.pt"
PHOTO_CELL_M = 0.005        # the surface photo's resolution
BLOB_SPACING_M = 0.012      # fill Gaussians this far apart
FEATHER_M = 0.20            # fade into the filmed surface over this distance
TUCK_M = 0.06               # and reach this far under furniture, out of sight
FRAME_STEP = 2              # every other frame is plenty for the photo
MAX_DISTANCE_M = 5.0        # farther views are too coarse to use
MIN_COS = 0.2               # nor views more oblique than ~78 degrees
CLUTTER_ABOVE_M = (0.10, 0.60)  # blobs this high over the floor are things on it


@dataclass
class Surface:
    """A rectangle in the scene frame: pixel (row, col) of its photo sits at
    origin + (col + 0.5) * cell * u + (row + 0.5) * cell * v."""
    name: str
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray       # facing into the room
    cols: int
    rows: int
    cell: float

    def points(self, cell: float | None = None) -> np.ndarray:
        cell = cell or self.cell
        cols, rows = int(round(self.cols * self.cell / cell)), int(round(self.rows * self.cell / cell))
        cc, rr = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
        return (self.origin + cc.ravel()[:, None] * cell * self.u
                + rr.ravel()[:, None] * cell * self.v)

    def pixel(self, scene: np.ndarray, cell: float | None = None):
        cell = cell or self.cell
        rel = scene - self.origin
        return ((rel @ self.v) / cell).astype(int), ((rel @ self.u) / cell).astype(int)


def floor_surface(room, height: float) -> Surface:
    m = room.metre
    cell = PHOTO_CELL_M * m
    lo = room.centre - room.half
    return Surface("floor", np.array([lo[0], lo[1], height]), np.array([1.0, 0, 0]),
                   np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                   int(round(2 * room.half[0] / cell)), int(round(2 * room.half[1] / cell)), cell)


# ------------------------------------------------------------------ photo
def project(points_c: np.ndarray, info: dict, cam: dict):
    """Camera-solve points -> (u, v) full-resolution pixels and depth, with
    COLMAP's OPENCV lens distortion."""
    local = points_c @ info["R"].T + info["t"]
    z = local[:, 2]
    x = local[:, 0] / np.maximum(z, 1e-9)
    y = local[:, 1] / np.maximum(z, 1e-9)
    p = cam["params"]
    fx, fy, cx, cy = p[:4]
    # The distortion polynomial folds points far outside the view back into
    # the image, so only points within (a margin of) the field of view count.
    outside = (np.abs(x) > 1.1 * cam["width"] / (2 * fx)) | (np.abs(y) > 1.1 * cam["height"] / (2 * fy))
    x, y = np.where(outside, 1e6, x), np.where(outside, 1e6, y)
    if cam["model"] == 4 and len(p) >= 8:
        k1, k2, p1, p2 = p[4:8]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    u, v = fx * x + cx, fy * y + cy
    return np.where(outside, -1.0, u), np.where(outside, -1.0, v), z


DEPTH_SHRINK = 16


def depth_buffer(dense_c: np.ndarray, info: dict, cam: dict, shrink: int = DEPTH_SHRINK) -> np.ndarray:
    """Nearest surface depth per shrink x shrink block of the frame, with gaps
    between points closed so furniture reliably hides what is behind it."""
    from scipy.ndimage import minimum_filter

    u, v, z = project(dense_c, info, cam)
    W, H = cam["width"] // shrink, cam["height"] // shrink
    ok = (z > 0) & (u >= 0) & (v >= 0) & (u < cam["width"]) & (v < cam["height"])
    cell = (v[ok] / shrink).astype(int) * W + (u[ok] / shrink).astype(int)
    order = np.lexsort((z[ok], cell))
    first = np.r_[True, np.diff(cell[order]) != 0]
    buf = np.full(H * W, np.inf)
    buf[cell[order][first]] = z[ok][order][first]
    return minimum_filter(buf.reshape(H, W), size=5)


def photograph(space: Path, room, surface: Surface, log=print, occluders=None):
    """(photo HxWx3 float, weight HxW): the surface as the frames saw it.
    `occluders` (camera-solve points) add to the dense cloud for deciding
    what blocks a view: the splat's blobs cover furniture sides that only
    frames between the dense cloud's keyframes saw."""
    from densify import read_cameras_bin, read_images_bin
    from pointcloud import load_ply
    from splat_export import model_dir

    model = model_dir(space)
    cameras = read_cameras_bin(model / "cameras.bin")
    images = sorted(read_images_bin(model / "images.bin").values(), key=lambda v: v["name"])
    dense = load_ply(space / "cloud-dense.ply").points
    dense = dense[np.random.default_rng(0).choice(len(dense), min(len(dense), 2_500_000),
                                                  replace=False)].astype(np.float64)
    if occluders is not None:
        dense = np.vstack([dense, occluders])
    m = room.metre
    pts_s = surface.points()
    pts_c = pts_s @ room.world          # scene -> camera-solve frame
    n = len(pts_s)
    samples = []                        # per frame: (cell indices, colours, weights)
    for info in images[::FRAME_STEP]:
        cam = cameras[info["camera_id"]]
        u, v, z = project(pts_c, info, cam)
        centre_s = room.world @ (-info["R"].T @ info["t"])
        ray = pts_s - centre_s
        dist = np.linalg.norm(ray, axis=1)
        cos = np.abs(ray @ surface.normal) / np.maximum(dist, 1e-9)
        # Only the side of the surface facing the camera, near enough, not too oblique.
        facing = (centre_s - surface.origin) @ surface.normal > 0
        ok = (facing & (z > 0.1 * m) & (dist < MAX_DISTANCE_M * m) & (cos > MIN_COS)
              & (u >= 0) & (v >= 0) & (u < cam["width"] - 1) & (v < cam["height"] - 1))
        if ok.sum() < 100:
            continue
        idx = np.flatnonzero(ok)
        buf = depth_buffer(dense, info, cam)
        blocked = z[idx] > buf[(v[idx] / DEPTH_SHRINK).astype(int),
                               (u[idx] / DEPTH_SHRINK).astype(int)] + 0.05 * m + 0.02 * z[idx]
        idx = idx[~blocked]
        if len(idx) < 100:
            continue
        img = np.asarray(Image.open(space / "workspace" / "images" / info["name"])
                         .convert("RGB").reduce(2), dtype=np.float32)
        uu, vv = u[idx] / 2, v[idx] / 2
        x0, y0 = np.floor(uu).astype(int), np.floor(vv).astype(int)
        x1, y1 = np.minimum(x0 + 1, img.shape[1] - 1), np.minimum(y0 + 1, img.shape[0] - 1)
        fx_, fy_ = (uu - x0)[:, None], (vv - y0)[:, None]
        colour = (img[y0, x0] * (1 - fx_) * (1 - fy_) + img[y0, x1] * fx_ * (1 - fy_)
                  + img[y1, x0] * (1 - fx_) * fy_ + img[y1, x1] * fx_ * fy_)
        footprint = dist[idx] / (cam["params"][0] / 2 * cos[idx])   # scene units per image pixel
        weight = (1.0 / np.maximum(footprint, 1e-9)) ** 2
        samples.append((idx.astype(np.int32), colour.astype(np.float32), weight.astype(np.float32)))
    log(f"  {surface.name} photo: {len(samples)} frames saw it")
    def accumulate(keep_fn=None):
        wsum, csum = np.zeros(n), np.zeros((n, 3))
        for idx, colour, weight in samples:
            if keep_fn is not None:
                keep = keep_fn(idx, colour)
                idx, colour, weight = idx[keep], colour[keep], weight[keep]
            wsum += np.bincount(idx, weights=weight, minlength=n)
            for ch in range(3):
                csum[:, ch] += np.bincount(idx, weights=weight * colour[:, ch], minlength=n)
        return wsum, csum

    wsum, csum = accumulate()
    mean = csum / np.maximum(wsum, 1e-12)[:, None]
    # Second pass without colours far from the first mean (reflections, motion).
    dev_sum = np.zeros(n)
    for idx, colour, weight in samples:
        dev_sum += np.bincount(idx, weights=weight * np.sum((colour - mean[idx]) ** 2, axis=1),
                               minlength=n)
    spread = np.sqrt(dev_sum / np.maximum(wsum, 1e-12))
    wsum2, csum2 = accumulate(lambda idx, colour: np.linalg.norm(colour - mean[idx], axis=1)
                              <= np.maximum(20.0, 1.2 * spread[idx]))
    photo = (csum2 / np.maximum(wsum2, 1e-12)[:, None]).reshape(surface.rows, surface.cols, 3)
    # How well seen, relative to a frame looking straight down from 1.5 m.
    reference = (1.0 / ((1.5 * m) / (cameras[images[0]["camera_id"]]["params"][0] / 2))) ** 2
    return photo, (wsum2 / reference).reshape(surface.rows, surface.cols)


# ---------------------------------------------------------------- inpaint
def inpaint(photo: np.ndarray, unknown: np.ndarray, log=print) -> np.ndarray:
    """LaMa fills `unknown` pixels of an HxWx3 0-255 photo."""
    import torch

    if not LAMA_PATH.exists():
        sys.exit(f"LaMa weights not found at {LAMA_PATH}")
    model = torch.jit.load(str(LAMA_PATH), map_location="cpu").eval()
    H, W = unknown.shape
    ph, pw = (-H) % 8, (-W) % 8
    img = np.pad(np.clip(photo, 0, 255) / 255.0, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mask = np.pad(unknown.astype(np.float32), ((0, ph), (0, pw)), mode="edge")
    img_t = torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32))
    img_t = img_t * (1 - torch.from_numpy(mask)[None, None])
    with torch.inference_mode():
        out = model(img_t, torch.from_numpy(mask)[None, None])
    filled = out[0].permute(1, 2, 0).numpy()[:H, :W] * 255.0
    log(f"  inpainted {unknown.mean():.0%} of the {H}x{W} photo")
    return np.where(unknown[..., None], filled, photo)


# ------------------------------------------------------------- floor fill
def splat_floor(room, scene, colours, alpha, scale, height, cell):
    """The splat's own floor as an image on the `cell` grid (colour, weight),
    each blob spread over its footprint."""
    m = room.metre
    lo = room.centre - room.half
    W, H = int(round(2 * room.half[0] / cell)), int(round(2 * room.half[1] / cell))
    sel = np.flatnonzero((np.abs(scene[:, 2] - height) < 0.08 * m) & (alpha > 0.2))
    wsum = np.zeros((H, W))
    csum = np.zeros((H, W, 3))
    for i in sel:
        cx, cy = (scene[i, 0] - lo[0]) / cell, (scene[i, 1] - lo[1]) / cell
        r = max(1.0, 2 * scale[i] / cell)
        x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, W))
        y0, y1 = int(max(cy - r, 0)), int(min(cy + r + 1, H))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = alpha[i] * np.exp(-2 * ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2)
        wsum[y0:y1, x0:x1] += w
        csum[y0:y1, x0:x1] += w[..., None] * colours[i]
    return csum / np.maximum(wsum, 1e-9)[..., None], wsum


def raster(surface: Surface, scene: np.ndarray, rows: int, cols: int, cell: float) -> np.ndarray:
    """Point count per cell of the surface grid."""
    r, c = surface.pixel(scene, cell)
    ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    return np.bincount(r[ok] * cols + c[ok], minlength=rows * cols).reshape(rows, cols)


def fill_floor(room, arr: np.ndarray, log=print) -> np.ndarray:
    """The splat with its missing floor filled (see the module notes)."""
    from scipy.ndimage import (binary_closing, binary_dilation, distance_transform_edt,
                               gaussian_filter, zoom)
    from pointcloud import load_ply
    from splat_edit import discs, splat_colours

    space = room.space
    m = room.metre
    scene = room.to_scene(np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64))
    colours = splat_colours(arr)
    alpha = 1 / (1 + np.exp(-arr["opacity"].astype(np.float64)))
    scale = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1)
                   .astype(np.float64)).max(axis=1)
    inside = np.all(np.abs(scene[:, :2] - room.centre) < room.half, axis=1)
    band = inside & (np.abs(scene[:, 2] - room.floor_z) < 0.10 * m) & (alpha > 0.3)
    height = float(np.median(scene[band, 2])) if band.sum() > 100 else room.floor_z
    log(f"  splat floor at {(height - room.floor_z) / m * 100:+.1f} cm from stage 3's floor level")

    surface = floor_surface(room, height)
    rows, cols = surface.rows, surface.cols
    # What the dense cloud says about each spot: floor there, or something standing on it.
    dense = room.to_scene(load_ply(space / "cloud-dense.ply").points.astype(np.float64))
    level = dense[:, 2] - height
    standing = raster(surface, dense[(level > 0.15 * m) & (level < 1.8 * m)], rows, cols,
                      surface.cell) >= 2
    standing = binary_closing(binary_dilation(standing, iterations=6), iterations=3)
    under = np.zeros((rows, cols), bool)
    for box in room.shapes["boxes"]:
        if box.get("build", True):
            r0, c0 = surface.pixel(np.array([box["min"]]))
            r1, c1 = surface.pixel(np.array([box["max"]]))
            under[max(r0[0], 0):max(r1[0], 0), max(c0[0], 0):max(c1[0], 0)] = True
    blocked = standing | under

    solid = (alpha > 0.3) & (scene[:, 2] > height + 0.05 * m)
    blob_points = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)[solid].astype(np.float64)
    photo, seen_weight = photograph(space, room, surface, log, occluders=blob_points)
    confident = (seen_weight > 0.02) & ~blocked
    save = lambda image, name: Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)[::-1]).save(
        space / f"{surface.name}-{name}.png")
    save(np.where(confident[..., None], photo, (255, 0, 200)), "photo")
    log(f"  floor photo: {confident.mean():.0%} clean filmed floor, "
        f"{standing.mean():.0%} with something standing on it, {under.mean():.0%} under furniture")

    # The splat's own floor, which is what the fill must sit next to.
    grid = BLOB_SPACING_M * m
    k = grid / surface.cell
    splat_img, splat_w = splat_floor(room, scene, colours, alpha, scale, height, grid)
    splat_ok = splat_w >= 0.3
    grow = lambda image: zoom(image.astype(float), (k, k) + (1,) * (image.ndim - 2),
                              order=0 if image.dtype == bool else 1)[:rows, :cols]
    pad = lambda image: np.pad(image, [(0, max(0, rows - image.shape[0])),
                                       (0, max(0, cols - image.shape[1]))]
                               + [(0, 0)] * (image.ndim - 2), mode="edge")
    splat_ok_fine = pad(grow(splat_ok)) > 0.5
    splat_fine = pad(grow(splat_img))

    # Broad colour from the splat's floor (smoothly spread), fine detail from the photo.
    sigma_base = 0.15 * m / surface.cell
    known_base = (splat_ok_fine & ~blocked).astype(float)
    spread = gaussian_filter(known_base, sigma_base)
    base = np.stack([gaussian_filter(splat_fine[..., ch] * known_base, sigma_base)
                     / np.maximum(spread, 1e-6) for ch in range(3)], axis=-1)
    photo_low = np.stack([gaussian_filter(photo[..., ch] * confident, 0.03 * m / surface.cell)
                          / np.maximum(gaussian_filter(confident.astype(float), 0.03 * m / surface.cell), 1e-6)
                          for ch in range(3)], axis=-1)
    detail = np.where(confident[..., None], photo - photo_low, 0.0)
    if (spread > 0.05).any():
        # Where no splat floor is near, the photo's broad colour stands in, scaled to the splat's.
        both = confident & (spread > 0.2)
        gain = (np.array([base[..., ch][both].mean() / max(photo_low[..., ch][both].mean(), 1e-6)
                          for ch in range(3)]) if both.sum() > 200 else np.ones(3))
        fallback = photo_low * np.clip(gain, 0.5, 1.8)
        near = np.clip(spread / 0.2, 0, 1)[..., None]
        base = near * base + (1 - near) * np.where(confident[..., None], fallback, base)
    context = base + detail
    known = (splat_ok_fine | confident) & ~blocked
    filled = inpaint(np.where(known[..., None], context, 128), ~known, log)
    small = zoom(filled, (1 / k, 1 / k, 1), order=1)
    gh, gw = min(small.shape[0], splat_img.shape[0]), min(small.shape[1], splat_img.shape[1])
    small, splat_img, splat_w, splat_ok = small[:gh, :gw], splat_img[:gh, :gw], splat_w[:gh, :gw], splat_ok[:gh, :gw]
    fit = lambda mask: zoom(mask.astype(float), 1 / k, order=0)[:gh, :gw] > 0.5

    # Where the splat has no floor of its own, and nothing stands in the way.
    blocked_small = fit(blocked)
    holes = ~splat_ok & ~blocked_small
    log(f"  the splat's own floor covers {splat_ok.mean():.0%}; filling {holes.mean():.0%}")
    save(zoom(np.where(holes[..., None], small, np.where(splat_ok[..., None], splat_img, 60)),
              (k, k, 1), order=0), "filled")

    # Remove floor-level guesses in the holes, then lay the fill with a feathered edge.
    r, c = surface.pixel(scene, grid)
    in_grid = (r >= 0) & (r < gh) & (c >= 0) & (c < gw)
    guess = np.zeros(len(arr), bool)
    guess[in_grid] = holes[r[in_grid], c[in_grid]]
    guess &= inside & (scene[:, 2] < height + 0.04 * m)
    # Haze over the filled floor: soft or oversized blobs hanging low where
    # the dense cloud has nothing standing. They sat over floor no frame saw
    # clearly, and next to the clean fill they read as fog.
    near_fill = np.zeros((gh, gw), bool)
    near_fill[distance_transform_edt(~holes) * BLOB_SPACING_M <= FEATHER_M] = True
    near_fill &= ~blocked_small
    over = np.zeros(len(arr), bool)
    over[in_grid] = near_fill[r[in_grid], c[in_grid]]
    haze = (over & inside & (scene[:, 2] >= height + 0.04 * m) & (scene[:, 2] < height + 0.5 * m)
            & ((alpha < 0.6) | (scale > 0.05 * m)))
    log(f"  removed {int(haze.sum()):,} hazy blobs hanging over the filled floor")
    guess |= haze
    distance_in = distance_transform_edt(~holes) * BLOB_SPACING_M      # metres from the nearest hole
    opacity = np.where(holes, 0.9, 0.9 * np.clip(1 - distance_in / FEATHER_M, 0, 1))
    # Under furniture next to a hole, keep going a little at full strength, out of sight.
    opacity = np.where(blocked_small, np.where(distance_in <= TUCK_M, 0.9, 0.0), opacity)
    rr, cc = np.nonzero(opacity > 0.02)
    rng = np.random.default_rng(5)
    jitter = rng.uniform(-0.35, 0.35, (len(rr), 2))
    centres = (surface.origin + (cc[:, None] + 0.5 + jitter[:, :1]) * grid * surface.u
               + (rr[:, None] + 0.5 + jitter[:, 1:]) * grid * surface.v
               + rng.normal(0, 0.004 * m, (len(rr), 1)) * surface.normal)
    blobs = discs(centres, surface.normal, small[rr, cc], grid, room)
    blobs["opacity"] = np.log(opacity[rr, cc] / (1 - opacity[rr, cc]))
    size = rng.uniform(0.9, 1.4, len(rr))
    blobs["scale_0"] = np.log(grid * 0.75 * size)
    blobs["scale_1"] = np.log(grid * 0.75 * size * rng.uniform(0.7, 1.0, len(rr)))
    blobs["scale_2"] = np.log(np.full(len(rr), 0.003 * m))
    log(f"  removed {int(guess.sum()):,} floor-level guesses in the holes; added "
        f"{len(blobs):,} floor blobs ({int(holes.sum()):,} in holes, the rest fading out)")
    return np.concatenate([arr[~guess], blobs])
