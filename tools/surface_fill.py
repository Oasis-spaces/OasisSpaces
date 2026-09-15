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
     level, guesses the training made with no view, are removed first, and so
     is haze over the whole open floor: soft blobs within 60 cm of it that no
     dense-cloud surface supports.

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
HAZE_TOP_M = 0.60           # soft blobs this low over open floor may be haze...
HAZE_SUPPORT_M = 0.06       # ...when no dense-cloud surface is this close


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


def photograph(space: Path, room, surfaces: list, log=print, occluders=None,
               frame_step: int = FRAME_STEP):
    """For each surface: (photo HxWx3 float, weight HxW, see-through count HxW),
    the surface as the frames saw it. One pass over the frames serves every
    surface. `occluders` (camera-solve points) add to the dense cloud for
    deciding what blocks a view: the splat's blobs cover furniture sides that
    only frames between the dense cloud's keyframes saw. A spot counts as seen
    through in a frame when that frame sees a surface well behind it: an
    opening such as a doorway, never to be filled."""
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
    grids = []
    for surface in surfaces:
        pts_s = surface.points()
        grids.append((pts_s, pts_s @ room.world, len(pts_s)))   # scene, camera-solve frame
    samples = [[] for _ in surfaces]         # per surface, per frame: (cells, colours, weights)
    through = [np.zeros(n) for _, _, n in grids]
    frames_used = [0] * len(surfaces)
    for info in images[::frame_step]:
        cam = cameras[info["camera_id"]]
        centre_s = room.world @ (-info["R"].T @ info["t"])
        buf = None
        img = None
        for k, surface in enumerate(surfaces):
            pts_s, pts_c, n = grids[k]
            if (centre_s - surface.origin) @ surface.normal <= 0:
                continue  # the camera is behind this surface
            u, v, z = project(pts_c, info, cam)
            ray = pts_s - centre_s
            dist = np.linalg.norm(ray, axis=1)
            cos = np.abs(ray @ surface.normal) / np.maximum(dist, 1e-9)
            ok = ((z > 0.1 * m) & (dist < MAX_DISTANCE_M * m) & (cos > MIN_COS)
                  & (u >= 0) & (v >= 0) & (u < cam["width"] - 1) & (v < cam["height"] - 1))
            if ok.sum() < 100:
                continue
            if buf is None:
                buf = depth_buffer(dense, info, cam)
            idx = np.flatnonzero(ok)
            nearest = buf[(v[idx] / DEPTH_SHRINK).astype(int), (u[idx] / DEPTH_SHRINK).astype(int)]
            through[k] += np.bincount(idx[np.isfinite(nearest) & (nearest > z[idx] + 0.30 * m)],
                                      minlength=n)
            idx = idx[z[idx] <= nearest + 0.05 * m + 0.02 * z[idx]]
            if len(idx) < 100:
                continue
            if img is None:
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
            samples[k].append((idx.astype(np.int32), colour.astype(np.float32),
                               weight.astype(np.float32)))
            frames_used[k] += 1
    # How well seen, relative to a frame looking straight at the surface from 1.5 m.
    reference = (1.0 / ((1.5 * m) / (cameras[images[0]["camera_id"]]["params"][0] / 2))) ** 2
    results = []
    for k, surface in enumerate(surfaces):
        n = grids[k][2]

        def accumulate(keep_fn=None):
            wsum, csum = np.zeros(n), np.zeros((n, 3))
            for idx, colour, weight in samples[k]:
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
        for idx, colour, weight in samples[k]:
            dev_sum += np.bincount(idx, weights=weight * np.sum((colour - mean[idx]) ** 2, axis=1),
                                   minlength=n)
        spread = np.sqrt(dev_sum / np.maximum(wsum, 1e-12))
        wsum2, csum2 = accumulate(lambda idx, colour: np.linalg.norm(colour - mean[idx], axis=1)
                                  <= np.maximum(20.0, 1.2 * spread[idx]))
        shape = (surface.rows, surface.cols)
        results.append(((csum2 / np.maximum(wsum2, 1e-12)[:, None]).reshape(*shape, 3),
                        (wsum2 / reference).reshape(shape), through[k].reshape(shape)))
        log(f"  {surface.name} photo: {frames_used[k]} frames saw it")
    return results


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


# ------------------------------------------------------------- shared fill
def splat_surface(surface: Surface, scene, colours, alpha, scale, cell, band):
    """The splat's own blobs near a surface, as an image on the `cell` grid
    (colour, weight), each blob spread over its footprint. `band` is a
    distance either side of the plane, or a (behind, in front) pair."""
    rows = int(round(surface.rows * surface.cell / cell))
    cols = int(round(surface.cols * surface.cell / cell))
    rel = scene - surface.origin
    along, up, depth = rel @ surface.u / cell, rel @ surface.v / cell, rel @ surface.normal
    lo, hi = (-band, band) if np.isscalar(band) else band
    sel = np.flatnonzero((depth > lo) & (depth < hi) & (alpha > 0.2) & (along > -2) & (up > -2)
                         & (along < cols + 2) & (up < rows + 2))
    wsum = np.zeros((rows, cols))
    csum = np.zeros((rows, cols, 3))
    for i in sel:
        cx, cy = along[i], up[i]
        r = max(1.0, 2 * scale[i] / cell)
        x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, cols))
        y0, y1 = int(max(cy - r, 0)), int(min(cy + r + 1, rows))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = alpha[i] * np.exp(-2 * ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2)
        wsum[y0:y1, x0:x1] += w
        csum[y0:y1, x0:x1] += w[..., None] * colours[i]
    return csum / np.maximum(wsum, 1e-9)[..., None], wsum


def fill_surface(room, surface: Surface, scene, colours, alpha, scale, photo, seen_weight,
                 blocked, opening, slab, splat_band, log=print, region=None):
    """The shared steps for one surface, given what blocks it and where it
    opens: returns (blobs to add, mask of existing blobs to remove, stats).
    Broad colour comes from the splat's own surface, fine detail from the
    photo, LaMa continues both into what is unknown, and flat blobs go where
    the splat has none, feathered into its own and tucked behind furniture."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
    from splat_edit import discs

    m = room.metre
    space = room.space
    rows, cols = surface.rows, surface.cols
    # For a removed object (a region), texture and colour come from around it,
    # not from its own spot: that holds its shadow and what is left of it.
    outside = ~region if region is not None else np.ones((rows, cols), bool)
    confident = (seen_weight > 0.02) & ~blocked & ~opening & outside
    save = lambda image, name: Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)[::-1]).save(
        space / f"{surface.name}-{name}.png")
    save(np.where(confident[..., None], photo, (255, 0, 200)), "photo")

    grid = BLOB_SPACING_M * m
    k = grid / surface.cell
    splat_img, splat_w = splat_surface(surface, scene, colours, alpha, scale, grid, splat_band)
    splat_ok = splat_w >= 0.3
    grow = lambda image: zoom(image.astype(float), (k, k) + (1,) * (image.ndim - 2),
                              order=0 if image.dtype == bool else 1)[:rows, :cols]
    pad = lambda image: np.pad(image, [(0, max(0, rows - image.shape[0])),
                                       (0, max(0, cols - image.shape[1]))]
                               + [(0, 0)] * (image.ndim - 2), mode="edge")
    splat_ok_fine = pad(grow(splat_ok)) > 0.5
    splat_fine = pad(grow(splat_img))

    # Broad colour from the splat's surface (smoothly spread), fine detail from the photo.
    sigma_base = 0.15 * m / surface.cell
    known_base = (splat_ok_fine & ~blocked & ~opening & outside).astype(float)
    spread = gaussian_filter(known_base, sigma_base)
    base = np.stack([gaussian_filter(splat_fine[..., ch] * known_base, sigma_base)
                     / np.maximum(spread, 1e-6) for ch in range(3)], axis=-1)
    sigma_low = 0.03 * m / surface.cell
    photo_low = np.stack([gaussian_filter(photo[..., ch] * confident, sigma_low)
                          / np.maximum(gaussian_filter(confident.astype(float), sigma_low), 1e-6)
                          for ch in range(3)], axis=-1)
    detail = np.where(confident[..., None], photo - photo_low, 0.0)
    if (spread > 0.05).any():
        # Where none of the splat's surface is near, the photo's broad colour
        # stands in, scaled to the splat's.
        both = confident & (spread > 0.2)
        gain = (np.array([base[..., ch][both].mean() / max(photo_low[..., ch][both].mean(), 1e-6)
                          for ch in range(3)]) if both.sum() > 200 else np.ones(3))
        fallback = photo_low * np.clip(gain, 0.5, 1.8)
        near = np.clip(spread / 0.2, 0, 1)[..., None]
        base = near * base + (1 - near) * np.where(confident[..., None], fallback, base)
    known = (splat_ok_fine | confident) & ~blocked & ~opening & outside
    stats = {"splat": float(splat_ok.mean()), "clean_photo": float(confident.mean())}
    if known.mean() < 0.05:
        log(f"  {surface.name}: too little of it was filmed cleanly ({known.mean():.0%}); left as it is")
        return None, np.zeros(len(scene), bool), stats
    filled = inpaint(np.where(known[..., None], base + detail, 128), ~known, log)
    small = zoom(filled, (1 / k, 1 / k, 1), order=1)
    gh, gw = min(small.shape[0], splat_img.shape[0]), min(small.shape[1], splat_img.shape[1])
    small, splat_img, splat_ok = small[:gh, :gw], splat_img[:gh, :gw], splat_ok[:gh, :gw]
    fit = lambda mask: zoom(mask.astype(float), 1 / k, order=0)[:gh, :gw] > 0.5

    blocked_small, opening_small = fit(blocked), fit(opening)
    # A hole is where the splat has nothing at all near the surface: not on the
    # plane, and not set back or standing out from it. A door or window a few
    # centimetres behind a wall's plane is not wall to paint over.
    slab_img, slab_w = splat_surface(surface, scene, colours, alpha, scale, grid,
                                     (slab[0] * m, slab[1] * m))
    slab_img, slab_w = slab_img[:gh, :gw], slab_w[:gh, :gw]
    # Things in front (furniture, clutter) are kept out of the texture above, but
    # not out of the fill: blobs on the surface sit behind them, hidden where
    # they render and showing surface where the splat has nothing at all.
    holes = (slab_w < 0.3) & ~opening_small
    remnant = np.zeros_like(holes)
    if region is not None:
        region_small = fit(region)
        # Where an object was removed: fill wherever the surface is not solidly
        # covered, and replace what is left of the object near the surface (a
        # headboard's back, its shadow, set back or on the plane) wherever it
        # is far from the filled colour.
        remnant = (region_small & (slab_w >= 0.3) & ~opening_small
                   & (np.linalg.norm(slab_img - small, axis=-1) > REMNANT_COLOUR_GAP))
        holes = ((slab_w < REGION_SOLID_WEIGHT) & region_small & ~opening_small) | remnant
    stats["holes"] = float(holes.mean())
    save(zoom(np.where(holes[..., None], small, np.where(splat_ok[..., None], splat_img, 60)),
              (k, k, 1), order=0), "filled")

    # Faint leftovers in the holes (the splat's guesses with no view) go.
    rel = scene - surface.origin
    r, c = (rel @ surface.v / grid).astype(int), (rel @ surface.u / grid).astype(int)
    depth = rel @ surface.normal
    in_grid = (r >= 0) & (r < gh) & (c >= 0) & (c < gw)
    guess = np.zeros(len(scene), bool)
    guess[in_grid] = holes[r[in_grid], c[in_grid]]
    faint = alpha < 0.35
    if remnant.any():
        # In a removed object's leftovers, every blob on the surface goes.
        in_remnant = np.zeros(len(scene), bool)
        in_remnant[in_grid] = remnant[r[in_grid], c[in_grid]]
        faint |= in_remnant
    guess &= (depth > slab[0] * m) & (depth < slab[1] * m) & faint

    distance_in = distance_transform_edt(~holes) * BLOB_SPACING_M      # metres from the nearest hole
    # Fade out only over the splat's own surface, never over anything else nearby.
    opacity = np.where(holes, 0.9,
                       np.where(splat_ok, 0.9 * np.clip(1 - distance_in / FEATHER_M, 0, 1), 0.0))
    opacity[opening_small] = 0.0             # never across an opening
    rr, cc = np.nonzero(opacity > 0.02)
    if len(rr) == 0:
        return None, guess, stats
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
    stats["added"] = len(blobs)
    return blobs, guess, stats


def blob_arrays(room, arr):
    scene = room.to_scene(np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64))
    from splat_edit import splat_colours

    alpha = 1 / (1 + np.exp(-arr["opacity"].astype(np.float64)))
    scale = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1)
                   .astype(np.float64)).max(axis=1)
    return scene, splat_colours(arr), alpha, scale


def raster(surface: Surface, scene: np.ndarray, rows: int, cols: int, cell: float) -> np.ndarray:
    """Point count per cell of the surface grid."""
    r, c = surface.pixel(scene, cell)
    ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    return np.bincount(r[ok] * cols + c[ok], minlength=rows * cols).reshape(rows, cols)


# ------------------------------------------------------------- floor fill
def floor_masks(room, arr, dense, log=print, regions=None):
    """(surface, blocked, arr with haze cleared) for the floor; with `regions`,
    haze is only cleared near those boxes."""
    from scipy.ndimage import binary_closing, binary_dilation
    from scipy.spatial import cKDTree

    m = room.metre
    scene, colours, alpha, scale = blob_arrays(room, arr)
    inside = np.all(np.abs(scene[:, :2] - room.centre) < room.half, axis=1)
    band = inside & (np.abs(scene[:, 2] - room.floor_z) < 0.10 * m) & (alpha > 0.3)
    height = float(np.median(scene[band, 2])) if band.sum() > 100 else room.floor_z
    log(f"  splat floor at {(height - room.floor_z) / m * 100:+.1f} cm from stage 3's floor level")
    surface = floor_surface(room, height)
    rows, cols = surface.rows, surface.cols
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
    log(f"  floor: {standing.mean():.0%} with something standing on it, {under.mean():.0%} under furniture")

    # Haze over the floor: soft or oversized blobs hanging within HAZE_TOP_M
    # of it over open floor, with no dense-cloud surface near them.
    rise = scene[:, 2] - height
    band_idx = np.flatnonzero(inside & (rise > 0.04 * m) & (rise < HAZE_TOP_M * m))
    r, c = surface.pixel(scene[band_idx])
    ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    open_floor = np.zeros(len(band_idx), bool)
    open_floor[ok] = ~binary_dilation(blocked, iterations=4)[r[ok], c[ok]]
    band_idx = band_idx[open_floor]
    if regions:
        near = np.zeros(len(band_idx), bool)
        for lo, hi in regions:
            near |= np.all((scene[band_idx, :2] >= np.array(lo[:2]) - REGION_MARGIN_M * m)
                           & (scene[band_idx, :2] <= np.array(hi[:2]) + REGION_MARGIN_M * m), axis=1)
        band_idx = band_idx[near]
    above = dense[(level > 0.04 * m) & (level < (HAZE_TOP_M + 0.1) * m)]
    support = (cKDTree(above).query(scene[band_idx])[0] if len(above)
               else np.full(len(band_idx), np.inf))
    soft = (alpha[band_idx] < 0.6) | (scale[band_idx] > 0.05 * m)
    haze_idx = band_idx[((support > HAZE_SUPPORT_M * m) & soft) | (alpha[band_idx] < 0.2)]
    keep = np.ones(len(arr), bool)
    keep[haze_idx] = False
    log(f"  cleared {len(haze_idx):,} hazy blobs over the open floor")
    return surface, blocked, arr[keep]


# -------------------------------------------------------------- wall fill
def wall_surfaces(room, scene, alpha, log=print) -> list:
    """A Surface for each measured, built wall, at the depth where the
    splat's own wall actually is (stage 3 squares and moves walls by a few cm)."""
    m = room.metre
    level = room.shapes["room_level"]
    height = level["height"]
    out = []
    for i, p in enumerate(room.shapes["planes"]):
        if p.get("label") != "wall" or p.get("build") is False or p.get("source") == "inferred":
            continue
        a = np.array(p["axis_a"], float)
        a[2] = 0
        a /= np.linalg.norm(a)
        n = np.array(p["normal"], float)
        n[2] = 0
        n /= np.linalg.norm(n)
        c = np.array(p["center"], float)
        if (np.r_[room.centre, c[2]] - c) @ n < 0:
            n = -n                                   # face into the room
        half = p["half_a"]
        rel = scene - c
        on = ((np.abs(rel @ n) < 0.25 * m) & (np.abs(rel @ a) < half - 0.2 * m) & (alpha > 0.3)
              & (scene[:, 2] > level["floor_z"] + 0.9 * m) & (scene[:, 2] < level["floor_z"] + 2.0 * m))
        shift = float(np.median(rel[on] @ n)) if on.sum() > 100 else 0.0
        if abs(shift) > 0.15 * m:
            # Something else stands in front here (a wardrobe front stage 3 moved
            # this wall behind): keep stage 3's measured position.
            log(f"  W{i}: the nearest surface is {shift / m * 100:+.0f} cm off, not this wall; "
                "keeping stage 3's position")
            shift = 0.0
        cell = PHOTO_CELL_M * m
        origin = c - half * a + shift * n
        origin[2] = level["floor_z"]
        out.append(Surface(f"wall-W{i}", origin, a, np.array([0, 0, 1.0]), n,
                           int(round(2 * half / cell)), int(round(height / cell)), cell))
        log(f"  W{i}: {2 * half / m:.2f} m long; the splat's wall sits {shift / m * 100:+.1f} cm from stage 3's")
    return out


def wall_openings(room, surface: Surface, points, through):
    """Where a wall is open (a doorway, a window): the cameras reconstructed
    things more than 25 cm behind its plane there, which a solid wall would
    hide, or frames saw a surface well behind the spot."""
    from scipy.ndimage import binary_closing, binary_dilation

    m = room.metre
    rows, cols = surface.rows, surface.cols
    coarse = 0.04 * m
    rel = points - surface.origin
    depth = rel @ surface.normal
    behind = points[(depth < -0.25 * m) & (depth > -4.0 * m)]
    crow, ccol = int(np.ceil(rows * surface.cell / coarse)), int(np.ceil(cols * surface.cell / coarse))
    counts = raster(surface, behind, crow, ccol, coarse)
    k = coarse / surface.cell
    beyond = np.kron(counts >= 3, np.ones((int(round(k)), int(round(k))), bool))[:rows, :cols]
    beyond = np.pad(beyond, ((0, rows - beyond.shape[0]), (0, cols - beyond.shape[1])))
    opening = binary_closing(beyond | (through >= 2), iterations=6)
    return binary_dilation(opening, iterations=6)


def wall_masks(room, surface: Surface, dense, walls: list):
    """What stands in front of a wall, and the stretch hidden behind furniture."""
    from scipy.ndimage import binary_closing, binary_dilation

    m = room.metre
    rows, cols = surface.rows, surface.cols
    rel = dense - surface.origin
    depth = rel @ surface.normal
    up = rel @ surface.v
    near_other = np.zeros(len(dense), bool)
    for other in walls:
        if other is not surface:
            near_other |= np.abs((dense - other.origin) @ other.normal) < 0.08 * m
    front = ((depth > 0.06 * m) & (depth < 0.9 * m) & (up > 0.05 * m)
             & (up < rows * surface.cell - 0.05 * m) & ~near_other)
    standing = raster(surface, dense[front], rows, cols, surface.cell) >= 2
    standing = binary_closing(binary_dilation(standing, iterations=6), iterations=3)
    hidden = np.zeros((rows, cols), bool)
    for box in room.shapes["boxes"]:
        if not box.get("build", True):
            continue
        lo, hi = np.array(box["min"]), np.array(box["max"])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        crel = corners - surface.origin
        if (crel @ surface.normal).min() > 0.3 * m:
            continue                                 # not against this wall
        c0, c1 = int((crel @ surface.u).min() / surface.cell), int((crel @ surface.u).max() / surface.cell)
        r1 = int((crel @ surface.v).max() / surface.cell)
        hidden[:max(min(r1, rows), 0), max(c0, 0):max(min(c1, cols), 0)] = True
    return standing, hidden


# ------------------------------------------------------------------ run
REGION_MARGIN_M = 0.30        # a limited fill reaches this far past its boxes
REMNANT_COLOUR_GAP = 45       # surface this unlike the fill, where an object was, is left of it
REGION_SOLID_WEIGHT = 1.0     # and surface weaker than this there gets filled
REGION_ABOVE_M = 0.40         # and this far above them on a wall


def region_mask(room, surface: Surface, regions) -> np.ndarray:
    """The part of a surface near any of the scene boxes in `regions`."""
    m = room.metre
    pts = surface.points()
    mask = np.zeros(len(pts), bool)
    for lo, hi in regions:
        lo, hi = np.array(lo, float), np.array(hi, float)
        near = np.all((pts[:, :2] >= lo[:2] - REGION_MARGIN_M * m)
                      & (pts[:, :2] <= hi[:2] + REGION_MARGIN_M * m), axis=1)
        if surface.name != "floor":
            near &= pts[:, 2] <= hi[2] + REGION_ABOVE_M * m
        mask |= near
    return mask.reshape(surface.rows, surface.cols)


def fill_room(room, arr: np.ndarray, floor: bool = True, walls: bool = True, log=print,
              regions=None) -> np.ndarray:
    """The splat with its missing floor and/or wall surfaces filled; with
    `regions` (scene-frame (min, max) boxes), only near those boxes, and only
    on the surfaces they touch."""
    from pointcloud import load_ply

    m = room.metre
    dense = room.to_scene(load_ply(room.space / "cloud-dense.ply").points.astype(np.float64))
    jobs = []                                         # (surface, blocked, opening mask later)
    if floor:
        surface, blocked, arr = floor_masks(room, arr, dense, log, regions)
        jobs.append((surface, blocked, (-0.10, 0.04), 0.08))
    scene, colours, alpha, scale = blob_arrays(room, arr)
    if walls:
        surfaces = wall_surfaces(room, scene, alpha, log)
        for surface in surfaces:
            standing, hidden = wall_masks(room, surface, dense, surfaces)
            log(f"  {surface.name}: {standing.mean():.0%} with something in front, "
                f"{hidden.mean():.0%} behind furniture")
            jobs.append((surface, standing | hidden, (-0.35, 0.06), 0.06))
    masks = None
    if regions:
        masks = [region_mask(room, job[0], regions) for job in jobs]
        keep = [k for k, mask in enumerate(masks) if mask.any()]
        jobs, masks = [jobs[k] for k in keep], [masks[k] for k in keep]
        log("  limited to " + (", ".join(job[0].name for job in jobs) or "nothing"))
        if not jobs:
            return arr
    solid = (alpha > 0.3)
    occluders = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)[solid].astype(np.float64)
    photos = photograph(room.space, room, [j[0] for j in jobs], log, occluders=occluders,
                        frame_step=1 if walls else FRAME_STEP)
    remove = np.zeros(len(arr), bool)
    added = []
    behind_points = np.vstack([dense, scene[alpha > 0.3]])
    for k, ((surface, blocked, slab, splat_band), (photo, seen, through)) in enumerate(zip(jobs, photos)):
        opening = (wall_openings(room, surface, behind_points, through)
                   if surface.name != "floor" else np.zeros_like(blocked))
        blobs, guess, stats = fill_surface(room, surface, scene, colours, alpha, scale, photo, seen,
                                           blocked, opening, slab, splat_band, log,
                                           region=masks[k] if masks else None)
        log(f"  {surface.name}: the splat covers {stats['splat']:.0%}, "
            f"{stats['clean_photo']:.0%} filmed cleanly, "
            + (f"{opening.mean():.0%} open (seen through), " if surface.name != "floor" else "")
            + (f"filling {stats['holes']:.0%} with {stats.get('added', 0):,} blobs" if "holes" in stats
               else "not filled"))
        remove |= guess
        if blobs is not None:
            added.append(blobs)
    log(f"  removed {int(remove.sum()):,} faint leftover blobs in the holes")
    return np.concatenate([arr[~remove], *added])


def fill_floor(room, arr: np.ndarray, log=print) -> np.ndarray:
    return fill_room(room, arr, floor=True, walls=False, log=log)


def fill_walls(room, arr: np.ndarray, log=print) -> np.ndarray:
    return fill_room(room, arr, floor=False, walls=True, log=log)
