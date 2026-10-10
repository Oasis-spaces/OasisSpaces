"""Mirrors, windows and screens: the plane they sit in, instead of a hole.

Monocular depth on a mirror is the depth of the reflection, on a window the
view outside, on a screen whatever it shows. densify.py used to drop those
points, which deleted the front of any wardrobe with mirrored doors. What
Mirror3D (Tan et al., 2021) does instead works here too: take the depth of
a thin strip just outside the outline (the frame, the wall or the door
around it), fit a plane to it, and give the outline that plane's depth. The
points then stand where the surface is and take the name of what the outline
sits on (the wardrobe, or nothing, meaning the wall), so a mirrored wardrobe
keeps its front and a window stays in its wall.

Pure numpy, on the working-size depth map before back-projection.
"""

from __future__ import annotations

import numpy as np

RING_IN = 0.004          # the strip starts this share of the longest side outside the outline ...
RING_OUT = 0.016         # ... and ends here (the model smears depth across the outline's edge)
MIN_RING_PIXELS = 60     # fewer than this and the plane is a guess
ROUNDS = 3               # fit, drop the worst, refit
DEPTH_SPREAD = 0.25      # strip pixels this far (a share) from the strip's median depth are clutter
MAX_RMS_SHARE = 0.03     # the strip must lie on its plane to within 3% of its depth
MIN_INLIERS = 0.6        # ... with at least this share of it on the plane
DEPTH_RANGE = (0.3, 3.0) # a filled depth outside this multiple of the strip's is not believed
FARTHEST = 1.25          # ... nor one beyond this multiple of the frame's own farthest depth (the
                         # 98th percentile outside the outlines): a window's strip can be the view outside


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a (2r+1)-square, by a box sum over a padded cumsum."""
    if radius <= 0:
        return mask.copy()
    rows, cols = mask.shape
    padded = np.zeros((rows + 2 * radius + 1, cols + 2 * radius + 1), np.int32)
    padded[radius + 1:radius + 1 + rows, radius + 1:radius + 1 + cols] = mask
    summed = padded.cumsum(0).cumsum(1)
    k = 2 * radius + 1
    box = (summed[k:, k:] - summed[:-k, k:] - summed[k:, :-k] + summed[:-k, :-k])
    return box[:rows, :cols] > 0


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """A plane through `points` (N x 3): centre, unit normal, RMS distance of
    the inliers and their share, after ROUNDS of dropping the worst. The
    strip round a mirror lies at one depth, so points far from the strip's
    median depth (clutter in front, a doorway behind) are left out before the
    first fit, which a few of them would otherwise tilt. None when too few
    points remain."""
    z = points[:, 2]
    # The reference depth is the strip's densest one, not its median: a strip
    # half in a doorway has a median between its two depths, on neither.
    bins = np.arange(z.min(), z.max() + 0.05 * np.median(z) + 1e-9, 0.05 * np.median(z))
    counts, edges = np.histogram(z, bins=bins if len(bins) > 1 else 1)
    densest = edges[np.argmax(counts)] + 0.025 * np.median(z)
    keep = np.abs(z - densest) <= DEPTH_SPREAD * densest
    centre = normal = None
    for _ in range(ROUNDS):
        pts = points[keep]
        if len(pts) < 3:
            return None
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
        normal = vt[-1]
        distance = np.abs((points - centre) @ normal)
        inlier = distance[keep]
        limit = max(0.02, 2.5 * np.median(inlier) + 1e-6)   # 2 cm, or what most of the strip does
        keep = distance <= limit
    pts = points[keep]
    if len(pts) < 3:
        return None
    rms = float(np.sqrt(np.mean(((pts - centre) @ normal) ** 2)))
    return centre, normal, rms, float(keep.mean())


def mask_at(det: dict, shape: tuple[int, int], k: float) -> np.ndarray:
    """The detection's outline (or rectangle) as booleans at the depth map's raster."""
    rows, cols = shape
    mask = det.get("mask")
    if mask is not None:
        if mask.shape != shape:
            from PIL import Image

            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                              .resize((cols, rows), Image.NEAREST)) > 0
        return mask
    x0, y0, x1, y1 = (int(round(v * k)) for v in det["box"])
    out = np.zeros(shape, bool)
    out[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
    return out


def fill_unreliable(depth: np.ndarray, k: float, detections: list[dict], unreliable: set[str],
                    intrinsics: tuple[float, float, float, float], normal: np.ndarray | None = None):
    """Give each unreliable detection's outline the depth of the plane its
    surroundings lie in. `depth` is the working-size map (0 = invalid), `k`
    the scale from frame pixels to it, `intrinsics` the frame's fx, fy, cx,
    cy. Returns (depth, normal, filled, records): copies with the outlines
    filled, the mask of pixels filled, and one record per detection."""
    rows, cols = depth.shape
    fx, fy, cx, cy = (v * k for v in intrinsics)
    long_side = max(rows, cols)
    r_in, r_out = max(2, int(round(RING_IN * long_side))), max(6, int(round(RING_OUT * long_side)))
    targets = [d for d in detections if d["label"] in unreliable]
    if not targets:
        return depth, normal, np.zeros(depth.shape, bool), []
    depth = depth.copy()
    normal = normal.copy() if normal is not None else None
    original = depth.copy()
    masks = [mask_at(d, depth.shape, k) for d in targets]
    all_unreliable = np.zeros(depth.shape, bool)
    for m in masks:
        all_unreliable |= m
    filled = np.zeros(depth.shape, bool)
    records = []
    vv, uu = np.mgrid[:rows, :cols]
    solid = original[(original > 0) & ~dilate(all_unreliable, r_out)]
    farthest = float(np.percentile(solid, 98)) if len(solid) else np.inf
    for det, mask in zip(targets, masks):
        record = {"label": det["label"], "pixels": int(mask.sum()), "filled": 0, "why": None}
        records.append(record)
        if not mask.any():
            record["why"] = "empty outline"
            continue
        ring = dilate(mask, r_out) & ~dilate(mask, r_in) & ~all_unreliable & (original > 0)
        n_ring = int(ring.sum())
        record["ring"] = n_ring
        if n_ring < MIN_RING_PIXELS:
            record["why"] = f"only {n_ring} pixels of surroundings"
            continue
        z = original[ring]
        u, v = uu[ring].astype(np.float64), vv[ring].astype(np.float64)
        points = np.column_stack([(u - cx) / fx * z, (v - cy) / fy * z, z])
        fit = fit_plane(points)
        if fit is None:
            record["why"] = "no plane"
            continue
        centre, plane_normal, rms, inliers = fit
        median_z = float(np.median(z))
        record.update({"rms_m": round(rms, 4), "inliers": round(inliers, 2)})
        if rms > MAX_RMS_SHARE * median_z or inliers < MIN_INLIERS:
            record["why"] = f"surroundings not flat (rms {rms:.3f} m, {inliers:.0%} on the plane)"
            continue
        if median_z > FARTHEST * farthest:
            record["why"] = f"surroundings at {median_z:.1f} m, beyond the rest of the frame ({farthest:.1f} m)"
            continue
        if plane_normal[2] > 0:            # face the camera
            plane_normal = -plane_normal
        u, v = uu[mask].astype(np.float64), vv[mask].astype(np.float64)
        rays = np.column_stack([(u - cx) / fx, (v - cy) / fy, np.ones(len(u))])
        along = rays @ plane_normal
        with np.errstate(divide="ignore", invalid="ignore"):
            z_new = (centre @ plane_normal) / along
        ok = np.isfinite(z_new) & (z_new > 0.05) \
            & (z_new >= DEPTH_RANGE[0] * median_z) & (z_new <= DEPTH_RANGE[1] * median_z)
        if ok.sum() < 0.5 * len(ok):
            record["why"] = "the plane does not cover the outline"
            continue
        target = np.zeros(depth.shape, bool)
        target[mask] = ok
        depth[target] = z_new[ok]
        filled |= target
        if normal is not None:
            normal[target] = plane_normal.astype(normal.dtype)
        record["filled"] = int(ok.sum())
        record["plane_depth_m"] = round(float(np.median(z_new[ok])), 3)
    return depth, normal, filled, records
