#!/usr/bin/env python3
"""Shape detection & regeneration: point cloud -> structured room model.

Detects the geometric structure of a reconstructed space and regenerates
it as clean parametric shapes:

  1. Estimate the up direction from the solved camera poses.
  2. RANSAC plane detection -> walls; floor and ceiling as height levels
     inside the walls.
  3. Cluster the remaining points -> furniture volumes (oriented boxes).
  4. Emit shapes.json (parameters) — the Blender builder
     (tools/blender_room.py) turns it into an editable .blend scene.

Usage:
    python3 pipeline/shapes.py spaces/<name> [--cloud cloud-dense.ply]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pointcloud import load_ply
from semantics import FURNITURE, HANGING


def read_camera_rotations(path):
    rotations = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", f.read(4))
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            f.read(24)
            struct.unpack("<I", f.read(4))
            while f.read(1) != b"\x00":
                pass
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)
            rotations.append(np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
            ]))
    return rotations


def estimate_up(model_dir: Path) -> np.ndarray:
    """A phone held upright films with its image-Y pointing at the floor,
    so world-up is the average of the cameras' -Y axes."""
    ups = [-R.T @ np.array([0.0, 1.0, 0.0]) for R in
           read_camera_rotations(model_dir / "images.bin")]
    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)


def ransac_plane(points, threshold, iterations=400, rng=None, normals=None,
                 min_cos=0.85, max_abs_up=None):
    """Plane with the most inliers. Given per-point normals, an inlier must
    also face the plane's way (within ~30°), so a plane stops counting the
    furniture it slices through. With max_abs_up, only planes whose normal's
    up component stays below it are tried (0.35: walls only)."""
    rng = rng or np.random.default_rng(0)

    def inliers(normal, d):
        mask = np.abs(points @ normal + d) < threshold
        if normals is not None:
            mask &= np.abs(normals @ normal) > min_cos
        return mask

    best_inliers, best = 0, None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if max_abs_up is not None and abs(normal[2]) > max_abs_up:
            continue
        d = -normal @ sample[0]
        count = int(inliers(normal, d).sum())
        if count > best_inliers:
            best_inliers, best = count, (normal, d)
    if best is None:
        return None, None, np.zeros(len(points), bool)
    normal, d = best
    mask = inliers(normal, d)
    # refine with least squares on inliers
    inlier_pts = points[mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = Vt[2]
    d = -normal @ centroid
    mask = inliers(normal, d)
    return normal, d, mask


# Walls this close to parallel, this near each other (metres, or 3% of the
# scene without a metric scale) and overlapping along their length are one
# wall found twice, or a wardrobe front beside it.
WALL_MERGE_DEGREES = 10.0
WALL_MERGE_METRES = 0.30


# Floor and ceiling are found as height levels, not by RANSAC: a pan often
# sees only a strip of the room's floor, and a plane fitter then prefers a
# tilted plane joining it to a lower corridor floor seen through the door.
# A level is a peak in the heights of points facing up (floor) or down
# (ceiling) inside the walls, holding at least LEVEL_MIN_SHARE of them.
LEVEL_NORMAL_DEGREES = 20.0
LEVEL_MIN_SHARE = 0.10
MIN_CEILING_METRES = 1.8   # or MIN_CEILING_FRAC of the scene without a scale
MIN_CEILING_FRAC = 0.30


def find_level(heights, bin_width, lowest):
    """Height of the lowest (or highest) well-populated level, or None."""
    if len(heights) < 1000:
        return None
    edges = np.arange(heights.min(), heights.max() + bin_width, bin_width)
    if len(edges) < 2:
        return float(np.median(heights))
    counts, _ = np.histogram(heights, edges)
    near = np.convolve(counts, [1, 1, 1], mode="same")  # a level may straddle two bins
    peaks = np.where(near >= LEVEL_MIN_SHARE * len(heights))[0]
    if len(peaks) == 0:
        return None
    k = peaks[0] if lowest else peaks[-1]
    lo, hi = edges[max(k - 1, 0)], edges[min(k + 2, len(edges) - 1)]
    return float(np.median(heights[(heights >= lo) & (heights <= hi)]))


def oriented_rect(points_2d, trim=1.0):
    """Rectangle (center, half-sizes) around 2D points, ignoring the outer
    `trim` percent on each side."""
    lo, hi = np.percentile(points_2d, [trim, 100 - trim], axis=0)
    return (lo + hi) / 2, (hi - lo) / 2


def along_wall(normal):
    """Horizontal direction running along a wall."""
    a = np.cross(normal, [0, 0, 1.0])
    return a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0, 0])


def overlapping(P, wall, other):
    """Do two walls share part of their length?"""
    a = along_wall(wall["normal"])
    lo1, hi1 = np.percentile(P[wall["idx"]] @ a, [5, 95])
    lo2, hi2 = np.percentile(P[other["idx"]] @ a, [5, 95])
    return max(lo1, lo2) < min(hi1, hi2)


def dominant_wall_axis(walls):
    """The room's main horizontal direction: wall directions folded to 90
    degrees and averaged, weighted by points, so floors can line up with it."""
    if not walls:
        return None
    total = np.zeros(2)
    for w in walls:
        a = along_wall(w["normal"])
        angle = np.arctan2(a[1], a[0])
        total += len(w["idx"]) * np.array([np.cos(4 * angle), np.sin(4 * angle)])
    angle = np.arctan2(total[1], total[0]) / 4
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def cluster_grid(points, cell):
    """Connected-component clustering on a voxel grid (26-connectivity)."""
    grid = np.floor(points / cell).astype(np.int64)
    grid -= grid.min(axis=0)
    dims = grid.max(axis=0) + 2
    keys = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
    unique, inverse = np.unique(keys, return_inverse=True)
    index = {k: i for i, k in enumerate(unique.tolist())}
    parent = np.arange(len(unique))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    offsets = [(dx * dims[1] + dy) * dims[2] + dz
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0)]
    for i, k in enumerate(unique.tolist()):
        for off in offsets:
            j = index.get(k + off)
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    labels = np.array([find(i) for i in range(len(unique))])
    return labels[inverse]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space")
    parser.add_argument("--cloud", default=None,
                        help="which cloud to analyze (default: densest available)")
    parser.add_argument("--max-walls", type=int, default=4)
    args = parser.parse_args()

    space = Path(args.space)
    cloud_path = (space / args.cloud) if args.cloud else next(
        p for p in [space / "cloud-dense.ply", space / "cloud.ply"] if p.exists())
    cloud = load_ply(cloud_path)
    print(f"{len(cloud):,} points from {cloud_path.name}")

    # cap points for speed
    if len(cloud) > 800_000:
        cloud = cloud.subset(
            np.random.default_rng(0).choice(len(cloud), 800_000, replace=False))
    pts, cols = cloud.points.astype(np.float64), cloud.colors

    sparse_dir = space / "workspace" / "sparse"
    models = [d for d in sparse_dir.iterdir() if (d / "points3D.bin").exists()] \
        if sparse_dir.exists() else []
    densify_meta = space / "densify.json"
    if densify_meta.exists():
        # The dense cloud lives in the frame of the model densify used.
        up = estimate_up(Path(json.loads(densify_meta.read_text())["model_dir"]))
    elif models:
        up = estimate_up(max(models, key=lambda d: (d / "points3D.bin").stat().st_size))
    else:
        up = np.array([0.0, 0.0, 1.0])  # no camera solve: assume Z-up
    print(f"up vector: {np.round(up, 3)}")

    # world frame: up = +Z
    z = up
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    world = np.stack([x, y, z])
    P = pts @ world.T
    N = cloud.normals.astype(np.float64) @ world.T if cloud.normals is not None else None

    extent = np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0))
    threshold = extent * 0.012
    remaining = np.ones(len(P), bool)
    point_labels = cloud.labels
    names = cloud.label_names or []
    furniture_ids = [names.index(n) for n in FURNITURE if n in names]
    if point_labels is not None and furniture_ids:
        detected_points = np.isin(point_labels, furniture_ids)
        # Detected furniture is neither wall nor floor: keep it out of the
        # plane fitting, and build boxes from it below.
        remaining[detected_points] = False
        print(f"detected furniture on {int(detected_points.sum()):,} points")
    hanging_ids = [names.index(n) for n in HANGING if n in names]
    if point_labels is not None and hanging_ids:
        hanging = np.isin(point_labels, hanging_ids)
        # Curtains: neither a wall to fit nor furniture to box.
        remaining[hanging] = False
        print(f"leaving out {int(hanging.sum()):,} curtain points")

    if N is not None:
        # Trust the cloud's normals only if they agree with the geometry: fit
        # the dominant plane from positions alone and check its inliers.
        n0, _, m0 = ransac_plane(P, threshold, rng=np.random.default_rng(3))
        agreement = float(np.median(np.abs(N[m0] @ n0)))
        if agreement <= 0.8:
            N = None
        print(f"normal agreement on the dominant plane: {agreement:.2f} "
              f"({'using' if N is not None else 'ignoring'} normals)")
    shapes = {"up": up.tolist(), "world": world.tolist(), "planes": [], "boxes": []}
    rng = np.random.default_rng(7)
    units_per_metre = None
    if densify_meta.exists():
        units_per_metre = json.loads(densify_meta.read_text()).get("colmap_units_per_metre")
    merge_distance = (WALL_MERGE_METRES * units_per_metre if units_per_metre
                      else extent * 0.03)

    # Walls by RANSAC (vertical planes only); floor and ceiling come later,
    # as height levels inside the walls.
    found = []
    for _ in range(args.max_walls + 2):
        active = np.where(remaining)[0]
        if len(active) < 5000:
            break
        normal, d, mask = ransac_plane(
            P[active], threshold, rng=rng,
            normals=N[active] if N is not None else None, max_abs_up=0.35)
        if normal is None or mask.sum() < len(P) * 0.02:
            break
        found.append({"normal": normal, "idx": active[mask], "kind": "wall"})
        remaining[active[mask]] = False

    # One wall found twice, or a wardrobe front standing beside it: keep the
    # plane with more points; the other's points return to the leftovers.
    walls = sorted((f for f in found if f["kind"] == "wall"), key=lambda f: -len(f["idx"]))
    for i, keep in enumerate(walls):
        if keep.get("merged"):
            continue
        for other in walls[i + 1:]:
            if other.get("merged"):
                continue
            if abs(keep["normal"] @ other["normal"]) < np.cos(np.radians(WALL_MERGE_DEGREES)):
                continue
            gap = abs(np.median((P[other["idx"]] - P[keep["idx"]].mean(axis=0)) @ keep["normal"]))
            if gap > merge_distance or not overlapping(P, keep, other):
                continue
            other["merged"] = True
            remaining[other["idx"]] = True
            shown = f"{gap / units_per_metre:.2f} m" if units_per_metre else f"{gap:.2f} units"
            print(f"  merged a duplicate wall ({len(other['idx']):,} pts, {shown} "
                  f"from a larger one)")
    found = [f for f in found if not f.get("merged")]
    wall_axis = dominant_wall_axis([f for f in found if f["kind"] == "wall"])
    if wall_axis is not None:
        # Turn the scene about the vertical so the room's main wall direction
        # runs along +x. Walls, floor and furniture boxes (which are built along
        # the x/y axes) then all share the room's orientation. A pure rotation:
        # nothing is re-measured.
        angle = float(np.arctan2(wall_axis[1], wall_axis[0]))
        cos_t, sin_t = np.cos(angle), np.sin(angle)
        turn = np.array([[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
        P = P @ turn.T
        if N is not None:
            N = N @ turn.T
        for f in found:
            f["normal"] = turn @ f["normal"]
        world = turn @ world
        shapes["world"] = world.tolist()
        wall_axis = np.array([1.0, 0.0, 0.0])
        print(f"turned the scene {np.degrees(angle):.1f} deg to line up with the walls")

    walls_found = [f for f in found if f["kind"] == "wall"]
    inside = np.ones(len(P), bool)
    if len(walls_found) >= 2:
        wall_xy = P[np.concatenate([f["idx"] for f in walls_found]), :2]
        lo, hi = np.percentile(wall_xy, [1, 99], axis=0)
        if np.all(hi - lo > extent * 0.05):
            inside = np.all((P[:, :2] >= lo) & (P[:, :2] <= hi), axis=1)
    cos_level = np.cos(np.radians(LEVEL_NORMAL_DEGREES))
    faces = {"floor": N[:, 2] > cos_level if N is not None else np.ones(len(P), bool),
             "ceiling": N[:, 2] < -cos_level if N is not None else np.ones(len(P), bool)}
    levels = {}
    for level in ("floor", "ceiling"):
        pool = np.where(remaining & inside & faces[level])[0]
        height = find_level(P[pool, 2], threshold, lowest=level == "floor") \
            if len(pool) else None
        if height is None:
            continue
        if level == "ceiling" and "floor" in levels:
            min_height = (MIN_CEILING_METRES * units_per_metre if units_per_metre
                          else MIN_CEILING_FRAC * extent)
            if height - levels["floor"] < min_height:
                continue  # a table or bed top, not the ceiling
        idx = pool[np.abs(P[pool, 2] - height) < threshold]
        levels[level] = height
        found.append({"normal": np.array([0.0, 0.0, 1.0]), "idx": idx,
                      "kind": "floor_or_ceiling", "level": level})
        remaining[idx] = False
        shown = f"{height / units_per_metre:.2f} m" if units_per_metre else f"{height:.2f}"
        print(f"  {level} level at z = {shown} ({len(idx):,} pts inside the walls)")

    for f in found:
        normal, idx, kind = f["normal"], f["idx"], f["kind"]
        vertical = abs(normal[2])
        n = normal if normal[2] >= 0 or vertical <= 0.85 else -normal
        # Floors and ceilings take their axes from the walls, so they line up
        # with the room instead of sitting at an arbitrary angle.
        if kind == "floor_or_ceiling" and wall_axis is not None:
            a = wall_axis - n * (wall_axis @ n)
        else:
            a = np.cross(n, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.array([1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        uv = np.column_stack([P[idx] @ a, P[idx] @ b])
        # Floors pick up stray points far from the room; trim harder.
        center_uv, half = oriented_rect(uv, trim=5.0 if kind == "floor_or_ceiling" else 1.0)
        offset = float(np.median(P[idx] @ n))
        center = center_uv[0] * a + center_uv[1] * b + offset * n
        color = cols[idx].mean(axis=0)
        shapes["planes"].append({
            "kind": kind, "normal": n.tolist(), "center": center.tolist(),
            "axis_a": a.tolist(), "axis_b": b.tolist(),
            "half_a": float(half[0]), "half_b": float(half[1]),
            "points": int(len(idx)), "color": color.astype(int).tolist(),
            **({"level": f["level"]} if "level" in f else {}),
        })
        print(f"  plane: {kind}, {len(idx):,} pts, "
              f"{2*half[0]:.1f} x {2*half[1]:.1f} units")

    def add_box(members, source, detected=None):
        lo, hi = np.percentile(P[members], [2, 98], axis=0)
        if np.any(hi - lo < extent * 0.01):
            return
        box = {"min": lo.tolist(), "max": hi.tolist(),
               "points": int(len(members)), "source": source,
               "color": cols[members].mean(axis=0).astype(int).tolist()}
        if detected:
            box["detected"] = detected
        shapes["boxes"].append(box)
        print(f"  box ({source}): {detected or 'unlabelled'}, {len(members):,} pts, "
              f"size {np.round(hi - lo, 1).tolist()}")

    # Detected furniture: one box per object, clustered so that two chairs
    # side by side do not merge into one.
    min_points = max(2000, int(len(P) * 0.002))
    if point_labels is not None:
        for name in FURNITURE:
            if name not in names:
                continue
            idx = np.where(point_labels == names.index(name))[0]
            if len(idx) < min_points:
                continue
            groups = cluster_grid(P[idx], cell=extent * 0.02)
            for group in np.unique(groups):
                members = idx[groups == group]
                if len(members) >= min_points:
                    add_box(members, "detected", name)

    # Whatever the planes and the detector left over, grouped geometrically.
    leftover = np.where(remaining)[0]
    if len(leftover):
        groups = cluster_grid(P[leftover], cell=extent * 0.02)
        for group in np.unique(groups):
            members = leftover[groups == group]
            if len(members) >= len(P) * 0.005:
                add_box(members, "geometry")

    out = space / "shapes.json"
    out.write_text(json.dumps(shapes, indent=1))
    print(f"{len(shapes['planes'])} planes + {len(shapes['boxes'])} boxes -> {out}")


if __name__ == "__main__":
    main()
