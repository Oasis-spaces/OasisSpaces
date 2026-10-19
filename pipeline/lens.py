"""A solved camera's lens: from a pixel to its ray, and from a point to its pixel.

Stage 1 solves every video with COLMAP's OPENCV camera: fx, fy, cx, cy and
the lens distortion k1, k2, p1, p2. The later stages read the raw frames, so
those pixels are still bent by the lens. A pinhole ray, (u - cx) / fx, puts
every point off the middle of the picture in the wrong direction (walls bow,
keyframes disagree where they overlap), and a pinhole projection draws a
silhouette where the frame does not show the object. Both directions live
here, so every stage uses the one model the poses and sparse points were
solved with (tools/surface_fill.py projects through it too).

A bad solve's polynomial can fold inside the frame: past some radius it turns
rays back toward the middle of the picture. The lens is taken to end at the
fold (fold_radius), in both directions, so no pixel gets a ray from beyond it
and no point beyond it is projected back into the picture.

Pixel positions follow COLMAP: the centre of the pixel in column i, row j is
at (i + 0.5, j + 0.5).

    l = from_colmap(cam)                      # fx, fy, cx, cy, distortion, ...
    x, y = unproject(l, us + 0.5, vs + 0.5)   # the rays (x, y, 1) of pixel centres
    u, v = project(l, X / Z, Y / Z)           # camera-space points -> pixels
"""

from __future__ import annotations

import numpy as np

# COLMAP's camera models: id -> (focal lengths, principal point, distortion), as
# indices into its parameter list. The pipeline writes OPENCV.
SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV = 0, 1, 2, 3, 4
NO_DISTORTION = (0.0, 0.0, 0.0, 0.0)
NEWTON_STEPS = 50        # the solves here converge in under 10; a broken one may need more
TOLERANCE = 1e-9         # normalised units: well under a thousandth of a pixel
FOLD_DIRECTIONS = 64     # directions scanned for the fold ...
FOLD_STEPS = 256         # ... at this many radii each, then the first fold on each is bisected


def from_colmap(cam: dict) -> dict:
    """A COLMAP camera record (read_cameras_bin) as the lens the functions here
    take: fx, fy, cx, cy, distortion (k1, k2, p1, p2), width, height, and
    fov_radius, the widest ray the frame sees."""
    p = [float(v) for v in cam["params"]]
    model = cam["model"]
    if model in (SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        distortion = (p[3] if model != SIMPLE_PINHOLE else 0.0,
                      p[4] if model == RADIAL else 0.0, 0.0, 0.0)
    elif model in (PINHOLE, OPENCV):
        fx, fy, cx, cy = p[:4]
        distortion = tuple(p[4:8]) if model == OPENCV else NO_DISTORTION
    else:
        raise ValueError(f"COLMAP camera model {model} is not one the pipeline solves with "
                         "(SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL or OPENCV)")
    lens = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "distortion": distortion,
            "width": cam["width"], "height": cam["height"]}
    lens["fov_radius"] = fov_radius(lens)
    return lens


def distort(x: np.ndarray, y: np.ndarray, distortion=NO_DISTORTION):
    """Where the lens bends the ray (x, y, 1) to, in normalised coordinates."""
    k1, k2, p1, p2 = distortion
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def jacobian(x: np.ndarray, y: np.ndarray, distortion=NO_DISTORTION):
    """The Jacobian of distort at (x, y), [[a, b], [b, d]]: returns a, b, d."""
    k1, k2, p1, p2 = distortion
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    slope = 2 * (k1 + 2 * k2 * r2)
    return (radial + slope * x * x + 2 * p1 * y + 6 * p2 * x,
            slope * x * y + 2 * p1 * x + 2 * p2 * y,
            radial + slope * y * y + 6 * p1 * y + 2 * p2 * x)


def fold_radius(distortion, reach: float) -> float:
    """The smallest |(x, y)| within `reach` at which the lens polynomial folds:
    where the determinant of its Jacobian first reaches 0 going out from the
    middle, over FOLD_DIRECTIONS directions. Past it the polynomial turns rays
    back toward the middle of the picture, and further out the determinant
    can turn positive again where it mirrors rays through the middle, so no
    ray the lens sees lies beyond it. inf when the lens does not fold within
    reach, as a well-behaved one does not inside its frame."""
    if not any(distortion) or not reach > 0:
        return np.inf
    angle = np.linspace(0, 2 * np.pi, FOLD_DIRECTIONS, endpoint=False)
    radii = np.linspace(0, reach, FOLD_STEPS + 1)
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    a, b, d = jacobian(c * radii[1:], s * radii[1:], distortion)
    folded = a * d - b * b <= 0
    hit = folded.any(axis=1)
    if not hit.any():
        return np.inf
    first = np.argmax(folded[hit], axis=1)
    lo, hi = radii[first], radii[first + 1]
    c, s = c[hit, 0], s[hit, 0]
    for _ in range(30):                  # to well under a millionth of the reach
        mid = (lo + hi) / 2
        a, b, d = jacobian(c * mid, s * mid, distortion)
        past = a * d - b * b <= 0
        lo, hi = np.where(past, lo, mid), np.where(past, mid, hi)
    return float(lo.min())


def undistort(xd: np.ndarray, yd: np.ndarray, distortion=NO_DISTORTION):
    """The ray (x, y, 1) the lens bent to (xd, yd): `distort` inverted by
    Newton's method. Plain fixed-point iteration (OpenCV's undistortPoints)
    stalls on the strong distortion some phone solves have, a pixel or more
    off at the corners. NaN where no ray maps there on the lens's own side of
    the fold, which a well-behaved lens never has inside its frame."""
    xd, yd = np.asarray(xd, dtype=np.float64), np.asarray(yd, dtype=np.float64)
    if not any(distortion):
        return xd.copy(), yd.copy()
    x, y = xd.copy(), yd.copy()
    with np.errstate(all="ignore"):
        for _ in range(NEWTON_STEPS):
            ex, ey = distort(x, y, distortion)
            ex, ey = ex - xd, ey - yd
            a, b, d = jacobian(x, y, distortion)
            det = a * d - b * b
            step_x, step_y = (d * ex - b * ey) / det, (a * ey - b * ex) / det
            x, y = x - step_x, y - step_y
            if not np.any(np.abs(step_x) + np.abs(step_y) > TOLERANCE * 1e-3):
                break               # (a NaN step never converges, and is caught below)
        # Past the fold (the Jacobian's determinant turns negative) the
        # polynomial maps rays back toward the middle: such a root is no ray.
        a, b, d = jacobian(x, y, distortion)
        bx, by = distort(x, y, distortion)
        good = (np.hypot(bx - xd, by - yd) < TOLERANCE) & (a * d - b * b > 0)
        # Nor is one further out, where the determinant is positive again on
        # rays mirrored through the middle. The lens ends at its nearest fold,
        # in every direction as fov_radius does for project, so that within
        # it the lens is one-to-one and every ray projects back onto its pixel.
        r = np.hypot(x, y)
        if good.any():
            good &= r < fold_radius(distortion, float(r[good].max()))
    return np.where(good, x, np.nan), np.where(good, y, np.nan)


def unproject(lens: dict, u: np.ndarray, v: np.ndarray):
    """The rays (x, y, 1) through pixel positions (u, v): the pixel centre of
    column i is u = i + 0.5. NaN where the lens has no ray (see undistort)."""
    xd = (np.asarray(u, dtype=np.float64) - lens["cx"]) / lens["fx"]
    yd = (np.asarray(v, dtype=np.float64) - lens["cy"]) / lens["fy"]
    return undistort(xd, yd, lens.get("distortion", NO_DISTORTION))


def fov_radius(lens: dict) -> float:
    """The widest ray the frame sees (the largest |(x, y)| over its corners and
    edge midpoints): within it the lens polynomial holds. Never past the fold
    (fold_radius): a solve whose polynomial folds inside the frame leaves the
    border beyond the fold without rays, and its frame reaches the fold."""
    w, h = lens["width"], lens["height"]
    u = np.array([0.0, w, 0.0, w, w / 2, w / 2, 0.0, w])
    v = np.array([0.0, 0.0, h, h, 0.0, h, h / 2, h / 2])
    x, y = unproject(lens, u, v)
    r = np.hypot(x, y)
    pinhole = float(np.hypot(w / lens["fx"], h / lens["fy"]))
    widest = float(r.max()) if np.isfinite(r).all() else np.inf
    limit = min(widest, fold_radius(lens.get("distortion", NO_DISTORTION), min(widest, 4 * pinhole)))
    if np.isfinite(limit):
        return limit
    # No ray at the border and no fold found: Newton failed for some other reason.
    return float(np.nanmax(r)) if np.isfinite(r).any() else pinhole


def project(lens: dict, x: np.ndarray, y: np.ndarray):
    """Pixel positions (u, v) of the rays (x, y, 1), i.e. camera-space points
    divided by their depth. Within the frame's field of view this is COLMAP's
    own projection. Beyond it the polynomial would fold far-off points back into
    the picture, so there the distortion at the edge of the view is carried
    straight out: a point outside the view stays outside, in its direction."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    distortion = lens.get("distortion", NO_DISTORTION)
    if any(distortion):
        limit = lens.get("fov_radius") or fov_radius(lens)
        with np.errstate(divide="ignore", invalid="ignore"):
            shrink = np.minimum(1.0, limit / np.hypot(x, y))
        shrink = np.where(np.isfinite(shrink), shrink, 1.0)
        xd, yd = distort(x * shrink, y * shrink, distortion)
        x, y = xd / shrink, yd / shrink
    return lens["fx"] * x + lens["cx"], lens["fy"] * y + lens["cy"]
