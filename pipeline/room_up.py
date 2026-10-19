"""Which way is up in a camera solve.

COLMAP's frame knows nothing of gravity, so stage 3 (shapes.py), the splat
viewer's start views (splat_export.py) and the coverage map
(tools/capture_map.py) all find the room's up themselves, here:

  1. From the cameras (camera_up). A handheld phone pitches and turns but
     hardly rolls, so each frame's sideways axis (image x) stays level, and up
     is the direction most at right angles to all of them, whatever the pitch.
     Averaging the image's up axis instead leans back by the camera's pitch
     whenever a clip mostly looks one way: pexels-3769951, a pan filmed 12
     degrees down, came out 12 degrees off, and a room filmed like that came
     out 30% too tall with its bed 0.9 m too tall. When the camera never
     turned, the sideways axes are all parallel and say nothing about up; the
     mean image up is then all the cameras can give, pitch and all.
  2. From the room's level surfaces (refine_up), once a cloud is measured:
     floor, ceiling and the tops of furniture face straight up or down, so
     their normals correct what the cameras could not see.
"""

from __future__ import annotations

import numpy as np

# Below this, the camera never turned. The ratio of the sideways axes' second
# to largest spread is about the mean square of the turn (radians) around the
# clip's main direction, while hand roll alone scatters those axes by 1 to 3
# degrees (0.0003 to 0.003): a turn of less than about 3 degrees rms is lost in
# that jitter, and the plane the axes span could lean any way.
MIN_TURN = 0.003

# refine_up: the shrinking cones (degrees around the current up) whose
# normals are averaged; the support each needs, as a share of the normals and
# at least a count; the largest correction believed (a camera solve that is
# further off than this is less likely than a cloud whose level surfaces are
# not level); and the correction worth a warning.
REFINE_CONES_DEGREES = (35.0, 20.0, 10.0, 5.0)
REFINE_MIN_SHARE = 0.02
REFINE_MIN_POINTS = 1000
REFINE_MAX_DEGREES = 30.0
WARN_DEGREES = 3.0


def camera_up(cam_to_world: np.ndarray) -> tuple[np.ndarray, str]:
    """(up, how it was found) from cameras' rotations, camera-to-world (n, 3, 3):
    columns x right, y down, z forward, as COLMAP's R transposed."""
    rotations = np.asarray(cam_to_world, float)[:, :3, :3]
    sideways = rotations[:, :, 0]
    image_up = -rotations[:, :, 1].mean(axis=0)
    image_up /= np.linalg.norm(image_up)
    values, vectors = np.linalg.eigh(sideways.T @ sideways)
    if values[1] < MIN_TURN * values[2]:
        return image_up, "cameras' mean image up (the camera never turned)"
    up = vectors[:, 0]
    # The image's up points up, not down.
    return (up if up @ image_up > 0 else -up), "cameras' level sideways axes"


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two directions."""
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def refine_up(up: np.ndarray, normals: np.ndarray) -> np.ndarray | None:
    """Up corrected by the room's level surfaces: the mean of the normals that
    face within a cone of the current up (flipped to point up, so the ceiling
    counts with the floor), in shrinking cones so walls and slanted surfaces
    drop out as it settles. Stops at the last cone that holds enough normals;
    None when not even the widest one does. The caller decides whether to
    believe a correction (REFINE_MAX_DEGREES)."""
    normals = np.asarray(normals, float)
    length = np.linalg.norm(normals, axis=1)
    usable = np.isfinite(length) & (length > 0.5)
    normals = normals[usable] / length[usable, None]
    need = max(REFINE_MIN_POINTS, REFINE_MIN_SHARE * len(normals))
    up = np.asarray(up, float) / np.linalg.norm(up)
    refined = None
    for cone in REFINE_CONES_DEGREES:
        for _ in range(5):                      # settle inside this cone
            facing = normals @ up
            near = np.abs(facing) > np.cos(np.radians(cone))
            if near.sum() < need:
                break
            mean = (normals[near] * np.sign(facing[near])[:, None]).sum(axis=0)
            mean /= np.linalg.norm(mean)
            settled = angle_between(mean, up) < 0.01
            refined = up = mean
            if settled:
                break
        if near.sum() < need:
            break
    return refined
