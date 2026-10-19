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
     mean image up is then all the cameras can give, pitch and all. The few
     degrees a hand does roll are no turn, and after a narrow turn they can
     still tilt the axes' plane, so how far the levelling below may move up
     depends on how wide the turn was.
  2. From the room's level surfaces (refine_up), once a cloud is measured:
     floor, ceiling and the tops of furniture face straight up or down, so
     their normals correct what the cameras could not see.
"""

from __future__ import annotations

import numpy as np

# Below this, the camera never turned. Across the image up, the ratio of the
# sideways axes' second to largest spread is about the mean square of the turn
# (radians) around the clip's main direction: a turn of less than about 3
# degrees rms is no more than a hand wobbles, and the plane the axes span could
# lean any way.
MIN_TURN = 0.003

# A turn of more than about 10 degrees rms (a pan of 35 degrees or more)
# keeps the level axes' up within a few degrees of true even with 4 degrees of
# hand roll; a narrower one is still nearer true than the mean image up on
# average, but roll can tilt it 10 to 30 degrees, or further. This spread is
# taken in every direction: 4 degrees of roll adds only 0.005 to it, and
# across the image up a steep pitch would shrink the turn (by its cosine
# squared, half at 45 degrees down).
WIDE_TURN = 0.03

# After a narrow turn, a level-axes up further than this from the mean image
# up is not believed: few handheld clips look that far down or up on average,
# so the plane the sideways axes span was set by hand roll rather than by the
# turn. (A wide turn's plane is not, whatever the pitch.)
MAX_FROM_IMAGE_UP = 40.0

# How camera_up found up. Only TURNED is trusted to a few degrees; the level
# axes of a narrow turn can follow hand roll, and the mean image up leans by
# the camera's mean pitch, until the cloud's level surfaces correct them.
TURNED = "cameras' level sideways axes"
TURNED_A_LITTLE = "cameras' level sideways axes (a narrow turn, which hand roll can tilt)"
NEVER_TURNED = "cameras' mean image up (the camera never turned)"
ROLL_NOT_TURN = "cameras' mean image up (their level axes followed hand roll)"

# refine_up: the shrinking cones (degrees around the current up) whose normals
# are averaged, and the support each needs, as a share of the normals and at
# least a count. Then the largest correction believed, by how the cameras
# found up: after a wide turn a little more than such a solve is off
# (pexels-35308496 needed 6.9 degrees), otherwise a little more than a phone's
# mean pitch. A camera solve further off than that is less likely than a
# cloud whose most level surface is not level (an attic roof, a ramp). And
# the correction worth a warning.
REFINE_CONES_DEGREES = (35.0, 20.0, 10.0, 5.0)
REFINE_MIN_SHARE = 0.02
REFINE_MIN_POINTS = 1000
REFINE_MAX_TURNED_DEGREES = 10.0
REFINE_MAX_DEGREES = 30.0
WARN_DEGREES = 3.0


def camera_up(cam_to_world: np.ndarray) -> tuple[np.ndarray, str]:
    """(up, how it was found) from cameras' rotations, camera-to-world (n, 3, 3):
    columns x right, y down, z forward, as COLMAP's R transposed."""
    rotations = np.asarray(cam_to_world, float)[:, :3, :3]
    sideways = rotations[:, :, 0]
    image_up = -rotations[:, :, 1].mean(axis=0)
    image_up /= np.linalg.norm(image_up)
    # The turn is measured across the image up only: hand roll swings the
    # sideways axes along the image up, and counted as a turn it would let the
    # roll decide the plane they span.
    across = sideways - np.outer(sideways @ image_up, image_up)
    turn = np.linalg.eigvalsh(across.T @ across)
    if turn[1] < MIN_TURN * turn[2]:
        return image_up, NEVER_TURNED
    values, vectors = np.linalg.eigh(sideways.T @ sideways)
    up = vectors[:, 0]
    up = up if up @ image_up > 0 else -up       # the image's up points up, not down
    if values[1] >= WIDE_TURN * values[2]:
        return up, TURNED
    if angle_between(up, image_up) > MAX_FROM_IMAGE_UP:
        return image_up, ROLL_NOT_TURN
    return up, TURNED_A_LITTLE


def max_correction(how: str) -> float:
    """The largest levelling believed for an up camera_up found this way."""
    return REFINE_MAX_TURNED_DEGREES if how == TURNED else REFINE_MAX_DEGREES


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two directions."""
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def refine_up(up: np.ndarray, normals: np.ndarray) -> np.ndarray | None:
    """Up corrected by the room's level surfaces: the mean of the normals that
    face within a cone of the current up (flipped to point up, so the ceiling
    counts with the floor), in shrinking cones so walls and slanted surfaces
    drop out as it settles. None unless the result has the support of the
    narrow cones around it (on_a_surface): a wide cone over a floor and a
    sloped ceiling of about equal size settles halfway between them, where
    there is no surface at all. The caller decides whether to believe a
    correction (max_correction)."""
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
    if refined is None or not on_a_surface(refined, normals):
        return None
    return refined


def on_a_surface(direction: np.ndarray, normals: np.ndarray) -> bool:
    """Whether `direction` has the support of the narrow cones: the next to
    narrowest holds what a cone needs, and the narrowest about as many
    normals as the narrowest around any direction a cone's width away.
    Halfway between a floor and a sloped ceiling there are fewer than toward
    either. (Not a fixed count within the narrowest: normals fitted to noisy
    points spread so wide that fewer than 2% of them come within 5 degrees
    even of the true floor.)"""
    def facing(d, cone):
        return int((np.abs(normals @ d) > np.cos(np.radians(cone))).sum())

    wider, narrowest = REFINE_CONES_DEGREES[-2:]
    if facing(direction, wider) < max(REFINE_MIN_POINTS, REFINE_MIN_SHARE * len(normals)):
        return False
    here, a = facing(direction, narrowest), np.radians(narrowest)
    side = np.linalg.svd(direction[None])[2][1:]          # two directions at right angles to it
    for turn in np.radians(np.arange(0, 360, 45)):
        probe = np.cos(a) * direction + np.sin(a) * (np.cos(turn) * side[0] + np.sin(turn) * side[1])
        if facing(probe, narrowest) > 1.25 * here:
            return False
    return True
