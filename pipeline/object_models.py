#!/usr/bin/env python3
"""Per-object model jobs: a frame and a cut-out of each piece of furniture, for
an image-to-3D model to turn into a clean mesh.

The mixed scene shows a piece either as it was scanned (soft from every angle
the video did not see) or as a simple box model of its kind. This is the third
way: the object as a generated mesh, made from one frame of it.

What goes out, per piece, into <space>/object-models/<id>/:
  - view-N.jpg  the frames that saw the object best (tools/object_frames.py),
                from different sides where the video walked around it
  - view-N.png  that object in that frame, white on black
  - job.json    the cameras that took them, the measured box, the label

Several views go out because one is not enough: from a single frame an
image-to-3D model has to invent every side it cannot see, and a room's
furniture is filmed from one oblique angle, half behind clutter. The frames
are picked to be far apart around the object, which is what such a model can
actually use.

The mask is our own: the piece's Gaussians and the measured points inside its
box, projected into the frame, closed up and reduced to one blob. Stage 2's
detector outlines are not kept on disk, and this mask matches exactly what the
scene treats as that piece.

Usage:
    python3 pipeline/object_models.py spaces/<name> [--pieces B4 B13]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

VIEWS = 4                 # frames of one object, from as many sides as the video walked
MASK_GROW = 1 / 350       # close the gaps between a splat's blobs: this much of the frame's width
MASK_HOLE_SHARE = 0.02    # a hole smaller than this share of the object is filled in
MIN_MASK_SHARE = 0.005    # a mask smaller than this share of the frame is too small to model


def project(points_c: np.ndarray, info: dict, cam: dict):
    """Camera-solve points -> (u, v) pixels, keeping only what is in front."""
    local = points_c @ info["R"].T + info["t"]
    z = local[:, 2]
    ahead = z > 1e-6
    fx, fy, cx, cy = cam["params"][:4]
    u = fx * local[:, 0] / np.where(ahead, z, 1) + cx
    v = fy * local[:, 1] / np.where(ahead, z, 1) + cy
    return u, v, ahead


def blob_mask(room, points_s: np.ndarray, info: dict, cam: dict) -> np.ndarray:
    """A filled mask of the scene-frame points `points_s` in one frame: the
    points closed up into one blob, its small holes filled. Big holes are left
    open (the gap under a desk), and only the largest blob is kept, so a
    reflection or a stray does not come along."""
    from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label

    width, height = cam["width"], cam["height"]
    grow = max(2, round(width * MASK_GROW))
    u, v, ahead = project(room.to_splat(points_s), info, cam)
    on = ahead & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    mask = np.zeros((height, width), bool)
    mask[v[on].astype(int), u[on].astype(int)] = True
    mask = binary_closing(binary_dilation(mask, iterations=grow), iterations=grow * 3)
    blobs, found = label(mask)
    if found > 1:
        biggest = 1 + int(np.argmax(np.bincount(blobs.ravel())[1:]))
        mask = blobs == biggest
    holes, found = label(~mask)
    if found:
        sizes = np.bincount(holes.ravel())[1:]
        small = np.flatnonzero(sizes <= MASK_HOLE_SHARE * mask.sum()) + 1
        mask |= np.isin(holes, small)
    return mask


def viewer_to_scene(space: Path, room, manifest: dict):
    """The inverse of the scene's own frame: viewer metres -> camera-solve
    coordinates. scene.json records the frame; a scene built before it did has
    its floor worked out again, the way mixed_scene.build does."""
    from mixed_scene import TO_VIEWER

    known = manifest.get("solveFrame")
    if known:
        origin, metre = np.array(known["origin"], float), float(known["unitsPerMetre"])
    else:
        from pointcloud import load_ply
        from splat_tools import read_splat
        from surface_fill import floor_masks

        arr, _ = read_splat(space / "splat.ply")
        dense = room.to_scene(load_ply(space / "cloud-dense.ply").points.astype(np.float64))
        floor, _, _, _ = floor_masks(room, arr, dense, lambda *_: None)
        origin = np.array([room.centre[0], room.centre[1], float(floor.origin[2])])
        metre = room.metre
    return lambda viewer: (np.asarray(viewer, float) @ TO_VIEWER) * metre + origin


def piece_points(space: Path, piece: dict, to_scene) -> np.ndarray:
    """The scene-frame positions of a piece's own Gaussians, read back from
    scene/pieces/<id>.splat rather than cut again: what the scene calls the
    piece is what the model should be shown."""
    # The .splat layout mixed_scene.write_piece uses: 32 bytes a Gaussian.
    record = np.dtype([("position", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rotation", "u1", 4)])
    arr = np.frombuffer((space / "scene" / piece["file"]).read_bytes(), dtype=record)
    viewer = arr["position"].astype(np.float64)
    return to_scene(viewer + np.array(piece["anchor"], float))


def jobs(space: Path, only: list[str] | None = None, log=print) -> list[dict]:
    """Write a job per movable piece of the space's scene. Returns the jobs."""
    from densify import read_cameras_bin, read_images_bin
    from object_frames import best_frames
    from pointcloud import space_model_dir
    from splat_edit import Room

    scene_file = space / "scene" / "scene.json"
    if not scene_file.exists():
        raise SystemExit(f"no scene yet: run stage 4's scene step for {space.name}")
    manifest = json.loads(scene_file.read_text())
    room = Room(space)
    model = space_model_dir(space)
    cameras = read_cameras_bin(model / "cameras.bin")
    infos = {v["name"]: v for v in read_images_bin(model / "images.bin").values()}
    out_root = space / "object-models"
    out_root.mkdir(parents=True, exist_ok=True)
    boxes = room.shapes["boxes"]
    to_scene = viewer_to_scene(space, room, manifest)
    written = []
    for piece in manifest["pieces"]:
        ident = piece["id"]
        if not piece.get("movable") or not ident.startswith("B"):
            continue
        if only and ident not in only:
            continue
        index = int(ident[1:])
        box = boxes[index]
        picks = best_frames(space, {index: box}, per_box=VIEWS).get(index) or []
        if not picks:
            log(f"  {ident} {piece['label']}: no frame saw it whole; skipped")
            continue
        points = piece_points(space, piece, to_scene)
        folder = out_root / ident
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("view-*"):
            old.unlink()
        views = []
        for pick in picks:
            info = infos.get(pick["frame"])
            if info is None:
                continue
            cam = cameras[info["camera_id"]]
            mask = blob_mask(room, points, info, cam)
            share = float(mask.mean())
            if share < MIN_MASK_SHARE:
                continue
            n = len(views)
            Image.open(space / "workspace" / "images" / pick["frame"]).convert("RGB") \
                .save(folder / f"view-{n}.jpg", quality=95)
            Image.fromarray((mask * 255).astype(np.uint8)).save(folder / f"view-{n}.png")
            views.append({"frame": pick["frame"], "mask_share": round(share, 4),
                          "detected": bool(pick.get("detected")),
                          "camera": {"width": cam["width"], "height": cam["height"],
                                     "params": [float(p) for p in cam["params"]],
                                     "R": info["R"].tolist(), "t": info["t"].tolist()}})
        if not views:
            log(f"  {ident} {piece['label']}: too small in every frame that saw it; skipped")
            continue
        size = (np.array(box["max"]) - np.array(box["min"])) / room.metre
        job = {"space": space.name, "id": ident, "label": piece["label"],
               "measured_size_m": [round(float(v), 3) for v in size], "views": views}
        (folder / "job.json").write_text(json.dumps(job, indent=1) + "\n")
        log(f"  {ident} {piece['label']}: {len(views)} view(s) "
            f"({', '.join(v['frame'].replace('frame_', '').replace('.jpg', '') for v in views)}), "
            f"largest {max(v['mask_share'] for v in views):.1%} of a frame, "
            f"measured {' x '.join(f'{v:.2f}' for v in size)} m")
        written.append(job)
    (out_root / "jobs.json").write_text(json.dumps(written, indent=1) + "\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--pieces", nargs="*", help="only these piece ids (B4 B13 ...)")
    args = parser.parse_args()
    written = jobs(args.space.resolve(), args.pieces)
    print(f"{len(written)} job(s) in {args.space / 'object-models'}")


if __name__ == "__main__":
    main()
