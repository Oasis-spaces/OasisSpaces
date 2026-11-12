#!/usr/bin/env python3
"""Find the gaps in a room's surfaces and fill them from the pattern around them.

After stage 3 the room has its walls, floor and ceiling as planes, but the cloud
on them has holes: floor the phone never pointed at, wall behind the wardrobe,
ceiling filmed in scraps. A person opening the room sees those holes, and taking
a piece out shows nothing behind it. This step:

  1. maps each surface (the floor and ceiling inside the room's outline, every
     built wall from floor to ceiling) on a 2 cm grid from the dense cloud, and
     marks what no frame saw;
  2. sorts each gap by what explains it: hidden behind or under a piece, an
     opening (a door in that wall), or simply never filmed;
  3. fills it from its surroundings. A surface with a repeating pattern (floor
     tiles, panels) has its period measured from what was seen, and a gap
     takes the colour one or more periods away; a plain one is inpainted;
  4. shows Claude each surface, as seen and as filled, with the frames that
     face it, and Claude decides per gap: fill it (and say what the pattern
     is), or leave it open (a doorway, an opening, a mirror). Claude's
     answers go into the room record.

Filled points are written apart from the measured ones, as cloud-gaps.ply, and
every one is inferred: tools/space_viewer_export.py marks them so the viewer can
show them as filled in. gaps.json records each surface's coverage, its pattern,
each gap and Claude's decision.

    python3 pipeline/gaps.py spaces/<name> [--no-claude] [--cell 0.02]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pointcloud import PointCloud, load_ply, save_ply  # noqa: E402

CELL_M = 0.02                 # grid cell on each surface, metres
NEAR_SURFACE_M = 0.05         # points this close to a surface belong to it
CLOSE_CELLS = 3               # seen cells closed over this many cells: sparse sampling is not a gap
MIN_GAP_M2 = 0.02             # smaller holes are filled without being reported
IN_FRONT_M = 0.9              # a piece this close in front of a wall hides it
PERIOD_RANGE_M = (0.15, 1.3)  # tile and panel sizes looked for
PERIOD_MIN_CORR = 0.3         # autocorrelation peak that counts as a repeating pattern
MAX_SHOWN = 8                 # gaps per surface Claude decides on, largest first
SHADING_CELLS = 15            # blur, in cells, that separates a surface's shading from its texture
TEXTURE_STRENGTH = 0.5         # how much of the copied fine texture is laid on the shading
PATCH_CELLS = 12              # texture patches laid into large gaps (24 cm at 2 cm cells)
SHOW_FRAMES = 3               # frames shown to Claude per surface
GAP_KINDS = {"hidden": "behind or under a piece", "opening": "an opening (a door in this wall)",
             "unfilmed": "never filmed"}


# ---------------------------------------------------------------- surfaces
def surfaces(shapes: dict, metre: float) -> list[dict]:
    """Each surface as a 2D frame in scene units: origin, two in-plane axes (u
    along, v up or across), normal, its size, and a mask of the cells that lie
    inside the room (the outline for floor and ceiling)."""
    level = shapes["room_level"]
    floor_z, height = level["floor_z"], level["height"]
    out = []
    polygons = (shapes.get("outline") or {}).get("polygons")
    if not polygons:
        room = shapes["room"]
        u, v, c = np.array(room["axis_u"]), np.array(room["axis_v"]), np.array(room["center"])
        polygons = [[(c + su * room["half_u"] * u + sv * room["half_v"] * v).tolist()
                     for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))]]
    pts = np.array([p for poly in polygons for p in poly], float)
    lo, hi = pts.min(0), pts.max(0)
    for name, z, up in (("floor", floor_z, 1.0), ("ceiling", floor_z + height, -1.0)):
        out.append({"name": name, "kind": name, "origin": np.array([lo[0], lo[1], z]),
                    "u": np.array([1.0, 0, 0]), "v": np.array([0, 1.0, 0]), "normal": np.array([0, 0, up]),
                    "size": (hi[0] - lo[0], hi[1] - lo[1]), "polygons": polygons})
    for i, p in enumerate(shapes.get("planes", [])):
        if p.get("label") != "wall" or p.get("build") is False:
            continue
        n = np.array(p["normal"], float)
        if abs(n[2]) > 0.2:
            continue
        a = np.array(p["axis_a"], float)
        a[2] = 0
        a /= np.linalg.norm(a)
        c = np.array(p["center"], float)
        origin = np.array([*(c[:2] - p["half_a"] * a[:2]), floor_z])
        # Faces into the room: towards the outline's middle.
        middle = np.array([*(pts.mean(0)), floor_z])
        n = np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2])
        if np.dot(middle - origin, n) < 0:
            n = -n
        out.append({"name": f"wall W{i}", "kind": "wall", "index": i, "origin": origin, "u": a,
                    "v": np.array([0, 0, 1.0]), "normal": n, "size": (2 * p["half_a"], height),
                    "inferred": p.get("source") == "inferred"})
    return out


def inside_polygons(polygons, x, y) -> np.ndarray:
    """Even-odd rule over several polygons, for arrays of points."""
    inside = np.zeros(len(x), bool)
    for poly in polygons:
        poly = np.asarray(poly, float)
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi)
            inside ^= crosses
            j = i
    return inside


def grid(surface: dict, scene: np.ndarray, colours: np.ndarray, cell: float, near: float, labels=None):
    """The surface's colour image (rows: v, cols: u), the cells seen, the cells
    that are part of the surface, and the in-plane coordinates of each cell."""
    w, h = surface["size"]
    cols, rows = max(2, int(np.ceil(w / cell))), max(2, int(np.ceil(h / cell)))
    rel = scene - surface["origin"]
    d = rel @ surface["normal"]
    su, sv = rel @ surface["u"], rel @ surface["v"]
    on = (np.abs(d) < near) & (su >= 0) & (su < w) & (sv >= 0) & (sv < h)
    c, r = (su[on] / cell).astype(int), (sv[on] / cell).astype(int)
    flat = r * cols + c
    count = np.bincount(flat, minlength=rows * cols).reshape(rows, cols)
    image = np.zeros((rows, cols, 3))
    for k in range(3):
        image[..., k] = np.bincount(flat, weights=colours[on, k], minlength=rows * cols).reshape(rows, cols)
    seen = count > 0
    image[seen] /= count[seen][:, None]
    named = np.zeros((rows, cols))
    if labels is not None:
        named = np.bincount(flat, weights=(labels[on] > 0).astype(float), minlength=rows * cols).reshape(rows, cols)
    surface["named_share"] = np.where(seen, named / np.maximum(count, 1), 0.0)
    uu, vv = np.meshgrid((np.arange(cols) + 0.5) * cell, (np.arange(rows) + 0.5) * cell)
    if "polygons" in surface:
        world = surface["origin"][:2] + uu[..., None] * surface["u"][:2] + vv[..., None] * surface["v"][:2]
        part = inside_polygons(surface["polygons"], world[..., 0].ravel(), world[..., 1].ravel()).reshape(rows, cols)
    else:
        part = np.ones((rows, cols), bool)
    return image.astype(np.float32), seen, part, (uu, vv)


# ---------------------------------------------------------------- gaps
def find_gaps(seen: np.ndarray, part: np.ndarray, cell: float):
    """Connected regions of the surface that nothing saw, after closing over
    the spacing of the cloud's own samples."""
    import cv2

    kernel = np.ones((2 * CLOSE_CELLS + 1, 2 * CLOSE_CELLS + 1), np.uint8)
    closed = cv2.morphologyEx(seen.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    holes = part & ~closed
    count, labels = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
    regions = []
    for k in range(1, count):
        mask = labels == k
        area = mask.sum() * cell * cell
        regions.append({"mask": mask, "area": area})
    return holes, closed, sorted(regions, key=lambda r: -r["area"])


def cover(surface: dict, boxes: list[dict], uv, metre: float, cell: float, shape) -> dict[str, np.ndarray]:
    """For each piece, the surface cells it stands on or in front of."""
    uu, vv = uv
    covered = {}
    for b in boxes:
        lo, hi = np.array(b["min"], float), np.array(b["max"], float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        rel = corners - surface["origin"]
        d = rel @ surface["normal"]
        if surface["kind"] == "wall" and (d.min() > IN_FRONT_M * metre or d.max() < -0.3 * metre):
            continue
        if surface["kind"] == "ceiling" and d.min() > 0.4 * metre:
            continue
        if surface["kind"] == "floor" and d.min() > 0.4 * metre:
            continue                       # a piece up on the wall does not hide the floor
        pu, pv = rel @ surface["u"], rel @ surface["v"]
        mask = (uu >= pu.min()) & (uu <= pu.max()) & (vv >= pv.min()) & (vv <= pv.max())
        if mask.any():
            covered[b["name"]] = mask
    return covered


# ---------------------------------------------------------------- pattern
def period(lum: np.ndarray, seen: np.ndarray, cell: float) -> tuple[int | None, int | None, float]:
    """The repeat along columns and along rows (in cells) of a pattern, from the
    masked autocorrelation of what was seen; None along an axis without one."""
    m = seen.astype(np.float64)
    if m.sum() < 400:
        return None, None, 0.0
    x = np.where(seen, lum - lum[seen].mean(), 0.0)
    shape = [2 * s for s in x.shape]
    fx, fm = np.fft.rfft2(x, shape), np.fft.rfft2(m, shape)
    num = np.fft.irfft2(fx * np.conj(fx), shape)
    den = np.fft.irfft2(fm * np.conj(fm), shape)
    var = num[0, 0] / max(den[0, 0], 1)
    corr = num / np.maximum(den, 1) / max(var, 1e-9)
    corr[den < 0.2 * den[0, 0]] = 0
    lo, hi = int(PERIOD_RANGE_M[0] / cell), int(PERIOD_RANGE_M[1] / cell)
    found, strength = [], []
    for axis in (1, 0):                     # columns (u), then rows (v)
        line = corr[0, lo:hi] if axis == 1 else corr[lo:hi, 0]
        if not len(line):
            found.append(None)
            continue
        k = int(np.argmax(line))
        if line[k] >= PERIOD_MIN_CORR and 0 < k < len(line) - 1 and line[k] >= line[k - 1] and line[k] >= line[k + 1]:
            found.append(lo + k)
            strength.append(float(line[k]))
        else:
            found.append(None)
    return found[0], found[1], (min(strength) if strength else 0.0)


def bare(image: np.ndarray, seen: np.ndarray, named_share: np.ndarray) -> np.ndarray:
    """The cells that show the surface itself: mostly unnamed points (no piece,
    pinboard or clothes there), in a colour near the surface's usual one."""
    plain = seen & (named_share < 0.3)
    if plain.sum() < 50:
        return plain
    median = np.median(image[plain], axis=0)
    dist = np.linalg.norm(image - median, axis=-1)
    spread = np.median(dist[plain]) + 1e-6
    return plain & (dist < 3.0 * spread + 12)


def seen_patches(ok: np.ndarray) -> list[tuple[int, int]]:
    """Top-left corners of PATCH_CELLS squares wholly bare surface, for texture."""
    import cv2

    whole = cv2.erode(ok.astype(np.uint8), np.ones((PATCH_CELLS, PATCH_CELLS), np.uint8),
                      anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    rr, cc = np.nonzero(whole[:ok.shape[0] - PATCH_CELLS + 1, :ok.shape[1] - PATCH_CELLS + 1])
    step = max(1, len(rr) // 4000)
    return list(zip(rr[::step], cc[::step]))


def smooth_over(image: np.ndarray, ok: np.ndarray, sigma: float) -> np.ndarray:
    """The surface's slow shading: a blur of the bare cells, normalised so the
    gaps take the shading around them."""
    import cv2

    w = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sigma)
    out = np.stack([cv2.GaussianBlur(np.where(ok, image[..., k], 0).astype(np.float32), (0, 0), sigma)
                    for k in range(3)], -1)
    base = out / np.maximum(w, 1e-4)[..., None]
    if (w < 1e-3).any():                      # far beyond any bare cell: the surface's usual colour
        base[w < 1e-3] = np.median(image[ok], axis=0) if ok.any() else 200
    return base


def fill(image: np.ndarray, seen: np.ndarray, todo: np.ndarray, pu, pv, ok=None) -> tuple[np.ndarray, int]:
    """Colours for the cells in `todo`. Where the surface repeats (tiles), each
    takes the bare cell one or more periods away. Elsewhere: the surface's
    slow shading carried in from around the gap, with fine texture laid on it
    from patches of bare surface (shading and texture apart, so patches do not
    show as blocks)."""
    out = image.copy()
    ok = seen if ok is None else ok
    rows, cols = seen.shape
    copied = 0
    if pu or pv:
        rr, cc = np.nonzero(todo)
        steps = sorted({(a, b) for a in range(-6, 7) for b in range(-6, 7) if (a or b)
                        and (pv or a == 0) and (pu or b == 0)}, key=lambda ab: abs(ab[0]) + abs(ab[1]))
        left = np.ones(len(rr), bool)
        for a, b in steps:
            r2, c2 = rr + a * (pv or 0), cc + b * (pu or 0)
            hit = left & (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < cols)
            hit[hit] &= ok[r2[hit], c2[hit]]
            out[rr[hit], cc[hit]] = image[r2[hit], c2[hit]]
            left &= ~hit
            if not left.any():
                break
        copied = int((~left).sum())
        todo = todo.copy()
        todo[rr[~left], cc[~left]] = False
    if todo.any():
        base = smooth_over(image, ok, SHADING_CELLS)
        texture = np.zeros_like(image)
        patches = seen_patches(ok)
        if patches:
            detail = image - smooth_over(image, ok, PATCH_CELLS / 2)
            rng = np.random.default_rng(0)
            rr, cc = np.nonzero(todo)
            for r0 in range(rr.min() - rr.min() % PATCH_CELLS, rr.max() + 1, PATCH_CELLS):
                for c0 in range(cc.min() - cc.min() % PATCH_CELLS, cc.max() + 1, PATCH_CELLS):
                    block = texture[r0:r0 + PATCH_CELLS, c0:c0 + PATCH_CELLS]
                    pr, pc = patches[rng.integers(len(patches))]
                    block[...] = detail[pr:pr + block.shape[0], pc:pc + block.shape[1]]
        out[todo] = base[todo] + TEXTURE_STRENGTH * texture[todo]
    return out, copied


# ---------------------------------------------------------------- Claude
def picture(surface: dict, image, seen, part, gaps, filled, path: Path, cell: float):
    """Two panels: the surface as seen with each gap outlined and numbered
    (red never filmed, yellow behind a piece, blue an opening), and as filled."""
    import cv2

    def panel(img):
        p = np.clip(img, 0, 255).astype(np.uint8)[::-1].copy()   # rows run up the surface: flip to look at it
        p[~part[::-1]] = (40, 40, 40)
        return cv2.cvtColor(p, cv2.COLOR_RGB2BGR)

    left = panel(np.where(seen[..., None], image, 0))
    right = panel(filled)
    colour = {"unfilmed": (60, 60, 255), "hidden": (0, 220, 255), "opening": (255, 140, 40)}
    for g in gaps:
        m = g["mask"][::-1].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, colour[g["kind"]], 2)
        ys, xs = np.nonzero(m)
        cv2.putText(left, str(g["id"]), (int(xs.mean()), int(ys.mean())), 0, 0.8, (255, 255, 255), 3)
        cv2.putText(left, str(g["id"]), (int(xs.mean()), int(ys.mean())), 0, 0.8, colour[g["kind"]], 2)
    gap = np.full((left.shape[0], 12, 3), 255, np.uint8)
    both = np.hstack([left, gap, right])
    scale = min(1400 / both.shape[1], 900 / both.shape[0])      # large enough for Claude to read
    both = cv2.resize(both, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    cv2.imwrite(str(path), both)
    return path


def facing_frames(space: Path, surface: dict, world: np.ndarray, metre: float, count: int) -> list[Path]:
    """The chosen frames that look most squarely at the surface's middle."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from densify import read_images_bin

    model = space / "workspace" / "sparse" / "0"
    chosen_path = space / "frames-chosen.json"
    chosen = set()
    if chosen_path.exists():
        data = json.loads(chosen_path.read_text())
        chosen = set(data.get("room", []) + data.get("objects", []))
    images = read_images_bin(model / "images.bin")
    w, h = surface["size"]
    middle = surface["origin"] + surface["u"] * w / 2 + surface["v"] * h / 2
    scored = []
    for info in images.values():
        if chosen and info["name"] not in chosen:
            continue
        centre = (-info["R"].T @ info["t"]) @ world.T
        look = (info["R"].T @ np.array([0, 0, 1.0])) @ world.T
        to = middle - centre
        dist = np.linalg.norm(to)
        if dist < 1e-6:
            continue
        facing = np.dot(look, to / dist) * max(0.0, -np.dot(surface["normal"], to / dist))
        scored.append((facing, info["name"]))
    return [space / "workspace" / "images" / n for _, n in sorted(scored, reverse=True)[:count]]


PROMPT = """You are finishing the room model of a real room filmed with a phone. This is the {name} ({kind}, {w:.2f} m x {h:.2f} m).

The first image has two panels. Left: the {kind} as the video measured it, looking straight at it (for the floor and ceiling, from above), black where no frame saw it, dark grey outside the room. Each gap is outlined and numbered: red never filmed, yellow behind or under a piece of furniture, blue an opening (a door detected in this wall). Right: the same {kind} with every gap filled from what surrounds it{pattern_note}. The other images are frames of the video that face this {kind}; use them to see what it really looks like.

The gaps:
{gaps}

For each gap decide whether the filled-in version is what is really there, as far as you can tell:
- "fill": the surface simply continues there (floor under a bed, wall behind a wardrobe, paint or tiles the phone missed). Filled points are always shown as inferred, never as measured.
- "open": it is not this surface there (a doorway or window, a mirror, a niche, a room beyond), so leave it empty.
Also say, in a few words, what the {kind} looks like (its pattern or finish: "beige 60 cm square tiles", "plain white paint", "textured white plaster"), and whether the fill on the right matches it.

Reply as JSON: {{"pattern": "...", "fill_matches": true or false, "gaps": [{{"id": 1, "decision": "fill" or "open", "what": "..."}}], "record": {{"doubts": [...]}}}}"""


def ask_claude(space: Path, surface: dict, gaps: list[dict], image, seen, part, filled, world, metre, cell,
               pattern: str | None, advisor) -> dict | None:
    work = space / "gaps"
    work.mkdir(exist_ok=True)
    pic = picture(surface, image, seen, part, gaps, filled, work / f"{surface['name'].replace(' ', '-')}.jpg", cell)
    frames = facing_frames(space, surface, world, metre, SHOW_FRAMES)
    listing = "\n".join(f"{g['id']}: {g['area'] / metre ** 2:.2f} m², {GAP_KINDS[g['kind']]}"
                        + (f" ({', '.join(g['behind'])})" if g.get("behind") else "") for g in gaps)
    w, h = (s / metre for s in surface["size"])
    note = f"; it repeats every {pattern}, and the fill copies that repeat" if pattern else ""
    return advisor.ask_json(PROMPT.format(name=surface["name"], kind=surface["kind"], w=w, h=h, gaps=listing,
                                          pattern_note=note), [pic, *frames], max_tokens=1500)


# ---------------------------------------------------------------- main
def run(space: Path, use_claude: bool = True, cell_m: float = CELL_M, log=print) -> dict:
    shapes = json.loads((space / "shapes.json").read_text())
    metre = json.loads((space / "densify.json").read_text())["colmap_units_per_metre"]
    world = np.array(shapes["world"], float)
    cloud = load_ply(space / "cloud-dense.ply")
    scene = cloud.points.astype(np.float64) @ world.T
    colours = cloud.colors.astype(np.float64)
    cell, near = cell_m * metre, NEAR_SURFACE_M * metre
    pieces = [dict(b, name=b.get("identity") or b.get("detected") or b.get("label") or f"B{i}")
              for i, b in enumerate(shapes.get("boxes", [])) if b.get("build", True)]
    advisor = None
    if use_claude:
        from advisor import BRIEF_PATH, Advisor
        advisor = Advisor(system=BRIEF_PATH.read_text() if BRIEF_PATH.exists() else None)
        if not advisor.available:
            log(f"Claude is not available ({advisor.reason}); gaps are filled without its check")
            advisor = None
    record = {"cell_m": cell_m, "surfaces": []}
    new_points, new_colours, new_normals = [], [], []
    for surface in surfaces(shapes, metre):
        image, seen, part, uv = grid(surface, scene, colours, cell, near, cloud.labels)
        ok = bare(image, seen, surface["named_share"])
        holes, closed, _ = find_gaps(seen, part, cell)
        covered = cover(surface, pieces, uv, metre, cell, seen.shape)
        # Each unseen cell takes its cause: an opening in the wall, the piece in
        # front of it (or standing on it), or none (never filmed). Gaps are the
        # connected regions of one cause.
        cause = np.full(seen.shape, "", object)
        for name, m in covered.items():
            cause[holes & m & (cause == "")] = name
        cause[holes & (cause == "")] = "-"
        gaps = []
        import cv2
        for name in sorted({c for c in cause.ravel() if c}):
            count, labels = cv2.connectedComponents((cause == name).astype(np.uint8), connectivity=8)
            for k in range(1, count):
                mask = labels == k
                g = {"mask": mask, "area": mask.sum() * cell * cell}
                if name == "-":
                    g["kind"] = "unfilmed"
                elif name in ("door", "window"):
                    g["kind"], g["behind"] = "opening", [name]
                else:
                    g["kind"], g["behind"] = "hidden", [name]
                g["small"] = g["area"] < MIN_GAP_M2 * metre ** 2
                gaps.append(g)
        gaps.sort(key=lambda g: -g["area"])
        lum = image @ np.array([0.299, 0.587, 0.114], np.float32)
        pu, pv, strength = period(lum, ok, cell_m)
        pattern = None
        if pu or pv:
            pattern = " by ".join(f"{p * cell_m * 100:.0f} cm" for p in (pu, pv) if p)
        todo = holes.copy()
        # Openings stay open unless Claude says otherwise; small holes are filled.
        for g in gaps:
            if g["kind"] == "opening":
                todo &= ~g["mask"]
        filled, copied = fill(image, seen | (closed & ~holes), todo | (closed & ~seen & part), pu, pv, ok)
        shown = [g for g in gaps if not g.get("small")][:MAX_SHOWN]     # the rest are filled as they are
        for k, g in enumerate(shown, 1):
            g["id"] = k
        verdict = None
        if advisor is None and shown:
            (space / "gaps").mkdir(exist_ok=True)
            picture(surface, image, seen, part, shown, filled,
                    space / "gaps" / f"{surface['name'].replace(' ', '-')}.jpg", cell)
        if advisor is not None and shown:
            verdict = ask_claude(space, surface, shown, image, seen, part, filled, world, metre, cell,
                                 pattern, advisor)
        decisions = {}
        if verdict:
            for d in verdict.get("gaps") or []:
                try:
                    decisions[int(d.get("id"))] = d
                except (TypeError, ValueError):
                    continue
        keep = todo | (closed & ~seen & part)
        for g in shown:
            d = decisions.get(g["id"])
            decision = (d or {}).get("decision") or ("open" if g["kind"] == "opening" else "fill")
            g["decision"], g["what"] = decision, (d or {}).get("what")
            if decision == "open":
                keep &= ~g["mask"]
            elif g["kind"] == "opening":
                keep |= g["mask"]
                filled, _ = fill(filled, seen | keep & ~g["mask"], g["mask"], pu, pv, ok)
        if verdict and verdict.get("fill_matches") is False and (pu or pv):
            # Claude saw the copied pattern does not match: fill plainly instead.
            filled, _ = fill(image, seen | (closed & ~holes), keep, None, None, ok)
            pattern_used = "plain (Claude: the repeat did not match)"
        else:
            pattern_used = pattern or "plain"
        rr, cc = np.nonzero(keep)
        uu, vv = uv
        pos = surface["origin"] + uu[rr, cc, None] * surface["u"] + vv[rr, cc, None] * surface["v"]
        new_points.append(pos)
        new_colours.append(np.clip(filled[rr, cc], 0, 255))
        new_normals.append(np.repeat(surface["normal"][None], len(pos), 0))
        area = part.sum() * cell_m ** 2
        entry = {"surface": surface["name"], "kind": surface["kind"], "area_m2": round(area, 2),
                 "seen_share": round(float((seen & part).sum() / max(part.sum(), 1)), 3),
                 "filled_m2": round(len(rr) * cell_m ** 2, 2), "pattern_measured": pattern,
                 "pattern_used": pattern_used, "pattern_claude": (verdict or {}).get("pattern"),
                 "fill_matches": (verdict or {}).get("fill_matches"),
                 "gaps": [{"id": g["id"], "area_m2": round(g["area"] / metre ** 2, 2), "kind": g["kind"],
                           "behind": g.get("behind"), "decision": g.get("decision"), "what": g.get("what")}
                          for g in shown]}
        record["surfaces"].append(entry)
        if verdict and (verdict.get("record") or {}).get("doubts"):
            entry["doubts"] = verdict["record"]["doubts"]
        log(f"  {surface['name']}: {entry['seen_share']:.0%} seen, {entry['filled_m2']:.2f} m² filled"
            f" ({pattern_used}); gaps: " + (", ".join(f"{g['id']} {g['kind']} {g['area_m2']} m² -> {g['decision']}"
                                                   for g in entry["gaps"]) or "none"))
    points = np.vstack(new_points) if new_points else np.zeros((0, 3))
    inferred = PointCloud((points @ world).astype(np.float32),        # back to the solve's frame
                          np.vstack(new_colours).astype(np.uint8) if new_colours else np.zeros((0, 3), np.uint8),
                          (np.vstack(new_normals) @ world).astype(np.float32) if new_normals else None)
    save_ply(inferred, space / "cloud-gaps.ply")
    record["points"] = int(len(points))
    record["claude"] = advisor is not None
    (space / "gaps.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"Filled {len(points):,} inferred points -> {space / 'cloud-gaps.ply'}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", type=Path)
    parser.add_argument("--no-claude", action="store_true")
    parser.add_argument("--cell", type=float, default=CELL_M, help="grid cell in metres")
    args = parser.parse_args()
    run(args.space, use_claude=not args.no_claude, cell_m=args.cell)


if __name__ == "__main__":
    main()
