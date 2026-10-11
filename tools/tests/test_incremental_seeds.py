"""The incremental mapper picks its first pair at random, so reconstruct.py
runs it with a few seeds (and, after a failed global solve, with the
small-baseline angles first) and keeps the best model.

    python3 tools/tests/test_incremental_seeds.py
"""

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import reconstruct  # noqa: E402


def fake_mapper(outcomes: dict, calls: list):
    """A stand-in for `colmap mapper`: writes one model whose frames and points
    depend on (small-baseline or not, seed), as outcomes says."""
    def run(command, log, check=True):
        small = "--Mapper.init_min_tri_angle" in command
        seed = int(command[command.index("--Mapper.random_seed") + 1])
        out = Path(command[command.index("--output_path") + 1])
        calls.append((small, seed))
        frames, points = outcomes.get((small, seed), (0, 0))
        if frames:
            (out / "0").mkdir()
            (out / "0" / "images.bin").write_bytes(struct.pack("<Q", frames))
            (out / "0" / "points3D.bin").write_bytes(struct.pack("<Q", points))
        return True
    return run


def solve(variants, outcomes):
    calls = []
    saved = reconstruct.run
    reconstruct.run = fake_mapper(outcomes, calls)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            images = workspace / "images"
            images.mkdir()
            for k in range(12):
                (images / f"frame_{k:05d}.jpg").write_bytes(b"")
            sparse = workspace / "sparse"
            sparse.mkdir()
            reconstruct.incremental_mapping(workspace, workspace / "database.db", images, sparse,
                                            variants, workspace / "log")
            kept = [(reconstruct.registered_images(m), reconstruct.model_points(m))
                    for m in reconstruct.solved_models(sparse)]
            leftovers = sorted(p.name for p in workspace.glob("sparse-incremental-*"))
            return kept, calls, leftovers
    finally:
        reconstruct.run = saved


def test_the_best_seed_is_kept_and_a_whole_solve_ends_the_search():
    # The white-bedroom slide with the small-baseline angles: 4, 4, then all 12.
    kept, calls, leftovers = solve([reconstruct.SMALL_BASELINE_MAPPER, []],
                                   {(True, 0): (4, 482), (True, 1): (4, 482), (True, 2): (12, 4266),
                                    (False, 0): (2, 591)})
    assert kept == [(12, 4266)]
    assert calls == [(True, 0), (True, 1), (True, 2)]          # never reached the default angles
    assert leftovers == []                                     # the losing runs are cleaned up


def test_frames_with_no_points_lose_to_fewer_frames_with_points():
    # The slow studio slide: 11 frames and no point is no solve; 9 with points is.
    kept, calls, _ = solve([[]], {(False, 0): (11, 0), (False, 1): (9, 2555), (False, 2): (7, 2307)})
    assert kept == [(9, 2555)] and len(calls) == 3
    # And the fallback goes on to the default angles when the small-baseline runs fall short.
    kept, calls, _ = solve([reconstruct.SMALL_BASELINE_MAPPER, []],
                           {(True, 0): (10, 6), (False, 1): (12, 4049)})
    assert kept == [(12, 4049)] and calls[-1] == (False, 1)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
