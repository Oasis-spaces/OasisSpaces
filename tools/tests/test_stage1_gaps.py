"""Stage 1 filling COLMAP's gaps with MapAnything: when to keep the joined solve.

    python3 tools/tests/test_stage1_gaps.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import agent  # noqa: E402

COLMAP = {"frames": 167, "total": 186, "models": 2, "reprojection": 0.87, "path_jump": 0.05}


def joined(**changes):
    return {"frames": 183, "total": 186, "models": 1, "reprojection": 0.78, "path_jump": 0.09, **changes}


def test_the_walkthrough_joined_is_kept():
    keep, why = agent.keep_filled(COLMAP, joined())
    assert keep and "183 of 186" in why


def test_no_more_frames_is_not_worth_it():
    assert not agent.keep_filled(COLMAP, joined(frames=167))[0]
    assert not agent.keep_filled(COLMAP, None)[0]
    assert not agent.keep_filled(COLMAP, joined(frames=0))[0]


def test_a_looser_solve_is_not_kept():
    assert agent.keep_filled(COLMAP, joined(reprojection=0.87 + agent.GAP_REPROJ_SLACK_PX - 0.01))[0]
    assert not agent.keep_filled(COLMAP, joined(reprojection=0.87 + agent.GAP_REPROJ_SLACK_PX + 0.01))[0]


def test_a_camera_leaping_across_the_room_is_not_kept():
    assert not agent.keep_filled(COLMAP, joined(path_jump=agent.MAX_PATH_JUMP + 0.01))[0]


def test_still_in_pieces_is_kept_only_when_nearly_every_frame_is_in():
    assert not agent.keep_filled(COLMAP, joined(models=2, frames=170))[0]
    assert agent.keep_filled(COLMAP, joined(models=2, frames=180))[0]           # 97% of frames


def test_without_a_gpu_the_gap_is_recorded_not_filled():
    said = []
    stub = agent.Agent.__new__(agent.Agent)
    stub.cuda = False
    stub.reconstruction_metrics = lambda: dict(COLMAP)
    stub.decide = lambda stage, action, why, *rest, **kw: said.append((stage, action, why))
    agent.Agent.fill_gaps(stub)
    assert said and said[0][:2] == ("reconstruct", "skip") and "19 of 186" in said[0][2]


def test_a_complete_solve_is_left_alone():
    said = []
    stub = agent.Agent.__new__(agent.Agent)
    stub.cuda = True
    stub.reconstruction_metrics = lambda: {**COLMAP, "frames": 185, "models": 1}
    stub.decide = lambda *a, **k: said.append(a)
    agent.Agent.fill_gaps(stub)
    assert said == []


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
