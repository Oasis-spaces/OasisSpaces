"""The frames Claude chooses from cover the whole capture, first to last.

    python3 tools/tests/test_frame_choice.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from agent import FRAME_CHOICES, spread  # noqa: E402


def test_a_48_frame_walkthrough_keeps_its_last_room():
    frames = [f"frame_{k:05d}.jpg" for k in range(1, 49)]
    chosen = spread(frames, FRAME_CHOICES)
    assert len(chosen) == FRAME_CHOICES
    assert chosen[0] == frames[0] and chosen[-1] == frames[-1]    # 37-48 were once never shown
    assert chosen == sorted(set(chosen))


def test_short_captures_and_edge_counts():
    assert spread(list(range(10)), FRAME_CHOICES) == list(range(10))
    assert spread(list(range(9)), 1) == [4]
    assert spread([], 5) == [] and spread([1, 2], 0) == []
    assert spread(list(range(100)), 3) == [0, 50, 99]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
