"""A wardrobe's door panels boxed as a poster or a curtain stay the wardrobe's.

    python3 tools/tests/test_furniture_fronts.py      (or pytest tools/tests)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from semantics import pixel_labels  # noqa: E402

INDEX = {"unlabelled": 0, "wardrobe": 1, "poster": 2, "curtain": 3, "pillow": 4, "mirror": 5}


def det(label, box, score):
    return {"label": label, "box": box, "score": score}


def grid():
    vs, us = np.mgrid[0:100, 0:100]
    return us.ravel().astype(float), vs.ravel().astype(float)


def test_a_poster_box_on_the_wardrobe_door_gives_its_pixels_to_the_wardrobe():
    us, vs = grid()
    detections = [det("wardrobe", (0, 0, 60, 100), 0.30),        # the whole wardrobe, a weak box
                  det("poster", (10, 10, 50, 90), 0.45),          # its patterned door, confidently a poster
                  det("poster", (70, 20, 90, 40), 0.45),          # a real poster on the wall beside it
                  det("curtain", (40, 0, 80, 100), 0.40)]         # half on the wardrobe, half on the window
    plain = pixel_labels(us, vs, detections, INDEX)
    fronts = pixel_labels(us, vs, detections, INDEX, {"poster", "curtain"}, {"wardrobe"})
    at = lambda u, v: int(v) * 100 + int(u)   # noqa: E731
    assert plain[at(30, 50)] == INDEX["poster"]                  # the old way: the door is a poster
    assert fronts[at(30, 50)] == INDEX["wardrobe"]               # now it is the wardrobe's
    assert fronts[at(80, 30)] == INDEX["poster"]                 # the real poster stays a poster
    assert fronts[at(70, 60)] == INDEX["curtain"]                # the curtain beyond the wardrobe stays
    assert fronts[at(50, 60)] == INDEX["wardrobe"]               # the curtain's half over the wardrobe does not


def test_things_on_furniture_and_mirrors_are_not_affected():
    us, vs = grid()
    detections = [det("wardrobe", (0, 0, 60, 100), 0.30), det("pillow", (10, 10, 30, 30), 0.5),
                  det("mirror", (35, 10, 55, 90), 0.5)]
    fronts = pixel_labels(us, vs, detections, INDEX, {"poster", "curtain"}, {"wardrobe"})
    assert fronts[20 * 100 + 20] == INDEX["pillow"]
    assert fronts[50 * 100 + 45] == INDEX["mirror"]   # its depth is a reflection: the plane fill decides, not this


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
