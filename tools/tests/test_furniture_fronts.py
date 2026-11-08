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


def test_claude_marks_on_the_grid_become_detections_in_frame_pixels():
    import tempfile
    from PIL import Image
    from semantics import ClaudeDetector, Vocabulary

    vocabulary = Vocabulary([{"name": "wardrobe", "role": "storage", "build_as": "wardrobe"},
                             {"name": "poster", "role": "fixture"}], "test")

    class Advisor:
        available, reason = True, None

        def __init__(self):
            self.asked = []

        def ask_json(self, prompt, images, max_tokens):
            self.asked.append((prompt, list(images)))
            if len(self.asked) > 1:
                return {"frames": []}
            return {"frames": [{"image": 1, "objects": [{"name": "wardrobe", "box": [10, 0, 60, 100]},
                                                         {"name": "sofa", "box": [0, 0, 5, 5]},          # not in the list
                                                         {"name": "poster", "box": [70, 20, 90, "x"]}]},  # unreadable
                               {"image": 2, "objects": [{"name": "Poster", "box": [-5, 10, 30, 40]}]}]}
    with tempfile.TemporaryDirectory() as tmp:
        frames = []
        for n in range(3):
            path = Path(tmp) / f"frame_{n}.jpg"
            Image.new("RGB", (2000, 1000), "white").save(path)
            frames.append(path)
        advisor = Advisor()
        found = ClaudeDetector(vocabulary, advisor=advisor, per_call=2, work_dir=Path(tmp) / "marks") \
            .detect_many(frames, log=lambda *_: None)
        assert len(advisor.asked) == 2 and len(advisor.asked[0][1]) == 2      # two frames a question
        assert found["frame_0.jpg"] == [{"label": "wardrobe", "score": 0.9, "box": [200.0, 0.0, 1200.0, 1000.0]}]
        assert found["frame_1.jpg"][0]["box"] == [0.0, 100.0, 600.0, 400.0]     # clamped to the frame
        assert found["frame_2.jpg"] == []                                       # asked, nothing marked
        assert "wardrobe" in advisor.asked[0][0] and "grid" in advisor.asked[0][0]


def test_the_outline_that_fills_the_box_is_the_object_not_its_frame_or_the_wall():
    from semantics import pick_outline
    frame = np.zeros((100, 100), bool)
    frame[10:90, 10:20] = frame[10:90, 50:60] = True                 # the wardrobe's dark frame only
    whole = np.zeros((100, 100), bool)
    whole[10:90, 10:60] = True                                        # frame and patterned doors
    wall = np.ones((100, 100), bool)                                  # the wall behind it too
    box = (10, 10, 59, 89)
    assert pick_outline(np.stack([frame, whole, wall]), box) == 1
    assert pick_outline(np.stack([frame, whole, wall]), box, iou=[0.95, 0.80, 0.70]) == 1  # SAM's favourite was the frame


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
