"""surface_fill.inpaint: LaMa works at most 512 px across, and only the unknown pixels change.

    python3 tools/tests/test_surface_fill.py      (or pytest tools/tests)
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import surface_fill  # noqa: E402


class FakeLama(torch.nn.Module):
    """Records what it was given and paints every pixel one colour."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, image, mask):
        self.seen.append((image.clone(), mask.clone()))
        return torch.full_like(image, 0.25)


def fill(photo, unknown):
    fake = FakeLama()
    loads, path = torch.jit.load, surface_fill.LAMA_PATH
    torch.jit.load = lambda *a, **k: fake
    surface_fill.LAMA_PATH = Path(__file__)          # any file that exists
    try:
        out = surface_fill.inpaint(photo, unknown, log=lambda *_: None)
    finally:
        torch.jit.load, surface_fill.LAMA_PATH = loads, path
    return out, fake.seen[0]


def test_a_big_photo_is_filled_at_512_and_only_its_unknown_pixels_change():
    rng = np.random.default_rng(0)
    photo = rng.uniform(0, 255, (1000, 760, 3))
    unknown = np.zeros((1000, 760), bool)
    unknown[300:700, 200:600] = True
    photo[unknown] = 128                            # as the callers pass it
    out, (image, mask) = fill(photo, unknown)
    assert max(image.shape[2:]) <= surface_fill.LAMA_WORK_PX + 7
    assert image.shape[2] % 8 == 0 and image.shape[3] % 8 == 0
    assert np.array_equal(out[~unknown], photo[~unknown])
    assert np.allclose(out[unknown], 0.25 * 255, atol=1)


def test_the_grey_of_the_unknown_does_not_bleed_into_known_pixels_when_shrunk():
    photo = np.full((1024, 1024, 3), 200.0)
    unknown = np.zeros((1024, 1024), bool)
    unknown[400:600, 333:701] = True               # edges that fall inside a shrunk pixel
    photo[unknown] = 128
    _, (image, mask) = fill(photo, unknown)
    known = mask[0, 0].numpy() == 0
    values = image[0].permute(1, 2, 0).numpy()[known]
    assert np.allclose(values, 200 / 255, atol=1e-6)
    # Every small pixel that held any unknown is unknown.
    assert mask[0, 0, 200:300, 166:351].min() == 1


def test_a_small_photo_is_filled_at_its_own_size():
    photo = np.full((300, 410, 3), 90.0)
    unknown = np.zeros((300, 410), bool)
    unknown[100:150, 100:200] = True
    out, (image, _) = fill(photo, unknown)
    assert image.shape[2:] == (304, 416)
    assert np.array_equal(out[~unknown], photo[~unknown])


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
