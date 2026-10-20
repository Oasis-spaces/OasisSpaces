"""Reading a splat drops the gaussians a trainer left with NaN or infinite values.

    python3 tools/tests/test_splat_read.py      (or pytest tools/tests)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import splat_export  # noqa: E402
from splat_tools import SPLAT_DTYPE, read_splat, write_splat  # noqa: E402


def splat_with_bad_rows(path: Path) -> None:
    arr = np.zeros(5, dtype=SPLAT_DTYPE)
    arr["x"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    arr["rot_0"] = 1.0
    arr["opacity"] = 2.0
    arr["x"][1] = np.nan                 # a position lost to NaN
    arr["scale_0"][3] = np.inf           # a scale that blew up
    write_splat(path, arr)


def test_read_splat_keeps_only_finite_gaussians():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splat.ply"
        splat_with_bad_rows(path)
        arr, _ = read_splat(path)
    assert len(arr) == 3
    assert arr["x"].tolist() == [0.0, 2.0, 4.0]


def test_the_viewer_copy_has_no_bad_gaussians_either():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splat.ply"
        splat_with_bad_rows(path)
        assert len(splat_export.read_splat_ply(path)) == 3
        count = splat_export.ply_to_splat(path, Path(tmp) / "splat.splat")
        records = np.fromfile(Path(tmp) / "splat.splat", dtype=[("position", "<f4", 3), ("scale", "<f4", 3),
                                                               ("rgba", "u1", 4), ("rotation", "u1", 4)])
    assert count == len(records) == 3
    assert np.isfinite(records["position"]).all() and np.isfinite(records["scale"]).all()


def test_a_clean_splat_reads_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splat.ply"
        arr = np.zeros(4, dtype=SPLAT_DTYPE)
        arr["x"] = [0.5, 1.5, 2.5, 3.5]
        write_splat(path, arr)
        back, _ = read_splat(path)
    assert np.array_equal(back, arr)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
