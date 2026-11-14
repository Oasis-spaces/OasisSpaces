"""tools/splat_noise.py: removing a share of the noisiest Gaussians keeps the rest untouched."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pipeline"))
import splat_noise  # noqa: E402
from splat_gsplat import read_ply, write_ply  # noqa: E402


def test_threshold_removes_the_asked_share():
    score = np.linspace(0, 1, 1000)
    assert np.isinf(splat_noise.threshold_for(score, 0))
    for percent in (1, 12.5, 40):
        assert abs((score >= splat_noise.threshold_for(score, percent)).mean() * 100 - percent) < 0.2


def test_apply_drops_only_the_noisiest(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    arrays = dict(means=rng.normal(size=(n, 3)), scales=rng.normal(size=(n, 3)), quats=rng.normal(size=(n, 4)),
                  opacities=rng.normal(size=n), sh0=rng.normal(size=(n, 1, 3)), shN=rng.normal(size=(n, 15, 3)))
    arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
    write_ply(tmp_path / "s.ply", **arrays)
    score = rng.uniform(size=n).astype(np.float32)
    np.savez(tmp_path / "splat-noise.npz", score=score)
    out = splat_noise.apply(tmp_path, tmp_path / "s.ply", 10, log=lambda *a: None)
    kept = read_ply(out)
    keep = score < np.quantile(score, 0.9)
    assert len(kept["means"]) == keep.sum()
    assert np.allclose(kept["shN"], arrays["shN"][keep])            # view-dependent colour kept as it was
