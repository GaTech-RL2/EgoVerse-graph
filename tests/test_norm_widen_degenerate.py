"""norm_stats.widen_degenerate_quantiles: a near-constant element gets min/max as its quantile range (2026-10-08)."""
import numpy as np

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


def test_widen_only_degenerate_elements():
    rng = np.random.default_rng(0)
    X = np.zeros((10000, 3))
    X[:, 0] = rng.normal(size=10000)            # ordinary element: untouched
    X[:50, 1] = 0.87                            # start-delay-like: 0 in 99.5 %, rare 0.87
    stats = MultiDataset._compute_stats_for_array(X)  # column 2 is constant: nothing to widen
    assert stats["quantile_99"][1] == 0 and stats["quantile_1"][1] == 0
    out, n = MultiDataset._widen_degenerate_quantiles(stats)
    assert n == 1
    assert np.isclose(out["quantile_99"][1], 0.87) and out["quantile_1"][1] == 0
    assert out["quantile_99"][0] == stats["quantile_99"][0] and out["quantile_1"][2] == stats["quantile_1"][2]
    q1, q99 = out["quantile_1"][1], out["quantile_99"][1]
    assert abs(2 * (0.87 - q1) / (q99 - q1 + 1e-6) - 1) <= 1.0 + 1e-6   # was ~1.7e6 before widening


def test_off_by_default_is_a_no_op():
    stats = MultiDataset._compute_stats_for_array(np.ones((100, 2)))
    out, n = MultiDataset._widen_degenerate_quantiles(stats)
    assert n == 0 and out is stats
