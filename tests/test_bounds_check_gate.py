"""`bounds_check` gates the per-sample quantile screen in MultiDataset.

The flag was already present in nine data configs but read by nothing: it fell
into **kwargs and was discarded, so the screen ran unconditionally on every
sample, train and validation alike. A violation does not raise -- it swaps in a
different sample via get_fallback_idx -- so a run could quietly train on a
different distribution than its config described.
"""

import numpy as np
import torch

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

_EMB = 7
_KEY = "actions_cartesian"


class _Leaf(torch.utils.data.Dataset):
    """A two-sample leaf whose second sample is out of bounds."""

    def __init__(self, bad_value: float = 1e6):
        self.episode_path = "/tmp/ep_0000.zarr"
        self.bad_value = bad_value
        self.served = []

    def __len__(self):
        return 2

    def __getitem__(self, index):
        self.served.append(index)
        value = 0.0 if index == 0 else self.bad_value
        return {
            "embodiment": _EMB,
            _KEY: np.full((4, 2), value, dtype=np.float32),
        }


def _dataset(*, bounds_check: bool, bad_value: float = 1e6) -> tuple:
    leaf = _Leaf(bad_value=bad_value)
    ds = MultiDataset(
        datasets={"src": leaf},
        mode="total",
        norm_mode="quantile",
        bounds_check=bounds_check,
    )
    # Stats that put the bad sample far outside the quantile band.
    ds.norm_stats = {
        _EMB: {
            _KEY: {
                "quantile_1": np.zeros((4, 2), dtype=np.float32),
                "quantile_99": np.ones((4, 2), dtype=np.float32),
                "mean": np.zeros((4, 2), dtype=np.float32),
                "std": np.ones((4, 2), dtype=np.float32),
                "min": np.zeros((4, 2), dtype=np.float32),
                "max": np.ones((4, 2), dtype=np.float32),
            }
        }
    }
    ds.zarr_keys = {_EMB: {_KEY: _KEY}}
    ds.key_types = {_EMB: {_KEY: "action_keys"}}
    ds.shapes = {_EMB: {_KEY: (4, 2)}}
    ds.embodiments = {_EMB}
    return ds, leaf


def test_the_flag_defaults_to_on_so_existing_behaviour_is_unchanged():
    ds, _ = _dataset(bounds_check=True)
    assert ds.bounds_check is True
    assert MultiDataset.bounds_check is True


def test_the_flag_is_actually_stored_rather_than_swallowed_by_kwargs():
    # This is the regression: it used to land in **kwargs and vanish.
    ds, _ = _dataset(bounds_check=False)
    assert ds.bounds_check is False


def test_a_state_only_instance_still_has_the_attribute():
    # Deploy-mode construction returns early; __getitem__ must not
    # AttributeError if it is ever reached.
    assert MultiDataset(state={}, norm_mode="quantile").bounds_check is True


def test_with_the_screen_on_an_out_of_bounds_sample_is_substituted():
    ds, leaf = _dataset(bounds_check=True)
    ds[1]
    # The bad index was read, rejected, and another index served in its place.
    assert leaf.served[0] == 1
    assert len(leaf.served) > 1, "no fallback substitution happened"


def test_with_the_screen_off_the_sample_is_returned_as_read():
    ds, leaf = _dataset(bounds_check=False)
    out = ds[1]
    assert leaf.served == [1], "the screen still substituted a sample"
    assert out is not None


def test_with_the_screen_off_nan_is_passed_through_rather_than_swapped():
    # NaN is the other thing the screen rejects; off means off.
    ds, leaf = _dataset(bounds_check=False, bad_value=float("nan"))
    ds[1]
    assert leaf.served == [1]


def test_with_the_screen_on_nan_is_substituted():
    ds, leaf = _dataset(bounds_check=True, bad_value=float("nan"))
    ds[1]
    assert len(leaf.served) > 1


def test_an_in_bounds_sample_is_unaffected_either_way():
    for flag in (True, False):
        ds, leaf = _dataset(bounds_check=flag)
        ds[0]
        assert leaf.served == [0], flag


def test_the_flag_reaches_the_dataset_through_from_resolver_kwargs():
    """_from_resolver forwards **kwargs to __init__, which is how the configs
    set this."""
    import inspect

    signature = inspect.signature(MultiDataset.__init__)
    assert "bounds_check" in signature.parameters
    assert signature.parameters["bounds_check"].default is True
