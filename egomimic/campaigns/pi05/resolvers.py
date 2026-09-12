"""Compatibility import; implementation lives in egomimic.rldb.zarr.zarr_dataset_multi."""

from importlib import import_module

_impl = import_module("egomimic.rldb.zarr.zarr_dataset_multi")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
