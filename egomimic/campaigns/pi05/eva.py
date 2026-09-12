"""Compatibility import; implementation lives in egomimic.rldb.embodiment.eva."""

from importlib import import_module

_impl = import_module("egomimic.rldb.embodiment.eva")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
