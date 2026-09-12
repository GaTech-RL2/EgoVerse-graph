"""Compatibility import; implementation lives in egomimic.rldb.embodiment.embodiment."""

from importlib import import_module

_impl = import_module("egomimic.rldb.embodiment.embodiment")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
