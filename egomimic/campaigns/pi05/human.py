"""Compatibility import; implementation lives in egomimic.rldb.embodiment.human."""

from importlib import import_module

_impl = import_module("egomimic.rldb.embodiment.human")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
