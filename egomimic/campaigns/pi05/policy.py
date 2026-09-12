"""Compatibility import; implementation lives in egomimic.models.pi05.policy."""

from importlib import import_module

_impl = import_module("egomimic.models.pi05.policy")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
