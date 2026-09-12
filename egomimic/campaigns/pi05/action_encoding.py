"""Compatibility import; implementation lives in egomimic.utils.action_encoding."""

from importlib import import_module

_impl = import_module("egomimic.utils.action_encoding")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
