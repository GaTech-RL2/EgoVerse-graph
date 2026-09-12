"""Compatibility import. Configure validation groups on the shared evaluator."""

from importlib import import_module

_impl = import_module("egomimic.eval.video")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
