"""Compatibility import; implementation lives in egomimic.eval.distribution_metrics."""

from importlib import import_module

_impl = import_module("egomimic.eval.distribution_metrics")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
