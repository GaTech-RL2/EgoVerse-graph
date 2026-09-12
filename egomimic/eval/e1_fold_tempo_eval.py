"""Compatibility import; implementation lives in egomimic.eval.bimanual_tempo_eval."""

from importlib import import_module

_impl = import_module("egomimic.eval.bimanual_tempo_eval")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
E1FoldTempoEval = _impl.BimanualTempoEval
