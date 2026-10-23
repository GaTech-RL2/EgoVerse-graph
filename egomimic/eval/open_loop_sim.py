"""Stable import path for the single explicit stream implementation."""

import egomimic.eval.open_loop_stream as _implementation

OpenLoopSimEval = _implementation.OpenLoopSimEval


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
