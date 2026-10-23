"""Stable import path for the single explicit stream implementation."""

import egomimic.robot.stream_decoder as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
