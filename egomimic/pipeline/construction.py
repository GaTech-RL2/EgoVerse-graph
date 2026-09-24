"""Scoped construction policy for strict checkpoint restoration.

Constructors still allocate the declared architecture. Only external parameter
initializers are skipped: a strict loader must supply every checkpoint tensor
before the caller can execute the graph. Tokenizers/configuration are not weights.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_restore_parameters = ContextVar("egoverse_restore_parameters", default=False)


def restoring_parameters():
    return _restore_parameters.get()


@contextmanager
def checkpoint_construction(enabled=True):
    if type(enabled) is not bool:
        raise TypeError(
            "Checkpoint construction must be explicitly enabled or disabled"
        )
    token = _restore_parameters.set(enabled)
    try:
        yield
    finally:
        _restore_parameters.reset(token)
