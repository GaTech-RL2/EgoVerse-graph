"""Historical Cartesian ARC deployment is withdrawn at this review level.

Baseline graph policies use decoder=None and remain available.
The explicit ARC integration later replaces this temporary boundary.
"""


def _withdrawn(*args, **kwargs):
    raise ValueError(
        "Legacy Cartesian ARC is withdrawn at this review level; use a time-domain baseline or the later explicit ARC integration level. Historical checkpoints must retain their original source."
    )


BimanualArcDecoder = _withdrawn
BimanualIntervalArcDecoder = _withdrawn


if __name__ == "__main__":
    _withdrawn()
