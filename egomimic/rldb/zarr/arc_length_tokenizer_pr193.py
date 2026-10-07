"""PR #193 compatibility contract, implemented by the shared canonical codec.

The donor copy at b1ecba6310fac41e7441a743f756362a7ca65517 is verbatim
67863ca5:egomimic/rldb/zarr/arc_length_tokenizer.py (SHA-256
4a908c44d0a338eab6139f492138cfbad81638a92bfc94729a45308769f5b625 including
the donor provenance header). Its AST matched the canonical codec at7ccb6096.
Only plain per-waypoint wide/stacked decoding is this historical contract.
The canonical M28 additions leave that path unchanged; keep parity fixtures
when changing it. Explicit module identity prevents shape-based codec guessing.
"""

from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian as _CanonicalCodec,
)
from egomimic.rldb.zarr.arc_length_tokenizer import (
    bimanual_arc_token_shape as _canonical_shape,
)
from egomimic.rldb.zarr.arc_length_tokenizer import (
    stack_arc_token as stack_arc_token,
)

SOURCE_COMMIT = "67863ca5e83885e1228eb23bfa8dfdd61f29ced5"
CODEC_VERSION = "pr193_plain_per_waypoint"


def bimanual_arc_token_shape(
    num_waypoints, velocity_mode="per_waypoint", velocity_layout=None
):
    if velocity_mode != "per_waypoint" or velocity_layout not in ("wide", "stacked"):
        raise ValueError(
            "PR193 compatibility requires explicit per_waypoint wide|stacked"
        )
    return _canonical_shape(num_waypoints, velocity_mode, velocity_layout)


class TokenizeBimanualArcLengthCartesian(_CanonicalCodec):
    def __init__(
        self,
        *,
        velocity_mode="per_waypoint",
        velocity_layout=None,
        rotation_distance_unit=None,
        arc_chunking_mode=None,
        **kwargs,
    ):
        bimanual_arc_token_shape(
            kwargs.get("resampled_vector_length", 20), velocity_mode, velocity_layout
        )
        if rotation_distance_unit is not None or arc_chunking_mode is not None:
            raise ValueError("PR193 compatibility is the plain no-R codec")
        super().__init__(
            velocity_mode=velocity_mode, velocity_layout=velocity_layout, **kwargs
        )
