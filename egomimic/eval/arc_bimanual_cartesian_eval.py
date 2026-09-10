"""Arc-tokenized twin of :class:`BimanualCartesianEval`.

An arc-tokenized run predicts (M+1, 14) rows that are NOT all poses: rows
0..M-1 are waypoints and row M is a velocity token. Every consumer downstream
of the model -- the revert transforms and the viz func -- assumes a stack of
poses, so the token has to be turned back into a time-indexed chunk before any
of them see it.

That conversion is the whole reason this class exists. The revert transform
rotates AND translates each row it is handed; applied to a velocity row it
returns the camera-to-gripper offset, i.e. a position. The resulting overlay
still renders, which is what makes the mistake expensive: the trajectory looks
plausible but the arm appears to teleport, and the paired MSE is computed
against a garbage final row.

Metrics are unchanged from the base class and stay in TOKEN space: both
prediction and target are tokens there, so ``Valid/MSE`` and
``Valid/Native_MSE`` remain apples-to-apples. Only the overlay path
detokenizes.
"""

from __future__ import annotations

import numpy as np
import torch

from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval

# Canonical bimanual cartesian width; an arc token keeps it.
_BIMANUAL_DIM = 14


class ArcBimanualCartesianEval(BimanualCartesianEval):
    """Detokenize arc rows before they reach the revert transforms.

    ``min_distance_unit`` and ``resampled_vector_length`` MUST match the values
    the data config gave the loader-side tokenizer. The detokenizer derives its
    replay duration from D and the velocity token, so a mismatch reconstructs at
    the wrong speed -- silently, since the shapes still line up. ``action_horizon``
    is how many control steps to emit.
    """

    def __init__(
        self,
        *args,
        min_distance_unit: float,
        resampled_vector_length: int,
        action_horizon: int = 100,
        dt: float = 1.0 / 30.0,
        velocity_mode: str = "mean",
        arc_metrics: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from egomimic.rldb.zarr.arc_length_tokenizer import (
            TokenizeBimanualArcLengthCartesian,
        )

        self.min_distance_unit = float(min_distance_unit)
        self.resampled_vector_length = int(resampled_vector_length)
        self.velocity_mode = str(velocity_mode)
        self.action_horizon = int(action_horizon)
        # Defaults ON here: an arc run is exactly the case the arc metric
        # families were built for. The base class owns the knobs, and it also
        # falls back to the action chunk when the preserved one is absent --
        # for an arc run that fallback must never trigger, because the
        # tokenizer overwrote actions_cartesian in place and recovering the GT
        # by detokenizing would be circular (a reconstruction spans D by
        # construction, so its distance says nothing about real travel).
        self.arc_metrics = bool(arc_metrics)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._tokenizer = TokenizeBimanualArcLengthCartesian(
            action_key=self.action_key,
            output_action_key=self.action_key,
            min_distance_unit=self.min_distance_unit,
            resampled_vector_length=self.resampled_vector_length,
            dt=float(dt),
            preserve_action_key=None,
            velocity_mode=self.velocity_mode,
        )

    def _viz_source(self, actions: torch.Tensor, embodiment_id: int) -> torch.Tensor:
        """Rows to hand the revert transforms.

        Arc tokens are detokenized first, because row M is a velocity and the
        revert would turn it into a position. A BASELINE run's prediction is
        already poses, so it passes straight through -- this evaluator serves
        both arms, and the overlay path needs the same predicate the metrics
        path uses or a baseline run dies here on a shape check meant for
        misconfiguration.
        """
        from egomimic.rldb.zarr.arc_length_tokenizer import (
            bimanual_arc_token_rows,
        )

        if not self._is_arc(actions):
            expected_rows = bimanual_arc_token_rows(
                self.resampled_vector_length, self.velocity_mode
            )
            if actions.ndim == 3 and int(actions.shape[-1]) != _BIMANUAL_DIM:
                # Not a bimanual chunk at all: that IS a misconfiguration.
                raise ValueError(
                    f"{type(self).__name__} expects (B, T, {_BIMANUAL_DIM}) "
                    f"rows, got {tuple(actions.shape)}. Arc tokens for M="
                    f"{self.resampled_vector_length} would be "
                    f"{expected_rows} rows."
                )
            # A row count matching the OTHER velocity mode's token is a
            # data/evaluator mode mismatch, not a baseline chunk. Passing it
            # through would score arc tokens as if they were poses and read
            # plausibly, so it stays a hard error.
            other = {"mean": "per_waypoint", "per_waypoint": "mean"}[
                self.velocity_mode
            ]
            if actions.ndim == 3 and int(actions.shape[-2]) == (
                bimanual_arc_token_rows(self.resampled_vector_length, other)
            ):
                raise ValueError(
                    f"{type(self).__name__} is configured for velocity_mode="
                    f"{self.velocity_mode!r} ({expected_rows} arc tokens) but "
                    f"got {tuple(actions.shape)}, which is the row count for "
                    f"{other!r}. The evaluator and the data config disagree."
                )
            return super()._viz_source(actions, embodiment_id)
        del embodiment_id
        native = actions.detach().cpu().numpy().astype(np.float64, copy=False)
        decoded = np.stack(
            [
                self._tokenizer.detokenize(sample, self.action_horizon)
                for sample in native
            ],
            axis=0,
        )
        return torch.from_numpy(decoded).to(dtype=actions.dtype)

    def _is_arc(self, actions) -> bool:
        """Whether these rows are an arc token rather than a pose chunk.

        One evaluator serves both arms of the ablation, so the run type is
        detected from the shape instead of configured twice: an arc token has
        exactly the row count this D/M and velocity mode imply, and the
        canonical bimanual width. A baseline chunk has the action horizon's
        rows, which only collides with the token count by coincidence -- and if
        it ever did, both readings would be the same rows anyway.
        """
        from egomimic.rldb.zarr.arc_length_tokenizer import (
            ARC_TOK_BIMANUAL_DIM,
            bimanual_arc_token_rows,
        )

        if actions is None or actions.ndim != 3:
            return False
        expected = bimanual_arc_token_rows(
            self.resampled_vector_length, self.velocity_mode
        )
        return (
            int(actions.shape[-2]) == expected
            and int(actions.shape[-1]) == ARC_TOK_BIMANUAL_DIM
        )

    def _arc_pred_time_indexed(self, prediction, embodiment_id: int):
        """Get the prediction into the time-indexed space the metrics score in.

        An ARC run predicts (M+1, 14) or (2M, 14) rows that must be walked back
        into control steps; the detokenizer already emits real control steps, so
        no de-interpolation follows. A BASELINE run predicts poses already, and
        takes the base class's path instead: unnormalize, then de-interpolate to
        the raw window, because arc length on an interpolated chunk reads short.

        Branching here rather than in a config is what lets one evaluator score
        both arms onto the same charts.
        """
        native = self._native(prediction, embodiment_id)
        if not self._is_arc(native):
            return super()._arc_pred_time_indexed(prediction, embodiment_id)
        decoded = self._viz_source(native.detach().cpu(), embodiment_id)
        return decoded.numpy().astype(np.float64, copy=False)
