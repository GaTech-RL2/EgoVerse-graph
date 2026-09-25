import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationDataRequirements:
    """Data capabilities checked before validation or expensive model setup."""

    ordered: bool = False
    complete_episodes: bool = False
    max_episodes: int | None = None
    sample_id_key: str | None = None
    frame_index_key: str | None = None
    source_fps: float | None = None
    required_keys: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.required_keys, tuple) or any(
            not isinstance(key, str) or not key for key in self.required_keys
        ):
            raise TypeError("required_keys must be a tuple of nonempty strings")
        if type(self.ordered) is not bool or type(self.complete_episodes) is not bool:
            raise TypeError("ordered and complete_episodes must be booleans")
        if self.max_episodes is not None and (
            type(self.max_episodes) is not int or self.max_episodes < 1
        ):
            raise ValueError("max_episodes must be a positive integer")
        for name in ("sample_id_key", "frame_index_key"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string")
        if self.source_fps is not None and (
            isinstance(self.source_fps, bool)
            or not isinstance(self.source_fps, (int, float))
            or not math.isfinite(self.source_fps)
            or self.source_fps <= 0
        ):
            raise ValueError("source_fps must be a finite positive number")


def validation_trainer_overrides(evaluator):
    """Validate evaluator-owned loop settings; resources belong to the launcher."""
    overrides = evaluator.trainer_overrides()
    allowed = {
        "limit_val_batches",
        "num_sanity_val_steps",
        "check_val_every_n_epoch",
        "val_check_interval",
        "log_every_n_steps",
        "inference_mode",
    }
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError(
            "Evaluator trainer_overrides may contain only validation-loop settings: "
            f"{sorted(allowed)}; got {overrides!r}"
        )
    return overrides


def validate_validation_loop(requirements, trainer_options, *, mode):
    """Reject loop truncation when the declared result needs complete episodes."""
    if not requirements.complete_episodes:
        return
    limit = trainer_options.get("limit_val_batches", 1.0)
    if limit is None:
        limit = 1.0
    if type(limit) is not float or limit != 1.0:
        raise ValueError(
            "Complete-episode evaluation requires limit_val_batches=1.0; "
            "limit episodes through the DataModule instead of truncating batches"
        )
    if trainer_options.get("fast_dev_run", False) or trainer_options.get(
        "overfit_batches", 0
    ):
        raise ValueError("Complete-episode evaluation cannot truncate the trainer loop")
    if mode == "train" and trainer_options.get("num_sanity_val_steps", 2) not in (
        0,
        -1,
    ):
        raise ValueError(
            "Complete-episode validation requires num_sanity_val_steps=0 or -1"
        )


class Eval(ABC):
    def set_validation_group(self, group: str):
        """Receive the opaque validation group selected by the data context."""
        self._validation_group = group

    def data_requirements(self) -> EvaluationDataRequirements:
        return EvaluationDataRequirements()

    def trainer_overrides(self) -> dict[str, object]:
        return {}

    @abstractmethod
    def __init__(self):
        pass

    def root_dir(self):
        return self.trainer.default_root_dir

    @abstractmethod
    def bind_data_context(self, *, normalizer):
        """Attach normalization and other data-owned evaluation state."""
        pass

    @abstractmethod
    def on_validation_start(self):
        pass

    @abstractmethod
    def on_validation_end(self):
        pass

    @abstractmethod
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass
