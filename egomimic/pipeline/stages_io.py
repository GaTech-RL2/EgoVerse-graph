"""Small data-boundary stages for configured pipelines."""

from __future__ import annotations

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar


class ActionTargetBuilder(Stage):
    """Move normalized loader actions into the pipeline target namespace.

    ``action_key`` is the batch key the dataset's transform list actually
    writes. It defaults to ``actions`` (Planar), but embodiment transform lists
    name it differently -- Eva/Human/Yam cartesian pipelines emit
    ``actions_cartesian`` -- and the graph must read the key that exists rather
    than force every embodiment to rename its output.
    """

    train_only = True
    writes = ("target",)

    def __init__(self, action_key: str = "actions"):
        super().__init__()
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        self.reads = (self.action_key,)

    def forward(self, batch: dict) -> dict:
        if self.action_key not in batch:
            raise ValueError(
                f"ActionTargetBuilder requires normalized actions at "
                f"{self.action_key!r}"
            )
        batch["target"] = batch.pop(self.action_key)
        return batch


class EmbodimentActionTargetBuilder(Stage):
    """Read each embodiment's native normalized action key without coercion."""

    train_only = True
    writes = ('target',)

    def __init__(self, action_keys: dict[str, str], selector_aliases: dict | None = None, selector_key: str = 'embodiment'):
        super().__init__()
        self.action_keys = {str(k): str(v) for k, v in action_keys.items()}
        if len(self.action_keys) < 2 or any(not k or not v for k, v in self.action_keys.items()):
            raise ValueError('Expected at least two nonempty embodiment/action-key pairs')
        self.selector_aliases = {str(k): str(v) for k, v in dict(selector_aliases or {}).items()}
        self.selector_key = str(selector_key)
        self.reads = (self.selector_key,)

    def forward(self, batch: dict) -> dict:
        raw = resolve_homogeneous_scalar(batch[self.selector_key], label=self.selector_key)
        name = self.selector_aliases.get(str(raw), str(raw))
        try:
            action_key = self.action_keys[name]
        except KeyError as exc:
            raise KeyError(f'No action interface for embodiment {name!r}') from exc
        if action_key not in batch:
            raise ValueError(f'Embodiment {name!r} requires normalized actions at {action_key!r}')
        batch['target'] = batch.pop(action_key)
        return batch
