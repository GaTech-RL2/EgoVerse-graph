"""Configured branch selection for homogeneous batches; source names are opaque."""

from collections.abc import Mapping

from torch import nn

from egomimic.pipeline.core import Pipeline, Stage, resolve_homogeneous_scalar


class BranchStage(Stage):
    """Select a declared subgraph, including branches with different action widths.

    Branches declare the same external keys by mode. Their tensor shapes and
    internal modules may differ. No embodiment registry or model type is used.
    """

    def __init__(
        self,
        branches: Mapping[str, Pipeline],
        reads_by_mode,
        writes_by_mode,
        selector_key="embodiment",
        selector_aliases=None,
    ):
        super().__init__()
        if not branches or any(
            not isinstance(value, Pipeline) for value in branches.values()
        ):
            raise TypeError("BranchStage requires configured Pipeline branches")
        self.branches = nn.ModuleDict(dict(branches))
        self.selector_key = selector_key
        self.selector_aliases = {
            str(k): str(v) for k, v in (selector_aliases or {}).items()
        }
        self.reads_by_mode = {
            mode: tuple(dict.fromkeys([selector_key, *keys]))
            for mode, keys in reads_by_mode.items()
        }
        self.writes_by_mode = {
            mode: tuple(keys) for mode, keys in writes_by_mode.items()
        }
        if set(self.reads_by_mode) != {"train", "inference"} or set(
            self.writes_by_mode
        ) != {"train", "inference"}:
            raise ValueError("BranchStage must declare train and inference boundaries")
        for name, pipeline in self.branches.items():
            for mode in ("train", "inference"):
                runnable, blocked = pipeline.plan(self.reads_by_mode[mode], mode=mode)
                blocked = [
                    (stage, missing)
                    for stage, missing in blocked
                    if missing not in (["<train-only>"], ["<inference-only>"])
                ]
                writes = {key for stage in runnable for key in stage.contract(mode)[1]}
                required = {
                    key for key in self.writes_by_mode[mode] if not key.endswith("*")
                }
                if blocked or not required <= writes:
                    raise ValueError(
                        f"Branch {name!r} cannot satisfy its {mode} boundary"
                    )

    def bind_data_context(self, *, normalizer):
        for branch in self.branches.values():
            branch.bind_data_context(normalizer=normalizer)

    def execute(self, batch, *, mode):
        selector = str(
            resolve_homogeneous_scalar(
                batch[self.selector_key], label=self.selector_key
            )
        )
        selected = self.selector_aliases.get(selector, selector)
        if selected not in self.branches:
            raise ValueError(
                f"No configured branch for selector {selector!r}; available: {list(self.branches)}"
            )
        return self.branches[selected].execute(batch, mode=mode)
