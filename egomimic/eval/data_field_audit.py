"""Configured zero/nonfinite data checks with exact recorded sample identities."""

import json
from pathlib import Path

import torch

from egomimic.eval.eval import Eval, EvaluationDataRequirements


class DataFieldAudit(Eval):
    def __init__(
        self, checks, *, sample_id_key="episode_hash", frame_index_key="frame_index"
    ):
        self.checks = checks
        self.sample_id_key, self.frame_index_key = sample_id_key, frame_index_key
        for source, rules in checks.items():
            for name, rule in rules.items():
                if rule["condition"] not in {"all_zero", "nonfinite"}:
                    raise ValueError(f"Unsupported data condition for {source}/{name}")
                columns = rule.get("columns")
                if columns is not None and (
                    len(columns) != 2
                    or any(type(v) is not int for v in columns)
                    or not 0 <= columns[0] < columns[1]
                ):
                    raise ValueError("columns must be [inclusive start, exclusive end]")

    def data_requirements(self):
        return EvaluationDataRequirements(
            ordered=True,
            complete_episodes=True,
            sample_id_key=self.sample_id_key,
            frame_index_key=self.frame_index_key,
        )

    def bind_data_context(self, *, normalizer):
        self.normalizer = normalizer

    def on_validation_start(self):
        self._counts = {}
        self.rows_path = Path(self.root_dir()) / "data-audit-rows.jsonl"
        with self.rows_path.open("x"):
            pass

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        for source, values in batch.items():
            rules = self.checks.get(source)
            if rules is None:
                raise ValueError(f"No data checks configured for source {source!r}")
            for name, rule in rules.items():
                data = torch.as_tensor(values[rule["key"]])
                if data.ndim < 2:
                    raise ValueError("Audited fields need batch and feature axes")
                if rule.get("columns") is not None:
                    start, end = rule["columns"]
                    if end > data.shape[-1]:
                        raise ValueError(
                            f"Data slice for {source}/{name} exceeds width {data.shape[-1]}"
                        )
                    data = data[..., start:end]
                flattened = data.flatten(1)
                if flattened.shape[1] == 0:
                    raise ValueError(f"Empty data field for {source}/{name}")
                bad = (
                    (flattened == 0).all(1)
                    if rule["condition"] == "all_zero"
                    else ~torch.isfinite(flattened).all(1)
                )
                count = (
                    self._counts.setdefault(self._validation_group, {})
                    .setdefault(source, {})
                    .setdefault(name, {"frames": 0, "matches": 0})
                )
                count["frames"] += len(data)
                count["matches"] += int(bad.sum())
                with self.rows_path.open("a") as handle:
                    for index in bad.nonzero().flatten().tolist():
                        json.dump(
                            {
                                "group": self._validation_group,
                                "source": source,
                                "check": name,
                                "episode": str(values[self.sample_id_key][index]),
                                "frame": int(values[self.frame_index_key][index]),
                            },
                            handle,
                        )
                        handle.write("\n")
        return {}

    def on_validation_end(self):
        with (Path(self.root_dir()) / "data-audit.json").open("x") as handle:
            json.dump(
                {"counts": self._counts, "rows": str(self.rows_path)}, handle, indent=2
            )
