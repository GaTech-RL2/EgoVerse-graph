"""Typed, model-declared runtime settings without model-family dispatch."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from omegaconf import OmegaConf


def validate_control_value(name, spec, value):
    kind = spec.get("type")
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"Inference control {name!r} requires a boolean")
    elif kind == "enum":
        choices = spec.get("choices")
        if (
            not isinstance(choices, (list, tuple))
            or not choices
            or any(not isinstance(choice, str) for choice in choices)
            or len(set(choices)) != len(choices)
        ):
            raise ValueError(f"Inference control {name!r} needs unique string choices")
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"Inference control {name!r} must be one of {choices}")
    elif kind in {"integer", "number"}:
        check = (
            (lambda v: type(v) is int)
            if kind == "integer"
            else (lambda v: type(v) in (int, float) and math.isfinite(v))
        )
        lower, upper = spec.get("min"), spec.get("max")
        if not check(lower) or not check(upper) or lower > upper:
            raise ValueError(f"Inference control {name!r} has invalid {kind} bounds")
        if not check(value) or not lower <= value <= upper:
            raise ValueError(
                f"Inference control {name!r} requires {kind} in [{lower}, {upper}]"
            )
        step = spec.get("step")
        if step is not None:
            if not check(step) or step <= 0:
                raise ValueError(f"Inference control {name!r} has invalid step")
            quotient = (value - lower) / step
            if not math.isclose(quotient, round(quotient), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Inference control {name!r} must use step {step}")
    else:
        raise ValueError(f"Inference control {name!r} has unsupported type {kind!r}")
    return value


def validate_control(name, spec, *, stage_ids):
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError("Inference controls require public setting names")
    if (
        not isinstance(spec, Mapping)
        or not isinstance(spec.get("label"), str)
        or not spec["label"]
    ):
        raise ValueError(
            f"Inference control {name!r} requires a declaration and UI label"
        )
    if not isinstance(spec.get("description", ""), str):
        raise ValueError(f"Inference control {name!r} description must be text")
    target = spec.get("target", {})
    if not isinstance(target, Mapping) or target.get("kind") not in {
        "stage_attribute",
        "policy_attribute",
    }:
        raise ValueError(
            f"Inference control {name!r} requires a declared setting target"
        )
    path = target.get("attribute_path")
    if not isinstance(path, str) or any(
        not (p.isidentifier() or p.isdecimal()) or p.startswith("_")
        for p in path.split(".")
    ):
        raise ValueError(f"Inference control {name!r} requires a public attribute path")
    if target["kind"] == "stage_attribute" and target.get("stage_id") not in stage_ids:
        raise ValueError(f"Inference control {name!r} refers to an undeclared stage_id")
    validate_control_value(name, spec, spec.get("default"))


def resolve_inference_profile(training, inference_profiles):
    """Resolve an explicitly declared stable stage identifier."""
    if not isinstance(inference_profiles, Mapping) or len(inference_profiles) != 1:
        raise ValueError("Declare exactly one resolved model-owned inference profile")
    name, profile = next(iter(inference_profiles.items()))
    if not isinstance(profile, Mapping):
        raise ValueError("Inference profile must be a mapping")
    stage_id = profile.get("stage_id")
    stage_ids = OmegaConf.select(training, "model.pipeline.stage_ids", default={})
    if not isinstance(stage_id, str) or stage_id not in stage_ids:
        raise ValueError(
            "Inference profile requires a declared stable stage_id. Legacy class-matched "
            "profiles must be migrated explicitly to a model-owned inference contract."
        )
    index = stage_ids[stage_id]
    stages = training.model.pipeline.stages
    if type(index) is not int or not 0 <= index < len(stages):
        raise ValueError(f"Invalid stage position for {stage_id!r}")
    return name, profile, stages[index]


@dataclass
class _InferenceControlBinding:
    name: str
    spec: dict
    value: object
    target_kind: str
    attribute_path: str
    owner: object | None = None
    attribute: str | None = None

    def validate(self, value):
        return validate_control_value(self.name, self.spec, value)

    def public(self):
        return {
            **{
                key: value
                for key, value in self.spec.items()
                if key not in {"target", "default"}
            },
            "value": self.value,
        }


def configure_profile_controls(graph, training, inference_profiles):
    """Bind declared settings by stable stage ID, with atomic preflight."""
    _, profile, _ = resolve_inference_profile(training, inference_profiles)
    stage_ids = OmegaConf.select(training, "model.pipeline.stage_ids", default={})
    bindings = []
    for name, spec in profile.get("overrides", {}).items():
        validate_control(name, spec, stage_ids=stage_ids)
        target = spec["target"]
        kind, path = target["kind"], target["attribute_path"]
        owner = attribute = None
        if kind == "stage_attribute":
            owner = graph.pipeline.stage_by_id(target["stage_id"])
            parts = path.split(".")
            for part in parts[:-1]:
                if not hasattr(owner, part):
                    raise ValueError(
                        f"Inference override target {path!r} does not exist"
                    )
                owner = getattr(owner, part)
            attribute = parts[-1]
            if not hasattr(owner, attribute):
                raise ValueError(f"Inference override target {path!r} does not exist")
        bindings.append(
            _InferenceControlBinding(
                name, dict(spec), spec["default"], kind, path, owner, attribute
            )
        )
    # No mutation occurs until every target and every default is valid.
    for binding in bindings:
        if binding.owner is not None:
            setattr(binding.owner, binding.attribute, binding.value)
    return tuple(bindings)
