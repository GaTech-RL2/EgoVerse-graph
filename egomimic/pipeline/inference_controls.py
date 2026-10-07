"""Typed, model-declared runtime settings without model-family dispatch."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

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
        value = validate_control_value(self.name, self.spec, value)
        validate = getattr(self.owner, "validate_inference_control", None)
        if callable(validate) and self.attribute is not None:
            validate(self.attribute, value)
        return value

    def public(self):
        return {
            **{
                key: value
                for key, value in self.spec.items()
                if key not in {"target", "default"}
            },
            "value": self.value,
        }


def resolve_control_attribute(owner, path):
    """Resolve an explicit public attribute path without interpreting its owner."""
    parts = path.split(".") if isinstance(path, str) else ()
    if not parts or any(
        not (part.isidentifier() or part.isdecimal()) or part.startswith("_")
        for part in parts
    ):
        raise ValueError("Inference controls require a public attribute path")
    for part in parts[:-1]:
        if part.isdecimal():
            index = int(part)
            if not isinstance(owner, (list, tuple)) or index >= len(owner):
                raise ValueError(f"Inference override target {path!r} does not exist")
            owner = owner[index]
            continue
        if not hasattr(owner, part):
            raise ValueError(f"Inference override target {path!r} does not exist")
        owner = getattr(owner, part)
    attribute = parts[-1]
    if attribute.isdecimal():
        raise ValueError(
            "Inference controls must target a public attribute, not a sequence slot"
        )
    if not hasattr(owner, attribute):
        raise ValueError(f"Inference override target {path!r} does not exist")
    return owner, attribute


def apply_control_values(bindings, values):
    """Preflight every declared setting, then apply or restore all touched values.

    Owners may expose ``validate_inference_control(attribute, value)`` for pure
    domain validation beyond the profile's type and bounds. Setter rejection
    also restores previously touched attributes, including the failing target.
    Operator-visible values change only after all setters succeed.
    """
    controls = {binding.name: binding for binding in bindings}
    if len(controls) != len(bindings):
        raise ValueError("Inference override names must be unique")
    if not isinstance(values, Mapping) or not values:
        raise ValueError("Inference overrides must be a nonempty mapping")
    unknown = set(values) - set(controls)
    if unknown:
        raise ValueError(
            "Inference override is not exposed by this model profile: "
            + ", ".join(sorted(unknown))
        )
    pending, targets = [], set()
    for name, value in values.items():
        binding = controls[name]
        if binding.owner is None or binding.attribute is None:
            raise ValueError(f"Inference control {name!r} has no bound target")
        target = (id(binding.owner), binding.attribute)
        if target in targets:
            raise ValueError("Inference controls cannot alias the same target")
        targets.add(target)
        value = binding.validate(value)
        previous = getattr(binding.owner, binding.attribute)
        pending.append((binding, value, previous))
    touched = []
    try:
        for binding, value, previous in pending:
            touched.append((binding, previous))
            setattr(binding.owner, binding.attribute, value)
    except Exception as error:
        rollback_errors = []
        for binding, previous in reversed(touched):
            try:
                setattr(binding.owner, binding.attribute, previous)
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                "Inference control update failed and its prior state could not be restored"
            ) from error
        raise
    for binding, value, _previous in pending:
        binding.value = value


def bind_policy_controls(policy, bindings, *, allowed_paths):
    """Bind only the public settings explicitly exposed by the policy adapter."""
    resolved, selected = [], []
    for binding in bindings:
        if binding.target_kind == "policy_attribute":
            if binding.attribute_path not in allowed_paths:
                raise ValueError(
                    f"Robot policy does not expose {binding.attribute_path!r}"
                )
            owner, attribute = resolve_control_attribute(policy, binding.attribute_path)
            binding = replace(binding, owner=owner, attribute=attribute)
            selected.append(binding)
        resolved.append(binding)
    if selected:
        apply_control_values(
            selected, {binding.name: binding.value for binding in selected}
        )
    return tuple(resolved)


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
            owner, attribute = resolve_control_attribute(owner, path)
        bindings.append(
            _InferenceControlBinding(
                name, dict(spec), spec["default"], kind, path, owner, attribute
            )
        )
    selected = [binding for binding in bindings if binding.owner is not None]
    if selected:
        apply_control_values(
            selected, {binding.name: binding.value for binding in selected}
        )
    return tuple(bindings)
