"""Nested, declared decoder settings remain atomic at the robot boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.inference_controls import (
    _InferenceControlBinding,
    apply_control_values,
    bind_policy_controls,
    resolve_control_attribute,
)
from egomimic.robot.graph_policy import GraphRobotPolicy


def _control(name, path, default, *, owner=None, attribute=None):
    return _InferenceControlBinding(
        name=name,
        spec={"label": name, "type": "integer", "min": 1, "max": 400, "step": 1},
        value=default,
        target_kind="policy_attribute",
        attribute_path=path,
        owner=owner,
        attribute=attribute,
    )


class Decoder:
    def __init__(self):
        self.speed_percent = 100
        self.execute_percent = 50
        self.steps = None

    def validate_inference_control(self, attribute, value):
        if attribute == "execute_percent" and (
            value * 16 % 100 or not 1 <= value <= 100
        ):
            raise ValueError("A cap must retain a whole prefix of the 16 waypoints")

    def execution_steps(self):
        return self.steps


class Adapter:
    """Keep the required policy-adapter boundary explicit in control fixtures."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.embodiment_id = 7
        self.action_key = "actions"
        self.proprio_key = "state"
        self.camera_keys = {"front": "front"}

    def input_constants(self):
        return {"embodiment": self.embodiment_id}


def test_declared_nested_defaults_and_updates_bind_to_decoder():
    decoder = Decoder()
    policy = SimpleNamespace(adapter=SimpleNamespace(decoder=decoder))
    controls = bind_policy_controls(
        policy,
        (
            _control("speed", "adapter.decoder.speed_percent", 125),
            _control("cap", "adapter.decoder.execute_percent", 50),
        ),
        allowed_paths={
            "adapter.decoder.speed_percent",
            "adapter.decoder.execute_percent",
        },
    )
    assert decoder.speed_percent == controls[0].value == 125
    apply_control_values(controls, {"speed": 150, "cap": 75})
    assert decoder.speed_percent == controls[0].value == 150
    assert decoder.execute_percent == controls[1].value == 75


@pytest.mark.parametrize("container", [list, tuple])
def test_sequence_decoder_controls_bind_to_the_declared_owner(container):
    decoder = Decoder()
    chain = SimpleNamespace(decoders=container([SimpleNamespace(), decoder]))
    policy = SimpleNamespace(adapter=SimpleNamespace(decoder=chain))
    path = "adapter.decoder.decoders.1.speed_percent"
    controls = bind_policy_controls(
        policy, (_control("speed", path, 125),), allowed_paths={path}
    )
    assert controls[0].owner is decoder
    apply_control_values(controls, {"speed": 150})
    assert decoder.speed_percent == controls[0].value == 150


def test_sequence_decoder_control_failure_restores_earlier_nested_target():
    class RejectingDecoder:
        def __init__(self):
            self._speed_percent = 100

        @property
        def speed_percent(self):
            return self._speed_percent

        @speed_percent.setter
        def speed_percent(self, value):
            self._speed_percent = value
            if value == 150:
                raise ValueError("retimer rejected")

    decoder, retimer = Decoder(), RejectingDecoder()
    policy = SimpleNamespace(
        adapter=SimpleNamespace(decoder=SimpleNamespace(decoders=[decoder, retimer]))
    )
    paths = [f"adapter.decoder.decoders.{index}.speed_percent" for index in (0, 1)]
    controls = bind_policy_controls(
        policy,
        tuple(
            _control(name, path, 100) for name, path in zip(("first", "second"), paths)
        ),
        allowed_paths=set(paths),
    )
    with pytest.raises(ValueError, match="retimer rejected"):
        apply_control_values(controls, {"first": 125, "second": 150})
    assert decoder.speed_percent == retimer.speed_percent == 100
    assert [control.value for control in controls] == [100, 100]


@pytest.mark.parametrize(
    "path",
    [
        "adapter.decoder.decoders.2.speed_percent",
        "adapter.decoder.decoders.-1.speed_percent",
        "adapter.decoder.decoders.1x.speed_percent",
        "adapter.decoder.decoders..speed_percent",
        "adapter.decoder.decoders.1",
        "adapter.decoder.0.speed_percent",
    ],
)
def test_invalid_sequence_paths_never_apply_earlier_defaults(path):
    decoder = Decoder()
    policy = SimpleNamespace(
        adapter=SimpleNamespace(decoder=SimpleNamespace(decoders=[None, decoder]))
    )
    valid = "adapter.decoder.decoders.1.speed_percent"
    with pytest.raises(ValueError):
        bind_policy_controls(
            policy,
            (_control("speed", valid, 125), _control("invalid", path, 1)),
            allowed_paths={valid, path},
        )
    assert decoder.speed_percent == 100


def test_robot_policy_exposes_declared_sequence_retimer_control():
    retimer = Decoder()
    adapter = Adapter(SimpleNamespace(decoders=[SimpleNamespace(), retimer]))
    normalizer = SimpleNamespace(
        validate_inference_schema=lambda *_args, **_kwargs: None
    )
    policy = GraphRobotPolicy(
        PipelineAlgo([Stage()], device="cpu"),
        normalizer,
        adapter,
        inference_controls=(
            _control("speed", "adapter.decoder.decoders.1.speed_percent", 125),
        ),
    )
    policy.apply_inference_overrides({"speed": 150})
    assert retimer.speed_percent == 150


def test_owner_validation_rejects_cap_before_any_control_changes():
    decoder = Decoder()
    controls = (
        _control(
            "speed",
            "adapter.decoder.speed_percent",
            100,
            owner=decoder,
            attribute="speed_percent",
        ),
        _control(
            "cap",
            "adapter.decoder.execute_percent",
            50,
            owner=decoder,
            attribute="execute_percent",
        ),
    )
    with pytest.raises(ValueError, match="whole prefix"):
        apply_control_values(controls, {"speed": 150, "cap": 51})
    assert (decoder.speed_percent, decoder.execute_percent) == (100, 50)
    assert [control.public()["value"] for control in controls] == [100, 50]


def test_failing_setter_restores_all_touched_attributes_and_display_values():
    class Setter:
        def __init__(self):
            self.first = 100
            self._second = 50

        @property
        def second(self):
            return self._second

        @second.setter
        def second(self, value):
            self._second = value
            if value == 60:
                raise ValueError("Setter rejected the value after touching its state")

    owner = Setter()
    controls = (
        _control("first", "first", 100, owner=owner, attribute="first"),
        _control("second", "second", 50, owner=owner, attribute="second"),
    )
    with pytest.raises(ValueError, match="Setter rejected"):
        apply_control_values(controls, {"first": 150, "second": 60})
    assert (owner.first, owner.second) == (100, 50)
    assert [control.value for control in controls] == [100, 50]


def test_invalid_declared_path_does_not_apply_earlier_defaults():
    decoder = Decoder()
    policy = SimpleNamespace(adapter=SimpleNamespace(decoder=decoder))
    with pytest.raises(ValueError, match="does not expose"):
        bind_policy_controls(
            policy,
            (
                _control("speed", "adapter.decoder.speed_percent", 150),
                _control("other", "adapter.decoder.arbitrary", 1),
            ),
            allowed_paths={"adapter.decoder.speed_percent"},
        )
    assert decoder.speed_percent == 100


@pytest.mark.parametrize(
    "path", ["adapter._decoder.speed_percent", "__dict__.x", "adapter..decoder"]
)
def test_private_or_invalid_paths_are_rejected(path):
    with pytest.raises(ValueError, match="public attribute"):
        resolve_control_attribute(SimpleNamespace(), path)


def test_aliasing_control_targets_is_rejected_without_mutation():
    decoder = Decoder()
    controls = tuple(
        _control(name, "speed_percent", 100, owner=decoder, attribute="speed_percent")
        for name in ("one", "two")
    )
    with pytest.raises(ValueError, match="alias"):
        apply_control_values(controls, {"one": 150, "two": 200})
    assert decoder.speed_percent == 100


def test_robot_policy_uses_declared_decoder_controls_and_runtime_limits():
    decoder = Decoder()
    graph = PipelineAlgo([Stage()], device="cpu")
    adapter = Adapter(decoder)
    normalizer = SimpleNamespace(
        validate_inference_schema=lambda *_args, **_kwargs: None
    )
    controls = (
        _control("speed", "adapter.decoder.speed_percent", 125),
        _control("replan", "replan_every", 30),
        _control("samples", "max_valid_samples", 1),
    )
    policy = GraphRobotPolicy(graph, normalizer, adapter, inference_controls=controls)
    policy.apply_inference_overrides({"speed": 150, "replan": 12})
    assert decoder.speed_percent == 150 and policy.replan_every == 12
    with pytest.raises(ValueError, match="max_valid_samples"):
        policy.apply_inference_overrides({"speed": 175, "samples": 32})
    assert decoder.speed_percent == 150
    assert policy.inference_controls()["speed"]["value"] == 150


def test_decoder_execution_boundary_takes_precedence_over_fixed_replan():
    policy = object.__new__(GraphRobotPolicy)
    decoder = Decoder()
    policy.adapter = SimpleNamespace(decoder=decoder)
    policy.replan_every = 30
    actions = np.zeros((100, 14))
    decoder.steps = 12
    assert len(policy.execution_plan(actions)) == 12
    decoder.steps = None
    assert len(policy.execution_plan(actions)) == 30


@pytest.mark.parametrize("count", [0, -1, 101, True, 2.5])
def test_invalid_decoder_execution_boundary_is_rejected(count):
    policy = object.__new__(GraphRobotPolicy)
    policy.adapter = SimpleNamespace(
        decoder=SimpleNamespace(execution_steps=lambda: count)
    )
    policy.replan_every = 30
    with pytest.raises(ValueError, match="outside the predicted plan"):
        policy.execution_plan(np.zeros((100, 14)))
