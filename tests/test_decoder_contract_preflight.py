"""Decoder shape mismatches reject before model loading or control binding."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.inference_controls import _InferenceControlBinding
from egomimic.pipeline.sequence_decoder import (
    SequentialActionDecoder,
    UniformTimeGridDecoder,
)
from egomimic.robot import graph_policy
from egomimic.robot.action_decoder import CartesianRotationDecoder
from egomimic.robot.arc_decoder import (
    BimanualArcDecoder,
    FirstStreamArcDecoder,
    TimeChunkRetimer,
)


class DeclaredDecoder:
    def __init__(self, native=(100, 16), output=(100, 14)):
        self.native_shape, self.output_shape = native, output
        self.speed_percent = 100
        self.contracts = []

    def validate_inference_contract(self, native, canonical):
        self.contracts.append((tuple(native), tuple(canonical)))
        if tuple(native) != self.native_shape or tuple(canonical) != self.output_shape:
            raise ValueError("Decoder dimensions differ from the model declaration")

    def __call__(self, values):
        raise AssertionError("Contract preflight must not run decoder inference")


def adapter(decoder):
    return graph_policy.CartesianGraphAdapter(
        base_T_model={"left": np.eye(4), "right": np.eye(4)},
        camera_keys={"camera": "front"},
        embodiment_id=7,
        rotation_mode="euler",
        action_frame="eef_frame",
        image_hw=(2, 3),
        proprio_key="state",
        action_key="actions",
        decoder=decoder,
    )


def contract(native=(100, 16), output=(100, 14)):
    return {
        "native_output": {"shape": list(native)},
        "output": {"shape": list(output), "representation": "cartesian"},
        "input": {"keys": ["state", "front", "embodiment"]},
        "profiles": {
            "default": {
                "stage_id": "sampler",
                "native_shape": list(native),
                "adapter": {},
            }
        },
    }


def test_graph_loader_checks_decoder_before_checkpoint_or_normalizer_loading(
    monkeypatch,
):
    inference = contract(native=(100, 18))
    training = OmegaConf.create(
        {
            "model": {
                "pipeline": {"stage_ids": {"sampler": 0}, "stages": [{}]},
                "inference": {"native_output": inference["native_output"]},
            }
        }
    )
    decoder = DeclaredDecoder()
    monkeypatch.setattr(graph_policy.OmegaConf, "load", lambda _path: training)
    monkeypatch.setattr(graph_policy, "load_inference_config", lambda *_args: inference)
    monkeypatch.setattr(graph_policy, "instantiate", lambda _config: adapter(decoder))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Decoder mismatch must reject before checkpoint loading")

    monkeypatch.setattr(graph_policy, "load_bound_graph", forbidden)
    with pytest.raises(ValueError, match="dimensions differ"):
        graph_policy.load_graph_policy(
            {
                "training_config": "no-read.yaml",
                "device": "cpu",
                "inference_config": "no-read-inference.yaml",
                "checkpoint": "no-read.ckpt",
                "normalizer_path": "no-read.json",
                "adapter": {},
            }
        )
    assert decoder.contracts == [((100, 18), (100, 14))]


def test_direct_policy_checks_decoder_before_applying_control_defaults():
    decoder = DeclaredDecoder()
    control = _InferenceControlBinding(
        "speed",
        {"label": "Speed", "type": "integer", "min": 1, "max": 400},
        125,
        "policy_attribute",
        "adapter.decoder.speed_percent",
    )
    with pytest.raises(ValueError, match="dimensions differ"):
        graph_policy.GraphRobotPolicy(
            PipelineAlgo([Stage()], device="cpu"),
            SimpleNamespace(validate_inference_schema=lambda *_args, **_kwargs: None),
            adapter(decoder),
            inference_graph=contract(native=(100, 18)),
            inference_controls=(control,),
        )
    assert decoder.speed_percent == 100 and control.owner is None


@pytest.mark.parametrize(
    "has_gripper,native_width,output_width", [(True, 20, 14), (False, 18, 12)]
)
def test_rotation_contract_preserves_rows_and_existing_gripper_semantics(
    has_gripper, native_width, output_width
):
    decoder = CartesianRotationDecoder(has_gripper)
    decoder.validate_inference_contract([100, native_width], [100, output_width])
    with pytest.raises(ValueError, match="without changing temporal rows"):
        decoder.validate_inference_contract([100, native_width], [50, output_width])
    with pytest.raises(ValueError):
        decoder.validate_inference_contract(
            [100, native_width + 2], [100, output_width]
        )


def grid(width=14, native_rows=100, output_rows=45):
    return UniformTimeGridDecoder(
        [native_rows, width],
        native_duration=44 / 30,
        output_horizon=output_rows,
        output_dt=1 / 30,
    )


def test_rotation_and_resampling_chain_validates_each_intermediate_shape():
    sequence = SequentialActionDecoder([CartesianRotationDecoder(True), grid()])
    sequence.validate_inference_contract([100, 20], [45, 14])
    with pytest.raises(ValueError, match="Rotation decoder"):
        sequence.validate_inference_contract([100, 18], [45, 14])
    with pytest.raises(ValueError, match="output shape"):
        sequence.validate_inference_contract([100, 20], [100, 14])


def test_grid_and_retimer_chain_checks_boundary_without_final_output_property():
    class Retimer:
        shape = (45, 14)

        def __init__(self):
            self.contracts = []

        def validate_inference_contract(self, native, canonical):
            self.contracts.append((tuple(native), tuple(canonical)))
            if tuple(native) != self.shape or tuple(canonical) != self.shape:
                raise ValueError("Retimer shape mismatch")

        def __call__(self, values):
            return values

    retimer = Retimer()
    sequence = SequentialActionDecoder([grid(), retimer])
    sequence.validate_inference_contract([100, 14], [45, 14])
    assert retimer.contracts == [((45, 14), (45, 14))]
    with pytest.raises(ValueError, match="Retimer shape mismatch"):
        sequence.validate_inference_contract([100, 14], [50, 14])


def test_sequence_rejects_incompatible_intermediate_decoder_input():
    sequence = SequentialActionDecoder([grid(), DeclaredDecoder(native=(100, 14))])
    with pytest.raises(ValueError, match="incompatible input"):
        sequence.validate_inference_contract([100, 14], [100, 14])


def test_uniform_grid_requires_native_and_canonical_shapes_to_match_implementation():
    decoder = grid()
    decoder.validate_inference_contract([100, 14], [45, 14])
    with pytest.raises(ValueError, match="native shape"):
        decoder.validate_inference_contract([50, 14], [45, 14])
    with pytest.raises(ValueError, match="output shape"):
        decoder.validate_inference_contract([100, 14], [45, 16])


def test_undeclared_intermediate_shape_fails_before_decoding():
    sequence = SequentialActionDecoder([lambda values: values, lambda values: values])
    with pytest.raises(ValueError, match="declared intermediate shape"):
        sequence.validate_inference_contract([100, 14], [100, 14])


@pytest.mark.parametrize(
    "decoder",
    [
        BimanualArcDecoder("e1_dur"),
        BimanualArcDecoder("e1_durhyb"),
        FirstStreamArcDecoder(
            "duration", "wide", 0.4, 1.0, codec_version="canonical200"
        ),
        FirstStreamArcDecoder(
            "duration", "wide", 0.4, 1.0, codec_version="m28_99be4af0"
        ),
        TimeChunkRetimer(),
    ],
)
def test_real_arc_and_time_decoder_contract_hooks_preflight_model_shapes(decoder):
    native = decoder.shape
    canonical = (decoder.action_horizon, 14)
    graph_policy.validate_adapter_inference_contract(
        adapter(decoder), contract(native, canonical)
    )
    with pytest.raises(ValueError, match="native_shape"):
        graph_policy.validate_adapter_inference_contract(
            adapter(decoder), contract((native[0], native[1] + 2), canonical)
        )
    with pytest.raises(ValueError, match="canonical_shape"):
        graph_policy.validate_adapter_inference_contract(
            adapter(decoder), contract(native, (canonical[0] + 1, canonical[1]))
        )


def test_real_time_retimer_composes_with_uniform_grid_preflight():
    sequence = SequentialActionDecoder([grid(), TimeChunkRetimer(action_horizon=45)])
    graph_policy.validate_adapter_inference_contract(
        adapter(sequence), contract((100, 14), (45, 14))
    )
    with pytest.raises(ValueError, match="canonical_shape"):
        graph_policy.validate_adapter_inference_contract(
            adapter(sequence), contract((100, 14), (50, 14))
        )
