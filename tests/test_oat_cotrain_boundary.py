"""Native routed observations preserve YAM and cannot mix embodiment batches."""

import pytest
import torch

from egomimic.pipeline.stages_oat import (
    EmbodimentOATObservationStage,
    OATObservationStage,
)


class Encoder(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))

    def forward(self, obs):
        return obs["state"] * self.weight


def stage():
    return EmbodimentOATObservationStage(
        branches={
            "yam": OATObservationStage(Encoder(6), ["robot"], {"robot": "state"}),
            "human": OATObservationStage(Encoder(4), ["hand"], {"hand": "state"}),
        },
        condition_dim=6,
        selector_aliases={"7": "yam", "3": "human"},
    )


@pytest.mark.parametrize("identity,key,width", [(7, "robot", 6), (3, "hand", 4)])
def test_route_padding_and_private_gradient(identity, key, width):
    s = stage()
    x = torch.randn(2, width)
    out = s({"embodiment": torch.tensor([identity] * 2), key: x})["condition"]
    torch.testing.assert_close(out[:, :width], x)
    assert torch.equal(out[:, width:], torch.zeros(2, 6 - width))
    out.sum().backward()
    active = "yam" if identity == 7 else "human"
    assert s.branches[active].encoder.weight.grad is not None
    assert s.branches["human" if active == "yam" else "yam"].encoder.weight.grad is None


def test_reject_mixed_batch_and_unknown_source():
    s = stage()
    with pytest.raises(ValueError, match="homogeneous"):
        s({"embodiment": torch.tensor([3, 7])})
    with pytest.raises(ValueError, match="No OAT observation"):
        s({"embodiment": torch.tensor([99, 99])})


def test_reject_condition_truncation():
    s = stage()
    s.condition_dim = 3
    with pytest.raises(ValueError, match="exceed"):
        s({"embodiment": torch.tensor([7, 7]), "robot": torch.ones(2, 6)})


def test_human_horizon_is_typed_and_metadata_matches():
    from egomimic.eval.yam_human_keypoints_action_flow_eval import (
        YamHumanKeypointsActionFlowEval,
    )

    with pytest.raises(ValueError, match="Human action horizon"):
        YamHumanKeypointsActionFlowEval(human_action_horizon=0)
    # An invalid stochastic chunk fails before accessing normalization state.
    evaluator = object.__new__(YamHumanKeypointsActionFlowEval)
    evaluator.human_action_horizon = 32
    with pytest.raises(ValueError, match="configured-horizon"):
        evaluator._energy_values(
            torch.zeros(32, 1, 100, 138), torch.zeros(1, 100, 138), 3, "human_bimanual"
        )
