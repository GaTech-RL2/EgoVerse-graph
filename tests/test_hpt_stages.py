"""HPT stems and trunk as graph stages."""

from functools import partial
import random

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.models.cores.hpt_transformer import MultiheadAttention, SimpleTransformer
from egomimic.models.stems.hpt_stems import MLPPolicyStem, PolicyStem
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_hpt import (
    AnnotationPromptStage,
    HPTStemStage,
    HPTTrunkStage,
    sample_annotation_prompt,
)

_DIM = 64
_LATENTS = 8
_POSE_KEY = "observations.state.ee_pose"
_IMG_KEY = "observations.images.front_img_1"


def _spec(embed_dim: int = _DIM, latents: int = _LATENTS):
    return OmegaConf.create(
        {
            "cross_attn": {
                "crossattn_latent": latents,
                "crossattn_heads": 4,
                "crossattn_dim_head": 16,
                "crossattn_modality_dropout": 0.0,
                "modality_embed_dim": embed_dim,
            }
        }
    )


def _stem(input_dim: int, embed_dim: int = _DIM, latents: int = _LATENTS):
    return MLPPolicyStem(
        input_dim=input_dim,
        output_dim=embed_dim,
        widths=[embed_dim],
        specs=_spec(embed_dim, latents),
    )


def _trunk(embed_dim: int = _DIM, num_blocks: int = 2):
    return SimpleTransformer(
        attn_target=partial(
            MultiheadAttention, embed_dim=embed_dim, num_heads=4, batch_first=True
        ),
        embed_dim=embed_dim,
        num_blocks=num_blocks,
    )


# -- stems ------------------------------------------------------------------


def test_stem_stage_declares_its_modalities_as_reads():
    stage = HPTStemStage(stems={_POSE_KEY: _stem(14)})
    assert stage.contract("train") == ((_POSE_KEY,), ("hpt/tokens",))


def test_stem_stage_compresses_a_modality_to_the_configured_latents():
    stage = HPTStemStage(stems={_POSE_KEY: _stem(14)})
    out = stage.forward({_POSE_KEY: torch.zeros(4, 14)})
    assert out["hpt/tokens"].shape == (4, _LATENTS, _DIM)


def test_stem_stage_concatenates_modalities_along_the_token_axis():
    # The point of the cross-attention pooling: a 14-D pose and a 512-D image
    # feature both become `latents` tokens of the same width.
    stage = HPTStemStage(stems={_POSE_KEY: _stem(14), _IMG_KEY: _stem(512)})
    out = stage.forward({_POSE_KEY: torch.zeros(3, 14), _IMG_KEY: torch.zeros(3, 512)})
    assert out["hpt/tokens"].shape == (3, 2 * _LATENTS, _DIM)


def test_stem_stage_order_is_sorted_not_dict_insertion():
    """Token layout must not depend on dict ordering, or DDP ranks diverge."""
    forward = HPTStemStage(stems={_POSE_KEY: _stem(14), _IMG_KEY: _stem(512)})
    reverse = HPTStemStage(stems={_IMG_KEY: _stem(512), _POSE_KEY: _stem(14)})
    assert (
        forward.shared_keys
        == reverse.shared_keys
        == tuple(sorted([_POSE_KEY, _IMG_KEY]))
    )


def test_stem_stage_handles_dotted_batch_keys():
    # ModuleDict rejects "." so the stage sanitizes; the reads must still be
    # the real batch keys.
    stage = HPTStemStage(stems={_POSE_KEY: _stem(14)})
    assert _POSE_KEY in stage.reads
    assert stage.forward({_POSE_KEY: torch.zeros(2, 14)})["hpt/tokens"].shape[0] == 2


def test_stem_stage_rejects_stems_that_disagree_on_width():
    stage = HPTStemStage(
        stems={_POSE_KEY: _stem(14, embed_dim=_DIM), _IMG_KEY: _stem(512, embed_dim=32)}
    )
    with pytest.raises(ValueError, match="disagree on embed dim"):
        stage.forward({_POSE_KEY: torch.zeros(2, 14), _IMG_KEY: torch.zeros(2, 512)})


def test_stem_stage_needs_at_least_one_stem():
    with pytest.raises(ValueError, match="at least one stem"):
        HPTStemStage(stems={})


def test_stem_stage_accepts_a_plain_dict_cross_attn_spec():
    # Hydra gives DictConfig, hand-built stems may give dict; both must work.
    stem = MLPPolicyStem(
        input_dim=14,
        output_dim=_DIM,
        widths=[_DIM],
        specs={
            "cross_attn": {
                "crossattn_latent": 4,
                "crossattn_heads": 2,
                "crossattn_dim_head": 8,
                "crossattn_modality_dropout": 0.0,
                "modality_embed_dim": _DIM,
            }
        },
    )
    out = HPTStemStage(stems={_POSE_KEY: stem}).forward({_POSE_KEY: torch.zeros(2, 14)})
    assert out["hpt/tokens"].shape == (2, 4, _DIM)


# -- per-domain stems -------------------------------------------------------


def test_domain_stems_add_tokens_and_read_the_selector():
    stage = HPTStemStage(
        stems={_POSE_KEY: _stem(14)},
        domain_stems={"yam_bimanual": {_IMG_KEY: _stem(512)}},
        selector_aliases={"7": "yam_bimanual"},
    )
    assert "embodiment" in stage.reads
    out = stage.forward(
        {_POSE_KEY: torch.zeros(2, 14), _IMG_KEY: torch.zeros(2, 512), "embodiment": 7}
    )
    assert out["hpt/tokens"].shape == (2, 2 * _LATENTS, _DIM)


def test_domain_stems_reject_an_unconfigured_domain():
    stage = HPTStemStage(
        stems={_POSE_KEY: _stem(14)},
        domain_stems={"yam_bimanual": {_IMG_KEY: _stem(512)}},
    )
    with pytest.raises(KeyError, match="no stems for domain"):
        stage.forward(
            {
                _POSE_KEY: torch.zeros(2, 14),
                _IMG_KEY: torch.zeros(2, 512),
                "embodiment": 3,
            }
        )


def test_a_batched_selector_resolves_to_one_domain():
    stage = HPTStemStage(
        stems={_POSE_KEY: _stem(14)},
        domain_stems={"yam_bimanual": {_IMG_KEY: _stem(512)}},
        selector_aliases={"7": "yam_bimanual"},
    )
    out = stage.forward(
        {
            _POSE_KEY: torch.zeros(2, 14),
            _IMG_KEY: torch.zeros(2, 512),
            "embodiment": torch.tensor([7, 7]),
        }
    )
    assert out["hpt/tokens"].shape[1] == 2 * _LATENTS


# -- trunk ------------------------------------------------------------------


def test_trunk_stage_writes_the_same_condition_key_the_heads_read():
    stage = HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM)
    assert stage.contract("train") == (("hpt/tokens",), ("condition",))


@pytest.mark.parametrize("pooling", ["action_token", "mean", "last"])
def test_trunk_stage_pools_to_one_vector_per_sample(pooling):
    stage = HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM, token_postprocessing=pooling)
    out = stage.forward({"hpt/tokens": torch.zeros(5, _LATENTS, _DIM)})
    assert out["condition"].shape == (5, _DIM)


def test_trunk_stage_rejects_an_unknown_pooling():
    with pytest.raises(ValueError, match="token_postprocessing"):
        HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM, token_postprocessing="cls")


def test_action_token_pooling_prepends_a_learned_token():
    stage = HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM)
    assert stage.action_token is not None and stage.action_token.requires_grad
    # mean pooling has no such parameter
    assert (
        HPTTrunkStage(
            trunk=_trunk(), embed_dim=_DIM, token_postprocessing="mean"
        ).action_token
        is None
    )


def test_trunk_stage_rejects_a_width_mismatch_with_the_stems():
    stage = HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM)
    with pytest.raises(ValueError, match="embed_dim is"):
        stage.forward({"hpt/tokens": torch.zeros(2, _LATENTS, _DIM * 2)})


def test_trunk_stage_rejects_a_non_token_tensor():
    stage = HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM)
    with pytest.raises(ValueError, match=r"\(B, tokens,"):
        stage.forward({"hpt/tokens": torch.zeros(2, _DIM)})


def test_position_embedding_is_added_and_bounded_by_max_tokens():
    stage = HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM, max_tokens=_LATENTS + 1)
    stage.forward({"hpt/tokens": torch.zeros(2, _LATENTS, _DIM)})
    with pytest.raises(ValueError, match="max_tokens"):
        stage.forward({"hpt/tokens": torch.zeros(2, _LATENTS * 4, _DIM)})


def test_position_embedding_can_be_disabled():
    stage = HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM, use_position_embedding=False)
    assert stage.position_embedding is None
    assert stage.forward({"hpt/tokens": torch.zeros(2, _LATENTS, _DIM)})[
        "condition"
    ].shape


def test_position_embedding_is_not_a_trained_parameter():
    stage = HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM)
    assert "position_embedding" not in dict(stage.named_parameters())


def test_domain_embedding_shifts_the_condition_per_domain():
    torch.manual_seed(0)
    stage = HPTTrunkStage(
        trunk=_trunk(),
        embed_dim=_DIM,
        domains=["yam_bimanual", "human_bimanual"],
        use_domain_embedding=True,
        selector_aliases={"7": "yam_bimanual", "3": "human_bimanual"},
    ).eval()
    tokens = torch.randn(1, _LATENTS, _DIM)
    with torch.no_grad():
        a = stage.forward({"hpt/tokens": tokens, "embodiment": 7})["condition"]
        b = stage.forward({"hpt/tokens": tokens, "embodiment": 3})["condition"]
    assert not torch.allclose(a, b), "domain embedding had no effect"


def test_domain_embedding_requires_a_domain_list():
    with pytest.raises(ValueError, match="non-empty domains"):
        HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM, use_domain_embedding=True)


def test_domain_embedding_rejects_an_unlisted_domain():
    stage = HPTTrunkStage(
        trunk=_trunk(),
        embed_dim=_DIM,
        domains=["yam_bimanual"],
        use_domain_embedding=True,
    )
    with pytest.raises(KeyError, match="no domain embedding"):
        stage.forward({"hpt/tokens": torch.zeros(1, _LATENTS, _DIM), "embodiment": 99})


# -- the two together in a graph -------------------------------------------


def test_stems_and_trunk_compose_into_a_runnable_subgraph():
    pipeline = Pipeline(
        [
            HPTStemStage(stems={_POSE_KEY: _stem(14)}),
            HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM),
        ]
    )
    runnable, excluded = pipeline.plan([_POSE_KEY], mode="train")
    assert len(runnable) == 2 and not excluded
    out = pipeline.execute({_POSE_KEY: torch.zeros(2, 14)}, mode="train")
    assert out["condition"].shape == (2, _DIM)


def test_the_subgraph_runs_in_inference_too():
    """Neither stage is mode-restricted: HPT encodes identically either way."""
    pipeline = Pipeline(
        [
            HPTStemStage(stems={_POSE_KEY: _stem(14)}),
            HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM),
        ]
    )
    out = pipeline.execute({_POSE_KEY: torch.zeros(2, 14)}, mode="inference")
    assert out["condition"].shape == (2, _DIM)


def test_a_missing_modality_blocks_the_graph_rather_than_running_partially():
    pipeline = Pipeline(
        [HPTStemStage(stems={_POSE_KEY: _stem(14), _IMG_KEY: _stem(512)})]
    )
    with pytest.raises(RuntimeError, match="blocked stages"):
        pipeline.execute({_POSE_KEY: torch.zeros(2, 14)}, mode="train")


def test_condition_is_differentiable_back_to_the_stems():
    pipeline = Pipeline(
        [
            HPTStemStage(stems={_POSE_KEY: _stem(14)}),
            HPTTrunkStage(trunk=_trunk(), embed_dim=_DIM),
        ]
    )
    out = pipeline.execute({_POSE_KEY: torch.randn(2, 14)}, mode="train")
    out["condition"].sum().backward()
    grads = [
        p.grad for p in pipeline.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads, "no gradient reached the stems or trunk"


# -- dtype robustness -------------------------------------------------------
#
# Zarr stamps poses as float64 and torch.from_numpy keeps that. Autocast never
# downcasts Double, so a Double input reaches a stem's Linear unchanged and the
# matmul raises. The stage casts to its own parameter dtype rather than relying
# on an upstream normalisation that a given branch may not carry.


def test_stem_stage_accepts_a_float64_modality():
    stage = HPTStemStage(stems={_POSE_KEY: _stem(14)})
    out = stage.forward({_POSE_KEY: torch.zeros(2, 14, dtype=torch.float64)})
    assert out["hpt/tokens"].shape == (2, _LATENTS, _DIM)
    assert out["hpt/tokens"].dtype == torch.float32


def test_stem_stage_leaves_a_matching_dtype_untouched():
    stage = HPTStemStage(stems={_POSE_KEY: _stem(14)})
    out = stage.forward({_POSE_KEY: torch.zeros(2, 14, dtype=torch.float32)})
    assert out["hpt/tokens"].dtype == torch.float32


def test_stem_stage_handles_mixed_input_dtypes_across_modalities():
    # The real ABC batch: float32 images beside a float64 pose.
    stage = HPTStemStage(stems={_POSE_KEY: _stem(14), _IMG_KEY: _stem(512)})
    out = stage.forward(
        {
            _POSE_KEY: torch.zeros(2, 14, dtype=torch.float64),
            _IMG_KEY: torch.zeros(2, 512, dtype=torch.float32),
        }
    )
    assert out["hpt/tokens"].shape == (2, 2 * _LATENTS, _DIM)


def test_a_half_precision_stem_casts_its_input_to_match():
    stage = HPTStemStage(stems={_POSE_KEY: _stem(14)}).half()
    out = stage.forward({_POSE_KEY: torch.zeros(2, 14, dtype=torch.float32)})
    assert out["hpt/tokens"].dtype == torch.float16


# -- language prompts -------------------------------------------------------


_ANN_KEY = "annotations"
_PROMPT_KEY = "observations.annotation"


class _ListStem(PolicyStem):
    """Stand-in for a text stem: consumes list[str], no HuggingFace download."""

    def __init__(self, embed_dim: int = _DIM, latents: int = _LATENTS):
        super().__init__(specs=_spec(embed_dim, latents))
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, prompts):
        return torch.ones(len(prompts), 1, _DIM)

    def compute_latent(self, prompts):
        feat = self(prompts)
        stem_tokens = self.tokens.repeat(feat.shape[0], 1, 1)
        return self.cross_attention(stem_tokens, feat)


def test_sample_annotation_prompt_falls_back_when_empty():
    assert sample_annotation_prompt([], default_prompt="idle") == "idle"
    assert sample_annotation_prompt(["", "  "], default_prompt="idle") == "idle"


def test_sample_annotation_prompt_takes_first_at_eval():
    item = ["fold the shirt", "grab the sleeve"]
    assert (
        sample_annotation_prompt(item, sampling_mode="random", training=False)
        == "fold the shirt"
    )
    assert sample_annotation_prompt(item, sampling_mode="first") == "fold the shirt"


def test_sample_annotation_prompt_can_be_seeded():
    item = ["a", "b", "c"]
    rng = random.Random(0)
    picked = {
        sample_annotation_prompt(item, training=True, rng=rng) for _ in range(20)
    }
    assert picked <= set(item)
    assert len(picked) > 1


def test_annotation_prompt_stage_writes_one_string_per_sample():
    stage = AnnotationPromptStage(sampling_mode="first")
    assert stage.contract("train") == ((_ANN_KEY,), (_PROMPT_KEY,))
    out = stage.execute(
        {_ANN_KEY: [["fold the shirt", "smooth"], [], "already a string"]},
        mode="train",
    )
    assert out[_PROMPT_KEY] == ["fold the shirt", "", "already a string"]


def test_annotation_prompt_stage_is_deterministic_in_inference():
    stage = AnnotationPromptStage(sampling_mode="random")
    batch = {_ANN_KEY: [["aaa", "bbb"], ["ccc", "ddd"]]}
    first = stage.execute(dict(batch), mode="inference")[_PROMPT_KEY]
    second = stage.execute(dict(batch), mode="inference")[_PROMPT_KEY]
    assert first == second == ["aaa", "ccc"]


def test_stem_stage_routes_list_valued_prompts_without_dtype_cast():
    stage = HPTStemStage(stems={_PROMPT_KEY: _ListStem()})
    out = stage.forward({_PROMPT_KEY: ["fold the shirt", "smooth the hem"]})
    assert out["hpt/tokens"].shape == (2, _LATENTS, _DIM)


def test_prompt_stage_and_list_stem_compose():
    pipeline = Pipeline(
        [
            AnnotationPromptStage(sampling_mode="first"),
            HPTStemStage(stems={_PROMPT_KEY: _ListStem()}),
        ]
    )
    out = pipeline.execute(
        {_ANN_KEY: [["fold the shirt"], ["pick up the cup"]]}, mode="train"
    )
    assert out["hpt/tokens"].shape == (2, _LATENTS, _DIM)


def test_domain_only_keys_are_not_required_on_every_source():
    """Yam wrists must not block a human batch that has no wrist cameras."""
    stage = HPTStemStage(
        stems={_POSE_KEY: _stem(14)},
        domain_stems={
            "yam_bimanual": {_IMG_KEY: _stem(512)},
            "human_bimanual": {"observations.state.gripper": _stem(2)},
        },
        selector_aliases={"7": "yam_bimanual", "3": "human_bimanual"},
    )
    assert _IMG_KEY not in stage.reads
    assert "embodiment" in stage.reads
    pipeline = Pipeline([stage])
    out = pipeline.execute(
        {
            _POSE_KEY: torch.zeros(2, 14),
            "observations.state.gripper": torch.zeros(2, 2),
            "embodiment": 3,
        },
        mode="train",
    )
    assert out["hpt/tokens"].shape == (2, 2 * _LATENTS, _DIM)
