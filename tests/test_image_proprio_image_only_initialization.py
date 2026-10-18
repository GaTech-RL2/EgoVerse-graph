"""Image-only -> two-expert readout migration with explicit CPU execution proof."""

import pytest
import torch

from egomimic.models.image_proprio import ImageProprioModel
from scripts.world_model.initialize_pixel_readout import (
    initialize_readout,
    sha256_file,
    transfer_shared_weights,
    validate_transition,
    verify_phase_a,
)
from tests.test_image_proprio_image_only import joint_batch


def configs(mode="continuous", modality=False, backend="sdpa"):
    source = dict(
        objective="visual_only",
        phase_a_condition_on_proprio=False,
        image_dim=12,
        width=16,
        heads=2,
        depth=2,
        max_cameras=2,
        max_tokens=12,
        time_mode=mode,
        use_modality_embedding=modality,
        attention_backend=backend,
        attention_dtype="bfloat16" if backend == "flash4" else "native",
    )
    target = dict(
        source,
        objective="visual_state_action",
        max_tokens=16,
        pose_dims={"gripper": 9 if mode == "continuous" else 36},
        action_dims={"gripper": 7 if mode == "continuous" else 28},
    )
    return source, target


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
@pytest.mark.parametrize("modality", [False, True])
def test_image_weights_copy_exactly_and_added_branch_keeps_seeded_initialization(
    mode, modality
):
    source_config, target_config = configs(mode, modality)
    validate_transition(source_config, target_config)
    source, target = (
        ImageProprioModel(**source_config),
        ImageProprioModel(**target_config),
    )
    initialized = {key: value.clone() for key, value in target.state_dict().items()}
    transfer = transfer_shared_weights(source, target)
    assert "proprio_role.weight" in transfer["new_tensors"]
    assert "pose_in.gripper.weight" in transfer["new_tensors"]
    assert "blocks.0.qkv.1.weight" in transfer["new_tensors"]
    for key in transfer["new_tensors"]:
        assert torch.equal(target.state_dict()[key], initialized[key])
    for key, value in source.state_dict().items():
        actual = target.state_dict()[key]
        if key in transfer["expanded_embeddings"]:
            actual = actual[: value.shape[0]]
        assert torch.equal(value, actual)
    proof = verify_phase_a(source, target)
    assert proof["bit_exact"] and len(proof["checks"]) == 2
    assert not proof["gpu_kernel_verified"]


def test_transition_cannot_enable_phase_a_proprio_or_relabel_source_architecture():
    source, target = configs()
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_transition(source, {**target, "phase_a_condition_on_proprio": True})
    with pytest.raises(ValueError, match="omit pose_dims"):
        validate_transition({**source, "pose_dims": {"gripper": 9}}, target)
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_transition(source, {**target, "image_dim": 8})


def test_full_twenty_step_generation_and_image_features_are_preserved():
    source_config, target_config = configs()
    source_config["image_dim"] = target_config["image_dim"] = 8
    target_config["pose_dims"] = {"gripper": 3}
    target_config["action_dims"] = {"gripper": 2}
    source, target = (
        ImageProprioModel(**source_config),
        ImageProprioModel(**target_config),
    )
    transfer_shared_weights(source, target)
    batch = joint_batch()
    image_features, target_features = [], []
    handles = [
        source.blocks[-1].register_forward_hook(
            lambda module, args, output: image_features.append(output[0].clone())
        ),
        target.blocks[-1].register_forward_hook(
            lambda module, args, output: target_features.append(output[0].clone())
        ),
    ]
    try:
        original = source.sample(
            {"image": batch["image"]},
            "gripper",
            steps=20,
            generator=torch.Generator().manual_seed(17),
        )
        migrated = target.sample(
            batch,
            "gripper",
            steps=20,
            generator=torch.Generator().manual_seed(17),
        )
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(original["image"], migrated["image"])
    assert len(image_features) == 20 and len(target_features) == 21
    assert all(torch.equal(a, b) for a, b in zip(image_features, target_features[:20]))


def test_fa4_cpu_probe_requires_declared_override_and_restores_model_execution():
    source_config, target_config = configs(backend="flash4")
    source, target = (
        ImageProprioModel(**source_config),
        ImageProprioModel(**target_config),
    )
    transfer_shared_weights(source, target)
    with pytest.raises(ValueError, match="explicit cpu_attention_override"):
        verify_phase_a(source, target)
    proof = verify_phase_a(source, target, cpu_attention_override="native")
    assert proof["bit_exact"] and proof["cpu_attention_override"] == "native"
    assert proof["declared_attention"] == ["flash4/bfloat16"]
    assert not proof["gpu_kernel_verified"]
    assert all(
        block.attention_backend == "flash4"
        for model in (source, target)
        for block in model.blocks
    )
    with torch.no_grad():
        target.image_out.bias.add_(1)
    with pytest.raises(ValueError, match="preservation failed"):
        verify_phase_a(source, target, cpu_attention_override="native")
    assert all(
        block.attention_backend == "flash4"
        for model in (source, target)
        for block in model.blocks
    )


def test_offline_fa4_artifact_binds_original_configs_and_records_only_cpu_proof(
    tmp_path,
):
    source_config, target_config = configs(backend="flash4")
    source = ImageProprioModel(**source_config)
    path = tmp_path / "source.pt"
    torch.save(source.state_dict(), path)
    kwargs = dict(
        source_checkpoint=path,
        source_sha256=sha256_file(path),
        source_config=source_config,
        target_config=target_config,
        output_path=tmp_path / "target.pt",
        receipt_path=tmp_path / "receipt.json",
        seed=42,
    )
    with pytest.raises(ValueError, match="explicit cpu_attention_override"):
        initialize_readout(**kwargs)
    assert not kwargs["output_path"].exists()
    rng_before = torch.random.get_rng_state().clone()
    receipt = initialize_readout(**kwargs, cpu_attention_override="native")
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert receipt["source_model"]["attention_backend"] == "flash4"
    assert receipt["target_model"]["attention_backend"] == "flash4"
    assert receipt["phase_a_preservation"]["cpu_attention_override"] == "native"
    assert not receipt["phase_a_preservation"]["gpu_kernel_verified"]
    target = ImageProprioModel(**target_config)
    target.load_state_dict(
        torch.load(kwargs["output_path"], weights_only=True), strict=True
    )
    assert verify_phase_a(source, target, "native")["bit_exact"]
    assert not receipt["optimizer_state_transferred"]
    assert not receipt["training_step_transferred"]


def test_unknown_added_parameter_cannot_be_silently_initialized():
    source_config, target_config = configs()
    source, target = (
        ImageProprioModel(**source_config),
        ImageProprioModel(**target_config),
    )
    target.extra_parameter = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="Unexpected parameter namespace"):
        transfer_shared_weights(source, target)
