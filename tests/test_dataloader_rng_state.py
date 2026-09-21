import torch

from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper


def _module(seed=42):
    return MultiDataModuleWrapper(
        train_datasets={},
        valid_datasets={},
        train_dataloader_params={},
        valid_dataloader_params={},
        seed=seed,
    )


def test_loader_generators_are_stable_and_source_scoped():
    left = _module()
    right = _module()
    assert torch.equal(
        left._loader_generator("train", "u_socket").get_state(),
        right._loader_generator("train", "u_socket").get_state(),
    )
    assert not torch.equal(
        left._loader_generator("train", "u_socket").get_state(),
        left._loader_generator("train", "chain").get_state(),
    )


def test_loader_generator_state_round_trips_through_datamodule_checkpoint():
    source = _module()
    generator = source._loader_generator("train", "u_socket")
    torch.rand(17, generator=generator)
    checkpoint = source.state_dict()

    restored = _module()
    restored.load_state_dict(checkpoint)
    restored_generator = restored._loader_generator("train", "u_socket")
    assert torch.equal(generator.get_state(), restored_generator.get_state())
    assert torch.equal(
        torch.rand(8, generator=generator),
        torch.rand(8, generator=restored_generator),
    )


def test_loader_checkpoint_rejects_a_different_seed():
    source = _module(seed=42)
    source._loader_generator("train", "u_socket")
    restored = _module(seed=43)
    try:
        restored.load_state_dict(source.state_dict())
    except ValueError as error:
        assert "seed differs" in str(error)
    else:
        raise AssertionError("different loader seed was accepted")
