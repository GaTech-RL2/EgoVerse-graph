from egomimic.rldb.zarr.proportional_batch_sampler import (
    ProportionalHomogeneousBatchSampler,
)


def test_every_window_once_and_homogeneous():
    sampler = ProportionalHomogeneousBatchSampler(
        {"yam": 17, "human": 11}, batch_size=4, seed=42
    )
    batches = list(sampler)
    assert len(batches) == len(sampler) == 8
    assert sorted(i for batch in batches for i in batch) == list(range(28))
    assert all(
        len({sampler.source_for_index(i) for i in batch}) == 1 for batch in batches
    )
    assert (
        sum(sampler.source_for_index(i) == "yam" for batch in batches for i in batch)
        == 17
    )
    assert (
        sum(sampler.source_for_index(i) == "human" for batch in batches for i in batch)
        == 11
    )
    assert sorted(len(batch) for batch in batches) == [1, 3, 4, 4, 4, 4, 4, 4]


def test_epoch_replay_and_seed_change():
    sampler = ProportionalHomogeneousBatchSampler(
        {"yam": 17, "human": 11}, batch_size=4, seed=42
    )
    first = list(sampler)
    second = list(sampler)
    assert first != second
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_rejects_invalid_contract():
    for lengths, batch in [
        ({"yam": 2}, 2),
        ({"yam": 2, "human": 0}, 2),
        ({"yam": 2, "human": 2}, 0),
    ]:
        try:
            ProportionalHomogeneousBatchSampler(lengths, batch)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mixture was accepted")


def test_resume_replays_active_epoch_for_lightning_batch_skip():
    original = ProportionalHomogeneousBatchSampler({"yam": 17, "human": 11}, 4, 42)
    iterator = iter(original)
    consumed = [next(iterator), next(iterator)]
    state = original.state_dict()
    resumed = ProportionalHomogeneousBatchSampler({"yam": 17, "human": 11}, 4, 42)
    resumed.load_state_dict(state)
    replay = iter(resumed)
    assert [next(replay), next(replay)] == consumed
    assert list(replay) == list(iterator)
    try:
        ProportionalHomogeneousBatchSampler(
            {"yam": 17, "human": 10}, 4, 42
        ).load_state_dict(state)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed source lengths were accepted on resume")
