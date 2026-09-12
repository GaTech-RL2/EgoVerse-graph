"""Human's arc-tokenizer action mode, the counterpart of Yam's."""

import pytest

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.action_chunk_transforms import PadGripperZeros

_ARC = "arc_tokenizer_cartesian_gripper_padded"
_D = 0.40
_M = 100


def _transforms(**kwargs):
    params = dict(
        action_mode=_ARC,
        coord_frame="eef_frame",
        rotation_mode="euler",
        stride=1,
        min_distance_unit=_D,
        resampled_vector_length=_M,
    )
    params.update(kwargs)
    return Human.get_transform_list(**params)


def _tokenizer(transforms):
    found = [t for t in transforms if "ArcLength" in type(t).__name__]
    assert len(found) == 1, f"expected one tokenizer, got {len(found)}"
    return found[0]


# -- the mode exists and is wired -------------------------------------------


def test_arc_mode_appends_exactly_one_tokenizer():
    assert _tokenizer(_transforms())


def test_gripper_padding_runs_before_the_tokenizer():
    """Human has no gripper signal.

    The tokenizer's layout routes gripper into slot 6 per arm, so the zero
    column must already exist when it runs -- otherwise the chunk is 12D and
    the 14D layout cannot be built.
    """
    transforms = _transforms()
    pads = [t for t in transforms if isinstance(t, PadGripperZeros)]
    assert len(pads) == 2  # action chunk and proprio
    assert transforms.index(pads[0]) < transforms.index(_tokenizer(transforms))


def test_bare_arc_cartesian_is_rejected_with_a_pointer_to_the_padded_mode():
    """12D in, 14D needed: fail at config time, not at the first batch."""
    with pytest.raises(ValueError, match="gripper_padded"):
        Human.get_transform_list(action_mode="arc_tokenizer_cartesian")


def test_plain_cartesian_modes_are_untouched():
    for mode in ("cartesian", "cartesian_gripper_padded"):
        transforms = Human.get_transform_list(action_mode=mode, stride=1)
        assert not [t for t in transforms if "ArcLength" in type(t).__name__]


# -- the part that is easy to get wrong: dt tracks stride -------------------


@pytest.mark.parametrize("stride, expected_dt", [(1, 1 / 30), (2, 2 / 30), (3, 3 / 30)])
def test_tokenizer_dt_follows_the_stride(stride, expected_dt):
    """The chunk is subsampled by actions[::stride].

    Consecutive samples are stride/30 s apart, so leaving the tokenizer's 1/30
    default inflates the velocity channel by exactly `stride`. It cancels
    inside tokenize -> detokenize, but it is what the model learns and what a
    deployed policy would command.
    """
    tokenizer = _tokenizer(_transforms(stride=stride))
    assert tokenizer.tokenizer.config.dt == pytest.approx(expected_dt)


def test_yam_keeps_the_unstrided_default_for_contrast():
    from egomimic.rldb.embodiment.yam import Yam

    transforms = Yam.get_transform_list(
        action_mode="arc_tokenizer_cartesian",
        coord_frame="eef_frame",
        min_distance_unit=_D,
        resampled_vector_length=_M,
    )
    assert _tokenizer(transforms).tokenizer.config.dt == pytest.approx(1 / 30)


# -- the raw window ---------------------------------------------------------


def test_arc_keymap_widens_the_raw_action_window():
    """Arc needs room to reach D before the padded tail begins."""
    plain = Human.get_keymap(keymap_mode="cartesian")
    arc = Human.get_keymap(keymap_mode="arc_tokenizer_cartesian")

    def horizons(keymap):
        return {
            v.get("horizon")
            for v in keymap.values()
            if v.get("key_type") == "action_keys"
        }

    assert horizons(plain) == {Human.ACTION_HORIZON}
    assert horizons(arc) == {Human.ARC_TOK_ACTION_HORIZON}
    assert Human.ARC_TOK_ACTION_HORIZON > Human.ACTION_HORIZON


def test_the_arc_keymap_is_otherwise_identical_to_cartesian():
    """Only the horizon differs, so the same transform list works on both."""
    plain = Human.get_keymap(keymap_mode="cartesian")
    arc = Human.get_keymap(keymap_mode="arc_tokenizer_cartesian")
    assert set(plain) == set(arc)
    for key in plain:
        assert plain[key]["zarr_key"] == arc[key]["zarr_key"], key
        assert plain[key]["key_type"] == arc[key]["key_type"], key


def test_arc_does_not_resample_before_tokenizing():
    """chunk_length defaults to the RAW window for arc modes.

    Interpolating to 100 first would decimate the human window, and arc length
    measured on a decimated path reads systematically short.
    """
    from egomimic.rldb.zarr.action_chunk_transforms import InterpolatePose

    transforms = _transforms()
    lengths = {
        t.new_chunk_length
        for t in transforms
        if isinstance(t, InterpolatePose) and hasattr(t, "new_chunk_length")
    }
    assert lengths == {Human.ARC_TOK_ACTION_HORIZON}, lengths


def test_an_explicit_chunk_length_overrides_the_arc_default():
    from egomimic.rldb.zarr.action_chunk_transforms import InterpolatePose

    transforms = _transforms(chunk_length=200)
    lengths = {
        t.new_chunk_length
        for t in transforms
        if isinstance(t, InterpolatePose) and hasattr(t, "new_chunk_length")
    }
    assert lengths == {200}


# -- the configs that were blocked on this ---------------------------------


@pytest.mark.parametrize(
    "experiment",
    [
        "abc_arc/abc_fstshirt_mecka_freefold_cotrain_arcD40M100",
        "abc_arc/abc_mecka_fold_multitask_cotrain_arcD40M100",
    ],
)
def test_the_arc_cotrain_configs_instantiate_their_transform_lists(experiment):
    """These failed outright before Human had an arc mode."""
    from pathlib import Path

    import hydra
    from hydra import compose, initialize_config_dir

    root = str(Path(__file__).resolve().parents[1] / "egomimic/hydra_configs")
    with initialize_config_dir(version_base=None, config_dir=root):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}", "++paths.root_dir=."],
        )
    assert set(cfg.data.train_datasets) == {"human_bimanual", "yam_bimanual"}
    for source in cfg.data.train_datasets:
        transforms = hydra.utils.instantiate(
            cfg.data.train_datasets[source].resolver.transform_list
        )
        assert [t for t in transforms if "ArcLength" in type(t).__name__], source


def test_both_embodiments_agree_on_d_and_m_in_a_cotrain_config():
    """A cotrain run must tokenize both sources the same way."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = str(Path(__file__).resolve().parents[1] / "egomimic/hydra_configs")
    with initialize_config_dir(version_base=None, config_dir=root):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=abc_arc/abc_fstshirt_mecka_freefold_cotrain_arcD40M100",
                "++paths.root_dir=.",
            ],
        )
    specs = [
        cfg.data.train_datasets[src].resolver.transform_list
        for src in cfg.data.train_datasets
    ]
    assert len({s.min_distance_unit for s in specs}) == 1
    assert len({s.resampled_vector_length for s in specs}) == 1


def test_cotrain_arc_row_count_follows_the_velocity_mode():
    """101 (M+1, odd) crashed ConditionalUnet1D's skip concat.

    Two downsample stages need the row count divisible by four, so the
    per_waypoint layout's 2*M = 200 is what makes these configs trainable at
    all. The model must read the knob rather than a hardcoded literal.
    """
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from egomimic.rldb.zarr.arc_length_tokenizer import bimanual_arc_token_rows

    root = str(Path(__file__).resolve().parents[1] / "egomimic/hydra_configs")
    for experiment in (
        "abc_arc/abc_fstshirt_mecka_freefold_cotrain_arcD40M100",
        "abc_arc/abc_mecka_fold_multitask_cotrain_arcD40M100",
    ):
        with initialize_config_dir(version_base=None, config_dir=root):
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=[f"+experiment={experiment}", "++paths.root_dir=."],
            )
        mode = cfg.abc.arc_velocity_mode
        specs = [
            cfg.data.train_datasets[src].resolver.transform_list
            for src in cfg.data.train_datasets
        ]
        # every source tokenizes the same way
        assert {OmegaConf.select(s, "velocity_mode") for s in specs} == {mode}
        expected = bimanual_arc_token_rows(int(specs[0].resampled_vector_length), mode)
        assert cfg.abc.arc_token_rows == expected, experiment
        assert expected % 4 == 0, "ConditionalUnet1D needs rows divisible by four"
        horizons = {
            int(s.action_horizon)
            for s in cfg.model.pipeline.stages
            if "action_horizon" in s
        }
        assert horizons == {expected}, (experiment, horizons)


def test_the_cotrain_arc_unet_accepts_the_token_width():
    """The check that 101 failed: run a real tensor through the denoiser."""
    from pathlib import Path

    import hydra
    import torch
    from hydra import compose, initialize_config_dir

    root = str(Path(__file__).resolve().parents[1] / "egomimic/hydra_configs")
    with initialize_config_dir(version_base=None, config_dir=root):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=abc_arc/abc_fstshirt_mecka_freefold_cotrain_arcD40M100",
                "++paths.root_dir=.",
            ],
        )
    algo = hydra.utils.instantiate(cfg.model.pipeline)
    denoiser = next(
        s for s in algo.pipeline.stages if type(s).__name__ == "DiffusionDenoiserStage"
    )
    rows = int(cfg.abc.arc_token_rows)
    out = denoiser.policy.model(
        torch.zeros(2, rows, denoiser.action_dim),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, denoiser.condition_input_dim),
    )
    assert tuple(out.shape) == (2, rows, denoiser.action_dim)


def test_original_abc_cotrain_study_shares_one_arcmatch_configuration():
    """All six arms must score in the same space or none of it is comparable.

    This is the regression that shipped twice: the two cotrain baselines kept
    the plain evaluator (no arc metrics at all), and the two arc cotrains had
    `evaluator: null`, so four of six runs produced zero arcmatch keys while
    looking like they had succeeded.
    """
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
    experiments = sorted(
        f"abc_arc/{p.stem}" for p in (root / "experiment/abc_arc").glob("abc_*.yaml")
        # Stationery is a separate source campaign with controller-facing
        # reconstruction metrics, covered by test_robot_arc_campaigns.py.
        if not p.stem.startswith("abc_stationery_")
    )
    assert len(experiments) == 10, experiments

    settings = {}
    for experiment in experiments:
        with initialize_config_dir(version_base=None, config_dir=str(root)):
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=[f"+experiment={experiment}", "++paths.root_dir=."],
            )
        evaluator = cfg.evaluator
        assert evaluator is not None, f"{experiment} has no evaluator"
        assert evaluator._target_.endswith("ArcBimanualCartesianEval"), experiment
        assert evaluator.arc_metrics is True, experiment
        assert evaluator.include_reconstruction_loss is False, experiment
        settings[experiment] = (
            float(evaluator.min_distance_unit),
            int(evaluator.resampled_vector_length),
            int(evaluator.arcmatch_points),
            int(evaluator.arc_chunk_rows),
            str(evaluator.velocity_mode),
            bool(evaluator.include_reconstruction_loss),
        )
    # One shared tuple across all arms.
    assert len(set(settings.values())) == 1, settings
