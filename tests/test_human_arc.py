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
