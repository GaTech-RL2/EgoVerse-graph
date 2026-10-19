"""Shared, checkout-local builders for the committed stationery recipes.

The episode manifests are input authority. Model leaves inherit the maintained
graphs instead of copying neural network YAML or writing another checkout.
"""

import json
import math
import os
import re
import socket
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "egomimic/hydra_configs"
ROTATION_DISTANCE_UNIT = 2 * math.pi


def require_compute_node():
    """Reject direct Python execution on known PACE login hosts."""
    host = socket.gethostname().lower()
    pace = any(x in host for x in ("phoenix", "pace", "ice", "atl1")) or bool(
        os.environ.get("PACE_CLUSTER")
    )
    if pace and "login" in host and not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError(
            "PACE login nodes are orchestration-only; run this builder through srun"
        )


def compose_recipe(name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(
            "train_zarr_cartesian", overrides=[f"+experiment=yam_arc_grid/{name}"]
        )


def robot_data(pool, variant):
    """Resolve inheritance before changing the codec on the selected robot pool."""
    base = "time" if variant == "time" else "arcdur"
    cfg = compose_recipe(f"scratch_rl2_stattempo_{pool}_{base}")
    data = OmegaConf.to_container(cfg.data, resolve=True)
    data.update(
        _target_="egomimic.rldb.zarr.data_module.ZarrDataModule",
        _recursive_=False,
        source_fps=30.0,
    )
    for group in ("train_datasets", "valid_datasets"):
        resolver = data[group]["yam_bimanual"]["resolver"]
        resolver["key_map"].update(drop_wrist_images=False, yam_source_frames=100)
        transforms = resolver["transform_list"]
        transforms.update(
            variant=variant,
            progress_smooth_hz=None,
            fixed_spacing=False,
            yam_source_frames=100,
        )
        if variant in ("arcdurhyb", "arcvelhyb"):
            transforms["rotation_distance_unit"] = ROTATION_DISTANCE_UNIT
    return data


def codec_overrides(
    variant, *, human=False, rotation_distance_unit=ROTATION_DISTANCE_UNIT
):
    """Only the transform fields that differ from the inherited TIME data."""
    transforms = {"variant": variant}
    if variant.endswith(("hyb", "tri", "evt", "ref")):
        transforms["rotation_distance_unit"] = rotation_distance_unit
    result = {
        group: {"yam_bimanual": {"resolver": {"transform_list": dict(transforms)}}}
        for group in ("train_datasets", "valid_datasets")
    }
    if human:
        result["train_datasets"]["human_bimanual"] = {
            "resolver": {"transform_list": dict(transforms)}
        }
    return result


def robot_data_config(pool, variant):
    """An inheriting YAML leaf, never a serialized resolved data tree."""
    return {
        "defaults": [
            f"/data/abc_arc/stationery_tempo_{pool}_time@_here_",
            "_self_",
        ],
        **(codec_overrides(variant) if variant != "time" else {}),
    }


def write_config(relative, cfg, header=""):
    path = CONFIGS / relative
    if not path.resolve().is_relative_to(CONFIGS.resolve()):
        raise ValueError(
            "A recipe builder may only write this checkout's Hydra configs"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + OmegaConf.to_yaml(OmegaConf.create(cfg)))
    print("wrote", path.relative_to(REPO))


def model_leaf(name, parent, **overrides):
    write_config(
        f"model/e1/{name}.yaml",
        {"defaults": [f"/model/e1/{parent}@_here_", "_self_"], **overrides},
        "# Shared graph inheritance preserves the observation encoder and training recipe.\n",
    )


# Slow-pace 366 (Elmo + Aidan robot pool incl. Elmo's 2026-10-04 batch, shared 24-episode val, 38 RL2 Aria
# episodes): the pool of the 10-05 slowpace366 runs, frozen in stationery_slowpace366_manifest.json.
SLOWPACE366_MANIFEST = REPO / "scripts/e1/stationery_slowpace366_manifest.json"
HYBRID_LAYOUTS = {
    "arcdurhyb": "e1_durhyb",
    "arcvelhyb": "e1_profhyb",
    "arcdurtri": "e1_durtri",
    "arcveltri": "e1_proftri",
}
# Event-stream variants (Ryan's stream codec with an event-list gripper): robot-only BC, own model leaf.
EVT_LAYOUTS = {"arcdurevt": "e1_durevt", "arcvelevt": "e1_stkevt"}
# Reference two-clock variants (Ryan's LiberoArcTimedCodec token with rotation6d, (M, 24)): robot-only BC, own model
# leaf, and normalization like his runs: fixed physical ranges instead of fitted quantiles, sampled under a clip.
REF_LAYOUTS = {"arcdurref": "e1_durref", "arcvelref": "e1_stkref"}
# The fixed ranges (norm_stats.fixed_ranges). rotation6d is in [-1, 1] and the gripper in [0, 1] by construction; the
# others bound the largest value on the slowpace366 validation windows and a 20 % training sample (2026-10-11):
# xyz 0.4525 / 0.5823 m, an interval 3.2667 s (it cannot exceed 99 frames), 1.826 m/s, 8.188 rad/s.
REF_XYZ_SCALE = 0.75  # m
REF_DURATION_SCALE = 3.3  # s = dt x (window frames - 1), Ryan's dt x horizon
REF_RATE_SCALES = (2.5, 12.0)  # m/s, rad/s
# 2026-10-11 decode fix F1: evaluator floors (m/s, rad/s, gripper 1/s) for the speed columns of arcvelhyb / arcveltri.
F1_SPEED_FLOORS = [0.01, 0.01, 0.001]


def ref_fixed_ranges(variant):
    """norm_stats.fixed_ranges of a ref variant: [low, high] per column of its (M, 24) token
    [xyz, rotation6d, grip] x 2 arms | translation timing L, R | rotation timing L, R."""
    a, b = (
        (REF_DURATION_SCALE, REF_DURATION_SCALE)
        if variant == "arcdurref"
        else REF_RATE_SCALES
    )
    arm = [REF_XYZ_SCALE] * 3 + [1.0] * 6 + [1.0]
    high = [*arm, *arm, a, a, b, b]
    low = [-x for x in high]
    low[9] = low[19] = 0.0
    return {"actions_cartesian": {"low": low, "high": high}}


def _with_episodes(lam, episodes):
    """The base filter with its explicit episode set replaced (the SQL row conditions are kept)."""
    members = ",".join(repr(x) for x in sorted(episodes))
    lam, count = re.subn(r"frozenset\(\{[^}]*\}\)", "frozenset({" + members + "})", lam)
    if count != 1:
        raise ValueError("Expected one explicit episode set in the filter")
    return lam


def build_slowpace366(
    variants, rotation_deg, kinds=("bc", "cotrain"), stamp="20261009", fixes=False
):
    """Data + experiment leaves for slow-pace 366 hybrid / tri runs, robot-only BC and RL2 Aria cotrain, with the
    hybrid rotation stream capped at rotation_deg. TIME data leaves own the frozen episode sets; codec leaves
    inherit them and change only the transform; experiments inherit _base/stattempo_openloop (BC) or
    _base/stattempo_aria (cotrain).

    The defaults reproduce the committed r72 recipes byte for byte. ``kinds`` picks BC and / or cotrain, ``stamp``
    is the date in the W&B id, and ``fixes`` turns on the 2026-10-11 evaluator options (the tick-prefix regime, and
    speed floors for the two speed variants). The event-stream variants (EVT_LAYOUTS) are BC only and always get the
    tick prefix and norm_stats.widen_degenerate_quantiles (their gripper column pads rows). The reference variants
    (REF_LAYOUTS) are BC only too and always get the tick prefix and norm_stats.fixed_ranges (ref_fixed_ranges); their
    hand-committed model leaf carries the matching sampling clip."""
    require_compute_node()
    bc_only = sorted(set(variants) & {*EVT_LAYOUTS, *REF_LAYOUTS})
    if bc_only and "cotrain" in kinds:  # before anything is written
        raise ValueError(
            f"{', '.join(bc_only)}: no cotrain recipe, the model leaf is BC only"
        )
    m = json.loads(SLOWPACE366_MANIFEST.read_text())
    sets = {k: m[k] for k in ("robot_train", "robot_val", "human_train")}
    if {k: len(set(v)) for k, v in sets.items()} != {
        "robot_train": 366,
        "robot_val": 24,
        "human_train": 38,
    } or set(sets["robot_train"]) & set(sets["robot_val"]):
        raise ValueError(
            "Expected the frozen, disjoint 366 / 24 robot split and 38 Aria episodes"
        )
    unit = math.radians(rotation_deg)
    tag = f"r{rotation_deg:g}"
    robot = compose_recipe("scratch_rl2_stattempo_slowpace_time").data
    human = compose_recipe("cotrain_rl2_stattempo_slowpace_aria_time").data
    write_config(
        "data/abc_arc/stationery_slowpace366_bc_time.yaml",
        {
            "defaults": [
                "/data/abc_arc/stationery_tempo_slowpace_time@_here_",
                "_self_",
            ],
            **{
                group: {
                    "yam_bimanual": {
                        "filters": {
                            "filter_lambdas": [
                                _with_episodes(
                                    robot[group].yam_bimanual.filters.filter_lambdas[0],
                                    sets[key],
                                )
                            ]
                        }
                    }
                }
                for group, key in (
                    ("train_datasets", "robot_train"),
                    ("valid_datasets", "robot_val"),
                )
            },
        },
        "# GENERATED by scripts/e1/recipe_builders.py (build_slowpace366) from stationery_slowpace366_manifest.json.\n"
        "# Slow-pace 366 robot pool, robot only: 366 train / 24 validation episodes; codec time.\n",
    )
    write_config(
        "data/abc_arc/stationery_slowpace366_aria_cotrain_time.yaml",
        {
            # The Aria cotrain base supplies the human domain; the 366 robot leaf replaces its robot episodes.
            "defaults": [
                "/data/abc_arc/stationery_slowpace_aria_cotrain_time@_here_",
                "/data/abc_arc/stationery_slowpace366_bc_time@_here_",
                "_self_",
            ],
            "train_datasets": {
                "human_bimanual": {
                    "filters": {
                        "filter_lambdas": [
                            _with_episodes(
                                human.train_datasets.human_bimanual.filters.filter_lambdas[
                                    0
                                ],
                                sets["human_train"],
                            )
                        ]
                    }
                }
            },
        },
        "# GENERATED by scripts/e1/recipe_builders.py (build_slowpace366) from stationery_slowpace366_manifest.json.\n"
        "# Slow-pace 366 robot pool + 38 RL2 Aria episodes; codec time; validation is robot-only.\n",
    )
    model_leaf(
        "hpt_flow_wrists_ft_arcdurtri",
        "hpt_flow_wrists_ft_arcdurhyb",
        action_token_dim=20,
    )
    model_leaf(
        "hpt_flow_wrists_ft_arcdurtri_cotrain",
        "hpt_flow_wrists_ft_arcdurhyb_cotrain",
        action_token_dim=20,
    )
    layouts = {**HYBRID_LAYOUTS, **EVT_LAYOUTS, **REF_LAYOUTS}
    for kind in kinds:
        time_data = "stationery_slowpace366_" + (
            "bc" if kind == "bc" else "aria_cotrain"
        )
        for v in variants:
            evt, ref = v in EVT_LAYOUTS, v in REF_LAYOUTS
            dim = 24 if ref else 18 if v.endswith("hyb") else 20
            data = f"{time_data}_{v}_{tag}"
            write_config(
                f"data/abc_arc/{data}.yaml",
                {
                    "defaults": [f"/data/abc_arc/{time_data}_time@_here_", "_self_"],
                    **codec_overrides(
                        v, human=kind == "cotrain", rotation_distance_unit=unit
                    ),
                },
                f"# GENERATED by scripts/e1/recipe_builders.py (build_slowpace366): codec {v}, rotation cap {rotation_deg:g} deg.\n",
            )
            name = f"{'bc_rl2_stattempo_slowpace366' if kind == 'bc' else 'cotrain_rl2_stattempo_slowpace366_aria'}_{v}_{tag}"
            leaf = (
                "arcdurref"
                if ref
                else "arcdurevt"
                if evt
                else "arcdurtri"
                if dim == 20
                else "arcdurhyb"
            )
            model = (
                f"e1/hpt_flow_wrists_ft_{leaf}{'_cotrain' if kind == 'cotrain' else ''}"
            )
            # Appended after the r72 keys, and only when they apply, so the r72 recipes regenerate unchanged.
            evaluator, extra = {"action_mode": "arc", "token_layout": layouts[v]}, {}
            if fixes or evt or ref:
                evaluator["tick_prefix"] = True
            if fixes and v in ("arcvelhyb", "arcveltri"):
                evaluator["e1_speed_floors"] = F1_SPEED_FLOORS
            if evt:
                extra["norm_stats"] = {"widen_degenerate_quantiles": True}
            if ref:
                # Spatial bounds and tick-prefix evaluation live in the common base.
                evaluator.pop("action_mode")
                evaluator.pop("tick_prefix", None)
            options = (
                [
                    f"evaluator.{k}"
                    for k in ("tick_prefix", "e1_speed_floors")
                    if k in evaluator
                ]
                + ["norm_stats.widen_degenerate_quantiles"] * evt
                + ["norm_stats.fixed_ranges"] * ref
            )
            write_config(
                f"experiment/yam_arc_grid/{name}.yaml",
                {
                    "defaults": [
                        f"/experiment/yam_arc_grid/_base/{'stattempo_reference' if ref else 'stattempo_openloop' if kind == 'bc' else 'stattempo_aria'}",
                        {"override /model": model},
                        {"override /data": f"abc_arc/{data}"},
                        "_self_",
                    ],
                    "name": name,
                    "description": f"h640t8_d384x10_cos10k_240k_s42_{name}",
                    "logger": {
                        "wandb": {
                            "id": f"{name}_{stamp}_s42",
                            "tags": [
                                "rl2",
                                "stationery",
                                "stattempo",
                                "slowpace366",
                                kind,
                                *(["aria"] if kind == "cotrain" else []),
                                v,
                                tag,
                                "openloop_val",
                            ],
                        }
                    },
                    "e1": {
                        "variant": v,
                        **(
                            {
                                "reference_translation_range": [
                                    -REF_DURATION_SCALE,
                                    REF_DURATION_SCALE,
                                ]
                                if v == "arcdurref"
                                else [-REF_RATE_SCALES[0], REF_RATE_SCALES[0]],
                                "reference_rotation_range": [
                                    -REF_DURATION_SCALE,
                                    REF_DURATION_SCALE,
                                ]
                                if v == "arcdurref"
                                else [-REF_RATE_SCALES[1], REF_RATE_SCALES[1]],
                            }
                            if ref
                            else {"action_dim": dim}
                        ),
                    },
                    **({} if ref else {"hpt": {"action_dim": dim}}),
                    "model": {"rotation_distance_unit": unit},
                    "evaluator": evaluator,
                    **extra,
                },
                f"# @package _global_\n# Slow-pace 366 {kind}, codec {v}, rotation cap {rotation_deg:g} deg; only codec, cap and identity differ\n"
                f"# from the shared recipe (GENERATED by scripts/e1/recipe_builders.py build_slowpace366).\n"
                + (f"# Also set here: {', '.join(options)}.\n" if options else ""),
            )
