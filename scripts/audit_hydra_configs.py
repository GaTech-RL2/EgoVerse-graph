"""Recursively compose shipped YAML with explicit, offline audit contexts.

Contexts supply campaign-owned variables for reusable group fragments. They
are not runtime model dispatch, skipped tests, or default training selections.
"""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "egomimic/hydra_configs"

EXPERIMENT_CONTEXTS = {
    "experiment/_shared/clock_settings": "real/multitask/arc_vel",
    "experiment/_shared/frozen_yam_arc": "real/wireless_earbuds/arc_vel",
    "experiment/real/arcdur_abstime/_base/arcdur_abstime": "real/arcdur_abstime/wireless_earbuds",
    "experiment/_shared/frozen_yam/base": "real/wireless_earbuds/baseline",
    "evaluator/frozen_yam_time": "real/wireless_earbuds/baseline",
    "model/abc_arc/frozen_yam_hpt_flow": "real/wireless_earbuds/baseline",
    "model/abc_arc/frozen_yam_hpt300_time": "real/wireless_earbuds/baseline",
    "model_contract/frozen_yam_wrists": "real/wireless_earbuds/baseline",
    "data/pusht/planar_chain_manual4919_standard_retimed": "sim/pusht/planar_usocket_chain_manual4919_standard_dp_retimed_masked",
    "data/pusht/parents/standard_dp_uc_h16": "sim/pusht/planar_usocket_chain_manual4919_standard_dp_retimed_masked",
    "model/bf/libero10_action_flow_latent_fm_sg_unite_h240_euler50_dithalf": "sim/libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42",
    "model/libero/oat_dp_matched": "sim/libero_historical/dp_libero10_oat_dp_matched_s42",
    "model/abc/yam_bimanual_dp": "abc/yam_fstshirt_dp",
    "model/abc/yam_bimanual_hpt": "abc/yam_fstshirt_hpt",
    "model_contract/abc_visual": "real/organize_stationary/variants/stationery_rl2_hpt300_visual_baseline_openloop",
    "model/abc_arc/hpt_yam_visual": "real/organize_stationary/variants/stationery_rl2_hpt300_visual_baseline_openloop",
    "model/abc_arc/hpt_bimanual_visual": "real/organize_stationary/baseline",
    "model/abc_arc/abc_dit_yam_visual": "real/multitask/variants/abc_multitask4_abcdit_baseline_visual_openloop",
    "data/abc_arc/abc_visual": "real/fold_and_stack_towels/variants/abc_towels_hpt180_baseline_visual_openloop",
    "data/abc_arc/stationery_rl2_hpt_baseline": "real/organize_stationary/variants/stationery_rl2_hpt300_visual_baseline_openloop",
    "data/abc_arc/stationery_rl2_hpt_arc_hybrid_D40_M100_R24deg": "real/organize_stationary/variants/stationery_rl2_hpt300_visual_hybrid_openloop",
    "data/abc_arc/mecka_fold_clothes_40h_human_baseline": "abc_arc/human_bc/mecka_fold_clothes_40h_human_visual_baseline_openloop",
    "data/abc_arc/mecka_fold_clothes_40h_human_hybrid_D40_M100_R24deg": "abc_arc/human_bc/mecka_fold_clothes_40h_human_visual_hybrid_openloop",
    "data/abc_arc/organize_rl2_elmo_cotrain": "real/organize_stationary/baseline",
    "experiment/_shared/visual_hpt300": "real/fold_and_stack_towels/variants/abc_towels_hpt180_baseline_visual_openloop",
    "model/bf/bf_planar_v2_arc_graph_tok": "sim/pusht/planar_v2_usocket_arc_graph_tok",
    "model/bf/us_action_flow_latent_fm_sg_unite_h384": "sim/pusht/parents/action_flow_usocket_h384",
    "data/libero/action_flow_libero10_val01_h16": "sim/libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42",
    "data/libero/oat": "sim/oat/libero_oattok",
    "model/oat/tokenizer": "sim/oat/libero_oattok",
    "model/oat/policy": "sim/oat/libero_oatpolicy",
    "model/oat/arc_tokenizer": "sim/oat/libero_arc_oattok",
    "model/oat/arc_policy": "sim/oat/libero_arc_policy",
    "model/oat/arc_oat_policy": "sim/oat/libero_arc_oatpolicy",
    "model/oat/arc_oat_dp_policy": "sim/oat/libero_arc_oat_dp_policy",
}
MODEL_CONTEXTS = {
    "experiment/sim/pusht/action_flow_usocket_candidate_common": "bf/us_action_flow_latent_fm_sg_unite_h384",
    "experiment/sim/pusht/action_latent_vfm_usocket_val01_h16": "bf/us_action_latent_vfm_nt16_d8_h512_s42",
    "experiment/sim/pusht/unite_cotrain_usocket_chain_val01_h16": "bf/ct_unite_register_separate_nt8_h384_s42",
    "experiment/sim/pusht/unite_usocket_register_sweep_val01_h16": "bf/us_unite_register_shared_nt4_s42",
    "experiment/sim/oat/libero_base": "oat/tokenizer",
}

# Explicit non-production composition inputs, scoped and restored even on error.
# These paths/hashes are never opened or accepted as launch/smoke evidence.
AUDIT_ENVIRONMENT = {
    **{
        name: "0" * 64
        for name in (
            "MATCHED_AF_SPLIT_SHA256",
            "MATCHED_DP_SPLIT_SHA256",
            "MATCHED_NORMALIZATION_SHA256",
            "MATCHED_DP_NORMALIZATION_SHA256",
            "MATCHED_SPLIT_SHA256",
            "DP_SPLIT_SHA256",
            "DP_CHAIN_SPLIT_SHA256",
        )
    },
    "MATCHED_SOURCE_COMMIT": "0" * 40,
    "MATCHED_DP_NORM": "/tmp/egoverse-config-audit/dp-normalizer.json",
    **{
        name: "/tmp/egoverse-config-audit/" + name.lower()
        for name in (
            "DP_SPLIT_MANIFEST",
            "DP_CHAIN_SPLIT_MANIFEST",
            "DP_U_SPLIT_MANIFEST",
        )
    },
    "DP_DATASET_DIR": "/tmp/egoverse-config-audit/chain",
    "EGOVERSE_ABC_DATASET_DIR": "/tmp/egoverse-config-audit/abc",
    "EGOVERSE_DINOV3_CHECKPOINT": "/tmp/egoverse-config-audit/dinov3.pth",
    "EGOVERSE_CLIP_CHECKPOINT": "/tmp/egoverse-config-audit/clip.pth",
    "DP_U_DATASET_DIR": "/tmp/egoverse-config-audit/usocket",
    "DP_CHAIN_DATASET_DIR": "/tmp/egoverse-config-audit/chain",
    "AF_COTRAIN_NORM": "/tmp/egoverse-config-audit/normalizer.json",
    "AF_HUMAN_DATA_ROOT": "/tmp/egoverse-config-audit/human",
    "YAM_HPTFLOW_TRAINING_CONFIG": "/tmp/egoverse-config-audit/resolved-config.yaml",
    "YAM_HPTFLOW_CHECKPOINT": "/tmp/egoverse-config-audit/model.ckpt",
    "YAM_HPTFLOW_NORMALIZER": "/tmp/egoverse-config-audit/data-context.json",
    "PUSHSHAPES_USOCKET_ROOT": "/tmp/egoverse-config-audit/usocket",
    "PUSHSHAPES_CHAIN_GRIPPER_ROOT": "/tmp/egoverse-config-audit/chain",
    **{
        name: "/tmp/egoverse-config-audit/" + name.lower()
        for name in (
            "LIBERO10_REPLAY_ROOT",
            "LIBERO_GOAL_REPLAY_ROOT",
            "LIBERO_OBJECT_REPLAY_ROOT",
            "LIBERO_SPATIAL_REPLAY_ROOT",
            "LIBERO_ORIGINAL_SPATIAL_ROOT",
            "LIBERO_SHARPA_SPATIAL_ROOT",
            "LIBERO_DIAGNOSTIC_SEED_BANK",
        )
    },
    **{
        name: "0" * 64
        for name in (
            "YAM_TRAIN_IDS_SHA256",
            "YAM_VALID_IDS_SHA256",
            "HUMAN_TRAIN_IDS_SHA256",
            "HUMAN_VALID_IDS_SHA256",
        )
    },
}


def audit_context(path):
    name = path.relative_to(CONFIGS).with_suffix("").as_posix()
    if name == "data/pusht/parents/episode_split":
        # This is a dataset fragment, not a selectable DataModule. Audit it
        # in the same nested package used by its real data parents.
        return "train_zarr_cartesian", [
            "+experiment=sim/pusht/planar_v2_usocket_dp_standard",
            "+data@data.train_datasets.pushshapes_sim_u_socket=pusht/parents/episode_split",
        ]
    group, _, entry = name.partition("/")
    if group == "visualization":
        # Standalone data-visualization roots composed from their own
        # directory (egomimic/scripts/data_visualization), not under a run.
        return None, [f"+visualization@_global_={entry}"]
    experiment = None
    if group == "experiment":
        experiment = entry
    elif group in {"model", "data"} and entry.startswith(("bf/", "pusht/")):
        experiment = "sim/pusht/planar_v2_usocket_dp_standard"
    elif group in {"model", "data"} and entry.startswith(("abc/", "abc_arc/")):
        # The retired language-conditioned ABC ablations are gone; the kept
        # 20-episode fstshirt baseline is the remaining shared ABC context.
        experiment = "abc/yam_fstshirt_dp"
    # Stationery fragments require the same FT sizes, source frames and fixed
    # ranges supplied by their real inheriting recipes, never invented values.
    if name.startswith("data/abc_arc/stationery_"):
        experiment = "real/wireless_earbuds/baseline"
    if name.startswith(("data/abc_arc/towels_", "data/abc_arc/earbuds_")):
        # Frozen YAM pools select their codec through the HPT300 recipe.
        experiment = "real/fold_and_stack_towels/baseline"
    experiment = EXPERIMENT_CONTEXTS.get(name, experiment)
    overrides = [] if experiment is None else [f"+experiment={experiment}"]
    config_name = name if not entry else "train_zarr_cartesian"
    if group in {"model", "evaluator"} and entry.startswith("pi05/"):
        config_name = "train_zarr_cartesian_pi"
    if name == "evaluator/dataset_video":
        config_name = "viz_language"
    if name in MODEL_CONTEXTS:
        overrides.append(f"model={MODEL_CONTEXTS[name]}")
    if group == "hydra":
        sub_group, _, sub_entry = entry.partition("/")
        overrides.append(f"hydra/{sub_group}={sub_entry}")
    elif group == "evaluator" and entry.startswith("viz/"):
        overrides.append("evaluator=eval_bimanual_cartesian")
        overrides.append(
            f"evaluator/viz@evaluator.viz_func={entry.removeprefix('viz/')}"
        )
    elif group not in {"experiment"} and entry:
        prefix = (
            ""
            if group
            in {
                "data",
                "model",
                "evaluator",
                "paths",
                "trainer",
                "callbacks",
                "logger",
                "debug",
            }
            else "+"
        )
        overrides.append(f"{prefix}{group}={entry}")
    return config_name, overrides


@contextmanager
def compose_for_audit(path):
    with patch.dict(os.environ, AUDIT_ENVIRONMENT):
        with _compose_for_audit(path) as cfg:
            yield cfg


@contextmanager
def _compose_for_audit(path):
    config_name, overrides = audit_context(path)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name=config_name, overrides=overrides, return_hydra_config=True
        )
    # Hydra runtime values are normally assigned by the launcher. Composition
    # has no launcher, so give these three fields explicit synthetic values.
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = "/tmp/egoverse-config-audit"
        cfg.hydra.job.num = 0
        cfg.hydra.job.id = "config-audit"
        if "pi05" in cfg and OmegaConf.is_missing(cfg.pi05, "pretrained_weights"):
            cfg.pi05.pretrained_weights = "/tmp/egoverse-config-audit/pi-weights"
        if "frozen_yam" in cfg:
            for key, value in {
                "variant": "time",
                "spread": "smoke",
                "train_root": "/tmp/audit/train",
                "valid_root": "/tmp/audit/valid",
                "data_root": "/tmp/audit/yam",
            }.items():
                if OmegaConf.is_missing(cfg.frozen_yam, key):
                    cfg.frozen_yam[key] = value
        for section, values in {
            "benchmark": {"dataset": "/tmp/audit/libero", "arc_basis": "chebyshev"},
            "speed_diagnostic": {"speed_reference": 0.1},
            "model": {"speed_reference": 0.1},
        }.items():
            if section in cfg:
                for key, value in values.items():
                    if OmegaConf.is_missing(cfg[section], key):
                        cfg[section][key] = value
        if (
            path.relative_to(CONFIGS).as_posix()
            == "experiment/sim/oat/libero_arc_shared_r384_d1p6_m36.yaml"
        ):
            # This reusable shortlist explicitly requires stk/dur selection.
            # Supplying an audit context does not make null a runtime default.
            assert cfg.benchmark.arc_mode is None
            cfg.benchmark.arc_mode = "stk"
        if "planar" in cfg:
            for key, value in {"action_dims": 4, "eval_native_decoder": None}.items():
                if OmegaConf.is_missing(cfg.planar, key):
                    cfg.planar[key] = value
        if path.relative_to(CONFIGS).parts[0] == "robot":
            # Station templates intentionally require physical values. These
            # named synthetic inputs prove composition only; never instantiate
            # hardware from this audit.
            station = cfg.robot
            cameras = OmegaConf.select(station, "robot.cameras", default={})
            for camera in cameras.values():
                if "serial_number" in camera:
                    camera.serial_number = "audit-no-hardware"
            policy = station.get("policy")
            if policy is not None:
                for key, value in {
                    "training_config": "/tmp/audit/resolved.yaml",
                    "checkpoint": "/tmp/audit/model.ckpt",
                    "normalizer_path": "/tmp/audit/normalizer.json",
                    "path": "/tmp/audit/replay.zarr",
                }.items():
                    if OmegaConf.is_missing(policy, key) or (
                        key == "path" and "path" in policy
                    ):
                        policy[key] = value
                adapter = policy.get("adapter")
                if adapter is not None:
                    if OmegaConf.is_missing(adapter, "prompt"):
                        adapter.prompt = "audit"
                    if "base_T_model" in adapter:
                        for arm in ("left", "right"):
                            if OmegaConf.is_missing(adapter.base_T_model, arm):
                                adapter.base_T_model[arm] = [
                                    [int(i == j) for j in range(4)] for i in range(4)
                                ]
    hydra = HydraConfig.instance()
    previous = hydra.cfg
    try:
        hydra.set_config(cfg)
        yield cfg
    finally:
        hydra.cfg = previous


def audit():
    records = []
    for path in sorted(CONFIGS.rglob("*.yaml")):
        row = {
            "path": str(path.relative_to(CONFIGS)),
            "context": audit_context(path)[1],
        }
        try:
            with compose_for_audit(path) as cfg:
                # Job-specific hydra.sweep and optional plugin internals are not
                # training fields; separately resolve the selected launcher.
                for key in cfg:
                    if key != "hydra":
                        value = cfg[key]
                        if OmegaConf.is_config(value):
                            OmegaConf.to_container(
                                value, resolve=True, throw_on_missing=True
                            )
                OmegaConf.to_container(
                    cfg.hydra.launcher, resolve=True, throw_on_missing=True
                )
            row["status"] = "passed"
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        records.append(row)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = audit()
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    failed = [r for r in results if r["status"] != "passed"]
    print(f"{len(results) - len(failed)}/{len(results)} YAML configs resolved")
    for row in failed:
        print(row["path"], row["error"].splitlines()[0])
    raise SystemExit(bool(failed))
