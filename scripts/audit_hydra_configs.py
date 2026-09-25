"""Recursively compose shipped YAML with explicit, offline audit contexts.

Contexts supply campaign-owned variables for reusable group fragments. They
are not runtime model dispatch, skipped tests, or default training selections.
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "egomimic/hydra_configs"

EXPERIMENT_CONTEXTS = {
    "model/e1/hpt_flow_wrists": "e1/abcs_time",
    "data/abc_arc/abc_fstshirt_arc_bc_D40_M100": "abc_arc/abc_fstshirt_arc_bc",
    "model/abc/yam_bimanual_dp": "abc/yam_fstshirt_dp",
    "model/abc/yam_bimanual_hpt": "abc/yam_fstshirt_hpt",
    "model/abc_arc/yam_bimanual_dp_arc_D40_M100": "abc_arc/abc_fstshirt_arc_bc",
    "model/bf/bf_planar_v2_arc_graph_tok": "pusht/planar_v2_usocket_arc_graph_tok",
    "model/bf/us_action_flow_latent_fm_sg_unite_h384": "pusht/action_flow_usocket_latent_fm_sg_unite_h384_s42",
    "evaluator/eval_arc_bimanual_cartesian_D40_M100": "abc_arc/abc_fstshirt_arc_bc",
}
MODEL_CONTEXTS = {
    "experiment/pusht/action_flow_usocket_candidate_common": "bf/us_action_flow_latent_fm_sg_unite_h384",
    "experiment/pusht/action_latent_vfm_usocket_val01_h16": "bf/us_action_latent_vfm_nt16_d8_h512_s42",
    "experiment/pusht/unite_cotrain_usocket_chain_val01_h16": "bf/ct_unite_register_separate_nt8_h384_s42",
    "experiment/pusht/unite_usocket_register_sweep_val01_h16": "bf/us_unite_register_shared_nt4_s42",
}


def audit_context(path):
    name = path.relative_to(CONFIGS).with_suffix("").as_posix()
    group, _, entry = name.partition("/")
    experiment = None
    if group == "experiment":
        experiment = entry
    elif "/e1/" in f"/{name}/":
        experiment = "e1/fold_time"
    elif group in {"model", "data"} and entry.startswith(("bf/", "pusht/")):
        experiment = "pusht/planar_v2_usocket_dp_standard"
    elif group in {"model", "data"} and entry.startswith(("abc/", "abc_arc/")):
        experiment = "abc_arc/abc_lang_fstshirt_mecka_freefold_cotrain_baseline"
        if "arc_D" in entry or "arcD" in entry or "arc_cotrain" in entry:
            experiment = "abc_arc/abc_lang_fstshirt_mecka_freefold_cotrain_arcD40M100"
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
        if "e1" in cfg:
            for key, value in {
                "variant": "time",
                "spread": "smoke",
                "train_root": "/tmp/audit/train",
                "valid_root": "/tmp/audit/valid",
            }.items():
                if OmegaConf.is_missing(cfg.e1, key):
                    cfg.e1[key] = value
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
                    "inference_graph": {
                        "status": "unsupported",
                        "reason": "station template audit",
                    },
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
    print(f"{len(results)-len(failed)}/{len(results)} YAML configs resolved")
    for row in failed:
        print(row["path"], row["error"].splitlines()[0])
    raise SystemExit(bool(failed))
