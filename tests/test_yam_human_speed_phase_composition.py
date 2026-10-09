"""Compose the maintained real argv for all phases before any corpus traversal.
Synthetic identities are schema regression fixtures, never launch receipts.
"""

import argparse
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from egomimic.rldb.zarr.episode_split import complete_window_count
from scripts.train.verify_yam_human_action_flow_launch import (
    aggregate_content,
    split_identities,
    verify_config,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "corpus_fixture", ROOT / "tests/test_yam_human_refreshed_corpus_contract.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("phase", ["norm", "smoke", "full"])
def test_real_shared_argv_composes_and_verifies(tmp_path, monkeypatch, phase):
    cp, sp, c, s, content = module.fixture(tmp_path)
    c.update(
        window_contract="physical_complete_native_virtual_views_v1",
        human_rates=[0.2, 0.4, 0.6, 0.8, 1.0],
        yam_rates=[1.0],
        sample_views_per_source=5,
    )
    for domain, horizon in (("yam", 100), ("human", 30)):
        for group in ("train", "valid"):
            count = sum(
                complete_window_count(r["num_frames"], horizon, 5)
                for r in s[domain]["episodes"]
                if r["split"] == group
            )
            s[domain]["summary"][group + "_frame_windows"] = count
            c["domains"][domain][group + "_frame_windows"] = count
    s["train_frame_windows_total"] = sum(
        s[d]["summary"]["train_frame_windows"] for d in ("yam", "human")
    )
    s["proportional_train_window_probabilities"] = {
        d: s[d]["summary"]["train_frame_windows"] / s["train_frame_windows_total"]
        for d in ("yam", "human")
    }
    module.save(cp, c)
    module.save(sp, s)
    ids = split_identities(sp, module.sha(sp), cp, module.sha(cp))
    aggregate = aggregate_content(
        content["yam"],
        module.sha(content["yam"]),
        content["human"],
        module.sha(content["human"]),
        cp,
        module.sha(cp),
    )
    norm = tmp_path / "norm.json"
    norm.write_text("{}")
    env = {
        "AF_YAM_RECIPE": "yam_human_keypoints",
        "AF_YAM_PHASE": phase,
        "AF_YAM_ACCUMULATION": "1",
        "AF_YAM_CHECKPOINT_POLICY": "dit_half",
        "AF_YAM_INFERENCE_METHOD": "euler",
        "AF_YAM_DATA_ROOT": str(tmp_path / "yam"),
        "AF_HUMAN_DATA_ROOT": str(tmp_path / "human"),
        "AF_YAM_NORM_JSON": str(norm),
        "AF_COTRAIN_NORM": str(norm),
        "AF_YAM_NORM_CACHE_DIR": str(tmp_path / "norm-cache"),
        "AF_YAM_RUN_DIR": str(tmp_path / phase),
        "AF_YAM_SOURCE_COMMIT": "0" * 40,
        "AF_YAM_NORM_SHA": module.sha(norm),
        "AF_YAM_SPLIT_SHA": module.sha(sp),
        "AF_YAM_WANDB_ENTITY": "schema_fixture",
        "AF_YAM_WANDB_PROJECT": "schema_fixture",
        "AF_YAM_WANDB_ID": "schema_fixture_" + phase,
        "content_sha": aggregate,
    }
    reference = tmp_path / "reference.json"
    module.save(
        reference,
        {
            "status": "TRAIN_ONLY_PHYSICAL_SPEED_REFERENCE_V1",
            "reference": 0.07,
            "train_ids_sha256": ids["yam_train"],
            "statistic": "refreshed_train_robot_command_both_arm_native_window_median_v1",
        },
    )
    env.update(
        AF_YAM_EXPERIMENT="yam_human_keypoints_speed_h816_private512_s42",
        AF_YAM_AUGMENTATION="human_speed_v1",
        AF_STATIONARY_SPEED_REFERENCE="0.07",
    )
    for d in ("yam", "human"):
        for group in ("train", "valid"):
            env[d + "_" + group + "_count"] = str(s[d]["summary"][group + "_episodes"])
            env[d + "_" + group + "_windows"] = str(
                s[d]["summary"][group + "_frame_windows"]
            )
            env[d.upper() + "_" + group.upper() + "_IDS_SHA256"] = ids[d + "_" + group]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    command = r'source "$1"; yam_action_flow_arguments; printf "%s\0" "${overrides[@]}"'
    assert chr(0) not in command, "shell executable argv must never embed a NUL"
    argv = (
        subprocess.check_output(
            [
                "bash",
                "-c",
                command,
                "argv",
                str(ROOT / "scripts/train/yam_action_flow_arguments.sh"),
            ],
            env={**os.environ, **env},
        )
        .decode()
        .rstrip("\0")
        .split("\0")
    )
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = compose(
            config_name="train_zarr_cartesian", overrides=argv, return_hydra_config=True
        )
        cfg.hydra.runtime.output_dir = str(tmp_path / phase)
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            del cfg["hydra"]
        OmegaConf.resolve(cfg)
    resolved = tmp_path / (phase + ".yaml")
    OmegaConf.save(cfg, resolved)
    args = argparse.Namespace(
        phase=phase,
        yam_root=tmp_path / "yam",
        human_root=tmp_path / "human",
        norm_json=norm,
        norm_sha=module.sha(norm),
        norm_cache_dir=tmp_path / "norm-cache",
        source_commit="0" * 40,
        split_sha=module.sha(sp),
        wandb_entity="schema_fixture",
        wandb_project="schema_fixture",
        wandb_id="schema_fixture_" + phase,
        run_dir=tmp_path / phase,
        inference_method="euler",
        checkpoint_policy="dit_half",
        corpus_contract=cp,
        corpus_contract_sha=module.sha(cp),
    )
    args.augmentation = "human_speed_v1"
    args.speed_reference = 0.07
    args.speed_reference_receipt = reference
    args.speed_reference_receipt_sha = module.sha(reference)
    assert isinstance(cfg.stationary_speed.reference, float)
    assert cfg.stationary_speed.reference == 0.07
    verify_config(resolved, args, ids, aggregate)
    # The profile name is strict and selected by the typed augmentation mode.
    # Reject accidentally pairing the speed graph with the noaug identity.
    speed_name = cfg.name
    assert speed_name == "yam_human_keypoints_speed_h816_private512_s42"
    cfg.name = "yam_human_keypoints_action_flow_h816_private512_s42"
    OmegaConf.save(cfg, resolved)
    with pytest.raises(ValueError, match="resolved name mismatch"):
        verify_config(resolved, args, ids, aggregate)
    cfg.name = speed_name
    OmegaConf.save(cfg, resolved)
    assert cfg.model.pipeline.stages[6].num_inference_steps == 50
    assert list(cfg.stationary_speed.human_rates) == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert (
        cfg.data.train_datasets.human_bimanual.resolver.key_map.extra_key_map._physical_timestamps_ns.horizon
        == 30
    )
    original = OmegaConf.to_container(cfg.stationary_speed.human_rates)
    cfg.stationary_speed.human_rates = [0.25, 0.4, 0.55, 0.7, 0.85]
    OmegaConf.save(cfg, resolved)
    with pytest.raises(ValueError):
        verify_config(resolved, args, ids, aggregate)
    cfg.stationary_speed.human_rates = original
    OmegaConf.save(cfg, resolved)
    for provenance in ("energy_score_provenance", "action_flow_diagnostics.provenance"):
        path = "evaluator." + provenance + ".sampler"
        OmegaConf.update(cfg, path, "dopri5")
        OmegaConf.save(cfg, resolved)
        with pytest.raises(ValueError):
            verify_config(resolved, args, ids, aggregate)
        OmegaConf.update(cfg, path, "euler")
    OmegaConf.save(cfg, resolved)
    assert not any("cuda" in value for value in argv)
    if phase == "norm":
        # Exercise the real trainHydra injected kwarg before expensive corpus traversal.
        from hydra.utils import instantiate

        key_cfg = OmegaConf.to_container(
            cfg.data.train_datasets.human_bimanual.resolver.key_map, resolve=True
        )
        regular = instantiate(key_cfg)
        key_cfg["norm_mode"] = True
        normalized = instantiate(key_cfg)
        assert "_physical_timestamps_ns" in normalized
        assert normalized == {
            k: v
            for k, v in regular.items()
            if v.get("key_type") not in ("camera_keys", "annotation_keys")
        }
        assert not any(
            v.get("key_type") in ("camera_keys", "annotation_keys")
            for v in normalized.values()
        )
        cfg.norm_stats.precomputed_norm_path = str(norm)
        OmegaConf.save(cfg, resolved)
        with pytest.raises(ValueError):
            verify_config(resolved, args, ids, aggregate)
