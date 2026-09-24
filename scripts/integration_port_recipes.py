"""One-time, source-pinned recipe migration; never used by training dispatch.

The reviewed output is ordinary graph YAML. The evidence snapshot was resolved
against EgoVerse ec5c903c; runtime code never imports legacy policy classes.
"""

import json
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "egomimic/hydra_configs/model"
EVIDENCE = ROOT / "docs/integration/evidence/legacy-model-recipes.json"
DOMAIN_IDS = {"eva_bimanual": 6, "human_bimanual": 3}
KEYS = {
    "front_img_1": "observations.images.front_img_1",
    "left_wrist_img": "observations.images.left_wrist_img",
    "right_wrist_img": "observations.images.right_wrist_img",
    "state_ee_pose": "observations.state.ee_pose",
    # The audited destination keymap emits Cartesian proprio; this stale
    # legacy reference never resolved with the shipped Cartesian data recipes.
    "state_joint_positions": "observations.state.ee_pose",
    "state_keypoints": "observations.state.keypoints",
    "annotation": "observations.annotation",
}


class Dumper(yaml.SafeDumper):
    def ignore_aliases(self, _data):
        return True


def relocated(config):
    result = deepcopy(config)
    if isinstance(result, dict):
        target = result.get("_target_", "")
        if target.startswith("egomimic.models.hpt_nets."):
            name = target.rsplit(".", 1)[1]
            home = "text_encoders" if name.startswith("Qwen") else "hpt_stems"
            result["_target_"] = f"egomimic.models.stems.{home}.{name}"
        if target.startswith("egomimic.utils.action_utils."):
            result["_target_"] = target.replace("action_utils", "action_encoding")
        return {key: relocated(value) for key, value in result.items()}
    if isinstance(result, list):
        return [relocated(value) for value in result]
    return result


def contract(
    domain,
    *,
    width,
    action_key,
    observation_keys,
    sampler_index,
    control_path,
    representation="cartesian",
    steps=50,
    action_stride=1,
):
    raw_horizon = 45 if domain == "eva_bimanual" else 30
    last_frame = ((raw_horizon - 1) // action_stride) * action_stride
    horizon = last_frame + 1
    timing = {"kind": "control_period", "temporally_resolved": True, "dt": 1 / 30}
    native_timing = {
        "kind": "uniform_resampled_target",
        "temporally_resolved": True,
        "dt": last_frame / (99 * 30),
        "first_source_frame": 0,
        "last_source_frame": last_frame,
        "source_fps": 30,
    }
    angular_indices = (
        [3, 4, 5, 10, 11, 12]
        if width == 14
        else ([3, 4, 5, 9, 10, 11] if width == 12 else [3, 4, 5, 72, 73, 74])
    )
    decoder = {
        "_target_": "egomimic.pipeline.sequence_decoder.UniformTimeGridDecoder",
        "native_shape": [100, width],
        "native_dt": native_timing["dt"],
        "output_horizon": horizon,
        "output_dt": timing["dt"],
        "angular_indices": angular_indices,
    }
    observations = list(dict.fromkeys([*observation_keys, "embodiment"]))
    adapter = {
        "_target_": "egomimic.pipeline.action_adapter.CanonicalSequenceAdapter",
        "shape": [horizon, width],
        "decoder": decoder,
    }
    if domain == "eva_bimanual" and representation == "cartesian":
        adapter = {
            "_target_": "egomimic.robot.graph_policy.CartesianGraphAdapter",
            "decoder": decoder,
            "rotation_mode": "euler",
            "action_frame": "model_frame",
        }
    controls = {
        "replan_every": {
            "label": "Actions before replanning",
            "description": "Control-rate execution prefix.",
            "type": "integer",
            "min": 1,
            "max": horizon,
            "step": 1,
            "default": min(30, horizon),
            "target": {"kind": "policy_attribute", "attribute_path": "replan_every"},
        }
    }
    if control_path is not None:
        controls["inference_steps"] = {
            "label": "Sampling steps",
            "description": "Configured sampler iterations.",
            "type": "integer",
            "min": 1,
            "max": 100,
            "step": 1,
            "default": steps,
            "target": {
                "kind": "stage_attribute",
                "stage_id": "sampler",
                "attribute_path": control_path,
            },
        }
    return {
        "input": {
            "history_length": 1,
            "keys": observations,
            "history_keys": [
                key for key in observations if key not in {"embodiment", "annotations"}
            ],
            "constants": {"embodiment": DOMAIN_IDS[domain]},
        },
        "native_output": {
            "key": "pred_action",
            "representation": "normalized_action",
            "shape": [100, width],
            "timing": native_timing,
        },
        "output": {
            "key": "actions",
            "representation": representation,
            "shape": [horizon, width],
            "timing": deepcopy(timing),
        },
        "compatibility": {
            "normalizer_schema": {
                "action_key": action_key,
                "native_shape": [100, width],
                "embodiment": DOMAIN_IDS[domain],
            },
            "tokenizer": None,
        },
        "profiles": {
            "default": {
                "stage_id": "sampler",
                "native_shape": [100, width],
                "adapter": adapter,
                "overrides": controls,
            }
        },
    }


def stem(spec, modality, policy):
    value = relocated(spec)
    if modality == "annotation":
        return value
    encoder = policy.get("encoder_specs", {}).get(modality)
    result = {
        "_target_": "egomimic.models.stems.composed.EncodedPolicyStem",
        "stem": value,
        "feature_position_encoding": True,
    }
    if encoder:
        result.update(
            encoder=relocated(encoder),
            train_transform=deepcopy(policy["train_image_augs"]),
            eval_transform=deepcopy(policy["eval_image_augs"]),
        )
    return result


def flow(head):
    if head["_target_"].endswith(".MultiBlockTransformerDecoder"):
        model = deepcopy(head)
        model["_target_"] = (
            "egomimic.models.heads.hpt_decoder.MultiBlockTransformerDecoder"
        )
        return [
            {
                "_target_": "egomimic.pipeline.stages_regression.SequenceRegressionStage",
                "model": model,
                "action_horizon": head["action_horizon"],
                "action_dim": head["output_dim"],
                "loss": "smooth_l1",
                "beta": 0.05,
            }
        ]
    width, horizon = head["model"]["act_dim"], head["action_horizon"]
    model = deepcopy(head["model"])
    if model["hidden_dim"] // 2 < width:
        model["allow_input_bottleneck"] = True
    return [
        {
            "_target_": "egomimic.pipeline.stages_flow.FlowNoisingStage",
            "action_horizon": horizon,
            "action_dim": width,
            "time_dist": head.get("time_dist", "beta"),
        },
        {
            "_target_": "egomimic.pipeline.stages_flow.FlowDenoiserStage",
            "action_horizon": horizon,
            "action_dim": width,
            "condition_input_dim": head["model"]["cond_dim"],
            "condition_ndim": 3,
            "condition_as_tokens": False,
            "num_inference_steps": head["num_inference_steps"],
            "model": model,
        },
        {"_target_": "egomimic.pipeline.stages_flow.FlowVelocityLossStage"},
    ]


def hpt_recipe(old, *, human_stride=3):
    policy = old["robomimic_model"]
    head_specs = {
        name: value for name, value in policy["head_specs"].items() if value is not None
    }
    deterministic = all(
        value["_target_"].endswith(".MultiBlockTransformerDecoder")
        for value in head_specs.values()
    )
    if deterministic:
        # The legacy YAML's latent_token_len=8 conflicts with its 64-token
        # trunk. Bind the decoder context to the declared trunk output.
        for value in head_specs.values():
            value["latent_token_len"] = policy["trunk"]["action_horizon"]
    elif not all(
        value["_target_"].endswith(".FMPolicy") for value in head_specs.values()
    ):
        raise ValueError("Unsupported retained head; add an explicit migration")
    result = {
        key: deepcopy(value) for key, value in old.items() if key != "robomimic_model"
    }
    trunk = policy["trunk"]
    stages = []
    language = "annotation" in policy["shared_stem_specs"]
    if language:
        stages.append(
            {
                "_target_": "egomimic.pipeline.stages_hpt.AnnotationPromptStage",
                "annotation_key": "annotations",
            }
        )
    stages.append(
        {
            "_target_": "egomimic.pipeline.stages_hpt.HPTStemStage",
            "domain_first": True,
            "preserve_config_order": True,
            "selector_aliases": {str(DOMAIN_IDS[d]): d for d in policy["domains"]},
            "stems": {
                KEYS[k]: stem(v, k, policy)
                for k, v in policy["shared_stem_specs"].items()
            },
            "domain_stems": {
                d: {KEYS[k]: stem(v, k, policy) for k, v in specs.items()}
                for d, specs in policy["stem_specs"].items()
            },
        }
    )
    stages.append(
        {
            "_target_": "egomimic.pipeline.stages_hpt.HPTTrunkStage",
            "embed_dim": trunk["embed_dim"],
            "token_postprocessing": trunk["token_postprocessing"],
            "action_token_count": trunk["action_horizon"],
            "squeeze_action_token": False,
            "use_domain_embedding": False,
            "trunk": {
                "_target_": "egomimic.models.cores.hpt_transformer.SimpleTransformer",
                "embed_dim": trunk["embed_dim"],
                "num_blocks": trunk["num_blocks"],
                "drop_path_rate": trunk["drop_path"],
                "weight_init_style": trunk["weight_init_style"],
                "attn_target": {
                    "_target_": "egomimic.models.cores.hpt_transformer.MultiheadAttention",
                    "_partial_": True,
                    "embed_dim": trunk["embed_dim"],
                    "num_heads": trunk["num_heads"],
                    "batch_first": True,
                    "bias": True,
                    "add_bias_kv": True,
                },
            },
        }
    )
    keys = set(policy["ac_keys"].values())
    action_key = (
        "actions_keypoints" if keys == {"actions_keypoints"} else "actions_cartesian"
    )
    stages.append(
        {
            "_target_": "egomimic.pipeline.stages_io.ActionTargetBuilder",
            "action_key": action_key,
        }
    )
    shared = "shared" in head_specs
    if shared:
        stages.append(
            {
                "_target_": "egomimic.pipeline.stages_layout.ActionLayoutStage",
                "key": "target",
                "layouts": {
                    "3": [0, 1, 2, 3, 4, 5, None, 6, 7, 8, 9, 10, 11, None],
                    "6": list(range(14)),
                },
                "input_widths": {"3": 12, "6": 14},
            }
        )
        sampler_index = len(stages) + int(not deterministic)
        stages.extend(flow(head_specs["shared"]))
        stages.append(
            {
                "_target_": "egomimic.pipeline.stages_layout.ActionLayoutStage",
                "key": "pred_action",
                "mode": "inference",
                "layouts": {
                    "3": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
                    "6": list(range(14)),
                },
                "input_widths": {"3": 14, "6": 14},
            }
        )
    elif len(head_specs) == 1:
        sampler_index = len(stages) + int(not deterministic)
        stages.extend(flow(next(iter(head_specs.values()))))
    else:
        sampler_index = len(stages)
        stages.append(
            {
                "_target_": "egomimic.pipeline.stages_routing.BranchStage",
                "selector_aliases": {str(DOMAIN_IDS[d]): d for d in policy["domains"]},
                "reads_by_mode": {
                    "train": ["target", "condition"],
                    "inference": ["condition"],
                },
                "writes_by_mode": {
                    "train": ["pred_action", "loss/regression"]
                    if deterministic
                    else ["loss/flow_velocity", "log/*"],
                    "inference": ["pred_action", "log/*"],
                },
                "branches": {
                    d: {
                        "_target_": "egomimic.pipeline.core.Pipeline",
                        "stages": flow(h),
                        "stage_ids": {"sampler": int(not deterministic)},
                    }
                    for d, h in head_specs.items()
                },
            }
        )
    result["pipeline"] = {
        "_target_": "egomimic.pipeline.algo.PipelineAlgo",
        "stages": stages,
        "stage_ids": {
            "stems": int(language),
            "trunk": int(language) + 1,
            "sampler": sampler_index,
        },
    }
    result["deployment_domain"] = policy["domains"][0]
    result["inference_domains"] = {}
    for domain in policy["domains"]:
        width = (
            138
            if action_key == "actions_keypoints"
            else (14 if domain == "eva_bimanual" else 12)
        )
        observations = [
            KEYS[k]
            for k in [*policy["stem_specs"][domain], *policy["shared_stem_specs"]]
        ]
        observations = [
            "annotations" if k == "observations.annotation" else k for k in observations
        ]
        control_path = (
            None
            if deterministic
            else (
                "num_inference_steps"
                if shared or len(head_specs) == 1
                else f"branches.{domain}.stages.1.num_inference_steps"
            )
        )
        result["inference_domains"][domain] = contract(
            domain,
            width=width,
            action_key=action_key,
            observation_keys=observations,
            sampler_index=sampler_index,
            control_path=control_path,
            representation="mano_keypoints" if width == 138 else "cartesian",
            action_stride=1 if domain == "eva_bimanual" else human_stride,
        )
    result["inference"] = "${model.inference_domains.${model.deployment_domain}}"
    if policy.get("ot"):
        left, right = policy["domains"]
        stages[int(language) + 1]["representation_block"] = policy["depth"]
        if policy.get("freeze_repr"):
            stages[int(language) + 1]["detach_condition_before_block"] = policy.get(
                "freeze_depth", 8
            )
        result["pipeline"]["training_passes"] = {
            "alignment": {
                "stage_ids": ["stems", "trunk"],
                "outputs": {"representation": "hpt/representation"},
            }
        }
        result["pipeline"]["loss_pipeline"] = {
            "_target_": "egomimic.pipeline.core.Pipeline",
            "stage_ids": {"alignment": 0, "reduction": 1},
            "stages": [
                {
                    "_target_": "egomimic.pipeline.stages_alignment.RepresentationAlignment",
                    "left_features": f"source/{left}/pass/alignment/representation",
                    "right_features": f"source/{right}/pass/alignment/representation",
                    "left_actions": f"input/{left}/{action_key}",
                    "right_actions": f"input/{right}/{action_key}",
                    "cost_layout": "legacy_broadcast",
                    "left_action_indices": list(
                        range(12 if policy.get("ot_6dof") else 3)
                    ),
                    "right_action_indices": list(
                        range(12 if policy.get("ot_6dof") else 3)
                    ),
                    "supervision": ("soft_dtw" if policy.get("dtw") else "mse")
                    if policy.get("supervised")
                    else None,
                    "match_weight": policy.get("lambda", 0.5),
                    "blur": 0.05,
                    "truncate": 18,
                    "dtw_gamma": 0.1,
                },
                {
                    "_target_": "egomimic.pipeline.stages_alignment.ScheduledLossReduction",
                    "components": {
                        "source_losses": {
                            "weight": 1.0,
                            "start_batch": policy.get("warm_start_steps", 0),
                        },
                        "alignment/loss": {
                            "weight": policy.get("temperature", 1.0),
                            "start_batch": policy.get("ot_warm_start_steps", 0),
                        },
                    },
                    "denominator_key": "source_count",
                },
            ],
        }
    return result


def pi_recipe(old, *, human_stride=3):
    policy = relocated(old["robomimic_model"])
    if "domains" not in policy:
        return (
            None  # Base fragment is emitted as a defaults alias after concrete recipes.
        )
    policy["_target_"] = "egomimic.models.pi05.policy.PI"
    policy["action_encoding"] = "legacy_normalized_ypr_rot6d"
    result = {
        key: deepcopy(value) for key, value in old.items() if key != "robomimic_model"
    }
    result["pipeline"] = {
        "_target_": "egomimic.pipeline.algo.PipelineAlgo",
        "stage_ids": {"sampler": 0},
        "stages": [
            {
                "_target_": "egomimic.pipeline.stages_pi05.PI05Stage",
                "_recursive_": False,
                "policy": policy,
            }
        ],
    }
    result["deployment_domain"] = policy["domains"][0]
    result["inference_domains"] = {}
    for domain in policy["domains"]:
        width = 14 if domain == "eva_bimanual" else 12
        declaration = contract(
            domain,
            width=width,
            action_key=policy["ac_keys"][domain],
            observation_keys=[
                "observations.state.ee_pose",
                "observations.images.front_img_1",
                "annotations",
            ],
            sampler_index=0,
            control_path="backend.num_steps",
            steps=10,
            action_stride=1 if domain == "eva_bimanual" else human_stride,
        )
        declaration["compatibility"].update(
            action_encoding=policy["action_encoding"],
            backend_native_output={
                "shape": [100, 32],
                "conversion": policy["action_converters"],
            },
        )
        result["inference_domains"][domain] = declaration
    result["inference"] = "${model.inference_domains.${model.deployment_domain}}"
    return result


def main():
    models = json.loads(EVIDENCE.read_text())["models"]
    for name, old in models.items():
        human_stride = 1 if "mecka" in name or "scale" in name else 3
        value = (pi_recipe if name.startswith("pi") else hpt_recipe)(
            old, human_stride=human_stride
        )
        if value is None:
            continue
        destination = CONFIG / f"{name}.yaml"
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite a reviewed recipe: {destination}"
            )
        destination.write_text(
            "# Graph port of EgoVerse ec5c903c; see docs/integration/RECIPE_PARITY.md.\n"
            + yaml.dump(value, Dumper=Dumper, sort_keys=False)
        )


if __name__ == "__main__":
    main()
