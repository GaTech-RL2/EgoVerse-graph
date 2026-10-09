#!/usr/bin/env bash
# The maintained launcher and CPU composition guard use this exact argv builder.
# Requires the launcher-verified corpus counts, identities and typed environment.
yam_action_flow_arguments() {
  overrides=(
    "trainer.accumulate_grad_batches=$AF_YAM_ACCUMULATION"
    "++run_provenance.grad_accumulation=$AF_YAM_ACCUMULATION"
    "e1.data_root=$AF_YAM_DATA_ROOT"
    "norm_stats.precomputed_norm_path=$AF_YAM_NORM_JSON"
    "hydra.run.dir=$AF_YAM_RUN_DIR"
    "paths.output_dir=$AF_YAM_RUN_DIR"
    "run_provenance.source_commit=$AF_YAM_SOURCE_COMMIT"
    "run_provenance.normalization_sha256=$AF_YAM_NORM_SHA"
    "run_provenance.dataset_content_aggregate_sha256=$content_sha"
    "++logger.wandb.entity=$AF_YAM_WANDB_ENTITY"
    "++logger.wandb.project=$AF_YAM_WANDB_PROJECT"
    "++logger.wandb.id=$AF_YAM_WANDB_ID"
    "++logger.wandb.name=$AF_YAM_WANDB_ID"
    '++logger.wandb.resume=allow'
    'trainer.devices=1'
    'trainer.num_nodes=1'
  )
  if test "$AF_YAM_RECIPE" = yam_human_keypoints; then
    overrides=(
      "+experiment=e1/${AF_YAM_EXPERIMENT:-yam_human_keypoints_action_flow_h816_private512_s42}"
      '++run_provenance.launch_recipe=yam_human_keypoints'
      "${overrides[@]}"
    )
    for domain in yam human; do
      if test "$domain" = yam; then source_name=yam_bimanual; else source_name=human_bimanual; fi
      train_var=${domain}_train_count; valid_var=${domain}_valid_count; windows_var=${domain}_train_windows
      train_count=${!train_var}; valid_count=${!valid_var}; train_windows=${!windows_var}
      for target in "data.train_datasets.$source_name" "data.valid_datasets.$domain.$source_name"; do
        overrides+=("$target.expected_train_episode_count=$train_count" "$target.expected_valid_episode_count=$valid_count")
      done
      overrides+=(
        "run_provenance.train_episode_count_per_domain.$source_name=$train_count"
        "run_provenance.valid_episode_count_per_domain.$source_name=$valid_count"
        "run_provenance.union_episode_count_per_domain.$source_name=$((train_count + valid_count))"
        "run_provenance.train_windows_per_domain.$source_name=$train_windows"
      )
    done
    overrides+=(
      "run_provenance.split_manifest_sha256=$AF_YAM_SPLIT_SHA"
      "model.pipeline.stages.6.inference_method=$AF_YAM_INFERENCE_METHOD"
      "run_provenance.inference.sampler=$AF_YAM_INFERENCE_METHOD"
    "evaluator.energy_score_provenance.sampler=$AF_YAM_INFERENCE_METHOD"
    "evaluator.action_flow_diagnostics.provenance.sampler=$AF_YAM_INFERENCE_METHOD"
      "++run_provenance.execution.checkpoint_policy=$AF_YAM_CHECKPOINT_POLICY"
    )
    if test "$AF_YAM_CHECKPOINT_POLICY" = dit_half; then
      overrides+=(
        '++model.pipeline.stages.4.encoders.yam_bimanual.backbone.checkpoint_policy=dit_half'
        '++model.pipeline.stages.4.encoders.human_bimanual.backbone.checkpoint_policy=dit_half'
        '++model.pipeline.stages.6.field.backbone.checkpoint_policy=dit_half'
        '++callbacks.yam_dit_half._target_=egomimic.utils.yam_dit_half.YamHumanDiTHalf'
      )
    fi
  else
    overrides=(
      '+experiment=e1/yam_stationary_action_flow_h816_s42'
      "${overrides[@]}"
    )
  fi
  if test "$AF_YAM_PHASE" = smoke; then
    smoke_checkpoint_interval=1
    if test "$AF_YAM_RECIPE" = yam_human_keypoints; then
      smoke_checkpoint_interval=4
    fi
    overrides+=(
      'trainer.max_steps=4'
      "trainer.val_check_interval=$((4 * AF_YAM_ACCUMULATION))"
      '+trainer.log_every_n_steps=1'
      'trainer.limit_val_batches=1'
      "callbacks.model_checkpoint.every_n_train_steps=$smoke_checkpoint_interval"
      'model.gradient_telemetry_cadence=3'
    )
  fi
  
  if test "$AF_YAM_PHASE" = norm; then
    if test -n "${AF_YAM_NORM_RECOVERY_JSON:-}${AF_YAM_NORM_RECOVERY_SHA:-}"; then
      test -n "${AF_YAM_NORM_RECOVERY_JSON:-}" && test -n "${AF_YAM_NORM_RECOVERY_SHA:-}" || return 64
      test "$(sha256sum "$AF_YAM_NORM_RECOVERY_JSON" | cut -d' ' -f1)" = "$AF_YAM_NORM_RECOVERY_SHA" || return 64
      overrides+=("++norm_stats.resume_partial_norm_path=$AF_YAM_NORM_RECOVERY_JSON")
    fi
    overrides+=(
      'norm_stats_only=true'
      'norm_stats.precomputed_norm_path=null'
      "++norm_stats.save_cache_dir=$AF_YAM_NORM_CACHE_DIR"
      'run_provenance.normalization_sha256=null'
    )
  fi
}
