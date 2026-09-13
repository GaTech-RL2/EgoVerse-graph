#!/usr/bin/env bash
# Task-local ICE port of the live Planar cotrain launcher:
# /coc/flash7/paphiwetsa3/scripts/train/planar_v2_dp_cotrain_clean_skynet_v1.sbatch
# live-launcher SHA-256: 4789831510204e14bfa298285c36f5bdbe9011c6af8d8d0cca8414c8adc2504c
# This variant keeps the same fresh-run and scheduled-validation discipline for
# current-main Paper DP ARC experiments, with no automatic requeue or resume.

#SBATCH --no-requeue
#SBATCH --output=/storage/ice1/2/9/tlertuss3/egoverse/paper-dp-arc-main-20260912/logs/pda-main-%j.out
#SBATCH --error=/storage/ice1/2/9/tlertuss3/egoverse/paper-dp-arc-main-20260912/logs/pda-main-%j.err

set -Eeuo pipefail
umask 077

MODE=${MODE:-smoke}
EXPECTED_HEAD=${EXPECTED_HEAD:?set exact source commit}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set the launcher sha256}
EXPERIMENT=${EXPERIMENT:?set a Hydra experiment, e.g. pusht/planar_v2_cotrain_obstacle_paper_dp}
PRECOMPUTED_NORM_DIR=${PRECOMPUTED_NORM_DIR:-}
RUN_SLUG=${RUN_SLUG:?set a unique run slug}
ROOT=${ICE_ARC_ROOT:-/storage/ice1/2/9/tlertuss3/egoverse/paper-dp-arc-main-20260912}
REPO=${ICE_ARC_REPO:-$ROOT/source/${EXPECTED_HEAD:0:8}}
PY=${ICE_ARC_PYTHON:-/storage/ice1/2/9/tlertuss3/egoverse/paper-dp-arc-waypoint-comparison-20260910/runtime/train/bin/python}
U_DATA=${ICE_ARC_U_DATA:-/storage/ice1/2/9/tlertuss3/egoverse/paper-dp-arc-obstacle-cotrain-20260911/datasets/u_socket_scripted_c3kb_ideal_3000_20260905_flat}
CHAIN_DATA=${ICE_ARC_CHAIN_DATA:-/storage/ice1/2/9/tlertuss3/egoverse/paper-dp-arc-obstacle-cotrain-20260911/datasets/chain_gripper_3000_v2_plus_gen_frozen1920_20260911}
LAUNCHER=${BASH_SOURCE[0]}

fail() { printf '[%s] FAIL line=%s status=%s command=%q\n' "$MODE" "$1" "$2" "$3" >&2; }
trap 'fail "$LINENO" "$?" "$BASH_COMMAND"' ERR

case "$MODE" in smoke|full) ;; *) printf 'MODE must be smoke or full\n' >&2; exit 64 ;; esac
test "${#EXPECTED_HEAD}" = 40
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -x "$PY" && test -d "$U_DATA" && test -d "$CHAIN_DATA"
test -z "${CKPT_PATH:-}" && test -z "${ICE_INITIAL_CHECKPOINT:-}"

test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "${SLURM_RESTART_COUNT:-0}" = 0
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1
case "$(nvidia-smi --query-gpu=name --format=csv,noheader)" in
  'NVIDIA H100 80GB HBM3'|'NVIDIA H200') ;;
  *) printf 'require exactly one H100 or H200\n' >&2; exit 64 ;;
esac

RUN_DIR=$ROOT/runs/${RUN_SLUG}-${MODE}-job-${SLURM_JOB_ID:?}
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/provenance"
cd "$REPO"
export PYTHONPATH=$REPO HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=online WANDB_SILENT=true

printf '%s\n' "$EXPECTED_HEAD" > "$RUN_DIR/provenance/source_head.txt"
printf '%s\n' "$EXPERIMENT" > "$RUN_DIR/provenance/experiment.txt"
printf '%s\n' "$RUN_SLUG" > "$RUN_DIR/provenance/run_slug.txt"
sha256sum "$LAUNCHER" > "$RUN_DIR/provenance/launcher.sha256"
scontrol show job -dd "$SLURM_JOB_ID" > "$RUN_DIR/provenance/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/provenance/nvidia-smi-q.txt"

COMMON=(
  --config-name=train_zarr_cartesian "+experiment=$EXPERIMENT" mode=train ckpt_path=null
  "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  "data.train_datasets.pushshapes_sim_chain_gripper.resolver.folder_path=$CHAIN_DATA"
  "data.valid_datasets.pushshapes_sim_chain_gripper.resolver.folder_path=$CHAIN_DATA"
  "++paths.root_dir=$RUN_DIR" "paths.output_dir=$RUN_DIR" "paths.work_dir=$REPO"
  "hydra.run.dir=$RUN_DIR" launch_params.gpus_per_node=1 launch_params.nodes=1
  trainer.devices=1 trainer.num_nodes=1 trainer.strategy=ddp trainer.precision=bf16
  norm_stats.sample_frac=1.0 norm_stats.save_cache_dir=null
  logger.wandb.entity=rl2-group logger.wandb.project=pushshapes-planar-v2
  logger.wandb.group=paper-dp-arc-main-20260912
  "++logger.wandb.id=${RUN_SLUG}-${MODE}-${SLURM_JOB_ID}"
  "++logger.wandb.name=${RUN_SLUG}-${MODE}-${SLURM_JOB_ID}" ++logger.wandb.resume=never
)

if test "$MODE" = smoke; then
  COMMON+=(trainer.max_steps=2 trainer.limit_train_batches=2 trainer.val_check_interval=1)
  COMMON+=(trainer.limit_val_batches=1 trainer.num_sanity_val_steps=0 trainer.log_every_n_steps=1)
else
  SMOKE_DIR=${PASSED_SMOKE_DIR:?full mode requires a passing same-experiment smoke directory}
  grep -q '"status": "passed"' "$SMOKE_DIR/SMOKE_RESULT.json"
  test "$(cat "$SMOKE_DIR/provenance/source_head.txt")" = "$EXPECTED_HEAD"
  test "$(cat "$SMOKE_DIR/provenance/experiment.txt")" = "$EXPERIMENT"
  test -s "$SMOKE_DIR/norm_stats/norm_stats.json"
  test -s "$SMOKE_DIR/provenance/split.json"
  COMMON+=("norm_stats.precomputed_norm_path=$SMOKE_DIR/norm_stats")
fi

"$PY" scripts/data/materialize_local_episode_split.py \
  --domain pushshapes_sim_u_socket="$U_DATA" \
  --domain pushshapes_sim_chain_gripper="$CHAIN_DATA" \
  --valid-ratio 0.01 --seed 42 --output "$RUN_DIR/provenance/split.json"
grep -q '"status": "PASS"' "$RUN_DIR/provenance/split.json"

"$PY" -m egomimic.trainHydra "${COMMON[@]}" --cfg job --resolve \
  > "$RUN_DIR/provenance/resolved_config.yaml"
grep -q 'every_n_train_steps: 60000' "$RUN_DIR/provenance/resolved_config.yaml"
grep -q 'save_weights_only: true' "$RUN_DIR/provenance/resolved_config.yaml"
grep -q 'save_last: false' "$RUN_DIR/provenance/resolved_config.yaml"
grep -q 'save_top_k: -1' "$RUN_DIR/provenance/resolved_config.yaml"

if test "$MODE" = smoke; then
  if test -n "$PRECOMPUTED_NORM_DIR"; then
    test -s "$PRECOMPUTED_NORM_DIR/norm_stats.json"
    mkdir -p "$RUN_DIR/norm_stats"
    cp --reflink=auto "$PRECOMPUTED_NORM_DIR/norm_stats.json" \
      "$RUN_DIR/norm_stats/norm_stats.json"
    test "$(sha256sum "$RUN_DIR/norm_stats/norm_stats.json" | awk '{print $1}')" = \
      "$(sha256sum "$PRECOMPUTED_NORM_DIR/norm_stats.json" | awk '{print $1}')"
    COMMON+=("norm_stats.precomputed_norm_path=$PRECOMPUTED_NORM_DIR")
    printf '%s\n' "$PRECOMPUTED_NORM_DIR" > "$RUN_DIR/provenance/precomputed_norm_dir.txt"
  else
    "$PY" -m egomimic.trainHydra "${COMMON[@]}" norm_stats_only=true \
      trainer.accelerator=cpu trainer.devices=1 "norm_stats.save_cache_dir=$RUN_DIR" \
      "hydra.run.dir=$RUN_DIR/norm_materialize"
    test -s "$RUN_DIR/norm_stats/norm_stats.json"
    COMMON+=("norm_stats.precomputed_norm_path=$RUN_DIR/norm_stats")
  fi
  COMMON+=(callbacks.model_checkpoint.every_n_train_steps=2)
  "$PY" -m egomimic.trainHydra "${COMMON[@]}" --cfg job --resolve \
    > "$RUN_DIR/provenance/train_resolved_config.yaml"
  grep -q 'train_log_on_step: true' "$RUN_DIR/provenance/train_resolved_config.yaml"
  grep -q 'log_every_n_steps: 1' "$RUN_DIR/provenance/train_resolved_config.yaml"
  grep -q 'every_n_train_steps: 2' "$RUN_DIR/provenance/train_resolved_config.yaml"
fi

srun --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 --unbuffered \
  "$PY" -m egomimic.trainHydra "${COMMON[@]}" \
  2>&1 | tee "$RUN_DIR/training.log"

if test "$MODE" = smoke; then
  "$PY" scripts/train/verify_planar_paper_dp_smoke.py "$RUN_DIR" \
    --expected-head "$EXPECTED_HEAD" --expected-experiment "$EXPERIMENT"
  sha256sum "$RUN_DIR/provenance/resolved_config.yaml" \
    "$RUN_DIR/provenance/split.json" "${PRECOMPUTED_NORM_DIR:-$RUN_DIR/norm_stats}/norm_stats.json" \
    > "$RUN_DIR/provenance/SHA256SUMS"
  touch "$RUN_DIR/SMOKE_PASS"
  printf '[smoke] ALL_PASS run=%s wandb=%s\n' "$RUN_DIR" \
    "https://wandb.ai/rl2-group/pushshapes-planar-v2/runs/${RUN_SLUG}-${MODE}-${SLURM_JOB_ID}"
else
  CHECKPOINT_COUNT=$(find "$RUN_DIR" -maxdepth 3 -type f -name '*.ckpt' | wc -l)
  test "$CHECKPOINT_COUNT" -eq 4
  printf '[full] PASS run=%s checkpoints=%s wandb=%s\n' "$RUN_DIR" "$CHECKPOINT_COUNT" \
    "https://wandb.ai/rl2-group/pushshapes-planar-v2/runs/${RUN_SLUG}-${MODE}-${SLURM_JOB_ID}"
fi
