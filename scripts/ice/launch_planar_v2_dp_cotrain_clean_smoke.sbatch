#!/bin/bash
# ICE smoke for standard DP cotraining on clean U-Socket + base ChainGripper.

set -Eeuo pipefail

MODE=${MODE:-smoke}
EXPECTED_HEAD=${EXPECTED_HEAD:?set EXPECTED_HEAD}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set EXPECTED_LAUNCHER_SHA}
ROOT=/storage/ice1/5/4/paphiwetsa3/egoverse/planar-v2-dp-cotrain-clean-20260903
REPO=$ROOT/source/EgoVerse-${EXPECTED_HEAD:0:8}
PY=/storage/ice1/5/4/paphiwetsa3/egoverse/unite-usocket-20260902/env/.venv/bin/python
LAUNCHER=$ROOT/tools/launch_planar_v2_dp_cotrain_clean_smoke.sbatch
U_DATA=$ROOT/data/u_socket_3000_v2_clean
CHAIN_DATA=$ROOT/data/chain_gripper_3000_v2
EXPERIMENT=pusht/planar_v2_cotrain_clean_dp_standard
PARAMETER_COUNT=51506981

fail() { printf '[%s] FAIL line=%s status=%s command=%q\n' "$MODE" "$1" "$2" "$3" >&2; }
trap 'fail "$LINENO" "$?" "$BASH_COMMAND"' ERR
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain)"
test -x "$PY" && test -d "$U_DATA" && test -d "$CHAIN_DATA"
cd "$REPO"
export PYTHONPATH=$REPO HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

preflight() {
  local output=$1
  mkdir -p "$output"
  "$PY" scripts/data/materialize_local_episode_split.py \
    --domain pushshapes_sim_u_socket="$U_DATA" \
    --domain pushshapes_sim_chain_gripper="$CHAIN_DATA" \
    --valid-ratio 0.01 --seed 42 --output "$output/split_audit.json"
  "$PY" -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    "+experiment=$EXPERIMENT" \
    "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA" \
    "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA" \
    "data.train_datasets.pushshapes_sim_chain_gripper.resolver.folder_path=$CHAIN_DATA" \
    "data.valid_datasets.pushshapes_sim_chain_gripper.resolver.folder_path=$CHAIN_DATA" \
    "++paths.root_dir=$output" "paths.output_dir=$output" "paths.work_dir=$REPO" \
    launch_params.gpus_per_node=1 launch_params.nodes=1 norm_stats.sample_frac=1.0 \
    --cfg job --resolve > "$output/resolved_config.yaml"
  "$PY" - "$output/resolved_config.yaml" "$output/split_audit.json" "$PARAMETER_COUNT" <<'PY'
import json, pathlib, sys
from hydra.utils import instantiate
from omegaconf import OmegaConf
c = OmegaConf.load(sys.argv[1]); s = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert s["status"] == "PASS" and s["cross_domain_train_valid_resolved_path_overlap_count"] == 0
assert {(d["total_count"], d["train_count"], d["valid_count"]) for d in s["domains"].values()} == {(2999, 2970, 29), (3000, 2970, 30)}
assert set(c.data.train_datasets) == {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
assert c.run_provenance.obstacle_data is False and c.norm_stats.sample_frac == 1.0
m = instantiate(c.model.pipeline)
assert sum(p.numel() for p in m.nets.parameters()) == int(sys.argv[3])
print("[preflight] READY exact source/data/split/config/parameters")
PY
}

if test "$MODE" = preflight; then
  PREFLIGHT=$ROOT/preflight/${EXPECTED_HEAD:0:12}
  test ! -e "$PREFLIGHT"
  preflight "$PREFLIGHT"
  exit 0
fi
test "$MODE" = smoke
test "${SLURM_JOB_NUM_NODES:?}" = 1 && test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8 && test "${SLURM_RESTART_COUNT:-0}" = 0
RUN_DIR=$ROOT/runs/smoke-job-${SLURM_JOB_ID:?}
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/provenance"
preflight "$RUN_DIR/provenance"

nvidia-smi -q > "$RUN_DIR/provenance/nvidia-smi-q.txt"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
assert torch.isfinite(y).all()
print("[hardware] CUDA BF16 PASS", torch.cuda.get_device_name())
PY

WANDB_ID=planar-v2-dp-cotrain-clean-ice-smoke-${SLURM_JOB_ID}
COMMON=(
  --config-name=train_zarr_cartesian "+experiment=$EXPERIMENT" mode=train ckpt_path=null
  "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  "data.train_datasets.pushshapes_sim_chain_gripper.resolver.folder_path=$CHAIN_DATA"
  "data.valid_datasets.pushshapes_sim_chain_gripper.resolver.folder_path=$CHAIN_DATA"
  "++paths.root_dir=$RUN_DIR" "paths.output_dir=$RUN_DIR" "paths.work_dir=$REPO"
  "hydra.run.dir=$RUN_DIR" launch_params.gpus_per_node=1 launch_params.nodes=1
  trainer.devices=1 trainer.num_nodes=1 trainer.strategy=ddp trainer.precision=bf16
  trainer.max_steps=2 trainer.limit_train_batches=2 trainer.val_check_interval=1
  trainer.limit_val_batches=1 trainer.num_sanity_val_steps=0 trainer.log_every_n_steps=1
  norm_stats.sample_frac=1.0 norm_stats.save_cache_dir=null
  callbacks.model_checkpoint.monitor=null callbacks.model_checkpoint.save_top_k=-1
  callbacks.model_checkpoint.save_last=true callbacks.model_checkpoint.every_n_epochs=null
  callbacks.model_checkpoint.every_n_train_steps=1 callbacks.model_checkpoint.train_time_interval=null
  callbacks.model_checkpoint.save_on_train_epoch_end=false
  "callbacks.model_checkpoint.filename='epoch-{epoch}-step-{step}'"
  logger.wandb.entity=rl2-group logger.wandb.project=pushshapes-planar-v2
  logger.wandb.group=planar-v2-dp-cotrain-clean-20260903
  "logger.wandb.id=$WANDB_ID" "logger.wandb.name=$WANDB_ID" logger.wandb.resume=never
  "logger.wandb.tags=[standard-dp,cotrain,clean-only,no-obstacle,planar-v2,h16,ddim100,smoke,ice]"
)
export WANDB_MODE=online WANDB_SILENT=true
"$PY" -m egomimic.trainHydra "${COMMON[@]}" norm_stats_only=true \
  trainer.accelerator=cpu trainer.devices=1 logger=null \
  "norm_stats.save_cache_dir=$RUN_DIR" "hydra.run.dir=$RUN_DIR/norm_materialize"
test -s "$RUN_DIR/norm_stats/norm_stats.json"
srun --kill-on-bad-exit=1 --unbuffered "$PY" -m egomimic.trainHydra "${COMMON[@]}" \
  "norm_stats.precomputed_norm_path=$RUN_DIR/norm_stats" 2>&1 | tee "$RUN_DIR/training.log"
"$PY" scripts/train/verify_planar_training_smoke.py "$RUN_DIR" \
  --expected-head "$EXPECTED_HEAD" --world-size 1 --parameter-count "$PARAMETER_COUNT"
sha256sum "$RUN_DIR/norm_stats/norm_stats.json" > "$RUN_DIR/provenance/norm_stats.sha256"
printf '[smoke] ALL_PASS job=%s wandb=%s\n' "$SLURM_JOB_ID" \
  "https://wandb.ai/rl2-group/pushshapes-planar-v2/runs/$WANDB_ID"
