#!/bin/bash
# Stationery mid-tempo smoke: both rows on the 4-episode subset (splits smoke_v2), then an
# eval-only open-loop pass of each smoke checkpoint on the 2-episode smoke test_mid.
# Exercises: data resolve (ABC + rl2), norm stats cache, 20 train steps, in-training
# open_loop_sim on 1 whole episode, checkpoint save, eval-only loading + open_loop_sim.json.
#
#   bash scripts/e1/stationery_midtempo_smoke.sh
set -euo pipefail
G=/storage/project/r-dxu345-0/agao81/EgoVerse-graph
S=/storage/project/r-dxu345-0/agao81/runs/stationery_midtempo/smoke
cd "$G"
for ROW in time arcdur; do
  EXTRA="data=abc_arc/stationery_midtempo_smoke_${ROW} trainer.max_steps=20 trainer.max_epochs=2 trainer.limit_train_batches=10 trainer.check_val_every_n_epoch=1 evaluator.limit_val_episodes=1 callbacks.model_checkpoint.every_n_epochs=1 description=smoke_${ROW}_s42"
  TJ=$(sbatch --parsable --job-name=midtempo-smoke-${ROW} -t 1:30:00 \
    --export=ALL,ROW=${ROW},RUN=${S}/${ROW},EXTRA="${EXTRA}" scripts/e1/stationery_midtempo_train.sbatch)
  EJ=$(sbatch --parsable --job-name=midtempo-smokeeval-${ROW} -t 1:00:00 --dependency=afterok:${TJ} \
    --export=ALL,ROW=${ROW},RUN=${S}/${ROW},SETS=test_mid,DATA_PREFIX=stationery_midtempo_smoke,CKPTS=${S}/${ROW}/train/checkpoints/last.ckpt \
    scripts/e1/stationery_midtempo_eval.sbatch)
  echo "${ROW} train=${TJ} eval=${EJ}"
done
