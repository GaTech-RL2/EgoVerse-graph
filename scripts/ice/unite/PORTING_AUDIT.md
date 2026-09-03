# UNITE ICE behavior audit

This branch starts from generic ICE foundation commit
`3a78bed15d8c320d08cd4bfc452c7bd063355d3e`. Historical code from
`2133a92614b24948fc4d273add588452d0e9f091` is behavioral evidence only.
Deployment identities remain manifest or command-line inputs.

| Historical behavior | Decision | Clean implementation | Replacement test |
| --- | --- | --- | --- |
| EMA resume baseline | keep | `egomimic/utils/ema_callback.py` restores the checkpoint update count and uses it as the resumed optimizer-step baseline. | `tests/test_ema_resume.py` |
| Explicit requeue ownership | keep | `egomimic/trainHydra.py`, `egomimic/utils/slurm_requeue.py`, and `scripts/ice/ice_requeue_runner.py` keep Lightning save-only while the runner alone calls `scontrol requeue`. | `tests/test_slurm_save_only_checkpoint.py`, `tests/test_train_hydra_slurm_environment.py`, `tests/test_ice_requeue_runner.py` |
| Single-GPU UNITE smoke validation | adapt | The removed legacy smoke verifier is not restored. The clean evaluator and UNITE contract validator accept an explicit world size and validate one-rank and multi-rank configurations through the current API. | `tests/test_unite_ice_contract.py` |
| Single-GPU barrier guards | reject | The clean `ModelWrapper` no longer performs the historical unconditional fit/validation barriers. Reintroducing them would restore dead synchronization rather than fix a current defect. Save-only checkpoint synchronization remains world-size aware. | `tests/test_slurm_save_only_checkpoint.py` |
| World-size-bound EnergyScore validation | keep | The UNITE experiment binds `validation_view.world_size` to `trainer.devices`; the clean contract validates `per_rank_batch_size * world_size == 32`. | `tests/test_unite_register.py`, `tests/test_unite_ice_contract.py` |
| World-size UNITE diagnostic validation | adapt | Diagnostics are optional on this clean stack. When present, the clean contract applies the same total-batch/world-size identity; absence is recorded and cannot be claimed as diagnostic coverage. | `tests/test_unite_ice_contract.py` |
| Hydra-safe string overrides | keep | `egomimic.utils.hydra_override` emits one exact quoted scalar override and rejects interpolation/control characters. | `tests/test_hydra_override.py` |

## Operational file decisions

| Historical file family | Decision | Reason |
| --- | --- | --- |
| Generic runner, mirror, runtime lock, GPU probe | keep | Already represented by the reviewed generic files under `scripts/ice/`; no duplicate is added. |
| Historical production launcher | reject | Repository launchers cannot replace the task-local SHA-verified mirror of the live Skynet launcher. |
| UNITE full-state validator | adapt | Retain strict model/optimizer/scheduler/EMA and adjacent-config validation behind explicit inputs. |
| Candidate staging and ICE-to-Skynet access selection | adapt | Remove fixed usernames, remotes, experiment roots, and row names; bind them through immutable artifacts and CLI arguments. |
| Resume/cutover records | adapt | Preserve create-only, hash-bound state transitions while removing account, user, W&B project, and job identities from code. |
| Fixed-step requeue harness | adapt | Preserve seed step 3, save boundary step 8, resume to step 11, and validation at the absolute terminal step. It remains an integration test, never a production launcher. |

## Acceptance boundary

Passing repository tests makes the pull request review-ready. Production
authority additionally requires a fresh clean-source optimizer-plus-validation
smoke, real requeue with `Restarts=1`, strict resume, terminal `COMPLETE.json`,
and an independently verified CPU mirror. Historical jobs do not satisfy that
gate.
