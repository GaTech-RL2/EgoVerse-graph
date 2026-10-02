# EgoVerse-graph consolidation

The three requested histories are integrated on
`codex/graph-consolidation-20261001`. Code commit
`96df46f1427cd08e0dd64ac534af133ee37b193c` passes **1,993 tests with no failures
or skips on macOS**, recursive composition of **439 YAML contexts**, and the included
constructor and installed-wheel checks. This is CPU acceptance; real-weight/data
GPU execution and the destination merge remain pending.

At the user's request, tests, documentation and validation artifacts are now
preserved on `codex/graph-validation-20261001`, separately from the lightweight
runtime branch. CI uses an immutable companion pin and the runtime checkout's
code. See [validation branch usage and the initial Linux hash failure](VALIDATION_BRANCH.md).
The original local receipt above is historical evidence, not a claim that the
first Linux CI run passed.

## Sources and review map

| Source | Pinned head | Model / data / experiment / evaluator YAMLs |
| --- | --- | --- |
| A, #197 and 22 ancestors | `20507c6866f8a167e0e2858141299dbf6ccddeb0` | 29 / 39 / 70 / 10 |
| B, #159 and nine ancestors | `eda6f8d177dec8df26ad62e3784d103b4de74d7d` | 55 / 63 / 72 / 26 |
| C, #160 | `bef887c4c46bae171711951849fbdbf1617ea534` | 44 / 136 / 130 / 10 |
| Integrated | code commit above | 67 / 153 / 151 / 27 |

Base main: `161e3a0c40182ba003434d3adaeaa6cbd12107d8`. An independent clone
isolates this work from other agents. B supplies the graph lifecycle and bound
checkpoint contracts, A the newer ARC semantics, and C unique training/data/grid
capabilities. History-preserving merges retain all 34 source PR heads as
ancestors. Published source branches and other worktrees remain intact.

- [All 34 PR dispositions](SOURCE_PROVENANCE_20261001.md).
- [All 128 source model dispositions](MODEL_DISPOSITIONS_20261001.md) and
  [shared definitions](CONFIG_DEDUPLICATION.md).
- [ARC schemas, clocks, windows and compatibility](ARC_CONSOLIDATED_CONTRACT.md).
- [48 conflict-path decisions](evidence/consolidation/conflict-resolutions.json).
- [Validation receipt](evidence/consolidation/validation.json),
  [JUnit cases](evidence/consolidation/cpu-junit.xml.gz),
  [64-grid equivalence](evidence/consolidation/grid-parity.json), and
  [byte-preserved pool manifests](evidence/consolidation/pool-manifests.json).

## Material decisions

| Area | Integrated behavior and reason |
| --- | --- |
| ARC math | A's per-arm rotation clocks, race/multistream/joint-distance modes, holds and M28 fixes are authoritative. Incoming obsolete methods that shadowed them are deleted. |
| Source window | Explicit validated `yam_source_frames` restores C's w400 behavior; default YAM time remains 100. Human and dynamic-distance modes remain distinct. |
| Homogeneous training | Compatible tensors stack through shared stages; outputs split back for losses/diagnostics. Incompatible shapes/routes execute independently. PI prepares each source separately, then groups compatible padded observations/actions. |
| Loss weighting | Existing B graphs explicitly retain source-mean loss. Weighted training and C grids use sample-mean loss, so mixture probabilities affect actual losses. |
| Sampling | Dataset probabilities are independent of length. Group balancing shards one seeded global draw and advances with epoch. Ordered evaluation rejects alternate sampling. |
| Normalization | IDs come from samples, not source labels. Same-ID sources fit pooled statistics in normalization mode or restore a full cache. Incompatible preprocessing/shape/action contracts reject. Export includes immutable data context. |
| Initialization | Exact key namespace; matching shapes transfer and changed shapes retain fresh initialization and are logged. EMA parameters overlay online buffers. Key/nonfinite errors reject before loading. Resume/requeue wins; weights-only loading does not import optimizer/step state. |
| Runtime ownership | B's model declarations, token order, diagnostics, preflight-before-model construction and checkpoint bindings survive. New formats are declared in YAML. |
| Visual recipes | A's pretrained ResNet+MLP and train-only jitter are canonical. Direct-ResNet checkpoints retain their saved architecture. E1 stems and proprio/action widths remain distinct. |
| Evaluation | A's full-frame metrics/video and DTW, B's generic lifecycle, and C's explicit E1 layouts/episode exports. Standalone eval restores a bound context without training-data access. |
| Recipes | B-referenced language/DP recipes remain callable. C's 64 grids and source lists remain intact. A visual recipes use 10% normalization; historical Lambda 20% preflight now selects it explicitly. |

C's additions remain in WeightedDataset, batch_utils, stage execute_batches,
ZarrDataModule, group balancing, fine-tune initialization, yam_arc_grid,
Lambda/E1 utilities and manifests. Its trainer source-name heuristic is replaced
by data-owned identity, with black-box pooled/restored-normalization tests.

## Removal and migration ledger

| Removed duplication | Replacement / evidence |
| --- | --- |
| Five identical B model copies | Thin aliases; all 254 composed contexts unchanged in that layer; callers migrated. |
| Two Qwen 180M copies | Shared 300M graphs with explicit flow-head sizes, retaining distinct capacity. |
| E1 arcdur FT, HPT300, pretrained-DP copies | Explicit width, trunk/head and pretraining overrides; six model and all 64 grid comparisons. |
| E1 inference/data declaration copies | model_contract/e1_wrists; binds source window, actual action width and sampler controls. |
| Duplicate hybrid methods / DualHPTTrunkStage | Per-arm and mode-aware implementations; independent numeric/parity tests and AST duplicate audit. |
| Identical duplicate PI _mask_from_batch | One implementation; PI graph and grouped preparation checks. |
| Absolute repository episode-list paths | Packaged allowlists, included in the wheel; identical contents and pre-split membership; missing/empty files fail. |
| Earlier A root visual recipe names | A already moved/replaced them under robot_bc/human_bc. Saved source/configuration reproduces older architectures; current callers use canonical destinations. |

No model command path from the three tips is silently removed. Nine C model
fragments lost their original caller knobs in C. Their raw declarations and
unresolved-context errors are recorded; they migrate to the retained B caller
and contract without a fabricated equivalence claim. A also has one standalone
evaluator fragment requiring caller ARC settings. Every integrated model and
retained experiment composes.

## Inventory and reproduction

`evidence/consolidation/config-inventory.json.gz` holds content-addressed
resolved models, data, evaluators, normalization, representation knobs, trainer
settings, initialization/resume fields, raw models and composition contexts.
`config-references.csv` maps all source/integrated callers; `model-dispositions.csv`
records each canonical destination and compatibility decision. Equal hashes
are grouping evidence, not pretrained-weight/GPU parity. Distinct stems,
augmentation, trunks, losses, timing and frames remain explicit.

```bash
source emimic/bin/activate
python -m scripts.integration.inventory_consolidation \
  --source A=/checkout/A --source B=/checkout/B --source C=/checkout/C \
  --source integrated=. --output inventory.json
python -m scripts.audit_hydra_configs --output config-audit.json
HF_HUB_OFFLINE=1 python -m scripts.audit_components --output component-audit.json
HF_HUB_OFFLINE=1 WANDB_MODE=disabled pytest tests -q
```

Validation used Python 3.11.14, torch 2.7.1 and the checked project environment.
Ruff 0.8.6 lint/format, Python-3.11 offline lockfile verification and whitespace
checks pass. CPU tests include real two-process Gloo training, bound reload,
independent ARC tests and an isolated wheel build/install. Meta constructors and
tiny backends are labeled substitutes. The suite's 42 warnings are not skips.

## Remaining merge gate and rollback

Main was refreshed at the base above; GitHub returned no branch protection or
repository rulesets. Source reviews/threads contain no human approval. The
provenance table classifies individually inspected failed source automation;
fresh PR CI and review must be assessed independently.

The handoff explicitly excludes GPU jobs, remote synchronization, checkpoint
conversion and deployment from this authorization. The applicable
[OSMO gate](OSMO_VALIDATION.md) requires five optimizer updates, validation and
bound restoration for 12 cases using pinned real episodes/weights/tokenizers.
The prepared L40 workflow uses one GPU per HPT/PI suite and two for DDP after
HPT: at most three concurrent GPUs, with a three-hour execution timeout.
Submission and tokenizer transfer require separate authorization. Main must
not merge by bypassing this gate. DQC stays stopped; the legacy runtime remains.

After the gate and applicable PR checks/review pass, merge and record the main
SHA and post-merge checks. Roll back through a reviewed revert of the final
integration merge (mainline 1), or the squash commit if maintainers choose squash.
Preserve source refs, checkpoints and datasets; never reset shared main.
