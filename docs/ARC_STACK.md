# ARC consolidation stack

This is a new draft stack on `GaTech-RL2/EgoVerse-graph/main` at
`6209dd5a58f0ba2a4e3692d67d73869106efe61b`. Each PR targets its predecessor.
Original branches, PRs and worktrees are retained. No original branch was
rebased, reset, force-pushed, closed or deleted. Source tips and previous local
builds are additionally pinned under `refs/consolidation/20260911/` in the
working repository.

## Review order

All new branches have prefix `codex/consolidate-arc-20260911/`.

| PR (merge order) | Branch suffix | Scope |
|---|---|---|
| [#81](https://github.com/GaTech-RL2/EgoVerse-graph/pull/81) | `01-data-foundation` | Rotation transforms and nested validation |
| [#82](https://github.com/GaTech-RL2/EgoVerse-graph/pull/82) | `02-hpt-graph` | HPT primitives, stems, flow stages and Qwen |
| [#83](https://github.com/GaTech-RL2/EgoVerse-graph/pull/83) | `03-abc-yam` | ABC conversion, Yam embodiment and graph recipes |
| [#84](https://github.com/GaTech-RL2/EgoVerse-graph/pull/84) | `04-arc-codecs` | ARC codecs, shared-D metrics, duration and reconstruction |
| [#85](https://github.com/GaTech-RL2/EgoVerse-graph/pull/85) | `05-arc-resampling` | Vectorized arc-length resampling |
| [#86](https://github.com/GaTech-RL2/EgoVerse-graph/pull/86) | `06-e1-data-codecs` | E1 codecs, sampling and dataset adapters |
| [#87](https://github.com/GaTech-RL2/EgoVerse-graph/pull/87) | `07-e1-evaluation` | E1 tempo metrics and ground-truth-span diagnostics |
| [#88](https://github.com/GaTech-RL2/EgoVerse-graph/pull/88) | `08-e1-campaigns` | Human/robot E1 HPT campaigns, diagnostics and launchers |
| [#89](https://github.com/GaTech-RL2/EgoVerse-graph/pull/89) | `09-robot-campaigns` | Stationery/shorts splits and robot token decoders |
| [#90](https://github.com/GaTech-RL2/EgoVerse-graph/pull/90) | `10-pi-action-data` | PI action geometry, normalization, bounds and scoring |
| [#91](https://github.com/GaTech-RL2/EgoVerse-graph/pull/91) | `11-pi-graph-campaigns` | Optional PI graph stage and campaign integration |
| [#92](https://github.com/GaTech-RL2/EgoVerse-graph/pull/92) | `12-stack-validation` | Source mapping, review order and verification record |

## Source mapping

| Original branches | New layers |
|---|---|
| `graph-transform_fixes`, `graph-nested-val` | [#81](https://github.com/GaTech-RL2/EgoVerse-graph/pull/81) |
| `graph-hpt-deps`, `graph-hpt-stems`, `graph-hpt-flow`, `graph-hpt-qwen`, `graph-bounds-gate` | [#82](https://github.com/GaTech-RL2/EgoVerse-graph/pull/82) |
| `graph-abc`, `graph-hpt-configs` | [#83](https://github.com/GaTech-RL2/EgoVerse-graph/pull/83) |
| `graph-arc`, `09-09-feat_arc_shared-d_arcmatch_with_optional_reconstruction_path` | 04, with the timm dependency declaration placed in 02 |
| `aidan/arc-tokenizer-fixes` (`209de8fd`) | 05; original EgoVerse PR #610 retained |
| `aidan/arc-e1-tempo-ablation` (`e8aa10fa`) | 06–08, the canonical E1 campaign |
| `aidan/arc-e1-abc` (`fff5d641`) | Shared E1 work in 06–08; distinct robot work in 09 |
| `aidan/shorts-extreme` (`55932d99`) | Duration/GT-span work in 06–08; exact shorts splits and native decoders in 09 |
| `aidan/arc-e1-fold-speed` (`b3ea94b3`) | Strict ancestor of the canonical tempo branch; no duplicate replay |
| `aidan/abc-stationery-pi` (`d5f72068`) | 10–11 |

Seventeen robot E1 commits have identical patch IDs to canonical E1 commits.
Their mapping, complete campaign commit lists and 189 inspected source files
are recorded in [arc_consolidation.json](arc_consolidation.json). The JSON
distinguishes direct copies, graph adapters and retained legacy provenance.
The exact graph tree at layer 04 equals Git's merge of current graph main and
the original graph stack tip. Original authorship is preserved for replayed
commits; relocated source modules identify their source branch and revision.

## Integration decisions

Graph main intentionally removed the old algorithm and controller runtime.
The stack retains graph training and ports the selected campaigns onto it.
E1 metrics preserve the source arithmetic and output schema while consuming
graph predictions and explicit normalization context. The HPT campaign models
retain the ResNet → MLP image path, camera differences, learning rates and
native token widths. Existing HPT checkpoints require their original runtime;
the graph recipes create new graph checkpoints.

The robot decoder accepts native `lab`, `e1_dur`, `e1_logdur` and `e1_profile`
tokens. Its library/CLI performs no robot I/O. Hardware control remains on the
original source branch; only the new decoding component is ported. It runs
after action unnormalization and before controller frame transforms.

Stationery and shorts remain separate campaigns from the original graph
cotrain study. Their reconstruction and E1 metric settings are explicit and
should not be treated as interchangeable chart series. The shorts train and
validation filters retain their exact 200/30 disjoint episode-ID lists.

PI uses a scoped source adapter and a pinned optional OpenPI dependency. Its
source data/normalization/rotation conventions stay together. Nested graph
validation replaces the legacy second evaluator dispatch. See
[PI05_GRAPH.md](PI05_GRAPH.md) for initialization, checkpoints and environment
requirements. Legacy ACT/HPT/latent evaluator dispatch is retained in original
history; its shared PI cadence/scoring behavior is carried into the PI port.

Source launchers that pruned checkpoints now retain them. The ported Mecka
converter refuses existing output directories and retains partial downloads.
No source scripts that delete data, launch training, sync datasets or command
a robot were executed during consolidation.

## Verification

655 focused CPU tests passed on the assembled stack in an isolated Python
3.11 environment, including graph/HPT/ARC regression tests, campaign config
composition, E1 reference metrics and the 54 PI source numerical tests. PI
integration tests use the actual source adapter with a tiny replacement
network/tokenizer and cover backward, inference, parameter registration,
strict checkpoints and Lightning evaluator binding. All ported shell and
Python tools pass syntax checks.

Full pretrained OpenPI training, distributed CUDA runs, real episode access
and robot actuation were not tested. Cluster launchers retain source lab
paths/partitions where those identify datasets or environments; supply paths
appropriate to the target machine.

The local audit directory is
`/Users/rpunamiya/Desktop/GEAR/handover_figures/arc-stack-20260911/`.
It contains original-ref snapshots, complete campaign patches, source file
snapshots, test logs, draft PR bodies and the final preservation report. The
user's two original checkouts remain untouched, and no existing file is
deleted relative to the selected graph main.
