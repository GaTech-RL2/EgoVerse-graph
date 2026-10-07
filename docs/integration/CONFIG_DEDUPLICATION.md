# Model configuration consolidation: first local layer

This is a partial implementation of the October 1 consolidation handoff.
The complete A/#197, B/#159 and C/#160 integration has **not** been assembled,
reviewed, pushed or merged. Main and the published source branches are unchanged.

## Available sources and scope

The isolated working branch is `codex/graph-consolidation-20261001`, based on
the complete B/#159 tip `eda6f8d177dec8df26ad62e3784d103b4de74d7d`.
Its main ancestor is `161e3a0c40182ba003434d3adaeaa6cbd12107d8`.
An independent clone preserves all other agents' checkouts and Git metadata.

The requested A/#197 commit `20507c6866f8a167e0e2858141299dbf6ccddeb0`
and C/#160 commit `bef887c4c46bae171711951849fbdbf1617ea534` are absent
from the available local EgoVerse and EgoVerse-graph object databases.
The GitHub ref refresh failed with `Could not resolve host: github.com`.
Current PR reviews, status checks and branch protection could not be inspected.
Consequently this layer makes only source-verified B config changes; it is not
a declaration that the full handoff has passed its acceptance criteria.

## Shared definitions and compatibility names

Each pair below was byte-identical at the pinned B source, including graph,
optimizer, initialization, inference and data contracts. Five copied model
definitions become five-line Hydra aliases. All 55 model command paths remain
available, while 1,356 duplicated YAML lines are removed.

| Compatibility name (`model=`) | Canonical definition (`model=`) |
| --- | --- |
| `hpt_bc_flow_aria` | `hpt_bc_flow_human` |
| `hpt_cotrain_scale_flow_shared_head` | `hpt_cotrain_mecka_flow_shared_head` |
| `pi0.5_bc_scale` | `pi0.5_bc_mecka` |
| `pi0.5_cotrain_mecka_scale` | `pi0.5_bc_mecka` |
| `pi05/pi0.5_ft_abc_eva_6d` | `pi05/pi0.5_bc_abc_eva_6d` |

Live launch and normalization scripts use the canonical names. Dataset
selection stays explicit; choosing a shared model does not change the selected
Aria, Scale, Mecka or EVA data. Historical source receipts and compatibility
tests retain their original names. Existing resolved checkpoint configurations
retain the same targets, parameter namespaces and contract values.

The aliases exist for external command/config compatibility. They contain no
second implementation or architecture override. Their eventual removal needs
a separately documented migration; consolidation does not silently invalidate
saved commands. The nested PI alias uses an explicit absolute Hydra config
path and `_here_` package, preserving its original model namespace.

## Evidence

The [machine-readable ledger](evidence/config-deduplication-20261001.json)
lists source hashes, all 55 B model dispositions, canonical commands, caller
migrations and remaining work. Other configurations are pending cross-stack
review; a unique filename is not evidence of an intentionally distinct model.

The local `.consolidation/` artifact directory contains the full resolved
254-context inventory, all 34 source PR entries, the archived source configs,
reproduction helpers, per-context before/after hashes and test logs.
Every non-Hydra field of all 254 composed contexts compares exactly equal
before and after this layer. The comparison freezes Hydra's `now` resolver in
both compositions, retaining W&B identity fields in the comparison. Synthetic
audit paths and no-weight constructor checks do not establish real input or
pretrained initialization parity.

Python 3.11.14 validation in the existing checked project environment:

- Baseline focused ARC/checkpoint/deployment/trunk tests: 202 passed.
- Source-pinned full-config equivalence: 254/254 unchanged.
- After the alias fix: 84 relevant tests passed, including all 254 recursive
  composition and constructor contexts, the isolated installed-wheel gate,
  retained HPT/PI training fixtures and diagnostics.
- Ruff 0.8.6 lint/format, offline lockfile check and `git diff --check`: passed.

The first broad CPU run had 1,434 passing tests, 18 failures and two errors.
Five failures exposed the nested PI alias path and were fixed; two packaging
errors were resolved by copying already cached public build dependencies into
a writable offline cache; one diagnostic failure was resolved by assigning a
writable Numba cache. All eight were included in the successful 84-test rerun.
The remaining 12 tests require loopback networking (11 dashboard cases and one
distributed checkpoint case), which this sandbox forbids. Their exact names
are in the ledger. They are unresolved gates, not passes or code exclusions.
The full CI gate is **not green**, and no test implementation was weakened.

Source-window reconciliation remains open. On B, an explicit
`get_keymap(horizon=100, embodiment="yam", yam_source_frames=400)` raises
`TypeError` at `Embodiment.get_keymap`. This differs from the handoff's reported
A silent-ignore/C explicit-window behavior and must be resolved against the
actual missing source trees. No source-window or ARC math is changed here.

## Remaining integration and rollback

Obtain the A/C source trees and refresh the exact source refs before continuing
the full merge. Review their code, contracts and all source PR discussions;
complete the config and implementation dispositions; reconcile ARC clocks,
layouts, source windows, reconstruction and checkpoint behavior; and rerun the
assembled-destination gates. This source-independent first cleanup does not
choose the eventual mechanical merge order or supersede either missing stack.

The handoff's compute/deployment boundary still applies. No GPU job, remote
sync, checkpoint conversion, hardware action or deployment was performed.
Required real-weight/data validation and merge review remain separate gates.

Rollback this layer with `git revert` of its integration commit. Its parent
preserves the complete original definitions; source branches remain intact.
Do not reset shared main or delete other agents' branches.
