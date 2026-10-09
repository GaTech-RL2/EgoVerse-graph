# c12 historical compatibility in the composed wrapper

Base: c69e5c32a5b6f4826f7a1d1557cd056e4305fd48. This is an unpublished patch,
not an approved training revision. Historical reference:
c12f5aaa91ab8594f191fb15a1a0a5041dd4ae74.

The opt-in experiment `pusht/action_flow_usocket_refactored_c12_compat_s42`
retains ModelWrapper + ActionFlowTrainingBehavior + ActionFlowDiagnosticProvider.
It selects `compatibility_mode: legacy_c12` at all four boundaries: data module,
training dataset, validation dataset, and PipelineAlgo. Current defaults remain
unchanged for other experiments.

Restored behaviors:

- Bounds filtering always on, strict full-vector checks, including nonfinite rejection.
- Historical child-local random replacement and candidate-count exhaustion.
- Default/global CPU generator for shuffled indices and worker base seeds.
- Preserve incoming floating-point dtype during device transfer.
- No optional validation-before-fit in this recipe.

The historical bounds flag was ignored by c12. This mode deliberately reproduces
that behavior rather than treating the old `bounds_check: false` as authoritative.
The original sample is still normalized normally after passing the filter.
BF16 trainer precision, objectives, architecture, optimizer and EMA are not changed.

Loader checkpoints record the mode and reject cross-mode restoration. Old
refactored checkpoints lacking a mode field are treated as `current`; do not
resume one under legacy mode and call that a historical reproduction. This mode
does not add exact mid-epoch worker/prefetch RNG restoration.

Validation: `tests/test_c12_compatibility.py` plus the existing bounds, loader RNG,
and pipeline tests. Set C12_DATASET_SOURCE to the pinned historical dataset file
to require direct AST-derived retry comparisons, including >25 failures,
cross-child distractors, exhaustion, and Python RNG state. The scheduled test
wrapper requires this file; absence must not silently skip historical checks.

Limitations: matching these code paths does not prove long-run reconstruction
equivalence, production initialization equality, identical runtime kernels, or
full-repository parity. Do not advertise a reproduced loss curve until measured.

Staging prevention: set GIT_LFS_SKIP_SMUDGE=1 for archive/worktree operations.
An initial staging attempt encountered an unrelated absent APK LFS object;
retrying with that setting succeeded. No pinned source files were modified.
Use fail-fast transfer pipelines (`set -euo pipefail`) so failed source transfer
cannot be hidden by a successful subsequent command.

CPU attempt 13550903 performed no tests: relocated activation selected
/usr/bin/python (without pytest). The wrapper now invokes the absolute pinned
interpreter after activation and asserts it is executable.

CPU attempt 13550934: 34 tests passed; the exact-signature architectural guard
failed on the newly added optional argument. Updated that expected signature and
asserted its unchanged default; all source-level model-routing prohibitions remain.

Final CPU job 13550968 completed 0:0: 35 tests passed, none skipped, in 17.63s.
This includes the AST-derived historical comparison. No full training was launched.
