# Integration validation gates

Owner: graph integration. These commands apply to the graph integration stack;
the final merge must repeat applicable gates on the assembled EgoVerse revision.

```bash
source emimic/bin/activate
uv lock --check
ruff check egomimic tests scripts tools
ruff format --check egomimic tests scripts tools
python scripts/audit_hydra_configs.py --output config-audit.json
HF_HUB_OFFLINE=1 python -m scripts.audit_components --output component-audit.json
HF_HUB_OFFLINE=1 WANDB_MODE=disabled pytest tests -q
```

Use pinned Ruff 0.8.6. CI creates the locked Python 3.11 environment with PI,
alignment and diagnostics extras, then installs hash-verified OpenPI source and
the required Transformers patches without changing the locked dependencies.
The CPU suite includes the isolated installed-wheel/resource/import gate.

The YAML audit recursively discovers every shipped file and supplies explicit
offline composition contexts for group fragments and station templates. It
restores Hydra's global state afterwards. The constructor audit builds graph
stages on meta tensors, constructs data keymaps/transforms/filters, instantiates
evaluators, and checks the declared inference dependency plan. It constructs
the actual model wrapper, training behavior, optimizer and scheduler too,
including named parameter binding for composite optimizers. A socket guard
rejects network access. It never resolves datasets or opens hardware.

Constructor preflight does **not** execute or train a model. External parameter
initialization is disabled, PI backends remain unbound, and text tokenizers are
explicit nonfunctional placeholders. Qwen architecture metadata comes from the
pinned JSON fixture and hash recorded in `evidence/qwen-architecture-input.json`.
The PI base fragment has an explicit nondeployable result; concrete PI models
remain fully checked. Every new YAML automatically enters both audits.
For a late-bound model with no parameters yet, optimizer construction uses one
explicit placeholder parameter; the audit records that substitution. It is not
evidence that the backend's actual parameter groups or updates have run.

The gate uncovered and repaired the stale EVA wrist keymap and eager Scale API
access during filter construction. Scale selection still queries completed
annotations when a resolver actually uses the filter, sharing that query within
one resolution scope. The CI layer also connects diagnostic prediction capture
to shared Cartesian metrics and removes the unused PI camera naming branch in
the Yam keymap. Neither change introduces a second model-specific evaluator.

The broad formatting portion is required to enforce the existing lint rules in
CI. Upstream-pinned Yam mapper/IK files remain byte-identical and are excluded
from automatic formatting. Scalar constructor validation explicitly runs on
CPU so meta-tensor architecture audits preserve its numeric checks.

The expanded constructor audit caught missing UNITE baseline identity fields
and nonexistent scheduler targets in the retained language recipes. Their YAML
now declares the existing objective and a canonical scheduler implementation;
this does not change the UNITE objective or launch a training run.

Source-pinned data/normalization/metric fixtures and all-epoch language scheduler
comparisons are described in `FIXED_FIXTURE_PARITY.md`. The restored root models
also reject incompatible declared data frames before constructing a network.
The deployment declaration gate covers every shipped ready model and checks
selected nested PI/Yam/ARC/E1 compositions against independent data settings.
Real-weight/data OSMO L40/L40S tests and assembled-tree validation remain separate
gates. Passing these CPU commands
does not authorize legacy cutover or assert a successful PI training run. DQC
is not restarted by any integration check.
