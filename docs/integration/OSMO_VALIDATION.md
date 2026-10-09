# Scheduled integration checks

`scripts/integration/matrix.yaml` declares five optimizer updates and one
validation batch for HPT (EVA, Aria, Mecka, Scale, Yam), PI (EVA, Aria, Mecka,
Scale), HPT/PI EVA+Aria cotraining, and a two-GPU HPT run. These checks are
execution and checkpoint gates, not policy-performance experiments.

The source manifests under `evidence/` pin 15 complete real episodes (two train
and one disjoint validation episode per source), public pretrained weights,
and the PI tokenizer. R2 GET requests use the recorded ETag as an `IfMatch`
condition; staged objects receive full SHA-256 receipts. Weight and tokenizer
bytes must match their full published/recorded hashes. Source objects are never
written. The tokenizer is transferred as six files, without any access token.

The workflow runs input staging on CPU before three downstream tasks: one L40
for each sequential model suite and two L40s for the distributed check. The
locked environment and pinned OpenPI source/Transformers patch are installed
inside the allocation. PI's optional gradient checkpointing is enabled, and
sampler compilation is disabled for this short gate; both defaults remain
unchanged for existing recipes. Training and strict restoration run in separate
processes to release model/optimizer memory between phases.

Each case preserves resolved configuration, complete data context, checkpoint,
held-out observations, finite losses/gradients, actual validation metrics,
parameter-update probes, GPU/world-size information, and artifact hashes.
Inference restoration uses the shared bound loader and checks sampling and
execution-prefix controls without a robot or reopening training data. Logs and
checkpoints are preserved at a new experiment prefix with conditional R2 writes;
an existing object cannot be replaced. No historical training run is resumed.

Use a local directory named `input-tokenizer` containing only the six manifest
files. Set `TOKENIZER_TRANSFER` to that directory. Render against a published
full source commit, validate, then submit:

```bash
source emimic/bin/activate
python -m scripts.integration.build_workflow \
  --source-commit "$INTEGRATION_COMMIT" --name "$INTEGRATION_RUN" \
  --output "$INTEGRATION_SPEC"
osmo workflow validate --pool groot-l40-01 "$INTEGRATION_SPEC"
osmo workflow submit --pool groot-l40-01 --format-type json \
  "$INTEGRATION_SPEC"
# Set INTEGRATION_WORKFLOW to the workflow ID returned by submit.
osmo workflow rsync "$INTEGRATION_WORKFLOW" stage-inputs \
  "${TOKENIZER_TRANSFER%/}:/tmp/" --once
```

Target `stage-inputs` explicitly. Submit-time `--rsync` selects the service's
first group, which can be `ddp-gates` even though staging appears first in the
YAML. That leaves staging waiting for tokenizer files in another task. The
OSMO copies the source directory beneath the destination, including its
basename; a trailing source slash does not remove that directory level. The
explicit transfer above creates `/tmp/input-tokenizer` in the running staging
task; `prepare_inputs` checks every file's size and hash.

The local synthetic harness test checks orchestration only. A submitted or
completed workflow is insufficient: every required case needs both training
and inference receipts, and failures remain failures. Additional fixed-fixture
parity, frame compatibility, and assembled-destination gates remain required
before cutover. DQC remains stopped; sky1/sky2 are not used.
