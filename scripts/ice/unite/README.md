# UNITE ICE continuation contracts

These tools add UNITE-specific, fail-closed validation around the generic ICE
runner introduced by PR #4. They are not a second training launcher and do not
replace either cluster's live launch authority.

## What each tool owns

- `unite_contract.py` validates the four clean register-sweep rows and their
  14:1 update ratio, noise start, Dopri5, EMA, checkpoint, and EnergyScore
  contracts.
- `select_ice_skynet_access.py` proves and records the usable ICE-to-Skynet SSH
  route from caller-supplied deployment identities.
- `stage_unite_candidates.py` copies only eligible scheduled checkpoints and
  their bound configuration into a create-only, content-addressed staging
  tree. It records the source run root and rejects paths outside that root.
- `unite_checkpoint_validate.py` performs structural or strict full-state
  validation of one immutable checkpoint.
- `unite_cutover_record_writer.py` creates the signed candidate, successor
  readiness, and source-cancellation authorization records.
- `unite_source_terminal_evidence.py` collects fresh, read-only terminal and
  queue-absence evidence using caller-pinned remote Slurm executables.
- `unite_ice_resume_child.py` binds a runner attempt to the exact authorized
  checkpoint and immutable continuation launcher.
- `requeue_integration_contract.py` validates the deterministic step 3 to 8,
  requeue, strict step 8 resume, and step 8 to 11 proof trace.
- `unite_checkpoint_validate_dispatch.sh` is the clean environment boundary
  for strict validation; the two `.sbatch` files are CPU wrappers only.

## Safety boundary

All deployment-specific values are inputs: account, partition, QoS, GPU model,
remote user and host, source run root, W&B identity, Python interpreter, source
checkout, and exact source commit. Records and checksum sidecars are
create-only. Missing first-attempt restart metadata means exactly zero;
malformed or ambiguous values fail closed.

Before a production continuation, use the live cluster protocol and launcher,
run their official preflight, and produce the real fixed-step requeue proof.
Passing this repository's unit tests does not authorize source cancellation or
submit a training job.

## Tests

Run the focused contract suite from the repository environment:

```bash
source emimic/bin/activate
python -m pytest -q \
  tests/test_hydra_override.py \
  tests/test_unite_ice_contract.py \
  tests/test_unite_ice_tooling.py
```

The strict checkpoint validator imports PyTorch and must also be exercised in
the supported Skynet environment before publication.
