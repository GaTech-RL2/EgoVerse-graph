# Native phase-observer proof admission

`phase_observer_proof.py` is the maintained read-only semantic verifier for
native Elmo recovery phase observers. It parses hash-bound evidence with the
standard library. It never imports external operations code, issues a native
command, loads a model/checkpoint, or authorizes a recovery allocation.

## API

```python
verified = verify_observer(
    observer_binding,
    producer_request=request_binding,
    producer_native=expected_native,
    terminal_native=native_binding,
)
```

`observer_binding` is `{path, sha256}` for the canonical **final commit**
`TERMINAL_DIRECTORY/native-JOB.observer.json`, not an observer specification.
`producer_request` is the actual materialized request `{path, sha256}`, not its
parsed JSON. `producer_native` is the complete expected native contract dict
including actual JobId. `terminal_native` is the raw canonical `native-JOB.txt`
`{path, sha256}`. These three comparison arguments are optional for generic
census and required as appropriate by stricter consumers.

`verified['result']` preserves the original final commit JSON. Other context
includes parsed request/spec, physical request/spec paths, actual job, accepted
held native record, observer acceptance/ID, terminal native fields,
`expected_native`, and `running_admission_status`. Physical paths are Path
objects; the original result remains JSON-compatible.

`verify_claim_terminal(claim, terminal_directory, run_id)` preserves the exact
reviewed original-parent afterany route and requires a complete spec-bound phase
observer for all new claims. `check_owned_steps(raw, job)` strictly parses a
complete owner inventory and rejects parent, parent+'.', and parent+'_' rows.
Callers must obtain fresh owner queue/steps separately before admission; the
proof reader itself performs no query.

## Evidence contract

The active service root, complete manifest and imported guard origins must match
the producer's exact service binding. Ops manifests are read and hashed as data;
no external code is imported. The reader reopens ID-free execution and observer
specifications, templates, actual materialized requests, accepted held-native
records, PREPARED/acceptance/submission outcomes, exact observer afterany
acceptance and the complete final terminal/step commit.

The producer spec's `release_observer_path` must match the bound observer spec.
The producer execution claim preserves `RELEASE_HELD_NATIVE.txt` and
`RELEASE_ATTEMPT.json`, whose fields are `job_id`, `spec_sha256`,
`request_sha256`, `held_native_sha256`, and `observer_readiness`. The actual
held native record and exact approved readiness query/identity are revalidated.
No operator-supplied PASS boolean replaces this chain.

Producer and observer owner/name/node/resources/Command, accepted SubmitTime and
spec comments must agree. Real StartTime/EndTime must put producer termination
before observer start. Native query success, producer host exit, observer
terminal state and scientific completion are distinct. Every metadata file is
bounded to 4 MiB, manifests to 128 members, and query stdout/stderr to
1 MiB/64 KiB. Oversized checkpoint-like inputs reject without loading.

The final commit is canonical, exclusive and produced only after fsync/readback
and successful full owned-step parsing. A raw or partial native file alone is
insufficient. Observer own terminal remains `UNKNOWN_NOT_REQUIRED` after purge.
Missing producer RUNNING provenance may remain `NOT_RECORDED` in generic failed
terminal census; it establishes neither zero work nor successful execution.

## Maintained completion consumers

`completed_producer` requires a complete observer binding, exact known
result-schema/status → request-schema/phase mapping, validated actual producer
RUNNING provenance, matching actual request/native bindings, canonical host exit
0, native COMPLETED0:0 and consistent result job IDs. Unknown schemas/statuses
have no compatibility bypass.

The canonical result files under `request['output']` are:

- capture: `interruption-capture.json`;
- inspect: `checkpoint-inspection.json`;
- diagnostic: `production-recovery-diagnostic.json`;
- finalize: `interruption-finalized.json`.

Producer host exits are `ROOT/status/MODE-JOB.exit`. Full terminal census uses
the shared proof API separately and receives no phase-completion authority.
All historical new-claim terminal consumers must use the complete proof reader;
original-parent terminal captures retain their separately pinned provenance.

Build service first, then bind its manifest in reviewed ops. Do not embed ops or
own service-manifest hashes inside this verifier. Dynamic specs/claims/results
stay outside immutable ops FILES, avoiding a hash cycle. A source publication is
not deployment or actual allocated recovery proof: controller/observer/runtime
assembly, current intent, source/cache/storage/native authentication, highest
checkpoint inspection, all-rank restore, finalizer and full preflight remain
separate gates.

Portable CPU fixtures live together in `tests/joint_phase_observer/`. They use
fresh temporary service copies and synthetic profiles/claims/native records,
block native subprocesses and ML imports, and exercise the real shared reader.
Retained campaign-specific original-profile audits are separate local evidence.
