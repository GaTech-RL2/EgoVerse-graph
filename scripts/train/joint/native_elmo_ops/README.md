# Native Elmo composite recovery operations — service C pinned candidate

This is a new, local 23-file operations assembly. It preserves the sealed runtime,
host-adapter, execution-v2, controller and observer fragments. No scientific source,
remote files, canonical campaign receipts, jobs or actual recovery bindings changed.
Both `execution_spec.SERVICE_COMMIT` / `SERVICE_SHA` and the wrapper equivalents
bind verified published service commit `abbb00499096faabf7035fde23bf4b1eba984377` and 35-file manifest
`adf45524de3c2e6dba186572e7151d48bfca0b3adc8e6cf9defd686b47744c87`. This operations closure
is a separate local candidate awaiting final review and its own publication.
The prepublication candidate and every sealed input remain unchanged.

The controller is CPU orchestration only. It never proves failure recoverability,
checkpoint validity, zero optimizer progress, scientific completion or child native
completion. Its gateway transport still reopens remote bytes and native state;
all scientific admission remains inside the actual allocated host guards. An
unknown submission/release/capture outcome remains single-use and requires exact
inspection, never an automatic retry or cancellation.

## Dependency and proof protocol

`FILES.sha256` covers all static operations bytes and excludes dynamic execution,
request, observer, controller, claim and status directories. The ID-free execution
spec binds this manifest and predeclares only `release_observer_path`. The observer
spec binds the execution spec SHA. Thus neither source module embeds the other's
SHA and no manifest/spec/observer hash cycle is introduced. Fixed-name operations
imports verify the whole immutable closure, physical module path and source bytes
captured at import; foreign or changed cached modules fail closed.

Every producer release requires an accepted, currently `PENDING/Dependency`
afterany observer and an exact `PENDING/JobHeldUser` producer. The release records
exclusive `RELEASE_HELD_NATIVE.txt`, its hash, and complete observer readiness
before the single release call. All five phases use the same observer contract;
the original production parents use their separate maintained `original_observer`
provenance path.

`phase_observer.verify_captured(observer_spec_path, observer_spec_sha, ops_sha)`
reopens the accepted binding and delegates to the active maintained service:

```
phase_observer_proof.verify_observer(
    {path: canonical native-JOB.observer.json, sha256: final commit hash},
    producer_request={path: actual request JSON, sha256: actual request hash},
    producer_native=exact native contract including actual JobId,
    terminal_native={path: canonical native-JOB.txt, sha256: raw native hash},
)
```

The service returns context including `result`, the exact observer commit JSON.
It reads and semantically validates evidence without importing operations code.
The operations wrapper preserves the existing `verify_captured -> result` API.
Both execution-claim and scientific-request-claim censuses require that final
commit, exact request/job/spec/native-path correspondence, and a fresh complete
owned step inventory. A raw terminal file alone never passes. Original-parent
claims require separately validated original-observer/failure evidence and their
actual original RUNNING admission; this is not substituted with a phase observer.

The canonical same-arm controller ledger is also reopened before new controller
submission/release. Earlier PREPARED/spec/manifest/accepted IDs and held bytes must
agree. Every previously registered execution root and terminal directory must be
covered by the new execution census, and prior GPU writer claim roots by the
scientific census. An old controller RESULT saying a child was released is never
terminal proof, even after that child disappears from the queue.

Missing RUNNING admission for a genuinely observed later terminal phase is only
terminal/quiescence evidence. It never proves zero progress or successful phase
completion. Maintained completion producers separately require actual RUNNING,
canonical host exit 0, native COMPLETED 0:0 and the exact phase-specific result.

Proof metadata is limited to 4 MiB per file, native stdout/stderr to 1 MiB/64 KiB,
and immutable manifests to 128 members. Checkpoint and data I/O remain in the
allocated maintained producer, outside these metadata-only guards.

## Local verification and remaining gates

`tests/native_elmo_ops/` uses portable synthetic requests and native records. Native subprocesses are intercepted. The real composed binder,
observer, release hook and maintained proof reader run together for all three
arms and all five modes. Additional cases exercise changed module bytes, cached
foreign modules, unknown claims, partial observer commits, conflicting release
proofs, surviving worker steps and current held/RUNNING interfaces. The temporary
local service snapshot is test input only; it is not publication or production
proof. The final 71-case suite uses the exact published service constants without overriding them.
Only synthetic original-parent fixture profiles and native subprocess results are replaced in memory.

Pending: independent final review; final operations publication; immutable actual deployment/readback; a genuine
recoverable parent failure; actual allocated controller/native authentication;
CPU highest-checkpoint inspection; all-rank GPU restoration and scheduled
validation; finalizer; human-visible resolved production preflight. No actual
request, claim, terminal certificate, checkpoint inspection or GPU proof is
created by this local candidate.

Portable publication tests live in `tests/native_elmo_ops/` and synthesize all
requests and native evidence. Local retained-request tests remain audit-only.
The final portable suite contains 71 cases, including two executable-wrapper packaging
regressions. It runs at the intended repository layout without path overrides.
Repeated runs exercise the same matrix and must not be summed as unique tests.
Native wrapper files must retain mode 0755 in Git/export/tar; direct executable
bind mounts otherwise fail before actual native authentication.
