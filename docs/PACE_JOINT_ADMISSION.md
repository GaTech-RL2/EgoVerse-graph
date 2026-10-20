# Joint training on ICE and Phoenix

The maintained joint service accepts an optional explicit `allocation` policy.
Requests without it retain the accepted Lambda node and health gate. PACE
requests require four ranks, batch eight per rank, global batch32,32CPU192G,
and a fresh no-requeue allocation. ICE uses coc/coc-ice and ice-gpu, at most16h;
Phoenix uses gts-dxu345-rl2/inferno, gpu-h200/gpu-h100, at most24h. These are
admission bounds, not a job submission API or launch authorization.

`scripts/clusters/pace/pace_podman.py` promotes the reviewed rootless container
boundary into a reusable source capability. Task files bind the exact service
files, publisher base receipt, tools, payload, output mounts and inner command.
The request schema is `pace-podman-boundary-v2`; its four boundary helpers are
pace_podman.py,gpu_admission.py,gpu_ownership.py,inside_pace.sh. Persistent mounts
must not shadow readonly payload contents. Extraction, Podman storage and JIT
are node-local; no credentials belong in public payloads or requests.

Ownership is established from the current native job and cluster identity,
owner/account/QoS, Slurm assignment, native job cgroup, kernel GPU device-open
permissions, and matching UUID/BDF/physical-minor inventories from NVML, XML and
/proc. Slurm's global IDX is never interpreted as a filtered NVML ordinal. CDI
is restricted to the owned UUIDs and physical minors. Missing or ambiguous proof
fails closed. Phoenix requires the native SLUID in the current cgroup. Container
ancestry must preserve that exact host step, not merely contain its job number.

Before running the opaque training command, four actual CUDA workers must prove
the expected Torch2.9.1+cu126 runtime, SM90 Hopper, BF16 operation and NCCL allreduce
sum10 on exactly those UUIDs. The resulting current attestation lives at
/run/pace-proof/attestation.json and its persistent proof-directory counterpart.

The joint request's allocation object binds cluster,user,uid,account,qos,
payload_archive_sha256 and base_archive_sha256. Joint preflight records the
attestation hash. A full request binds its smoke attestation as smoke.allocation;
the full admission verifier requires its hash to match preflight, its native
terminal job/association/node, and all four actual training-rank GPU UUIDs.
Source, data context, precision, optimizer and paired RGB controls remain
separate mandatory checks. A container admission PASS is not a training smoke.

`export_runtime.py` creates readonly portable archives from explicit cache mounts
inside a two-CPU, zero-GPU Slurm step. It preserves logical paths and literal
symlinks, rejects links outside the selected closure and excludes credentials,
logs and checkpoints. It never prepares data, installs dependencies or transfers
artifacts. Runtime export and container preparation do not alter model science.

Fresh run.py remains fresh-only. Do not enable automatic Slurm requeue against
it or claim recovery is implemented by this change. Before full ICE admission,
establish exact full-state recovery under the authorized checkpoint cadence and
a separate CPU checkpoint mirror with no pruning. Actual four-rank optimization,
scheduled validation, strict full-state reload and hidden-label-NaN sampling
smokes on the selected destinations still gate full launch. All four modality
arms must pass the actual paired RGB control verifier before any full arm starts.
