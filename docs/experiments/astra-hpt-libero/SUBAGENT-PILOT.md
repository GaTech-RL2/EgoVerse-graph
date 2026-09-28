# User-authorized Codex Astra transport pilot

On 2026-09-28 the user requested: “test pilot, can you use the astra here
directly as a subagent?” This authorizes replacing the unavailable NVIDIA
gateway with the current session's `gpt-6-astra` subagent for this new run.
The original specification and failed gateway attempts remain unchanged.

Run version: `astra-hpt-subagent-pilot-20260928-v1`. Source starts at `a5a16752`.
Use a fresh, context-isolated Astra child for each logical generation request;
archive its exact supplied prompt, requested model/reasoning setting, agent ID,
response and validation outcomes. A repair consumes another logical request.
Keep the eight-request phase limit and maximum two repairs. No other model
may substitute for Astra. Uniform requests must receive no learner feedback
or adaptive-arm archive.

This is a transport amendment, not evidence that Codex and the gateway expose
the same model snapshot or sampling behavior. Codex does not expose a hard
output-token cap, provider usage/billing or transport-attempt receipts here.
Record those fields as unavailable, request a concise response under 8,192
tokens, and bound accepted raw response size. Do not claim the original
gateway token/billing contract was measured or enforced.

First test: generated six-scene commissioning, independent schema/stock
inventory/novelty checks, bounded physical attempts (30 accepted, five per
scene, within 120 attempts), immutable data and a frozen normalizer, then a
discarded full HPT engineering update with those real observations. Keep all
successful and failed attempts. This tests the generator-to-simulator-to-HPT
path. It does not count as warm-start training or the U-versus-A comparison.

Production training still requires frozen evaluation partitions and the
remaining ordered work items. Commissioning episodes never enter imitation
replay. Use one OSMO L40/L40S and preserve all prior branches and artifacts.

## Measured first test and visual correction

The first OSMO run at `81ebaf3b` collected 30 accepted demonstrations in 31
attempts (one retained S1 approach timeout), passed all six exact fresh-process
replays, and passed the full HPT real-data update with gradients in all six
component groups. The normalizer used exactly five episodes per scene and ten
per stage. Production optimizer updates remain zero.

Visual inspection then identified a language-grounding issue: the optional
reference fixture had the same appearance as the yellow S3 marker. Those data
remain immutable evidence of the mechanical checks, but are superseded for
commissioning. Visual catalog version 2 makes the optional fixture neutral gray
and retains the yellow marker, without changing geometry, dynamics or teacher
control. The compiler records and verifies its source/asset hash in each bundle.

The replacement run at `41778ab32bb84d10bea8745763bfe8107f152b56` used those
same saved Astra proposals, with no new generation calls. It collected 30
accepted demonstrations in 31 attempts and passed all six exact fresh-process
replays. The first 31 acquisition attempts remain charged: cumulative use is
62 of the shared 120 cap. Both thirty-episode corpora remain preserved; only
the replacement corpus supplies the current frozen normalizer.

The full random 45,934,983-parameter HPT received one discarded engineering
update from the corrected observations. All six component groups received
gradients; update time was 0.797 seconds, peak allocated CUDA memory was
950,507,520 bytes, and inference returned `[4,10,7]`. These single-update
measurements do not establish steady-state throughput or learned competence.
The execution contract predicts ten commands and executes one before replanning.
No student policy rollout was performed, and production optimizer updates are
zero. The retained failure in each run is the same S1 approach timeout at
attempt 26; a later attempt completed that scene's five-episode quota.

The corrected commissioning receipt and frozen normalizer are in
[`evidence/subagent-pilot-v2/`](evidence/subagent-pilot-v2/). Normalizer SHA-256:
`433e6543f156d095adafc0890083f6fc5d510e3feb4ab702557dbd6dcc78b2b4`.
The initial checkpoint, saved before the discarded update, is unchanged from
the first run (`f274a13350e5e3b47f50c0737416bdefaa8d5cdffd6abba9a11241b510448bd8`).
Full data, snapshots, controller traces, videos and model receipts are preserved
under the separate source-pinned R2 prefixes recorded with each run.

Visual review covered the six representative successes and the retained
failure, in both external and wrist views. The corrected self-contained
preview is
`/Users/rpunamiya/Desktop/GEAR/handover_figures/astra-hpt-libero-20260928/subagent-pilot-v2.html`,
with all seven videos embedded; `subagent-pilot-v2.zip` is the shareable archive.
The earlier preview is retained as superseded evidence.

Both one-L40 workflows completed: the first took 757.864 seconds and preserved
522 files; the corrected run took 717.992 seconds and preserved 523 files.
The two jobs ran sequentially and have released their GPU allocations.
All 58 selected corrected artifacts match the completed upload's hashes.
Repository CI at the corrected executable source passed 1,489 CPU tests,
configuration/installed-wheel checks, static checks and lock validation in
[run 36463070158](https://github.com/GaTech-RL2/EgoVerse-graph/actions/runs/36463070158).

Three additional, isolated Astra requests tested W07 against synthetic report
fixtures. Allocations were S1/S2/S3 = 25/30/20 when all stages had constructed
zero success, 5/10/60 with constructed S1/S2 competence and S3 failures, and
25/25/25 for uniform with no report. All were valid without repair. These are
generation-contract evidence, not measured learner learning or curriculum
benefit. Total logical generation calls so far: four, with unknown billing.
