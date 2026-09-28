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

Rerun the same saved Astra proposals under the new source revision; do not call
Astra again. The first 31 acquisition attempts remain charged, leaving at most
89 attempts for the replacement commissioning test under the shared 120 cap.
The new normalizer must use only the replacement thirty-episode corpus. Report
both runs' compute, acquisition and accepted-data counts explicitly.

Three additional, isolated Astra requests tested W07 against synthetic report
fixtures. Allocations were S1/S2/S3 = 25/30/20 when all stages had constructed
zero success, 5/10/60 with constructed S1/S2 competence and S3 failures, and
25/25/25 for uniform with no report. All were valid without repair. These are
generation-contract evidence, not measured learner learning or curriculum
benefit. Total logical generation calls so far: four, with unknown billing.
