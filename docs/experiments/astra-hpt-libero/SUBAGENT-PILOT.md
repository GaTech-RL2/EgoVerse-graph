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
