# Verification without committed tests

The rebuilt stack removes the tracked regression suites, their fixtures and
test-only runners, including the vendored LeRobot suite. Do not commit new
tests, fixtures, or local verification scripts. Run focused regression checks
outside the application repository and include their results and limitations
in each pull request.

CI retains locked-environment checks, Ruff, recursive Hydra composition and
offline component construction. It no longer runs pytest or the former
installed-wheel regression gate. Passing CI therefore does not establish
behavioral parity, wheel-install compatibility, hardware safety, or successful
training. Those require separate evidence.

The Qwen architecture metadata used by the constructor audit is retained at
`scripts/resources/qwen3_embedding_06b_config.json` with the same checksum.
This is audit input, not a test suite. Existing dependency pins, runtime guards,
training/validation evaluators, and hardware-control examples are unchanged.
The vendored ARX examples named `test_*` are operational robot programs, not
automated regression tests.

Historical test references and validation receipts in older documents describe
the source revision that produced them. Their sources can be recovered from Git
history (for example the pre-rebuild main commit
`a525d73897bec64397048e7c8897fb886555567b`); they are no longer commands for the
current checkout. Do not rewrite historical provenance or weaken checkpoint
compatibility guards to accommodate a source cleanup.
