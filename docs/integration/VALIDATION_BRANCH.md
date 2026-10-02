# Tests and documentation companion branch

The runtime integration PR targets `main` from
`codex/graph-consolidation-20261001`. Tests, fixtures, design documents, migration
reports, validation tooling, notebooks and documentation images are preserved on
`codex/graph-validation-20261001`. The companion branch is an archive and a
validation source; do not merge its complete tree into the runtime branch.

Each runtime revision pins an immutable companion commit in
`.github/validation-ref`. That commit's `.github/validation-paths.json` enumerates
the externalized files and their SHA-256 hashes. The companion snapshot includes
runtime source for historical context, but the restore helper only retrieves
the manifest's validation/documentation files. It refuses to replace tracked
checkout files, modified local files or symlinks. CI therefore tests the actual
runtime revision under review, not the companion snapshot's runtime.

In a clean runtime checkout:

```bash
source emimic/bin/activate
python .github/scripts/restore_validation.py
HF_HUB_OFFLINE=1 WANDB_MODE=disabled pytest tests -q
python scripts/audit_hydra_configs.py --output config-audit.json
HF_HUB_OFFLINE=1 python -m scripts.audit_components --output component-audit.json
```

The restore command materializes untracked companion files in the same relative
locations expected by the existing tests and tools; running it again is safe
when those files are unchanged. Use an isolated checkout for validation and do
not stage those files into the runtime branch. Runtime licenses, third-party
package readmes required by their build metadata, and the dataset-card template
consumed by LeRobot remain with their packages.

To update tests or documentation, make the changes on the companion branch and
run `python scripts/integration/update_validation_manifest.py`. Commit the files
and manifest together, then update `.github/validation-ref` in the runtime PR
to that full commit. CI must pass with that pin. A moving branch name is not
accepted as a validation reference. The original tests and migration evidence
remain available through Git history; the separation does not waive CPU,
installed-wheel, constructor, GPU or review gates.

## Initial Linux CI failure

On runtime snapshot `9d8b6b87498ce95d15c53c6f7dafd8106992da10`, GitHub CI run
`36938596575` passed 1,977 tests and failed 16 tests in
`test_arc_token_bytes_match_frozen_reference`. All 439 config and constructor
contexts passed. Those 16 assertions compared raw floating-point bytes against
hashes captured on macOS arm64; Linux x86-64 produced different hashes using the
same NumPy 1.26.4 and SciPy 1.17.1 versions. The test files were unchanged from
source #197.

The corrected parity check executes the immutable tokenizer source from #197
(`20507c6866f8a167e0e2858141299dbf6ccddeb0`) with the same platform, dependencies
and inputs as the integrated tokenizer. It still requires exact token and
preserved-row bytes for every case, without adding tolerances or skipping cases.
The frozen source's SHA-256 is checked before execution. All 17 reference cases
match the original macOS hashes locally; those hashes are preserved as fixture
provenance. A subsequent Linux CI pass is required before claiming this failure
resolved there.

The separate PR Wiki Tracker failure is an empty `OBSIDIAN_VAULT_REPO` secret:
its API request goes to `/repos//contents/raw/prs/pr-0198.json` and returns 404.
The integration does not disable that check or fabricate a pass. Its destination
and authentication need to be configured by the repository maintainers.
