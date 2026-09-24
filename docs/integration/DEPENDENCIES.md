# Dependency decision

Retain the destination EgoVerse policy: install OpenPI's immutable source with
`--no-deps`, after installing a locked model runtime. Do not install OpenPI's
unrelated simulation/data-collection dependency bundle into this environment.

Evidence at the pinned audit heads:

- `uv lock --check` originally fails: `av==12.0.0` conflicts with Lerobot's
  `av>=14.2.0`.
- A local resolution probe with AV 14.4 also fails: OpenPI's `gym-aloha`
  requires a newer dm-control or older MuJoCo than the project pins. The probe
  was not retained; changing just AV does not solve the environment.
- EgoVerse `ec5c903c` already documents `--no-deps` in `pyproject.toml` and
  installs/tests the pinned OpenPI model runtime in `.github/workflows/ci.yml`.
- The revised lock resolves all declared extras without dependency overrides.
  Existing AV, Torch, torchvision, MuJoCo and dm-control pins are preserved.

Use a separate environment for the Yam SDK, as previously required by the
numpy/rerun conflicts. The PI extra now describes the supported model runtime;
it deliberately does not declare the incompatible full OpenPI distribution.
The source installation receipt makes this exception visible and reproducible.

```bash
uv venv emimic --python 3.11
source emimic/bin/activate
UV_PROJECT_ENVIRONMENT=emimic uv sync --locked --extra pi05
python scripts/install_pi05_source.py
```

The script verifies the source pin and locked runtime versions, installs only
the source distribution, and writes an installation receipt. It downloads the
five required Transformers files from the same immutable OpenPI commit,
verifies every SHA-256 in `scripts/pi05_transformers_manifest.json` before
changing any installed file, and preserves the originals by content hash.
Atomic inode replacement avoids modifying uv cache hardlinks or other virtual
environments. Import and real-weight train-step gates remain mandatory; a
resolved lock and verified patch alone do not establish PI parity.

Do not run `uv sync` between source installation and the PI smoke: synchronization
may remove the separately installed distribution. Re-run the source installer
after synchronization. A failed `uv pip check` for the intentionally excluded
OpenPI simulation bundle is not proof of a supported full OpenPI installation;
the supported surface is the tested PI model adapter described here.

The retained EgoBridge recipe uses the optional `alignment` extra, pinning
GeomLoss 0.3.1 and tslearn 0.8.1 to the audited EgoVerse environment. Add
`--extra alignment` to the sync command before installing OpenPI. Ordinary
HPT/PI graphs do not import these backends; the configured alignment stage does.

The optional `diagnostics` extra pins CPU `umap-learn==0.5.9.post2`. Its provider
uses the same PI model environment and normal inference graph. There is no
automatic cuML backend switch. Install `--extra diagnostics` before the source
installer when running latent reductions.
