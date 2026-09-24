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
the source distribution, and writes an installation receipt. Any required
transformers patch must be applied explicitly and verified by real PI import
and train-step gates; a resolved lock alone does not establish PI parity.

Do not run `uv sync` between source installation and the PI smoke: synchronization
may remove the separately installed distribution. Re-run the source installer
after synchronization. A failed `uv pip check` for the intentionally excluded
OpenPI simulation bundle is not proof of a supported full OpenPI installation;
the supported surface is the tested PI model adapter described here.
