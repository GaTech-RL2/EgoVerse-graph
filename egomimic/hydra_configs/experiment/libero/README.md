# Current LIBERO recipes

Only the four current Action Flow and four batch16 DP suite recipes belong here.
Action Flow uses 16x16 latents, velocity loss enabled, batch32, original optimizer,
clip3, fixed EMA0.9978, target120000 and validation every20000 updates.
DP uses batch16, accumulation1 and suite-specific retained optimizer-step targets.
Both use the released OAT observations/actions and 10% validation with seed42.

Superseded AV0, native legacy and batch1024 recipes are preserved under
`experiment/libero_historical`. They remain checkpoint reproduction recipes;
their names must not be used as current launch defaults. The historical native
launcher remains routed to its historical profile and cannot launch current
OAT-matched models without separate maintained launcher adoption and preflight.

No existing running source, checkpoint, dataset or split is migrated by this
configuration cleanup. Dataset paths remain required launch inputs. Equal split
seed alone does not prove equal raw demonstration membership across reordered
datasets: use the materialized actual episode lists.

## Contributor style gate

Before submitting recipe or regression-test changes, run the complete CI style
paths with the CI-pinned Ruff version, including `tools`:

```bash
uvx ruff@0.8.6 check egomimic tests scripts tools
uvx ruff@0.8.6 format --check egomimic tests scripts tools
```

A subset check is not the CI style gate. Formatting failures should be repaired
without changing the recipe or checkpoint contract.
