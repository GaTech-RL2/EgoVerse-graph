# UNITE diagnostic suite

This package keeps diagnostic capture separate from rendering. A stochastic
checkpoint inference request writes one immutable, hashed artifact; every plot
and animation is then derived from that artifact without rerunning the model.

## Layout

- `gradients.py`: reconstruction-versus-denoising gradient cosine. Set
  `UniteLatentPolicy.log_gradient_conflict=true` for future training runs; the
  resulting `log/unite_gradient_cosine` is suitable for W&B or the offline
  renderer.
- `latent_projection.py`: the shared deterministic PCA/UMAP backend. The
  existing `eval_latent.py` and the UNITE clean/generated latent map both use
  these exact functions.
- `trajectory.py`: native-action chunk-seam capture validation, metrics, plots,
  and command-line rendering.
- `unite_artifact.py`: immutable teacher-forced UNITE captures and offline
  reconstruction/generation, denoising-animation, diversity, PCA, and UMAP
  rendering.

## Artifact contents

Each selected validation example stores the demonstrated action, exact clean
latent reconstruction, generated action, clean and generated latent tokens,
decoded action after every configured sampler step, and all predictions for the
fixed-observation noise-diversity batch. The request manifest must bind the
checkpoint, config, split/order, normalization, RNG seed, sampler steps,
precision, horizon, and rendering selection.

The chunk-seam artifact stores the previous unused native-action tail, raw new
head, and actually executed head at every replan. Position, circular rotation,
grip, velocity, and jerk are derived offline from these raw arrays.

## Rendering

```text
python -m egomimic.eval.diagnostics.unite_artifact validate \
  --artifact <immutable-unite-artifact> --request <request.json>

python -m egomimic.eval.diagnostics.unite_artifact render \
  --artifact <immutable-unite-artifact> --output-dir <render-dir>

python -m egomimic.eval.diagnostics.trajectory render \
  --artifact <immutable-chunk-seam-artifact> --output-dir <render-dir>
```

Use the canonical PushShapes overlay and rollout launchers to create artifacts;
do not make a per-run inference or simulator script. Policy success still comes
from the canonical fixed-seed level-0 and OEC-56 rollouts, not from these
diagnostic plots.
