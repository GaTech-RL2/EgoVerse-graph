# Weighted mixtures and homogeneous graph training

`egomimic.rldb.weighted_dataset.WeightedDataset` wraps named datasets and
samples with replacement. Weights choose the **dataset**, independently of its
length; frames within each dataset are uniform. A 3:1 mixture therefore draws
approximately 75% of its samples from the first source even if it is smaller.
Weights must cover every active dataset, be finite and nonnegative, and include
at least one positive value. Zero disables sampling that source.

The existing MultiDataModuleWrapper owns the mixed loader. Normalization stays
in each original MultiDataset, once, before graph execution. Validation retains
its existing sources, groups and sampling order.

```yaml
data:
  dataset_weights:
    eva_bimanual: 1.0
    human_bimanual: 3.0
  weighted_dataloader_params:
    batch_size: 64
    num_workers: 6
    pin_memory: true
  samples_per_epoch: 100000
  sampling_seed: ${seed}
```

The mixed batch size is the **total per rank**, not a separate quota for each
source. Collation retains each source's schema; a batch may omit a source.
`samples_per_epoch` is a global draw count rounded up to a multiple of the
world size. By default it is the sum of active source lengths. Sampling is
reproducible by seed and epoch and independent of worker count. Lightning
advances the sampler epoch; standalone callers use `sampler.set_epoch(epoch)`.
Direct dataset indexing returns `(source_name, sample)` deterministically;
weights are applied by `WeightedDataset.sampler()`.

Without `dataset_weights`, the existing CombinedLoader is retained. The
per-source `train_dataloader_params` configure that loader; weighted mode uses
`weighted_dataloader_params` instead. It owns its sampler and collator and
does not apply the per-source anchor samplers.

The supplied Eva/human overlay uses the native graph API:

```sh
source emimic/bin/activate
python -m egomimic.trainHydra --config-name=train_zarr_cartesian_pi \
  data=pi05/cotrain_pi_lang_6d model=pi05/pi0.5_cotrain_eva_aria_6d \
  +experiment=weighted_cotrain data.dataset_weights.human_bimanual=3.0
```

Change the source names to match the selected data recipe. Distributed mixtures
require `trainer.strategy=ddp_find_unused_parameters_true` and
`trainer.sync_batchnorm=false`, as in the overlay: ranks may use different
source-specific modules. Invalid combinations fail before training starts.
Total loss is synchronized; conditional source diagnostics stay on rank zero,
avoiding collectives for metric names absent on another rank.

PipelineAlgo now executes training stage by stage across sources. Compatible
inputs concatenate along the existing batch dimension, and outputs split back
to their sources with gradients intact:

- HPT shared stems run once per compatible group. Domain-specific stems retain
  routing and combine sources that use the same stem. Domain embeddings are
  applied per source before a shared trunk forward.
- The shared observation encoder retains observation-episode boundaries.
  Encoders with custom forward context retain separate execution.
- Flow and diffusion denoisers share a larger forward. Noise generation,
  per-source loss reduction and diffusion scheduler checks remain explicit.
- PI converts each source with its own action encoding, normalization and
  camera masks, pads state to the policy width, then batches compatible inputs.
  Multiple datasets of the same embodiment remain independently identifiable.

Different non-batch shapes, dtypes or devices form separate groups. Custom
stages keep their existing `execute` behavior unless they implement the
`execute_batches` hook. Inference retains its existing execution path.
Use `++model.pipeline.homogeneous_training=false` for separate training
forwards. This changes execution only: losses always average by sample count,
so unequal source counts preserve the requested mixture proportions.

Batch-dependent layers and stochastic augmentations may produce different
draws when combined. Deterministic tests compare outputs, source losses and
parameter gradients; a two-rank CPU training test covers missing sources,
unused parameters, conditional logging and sampler epoch updates. PI tests
exercise the real adapter with a tiny OpenPI network; pretrained GPU runs
remain a separate validation step.
