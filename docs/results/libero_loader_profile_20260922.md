# LIBERO training bottleneck, September 22

Measurements at 22:32–22:39 UTC identify repeated per-sample Zarr access in the native LIBERO loader as a substantial bottleneck shared by ARC and OAT. Caching array handles and ARC targets did not cache decoded replay data. Current training still indexes and decompresses seven arrays for every sample.

## Live observations

Read-only 15-second process/GPU samples covered ARC STK1 Spatial, ARC DUR1 LIBERO-10, and OAT Spatial policy training on their existing two-L40S allocations.

- Each job has eight active training loader processes, generally using approximately one CPU core each. Eight additional validation workers were idle and are excluded from this observation.
- Each job has a 24-core quota. No CPU throttling occurred; aggregate CPU consumption was approximately 9–10 cores. This is a loader execution bottleneck, not an exhausted job CPU quota.
- Physical storage reads were zero in the sampled training processes. Cached reads still involved indexing, decompression, copying, and Zarr's synchronous-to-asynchronous dispatch.
- LIBERO-10 loader processes read about 7.2 GB through cached file reads in 15.5 seconds, versus approximately 2.2 GB for ARC Spatial.
- GPU activity was strongly asymmetric. A high utilization counter with low power/memory activity can include synchronization; these counters are not a measurement of useful model compute. No CUDA/NCCL trace was collected.
- OAT Spatial showed the same saturated loader pattern. ARC encoding alone cannot explain the shared bottleneck.

## Controlled data-access comparison

On each ARC worker, a bounded CPU-only probe used four fixed-seed random episodes, 256 fixed requests, one PyTorch thread, and the exact deployed `_ReplayEpisode.__getitem__`. Array handles were already warm. A second path indexed identical decoded episode arrays from RAM and retained the same episode padding, finite checks, contiguous conversion, and float tensors. Thirty-two requests per suite passed bitwise tensor equality for all seven keys. Neither path included normalization, collation, host-to-device transfer, or model compute.

| Suite | Current loader, ms/sample | Decoded RAM, ms/sample | Access speedup |
| --- | ---: | ---: | ---: |
| Spatial | 9.37 | 0.178 | 52.6× |
| LIBERO-10 | 17.26 | 0.188 | 91.9× |

These are sample-construction speedups, **not end-to-end training speedups**. They were measured alongside active training, over a small episode subset, and do not establish a new completion ETA. A separate profiled pass attributed about 96% of Spatial and 98% of LIBERO-10 sample-construction wall time to Zarr orthogonal indexing, including its internal waiting. The profile does not resolve all work in Zarr's background threads.

LIBERO-10's released image arrays use Zstd-compressed chunks containing 41 frames. A request for two observation frames therefore decompresses at least one much larger chunk; requests crossing a chunk boundary require two. The converted Spatial arrays use one-frame LZ4 chunks. This explains additional avoidable work in LIBERO-10; its training split also has 2.22× as many windows as Spatial and therefore more updates under the fixed 5001-epoch schedule.

## Next performance change

Decode replay arrays once into shared read-only RAM or memory-mapped arrays, preserving uint8 images and the exact current sampling, padding, splits, normalization, and values. Avoid a full independent decoded copy in every loader worker. The full two-camera uint8 image corpus occupies approximately 5.70 GiB for Spatial and 12.64 GiB for LIBERO-10, before other arrays and overhead, within the existing 128 GiB job limit when shared appropriately.

Validate full sample/batch parity, then measure complete optimizer steps on a checkpoint-preserving pilot before migrating the campaign or promising a speedup. Additional GPUs alone do not remove repeated CPU data preparation. Increasing loader workers may help temporarily, but does not remove this work.

No training source, running processes, checkpoints, budgets, or workflow allocations were changed by this diagnosis. The 28 training jobs remain on their previously recorded source pins. No production decoded cache has been deployed.

Raw profiles and probes are retained in `scratch/libero-bottleneck-20260922/`; the accompanying JSON preserves the data profiles and summarized live measurements. The prior sustained campaign rates and fixed update budgets are recorded in [the migration report](libero_arc_ddp_migration_20260922.md).
