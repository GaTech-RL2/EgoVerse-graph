# Future multi-embodiment experiments

The September 18 source contract replaces the old scripted U-Socket and
ChainGripper corpora with the exact data used by the current co-training runs.
Historical recipes and datasets remain available for reproducibility.

Use `planar_v2_multiemb_cotrain_flipper_holdout_*` for the next experiments.
Do not reuse the old `artic_cotrain7` roster: it trained on Flipper.

| Role | Embodiments / sources |
| --- | --- |
| Audited co-training sources | U-Socket: 2,999 clean episodes; ChainGripper: 3,000 clean plus the separately pinned 1,920 obstacle episodes |
| Newly generated candidates | Parallel gripper, UMI, small circle, straight bar (`stick`), triangle |
| Retained existing sources | Suction and spring, 3,000 ideal-control episodes each |
| Held out of training and validation | **Flipper** |
| Excluded from the new roster | Scoop |

The source contract and generation thresholds are in
[`multiemb_cotrain_sources_v1.yaml`](../egomimic/hydra_configs/data/pusht/multiemb_cotrain_sources_v1.yaml).
The baseline and all four uniform ARC D80/M16/M56 duration/stacked recipes share
the same dataset configuration. Stacked retains the existing independently
timed velocity representation; this migration does not change its codec.

## Generation and provenance

The simulator's `Tsimulation/sim_v2/generate/cross_embodiment.py` uses the
audited clean co-training episodes as sources. It regenerates actions and
images in the target embodiment's physics; it does not relabel source actions.
Gripper/UMI use the recorded object path after acquiring a physical grasp.
Pushers use the same recorded initial scene and goal with new contact planning.
The simulator physics are unchanged from the co-training data audit.

Each accepted episode must finish at at least 0.95 coverage, make useful
physical contact, pass motion/penetration checks, and reproduce its full state
trajectory in a separate rendered replay. All newly written observations are
**before action**, with `action_target_offset=1` for observation horizon 2.
The existing obstacle corpus retains its **after action** storage and offset 2.

Each derived episode records its source URI, content hash, exact source episode,
train/validation assignment, generation method and simulator source capsule.
Derived siblings inherit the source episode's original split. The existing
obstacle split is preserved; it has not been established to be disjoint by
original demonstration family.

The 24-scene pilot accepted 22 gripper, 22 UMI, 22 small-circle, 10 bar and
10 triangle episodes. Every accepted trajectory replayed with zero state error.
These are **generation acceptance counts**, not learned-policy scores.
Ten preview videos were inspected through ordered frames covering the clips.
Pusher contact replanning takes substantially more steps than grasp transport.

Bulk generation targets 2,970 training and 30 validation episodes per target,
using only the available source scenes. Exhausting the source pool produces an
explicit underfilled-quota receipt; sources are not duplicated to fill a count.

## Loading a completed release

`ManifestEpisodeResolver` checks the release SHA, episode bytes, source split
consistency and the held-out roster. It rejects pilot manifests and prevents
Flipper from entering training/validation loaders. Normalizers must be fitted
on training data only. Any later Flipper inference normalization must be derived
from training statistics; Flipper observations must not be used to fit it.

The release manifest includes archive hashes and relative episode paths.
Stage its immutable bytes without replacing any existing files:

```bash
source emimic/bin/activate
PYTHONPATH=. python scripts/data/stage_pinned_dataset.py \
  --manifest /workspace/multiemb/READY.json --sha256 <release-sha256> \
  --out /workspace/multiemb/data
```

Then select one of these experiments and supply
`planar.multiemb_manifest_sha256=<release-sha256>`:

- `pusht/planar_v2_multiemb_cotrain_flipper_holdout_dp`
- `pusht/planar_v2_multiemb_cotrain_flipper_holdout_arc_duration_D80_M16`
- `pusht/planar_v2_multiemb_cotrain_flipper_holdout_arc_duration_D80_M56`
- `pusht/planar_v2_multiemb_cotrain_flipper_holdout_arc_stacked_D80_M16`
- `pusht/planar_v2_multiemb_cotrain_flipper_holdout_arc_stacked_D80_M56`

The required manifest hash intentionally remains unset until the release is
ready. Dataset/source pins and offsets are shared across all five recipes.
DP executes half its native action chunk; ARC executes half its waypoint
supports using the predicted clocks and then replans, starting at index zero.
No new model-training job is launched by this data-generation work.

Artifacts are under `s3://rldb/experiments/multiemb-cotrain-20260918/`.
The immutable pilot lives in `pilot-v1/`; bulk outputs and source capsules have
separate versioned prefixes. See the campaign receipts for active workflow IDs.
