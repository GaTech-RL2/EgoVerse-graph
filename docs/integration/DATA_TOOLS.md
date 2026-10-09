# Data and visualization tools

Owner: graph integration. These tools use recorded data and do not load a model,
download pretrained weights, fit normalization statistics, or alter episodes.

## Recorded trajectories, language, images and arrays

```bash
source emimic/bin/activate
python -m egomimic.scripts.viz_language \
  data=cotrain_pi_lang split=valid output_dir=/path/to/new-output
```

The selected DataModule owns split construction, collation and complete-episode
limits. The tool uses its `prepare_visualization` and
`iter_visualization_batches` capabilities. It never edits dataset index maps or
chooses an embodiment class. Configured renderers paint every sample in a batch;
the shared video implementation preserves actual episode/frame identities.
By default, three complete episodes per source are selected. A partial-video
request must explicitly disable `evaluator.complete_episodes`.

Outputs include MP4s, sampled PNGs, safe NPZ arrays, the resolved tool config and
`visualization-receipt.json`. Sample cadence/count and exported fields are set
by `evaluator.sample_every`, `max_saved_samples`, and `artifact_keys`. Stored
arrays use the data transform's native units/frame. Optional per-source
`artifact_transforms` are explicit batch callables for exporting another frame;
the tool never guesses calibration or calls a camera pose a base pose.
An existing output directory is rejected to preserve previous work.

Python callers can compose `viz_language` and call
`egomimic.scripts.viz_language.visualize_data(cfg)`. This replaces the old
hardcoded `data_visualization.py` / notebook workflow with reusable data
selection, PNG/array export, trajectory/keypoint renderers and language videos.
The old scripts depended on absent `rldb.utils` and `egomimicUtils` imports at
the audit baseline; copying them would not preserve a runnable workflow.

For wrist-frame targets, configure the renderer's `transform_list` explicitly.
Prediction-renderer selections such as `pi_cartesian_lang_wrist` remain usable
with `evaluator=eval_hpt_wrist`; keypoints use `eval_keypoints_wrist`. These
renderer selections expect the evaluator to perform coordinate reversion,
as in the original runtime. They do not independently apply it a second time.

## Zero/nonfinite field audit

```bash
python -m egomimic.scripts.check_data data=eva_pi_lang \
  output_dir=/path/to/new-audit
```

Checks, source names, field keys and feature slices live in
`evaluator/data_fields.yaml`. The retained EVA recipe checks pose columns
`[0,6)` and `[7,13)` independently and reports nonfinite actions. Results are
streamed to `data-audit-rows.jsonl` with exact source/episode/frame identities;
`data-audit.json` contains counts. The old `eva_process/check_zero.py` command
delegates here with a migration warning. Its retained CPU Slurm wrapper requires
an explicit checkout instead of a developer's private path; integration jobs
continue to use OSMO.

## HDF5 conversion

The maintained converter is
`egomimic.scripts.eva_process.eva_to_zarr.convert_episode`. It retains raw
per-arm pose/joint/gripper fields, JPEG cameras, metadata and calibration in the
current Zarr format. It now rejects an existing episode or requested preview
before reading or writing conversion data. The input HDF5 remains unchanged.

The retained command below delegates to that implementation:

```bash
python -m egomimic.rldb.zarr.hdf5_to_zarr \
  --hdf5-path /path/to/timestamp.hdf5 --output-dir /path/to/new-data --arm both
```

The audited old debug converter imported nonexistent
`eva_process.zarr_utils`, ignored `extrinsics_key`, and supplied invented
identity intrinsics. It is replaced by the maintained converter, not reinstated.
Obsolete debug flags (`prestack`, `no_rot`, `low_res`, `extrinsics_key`) fail with
an explicit migration message. Trajectory interpolation, action encodings and
coordinate transforms belong in loader YAML. Local conversion imports do not
require SQL, cloud packages or a policy backend.

## Evidence and remaining gates

`tests/test_data_tool_parity.py` converts an actual HDF5 fixture, checks unchanged
source bytes and overwrite rejection, imports with cloud/model modules blocked,
encodes a complete annotated episode, validates video frame count/FPS and saved
array identities, and checks configurable field-audit slices. These are CPU
functional gates; combined-tree and real-data checks remain separate.
