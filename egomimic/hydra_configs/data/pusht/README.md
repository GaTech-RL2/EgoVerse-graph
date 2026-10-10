# PushT data configuration parents

The selected U-Socket + ChainGripper pair uses the maintained shared parents:

- `parents/standard_dp_uc_h16.yaml`: standard DP co-training data wiring.
- `parents/action_flow_uc_h16.yaml`: native Action Flow co-training data wiring;
  the manual4919 profile overrides only corpus identities/counts.
- `parents/episode_split.yaml`: shared resolver, 1% episode validation and bounds
  contract. Split seeds stay in each profile: DP follows `${seed}`, while the
  existing Action Flow profiles explicitly use42.
- `parents/action_flow_points6_uc_h16.yaml`: checkpoint-compatible frozen4920
  Action Flow view. It is distinct from manual4919, not a current pair default.

Training and validation definitions share YAML aliases. Keep per-representation
transforms, source counts/hashes and loader settings in the actual views; do not
combine rotvec4, common5 and points6 as if they were interchangeable.

The former manual3000 co-training, frozen4920 co-training and standard-retimed
co-training parent paths are removed. Their current references now select the
neutral parents. Historical checkouts retain the old names for their recorded
checkpoints; running sources and saved checkpoint configs are never migrated.

Before adding a file, extend the shared parent or an existing profile. A new
output path, scheduler retry or logging identifier does not require a new data
configuration. New scientifically different corpora/representations do.

`tests/test_pusht_parent_cleanup.py` compares all51 existing PushT recipes to
hashes captured from sourcef4ed4b97 before this cleanup. It also rejects dangling
references to removed parent paths and checks the selected2999/4919 pair retains
1% splits and full32+32 batches. CPU config equality is not a training smoke or
source-adoption approval.
