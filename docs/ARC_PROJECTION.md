# Optional ARC projection extension

This PR is a separate optional extension. It is not part of the main arrival-v2
stack's ancestry or default stream menu.

Select `abc.arc_streams=projection` only on this branch. Each arm has one clock,
giving M×16 tokens. Source progress is measured at `p + R @ (0.02, 0.02, 0.02)`
metres: a fixed point 2 cm along each local EEF axis. Rotation can therefore
advance the progress even when the EEF origin does not translate. XYZ, rotation
and gripper are emitted on those common supports.

The first interval starts at measured current proprio; its projected-point
distance is included in the incoming PW rate or duration speed bound. All other
arrival-v2 rules remain unchanged: next-frame source origin, raw absolute
duration or incoming speed, first-moving-stream termination, explicit decode
context and post-token normalization. Source D spans the future projected path,
excluding the separate current-to-first-future incoming segment.

The offset is fixed in this source revision, not an additional runtime tuning
parameter. Use fresh normalization and a new experiment identity. This branch
does not migrate historical projection checkpoints or claim performance gains.
