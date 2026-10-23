# Cartesian ARC stream selection

The former race, joint-distance and multistream selectors have been removed.
Cartesian ARC always uses independent stream supports and clocks, with
first-moving-stream replanning. Choose only the clock representation and stream
layout described in [ARC_CLOCK_CONTRACT.md](ARC_CLOCK_CONTRACT.md).

The supported layouts are full separation, translation/rotation,
translation-only and rotation-only. Projection is maintained on an optional
separate review branch, not in this stack's default source.

Historical checkpoints require their recorded Git source and configuration.
The historical mode documentation remains available in Git history; its timing,
normalization and evaluation contracts are not interchangeable with arrival-v2.
