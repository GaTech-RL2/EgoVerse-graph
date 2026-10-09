# Camera alignment monitor (front D405, YAM rollout station)

```
./run.sh [DATA_DIR]      # start in the background (default DATA_DIR ~/camera_align)
./stop.sh [DATA_DIR]
python -m egomimic.robot.camera_align.monitor --help
```

It listens on 127.0.0.1:8090 only. View it from your own machine with
`ssh -L 8090:127.0.0.1:8090 rl2-yam` and open http://localhost:8090.

Four times a second, on the rollout dashboard's own front-camera stream:

- **Camera vs the frame reference.** `Save frame reference` stores the current frame
  (take it with an empty table and the arms at home) under DATA_DIR and points
  `reference.json` at it. SIFT features give roll / tilt / pan in degrees and the
  scene shift in pixels, using the D405's own intrinsics from
  `yam_rl2_agentview_extrinsics.yaml` (the overlay is shown undistorted).
- **Desk vs camera and desk vs room.** The table's left and right edges are located
  to sub-pixel precision in two rows above the robot bases and tracked near their
  reference positions; SIFT features on the table top and in the background are
  fitted separately. Edges shifting = the desk moved relative to the camera; the
  table shifting while the background stays = the desk moved relative to the room;
  everything shifting together = the camera panned. Converted to millimetres at the
  table using the calibrated camera height.
- **Tags.** ArUco (default 4x4_50, 200 mm) detection with a pose per tag. After
  `Save tag reference` the same roll / tilt / pan is computed from the tag corners
  against the saved poses (`tag_reference.json`), which is the precise, repeatable
  check for later sessions; with two tags their distance flags a tag that moved.
- Advice lines for the camera, a verdict for the desk, and sparklines over the last
  three minutes.

**Caveat.** The rollout dashboard server keeps one websocket client. Starting this
monitor supersedes the rollout dashboard's browser tab; reopening that tab
supersedes the monitor (it then waits for `Reconnect stream`). The monitor never
sends a command to the rollout dashboard and never opens a camera or a robot.
Stop it before a real rollout.
