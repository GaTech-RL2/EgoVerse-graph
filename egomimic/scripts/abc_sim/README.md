# ABC sim on the graph: data + rollout

`abc_sim` (amazon-far/abc) data and rollouts for the E1 time / arcdur / arcvel policies.

1. **Data.** `convert_sim_to_zarr.py --src <extracted sim_224 tar> --model <abc>/abc_sim/models/yam_bimanual_empty.xml --out <zarr root>`
   writes one `yam_bimanual` zarr per episode with the real-ABC keys; `ee_pose` is FK through
   abc_sim's MJCF at `<side>_grasp_site` in the `left_arm` frame (left arm matches real ABC
   poses exactly). Data config: `data/abc_arc/abcsim13.yaml` (the 13 single-prompt tasks).
2. **Train.** `scripts/e1/launch_abcsim13.sh` -> `yam_arc_grid/multitask_abcsim13_{time,arcdur,arcvel}`.
3. **Rollout.** `rollout_lambda.sbatch` = `policy_server.py` (graph venv: GraphRobotPolicy from
   the run's exported inference-config, FK/IK through the same MJCF, 45 joint commands per
   request) + `sim_client.py` (ABC's venv, ABC's own `rollout_worlds` / `build_summary`, RTC off).
   `POLICY=hold` is the no-motion floor. `policy_server.py --selftest <episode dir>` checks a
   checkpoint end to end without the simulator.
