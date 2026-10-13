"""Bind the verified graph checkout before adding a simulator capsule."""
from pathlib import Path
import sys


def load_planar_runtime(source, simulator):
    source, simulator = Path(source).resolve(), Path(simulator).resolve()
    # Collection capsules may also contain a minimal egomimic writer package.
    # Appending preserves the policy checkout and prevents the simulator's
    # package initializer from adding its root ahead of that checkout.
    sys.path.append(str(simulator))
    from egomimic.eval.planar_rollout import load_planar_graph_policy
    from egomimic.rldb.zarr import zarr_dataset_multi
    from Tsimulation.sim_v2.pushshapes import env
    dataset_path = Path(zarr_dataset_multi.__file__).resolve()
    env_path = Path(env.__file__).resolve()
    if dataset_path != source / "egomimic/rldb/zarr/zarr_dataset_multi.py":
        raise ImportError("Dataset module did not resolve from the pinned policy checkout")
    if env_path != simulator / "Tsimulation/sim_v2/pushshapes/env.py":
        raise ImportError("Simulator module did not resolve from the pinned capsule")
    return env.PushShapesEnv, env.SIM_VERSION, load_planar_graph_policy, {
        "dataset_module": str(dataset_path), "simulator_module": str(env_path)}
