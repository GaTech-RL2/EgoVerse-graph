"""Compile and render an engineering starter; not a generated-scene novelty witness."""

import argparse
import importlib.metadata
import os
from pathlib import Path

import numpy as np
from PIL import Image

from egomimic.experiments.astra_push.artifacts import publish_json
from egomimic.experiments.astra_push.schemas import (
    SceneSpec,
    TaskSpec,
    canonical_hash,
    instruction_for,
)


def starter(stage="S1"):
    cubes = [
        {
            "name": "cube_a",
            "color": "red",
            "placement": {
                "center_xy": [-0.03, 0.0 if stage == "S1" else -0.09],
                "half_width": 0.03,
            },
        }
    ]
    targets = [
        {
            "name": "target" if stage == "S1" else "left",
            "center_xy": [0.09, 0.0 if stage == "S1" else -0.07],
            "half_width": 0.04,
        }
    ]
    if stage != "S1":
        cubes.append(
            {
                "name": "cube_b",
                "color": "blue" if stage == "S2" else "red",
                "placement": {"center_xy": [-0.03, 0.09], "half_width": 0.03},
            }
        )
        targets.append({"name": "right", "center_xy": [0.09, 0.07], "half_width": 0.04})
    scene = SceneSpec(
        schema_version="astrapush-1",
        scene_id=f"engineering_{stage.lower()}",
        stage=stage,
        cubes=cubes,
        targets=targets,
        fixtures=[{"name": "marker", "center_xy": [-0.18, 0.0]}]
        if stage == "S3"
        else [],
    )
    referent, destination = {
        "S1": ("block", "target"),
        "S2": ("red", "left"),
        "S3": ("left", "left"),
    }[stage]
    task = TaskSpec(
        schema_version="astrapush-1",
        task_id=f"engineering_{stage.lower()}",
        scene_hash=canonical_hash(scene),
        referent=referent,
        destination=destination,
        instruction=instruction_for(stage, referent, destination),
    )
    return scene, task


def probe(output, *, stage="S1"):
    from egomimic.experiments.astra_push.libero_scene import (
        compile_scene,
        load_environment,
    )

    output = Path(output)
    scene, task = starter(stage)
    bundle = compile_scene(scene, task, output / "bundle")
    env = load_environment(bundle)
    try:
        # ControlEnv.reset retries indefinitely on RandomizationError. One call
        # to the underlying reset preserves the experiment's bounded attempts.
        obs = env.env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        images = {}
        for source, target in [
            ("agentview_image", "external_rgb"),
            ("robot0_eye_in_hand_image", "wrist_rgb"),
        ]:
            array = np.ascontiguousarray(obs[source][::-1])
            assert array.shape == (224, 224, 3) and array.dtype == np.uint8
            if array.std() < 1:
                raise ValueError("Rendered camera is blank")
            Image.fromarray(array).save(output / f"{target}.png")
            images[target] = array
        proprioception = np.concatenate(
            [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
        )
        assert proprioception.shape == (9,) and np.isfinite(proprioception).all()
        np.savez_compressed(
            output / "observation.npz", **images, proprioception=proprioception
        )
        (output / "resolved.xml").write_text(env.sim.model.get_xml())
        controller = env.robots[0].controller
        spec = {
            key: np.asarray(getattr(controller, key)).tolist()
            for key in ["input_min", "input_max", "output_min", "output_max"]
        }
        publish_json(
            output / "receipt.json",
            {
                "kind": "engineering_starter_render",
                "stage": stage,
                "novelty_witness": False,
                "teacher_success": None,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ["mujoco", "robosuite", "numpy", "pydantic"]
                },
                "render_backend": os.environ.get("MUJOCO_GL"),
                "action_shape": list(env.env.action_spec[0].shape),
                "proprioception_shape": list(proprioception.shape),
                "control_hz": 10,
                "image_shapes": {
                    name: list(array.shape) for name, array in images.items()
                },
                "controller_scaling": spec,
                "scene_signature": scene.structural_signature(),
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["S1", "S2", "S3"], default="S1")
    args = parser.parse_args()
    probe(args.output, stage=args.stage)
