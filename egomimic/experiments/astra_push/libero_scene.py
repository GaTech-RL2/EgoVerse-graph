"""Trusted LIBERO registrations and BDDL compilation from bounded scene data.

Import only inside the separately locked simulator process. No generated code
is executed and no upstream source files or global user config are modified.
"""

import json
import random
from pathlib import Path

import numpy as np
from libero.libero.envs.base_object import OBJECTS_DICT
from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
from libero.libero.envs.regions import REGION_SAMPLERS, TableRegionSampler
from libero.libero.utils import mu_utils, task_generation_utils
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import InitialSceneTemplates
from robosuite.models.objects import BoxObject, CylinderObject

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.schemas import SceneSpec, TaskSpec, canonical_hash


class RedCube(BoxObject):
    def __init__(self, name="astra_red_cube", joints="default"):
        super().__init__(
            name=name,
            size=[0.02] * 3,
            rgba=[0.85, 0.08, 0.08, 1],
            density=500,
            friction=[1, 0.005, 0.0001],
            joints=joints,
        )
        self.rotation, self.rotation_axis = (0, 0), "z"
        self.category_name = "astra_red_cube"
        self.object_properties = {"vis_site_names": {}}


class BlueCube(BoxObject):
    def __init__(self, name="astra_blue_cube", joints="default"):
        super().__init__(
            name=name,
            size=[0.02] * 3,
            rgba=[0.08, 0.2, 0.85, 1],
            density=500,
            friction=[1, 0.005, 0.0001],
            joints=joints,
        )
        self.rotation, self.rotation_axis = (0, 0), "z"
        self.category_name = "astra_blue_cube"
        self.object_properties = {"vis_site_names": {}}


class ReferenceCylinder(CylinderObject):
    def __init__(self, name="astra_reference", joints=None):
        super().__init__(
            name=name, size=[0.02, 0.02], rgba=[0.9, 0.7, 0.12, 1], joints=joints
        )
        self.rotation, self.rotation_axis = (0, 0), "z"
        self.category_name = "astra_reference"
        self.object_properties = {"vis_site_names": {}}


for key, cls in {
    "astra_red_cube": RedCube,
    "astra_blue_cube": BlueCube,
    "astra_reference": ReferenceCylinder,
}.items():
    if key in OBJECTS_DICT and OBJECTS_DICT[key].__module__ != __name__:
        raise RuntimeError(f"Asset registration collision: {key}")
    OBJECTS_DICT[key] = cls


# Upstream register_problem stores the class and returns None. The module's
# decorated symbol is therefore not a class; use its registered class object.
class AstraPush(TASK_MAPPING["libero_tabletop_manipulation"]):
    def __init__(self, *args, scene_spec, **kwargs):
        self.scene_spec = SceneSpec.model_validate(scene_spec)
        super().__init__(*args, **kwargs)

    def _reset_internal(self):
        super()._reset_internal()
        for fixture in self.scene_spec.fixtures:
            body = self.fixtures_dict[fixture.name]
            body_id = self.sim.model.body_name2id(body.root_body)
            self.sim.model.body_pos[body_id] = [
                *fixture.center_xy,
                float(self.table_offset[2]) + 0.02,
            ]
        self.sim.forward()


register_problem(AstraPush)
REGION_SAMPLERS["astrapush"] = {"table": TableRegionSampler}


class CompiledScene(InitialSceneTemplates):
    scene_spec = None

    def __init__(self):
        # Primitive assets have no articulated affordance regions. Preserve
        # the upstream scene/BDDL interfaces without its scanned-asset crawler.
        self.workspace_name = "main_table"
        self.fixture_object_dict = {"table": ["main_table"]}
        if self.scene_spec.fixtures:
            self.fixture_object_dict["astra_reference"] = [
                f.name for f in self.scene_spec.fixtures
            ]
        self.movable_object_dict = {}
        for cube in self.scene_spec.cubes:
            self.movable_object_dict.setdefault(f"astra_{cube.color}_cube", []).append(
                cube.name
            )
        self.affordance_region_kwargs_list = []
        self.regions = {}
        self.define_regions()

    def define_regions(self):
        for cube in self.scene_spec.cubes:
            self.regions.update(
                self.get_region_dict(
                    cube.placement.center_xy,
                    f"init_{cube.name}",
                    region_half_len=cube.placement.half_width,
                )
            )
        for target in self.scene_spec.targets:
            self.regions.update(
                self.get_region_dict(
                    target.center_xy,
                    f"goal_{target.name}",
                    region_half_len=target.half_width,
                )
            )
            self.regions[f"goal_{target.name}"]["rgba"] = [0.15, 0.8, 0.3, 0.5]
        # Initial regions are invisible, targets are flat noncolliding sites.
        for name, region in self.regions.items():
            region.setdefault("rgba", [0, 0, 0, 0])
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        return [
            ("On", c.name, f"main_table_init_{c.name}") for c in self.scene_spec.cubes
        ]


def compile_scene(scene, task, output):
    scene, task = SceneSpec.model_validate(scene), TaskSpec.model_validate(task)
    task.validate_scene(scene)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene_type = type(
        "AstraScene" + canonical_hash(scene)[:12],
        (CompiledScene,),
        {"scene_spec": scene},
    )
    mu_utils.register_mu("astra_push")(scene_type)
    name = next(key for key, value in mu_utils.MU_DICT.items() if value is scene_type)
    if scene.stage == "S1":
        selected = scene.cubes[0].name
    elif scene.stage == "S2":
        selected = next(c.name for c in scene.cubes if c.color == task.referent)
    else:
        marker = next(f for f in scene.fixtures if f.name == "marker")
        selected = next(
            c.name
            for c in scene.cubes
            if (c.placement.center_xy[1] < marker.center_xy[1])
            == (task.referent == "left")
        )
    saved = dict(task_generation_utils.TASK_INFO)
    try:
        task_generation_utils.TASK_INFO.clear()
        task_generation_utils.register_task_info(
            task.instruction,
            name,
            objects_of_interest=[c.name for c in scene.cubes],
            goal_states=[("On", selected, f"main_table_goal_{task.destination}")],
        )
        paths, failures = task_generation_utils.generate_bddl_from_task_info(
            str(output)
        )
    finally:
        task_generation_utils.TASK_INFO.clear()
        task_generation_utils.TASK_INFO.update(saved)
    if failures or len(paths) != 1:
        raise RuntimeError(f"LIBERO BDDL generation failed: {failures}")
    generated = Path(paths[0])
    bddl = generated.read_text().replace(
        "(problem LIBERO_Tabletop_Manipulation)", "(problem AstraPush)", 1
    )
    canonical = output / "task.bddl"
    with canonical.open("x") as stream:
        stream.write(bddl)
    publish_json(output / "scene.json", scene.model_dump(mode="json"))
    publish_json(output / "task.json", task.model_dump(mode="json"))
    publish_json(
        output / "compiler.json",
        {
            "schema_version": 1,
            "scene_hash": canonical_hash(scene),
            "task_hash": canonical_hash(task),
            "structural_signature": scene.structural_signature(),
            "registered_scene": name,
            "problem": "AstraPush",
            "bddl_sha256": file_hash(canonical),
            "generator": "LIBERO register_mu/register_task_info/generate_bddl_from_task_info",
            "physical_checks": "not_run",
        },
    )
    return output


def load_environment(bundle, *, seed=17):
    from libero.libero.envs import OffScreenRenderEnv

    bundle = Path(bundle)
    scene = SceneSpec.model_validate_json((bundle / "scene.json").read_text())
    meta = json.loads((bundle / "compiler.json").read_text())
    if (
        file_hash(bundle / "task.bddl") != meta["bddl_sha256"]
        or canonical_hash(scene) != meta["scene_hash"]
    ):
        raise ValueError("Scene bundle content changed")
    np.random.seed(seed)
    random.seed(seed)
    return OffScreenRenderEnv(
        bddl_file_name=str(bundle / "task.bddl"),
        scene_spec=scene.model_dump(mode="json"),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=224,
        camera_widths=224,
        control_freq=10,
        horizon=150,
        ignore_done=True,
        hard_reset=False,
    )
