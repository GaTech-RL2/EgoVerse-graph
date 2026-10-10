"""Explicit native LIBERO suite recipes; shared model/science stays unchanged."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeLaunchProfile:
    suite: str
    name: str
    replay_environment: str
    task_uids: tuple
    action_velocity_weight: float = 1.0

    @property
    def experiment(self):
        return "libero_historical/" + self.name


PROFILES = {
    "libero10": NativeLaunchProfile(
        "libero10",
        "action_flow_libero10_h240_euler50_dithalf_80k_s42",
        "LIBERO10_REPLAY_ROOT",
        tuple(range(30, 40)),
    ),
    "libero_object": NativeLaunchProfile(
        "libero_object",
        "action_flow_libero_object_h240_euler50_dithalf_80k_s42",
        "LIBERO_OBJECT_REPLAY_ROOT",
        tuple(range(10, 20)),
    ),
    "libero_goal": NativeLaunchProfile(
        "libero_goal",
        "action_flow_libero_goal_h240_euler50_dithalf_80k_s42",
        "LIBERO_GOAL_REPLAY_ROOT",
        tuple(range(20, 30)),
    ),
    "libero_spatial": NativeLaunchProfile(
        "libero_spatial",
        "action_flow_libero_spatial_h240_euler50_dithalf_80k_s42",
        "LIBERO_SPATIAL_REPLAY_ROOT",
        tuple(range(10)),
    ),
}

AV0_PROFILES = {
    suite: NativeLaunchProfile(
        p.suite,
        p.name.replace("_80k_s42", "_80k_av0_s42"),
        p.replay_environment,
        p.task_uids,
        0.0,
    )
    for suite, p in PROFILES.items()
}


def profile_for_suite(suite):
    try:
        return PROFILES[suite]
    except (KeyError, TypeError):
        raise ValueError("unsupported native LIBERO suite: " + str(suite)) from None


def profile_for_experiment(experiment):
    for profile in (*PROFILES.values(), *AV0_PROFILES.values()):
        if profile.experiment == experiment:
            return profile
    raise ValueError("unsupported native LIBERO experiment: " + str(experiment))


def profile_for_config(config):
    profile = profile_for_experiment("libero_historical/" + str(config.get("name")))
    if config["benchmark"]["suite"] != profile.suite:
        raise ValueError("native suite/recipe identity mismatch")
    expected_ablation = (
        "action_velocity_off" if profile.action_velocity_weight == 0.0 else None
    )
    if config.get("run_provenance", {}).get("ablation") != expected_ablation:
        raise ValueError("native ablation provenance/profile mismatch")
    if (
        config.get("run_provenance", {})
        .get("objective", {})
        .get(
            "action_velocity_weight",
            1.0 if profile.action_velocity_weight == 1.0 else None,
        )
        != profile.action_velocity_weight
    ):
        raise ValueError("native ablation objective provenance mismatch")
    for mode in ("train", "valid"):
        if (
            config["data"][mode + "_datasets"]["libero_panda"]["resolver"]["suite"]
            != profile.suite
        ):
            raise ValueError("native suite/resolver identity mismatch")
    return profile


def profile_for_argv(argv):
    selected = [
        arg.split("=", 1)[1]
        for arg in argv
        if arg.split("=", 1)[0].lstrip("+") == "experiment" and "=" in arg
    ]
    if len(selected) != 1:
        raise ValueError("exactly one native experiment override required")
    return profile_for_experiment(selected[0])
