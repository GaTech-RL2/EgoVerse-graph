"""Follower force updates preflight both arms and verify any restoration."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.yam.interface import YamInterface


class Driver:
    def __init__(self, force=50.0, behavior=None):
        self.force = force
        self.behavior = behavior
        self.calls = []

    def get_robot_info(self):
        return {"limit_gripper_effort": self.force}

    def set_gripper_force_limit(self, value):
        self.calls.append(value)
        if self.behavior is not None:
            self.behavior(self, value)
        else:
            self.force = value


def interface(left=None, right=None):
    result = object.__new__(YamInterface)
    result.arms = ["left", "right"]
    result.controller = {"left": left or Driver(), "right": right or Driver()}
    result.gripper_force_limit = 50.0
    return result


def test_force_updates_both_drivers_and_returns_verified_active_limits():
    yam = interface()
    assert yam.set_gripper_force_limit(35) == {"left": 35, "right": 35}
    assert yam.gripper_force_limit == 35


def test_single_arm_update_preserves_other_driver_and_station_default():
    yam = interface()
    assert yam.set_gripper_force_limit(40, arm="right") == {"right": 40}
    assert yam.controller["left"].calls == []
    assert yam.gripper_force_limit == 50


@pytest.mark.parametrize("value", [0, -1, 50.01, np.nan, np.inf])
def test_force_rejects_invalid_range_before_driver_mutation(value):
    yam = interface()
    with pytest.raises(ValueError):
        yam.set_gripper_force_limit(value)
    assert all(not driver.calls for driver in yam.controller.values())


def test_unknown_arm_rejects_before_driver_mutation():
    yam = interface()
    with pytest.raises(ValueError, match="Unknown Yam arm"):
        yam.set_gripper_force_limit(30, arm="third")
    assert all(not driver.calls for driver in yam.controller.values())


@pytest.mark.parametrize(
    "right",
    [
        SimpleNamespace(),
        SimpleNamespace(get_robot_info=lambda: {"limit_gripper_effort": 50}),
        Driver(force=np.nan),
        Driver(force=51),
    ],
)
def test_all_arm_capabilities_and_previous_limits_preflight_before_writes(right):
    left = Driver()
    yam = interface(left, right)
    with pytest.raises(RuntimeError):
        yam.set_gripper_force_limit(30)
    assert left.calls == []


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_failing_target_and_prior_arm_restore_on_setter_failure_or_cancel(error_type):
    def mutate_then_fail(driver, value):
        driver.force = value
        if value == 30:
            raise error_type("failed after mutation")

    left, right = Driver(force=45), Driver(force=40, behavior=mutate_then_fail)
    yam = interface(left, right)
    with pytest.raises(error_type, match="failed after mutation"):
        yam.set_gripper_force_limit(30)
    assert (left.force, right.force, yam.gripper_force_limit) == (45, 40, 50)
    assert left.calls == [30, 45] and right.calls == [30, 40]


def test_verification_failure_restores_both_arms():
    def ignore_requested_limit(driver, value):
        if value != 30:
            driver.force = value

    left, right = Driver(force=45), Driver(force=40, behavior=ignore_requested_limit)
    yam = interface(left, right)
    with pytest.raises(RuntimeError, match="did not match"):
        yam.set_gripper_force_limit(30)
    assert (left.force, right.force, yam.gripper_force_limit) == (45, 40, 50)
    assert left.calls == [30, 45] and right.calls == [30, 40]


def test_rollback_setter_failure_reports_arm_and_continues_remaining_restores():
    def fail_update_and_restore(driver, value):
        if value == 30:
            driver.force = value
            raise ValueError("update rejected")
        raise RuntimeError("restore rejected")

    left, right = Driver(force=45), Driver(force=40, behavior=fail_update_and_restore)
    yam = interface(left, right)
    with pytest.raises(
        RuntimeError, match="rollback failed: right: restore rejected"
    ) as caught:
        yam.set_gripper_force_limit(30)
    assert isinstance(caught.value.__cause__, ValueError)
    assert left.force == 45 and left.calls == [30, 45]
    assert right.force == 30 and yam.gripper_force_limit == 50


def test_silent_rollback_failure_reports_verified_active_limit():
    def fail_update_ignore_restore(driver, value):
        if value == 30:
            driver.force = value
            raise ValueError("update rejected")

    left, right = (
        Driver(force=45),
        Driver(force=40, behavior=fail_update_ignore_restore),
    )
    yam = interface(left, right)
    with pytest.raises(
        RuntimeError, match="right: active limit 30 N differs from prior 40 N"
    ):
        yam.set_gripper_force_limit(30)
    assert left.force == 45 and left.calls == [30, 45]
    assert right.force == 30 and yam.gripper_force_limit == 50


def test_constructor_cap_rejects_before_driver_or_camera_startup():
    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid force must be rejected before device startup")

    with pytest.raises(ValueError, match=r"\(0, 50\]"):
        YamInterface(
            arms=["left", "right"],
            channels={"left": "can0", "right": "can1"},
            cameras={},
            kinematics={},
            home={"left": np.zeros(7), "right": np.zeros(7)},
            gripper_force_limit=51,
            driver_factory=forbidden,
            camera_validator=forbidden,
        )


@pytest.mark.parametrize(
    "failure", ["setter", "cancel", "missing_limiter", "missing_map"]
)
@pytest.mark.parametrize("opened_previous_arm", [False, True])
def test_default_factory_closes_unregistered_driver_once_on_setup_failure(
    monkeypatch, failure, opened_previous_arm
):
    class StartupDriver(Driver):
        xml_path = "unused-mock.xml"

        def __init__(self):
            super().__init__()
            self.close_calls = 0

        def get_joint_pos(self):
            return np.zeros(7)

        def get_robot_info(self):
            return {
                "gripper_index": 6,
                "kp": np.full(7, 20.0),
                "kd": np.full(7, 0.5),
                "limit_gripper_effort": self.force,
            }

        def close(self):
            self.close_calls += 1

    prior, failing = StartupDriver(), StartupDriver()
    if failure in {"setter", "cancel"}:
        error_type = KeyboardInterrupt if failure == "cancel" else RuntimeError

        def reject(value):
            failing.force = value
            raise error_type("gripper setup interrupted")

        failing.set_gripper_force_limit = reject
    else:
        failing.set_gripper_force_limit = None
        error_type = RuntimeError if failure == "missing_limiter" else AttributeError
        if failure == "missing_map":
            failing._gripper_force_limiter = SimpleNamespace()
    created = []

    def get_yam_robot(*, channel, **_kwargs):
        driver = prior if opened_previous_arm and channel == "can0" else failing
        created.append(driver)
        return driver

    # Exercise the actual default-factory path with a fully mocked vendor import.
    for name in ("i2rt", "i2rt.robots"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    factory_module = ModuleType("i2rt.robots.get_robot")
    factory_module.get_yam_robot = get_yam_robot
    utilities = ModuleType("i2rt.robots.utils")
    utilities.ArmType = SimpleNamespace(YAM="mock-yam")
    utilities.GripperType = SimpleNamespace(from_string_name=lambda value: value)
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    monkeypatch.setitem(sys.modules, utilities.__name__, utilities)

    def forbidden_cameras(*_args, **_kwargs):
        pytest.fail("Configuration failure must precede camera startup")

    monkeypatch.setattr("egomimic.robot.yam.interface.open_cameras", forbidden_cameras)
    arms = ["left", "right"] if opened_previous_arm else ["left"]
    with pytest.raises(error_type):
        YamInterface(
            arms=arms,
            channels={"left": "can0", "right": "can1"},
            cameras={},
            kinematics={},
            home={arm: np.zeros(7) for arm in arms},
            gripper_force_limit=35,
            solver_factory=lambda **_kwargs: object(),
            camera_validator=lambda _config: None,
        )
    assert failing.close_calls == 1
    assert prior.close_calls == int(opened_previous_arm)
    assert created == ([prior, failing] if opened_previous_arm else [failing])
