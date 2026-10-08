"""Shared validity and overlap checks for the explicitly new comparison bank."""
import math

import numpy as np

from Tsimulation.sim_v2.collect.obstacle_init import (
    _arena_clearance, _obstacle_clearance, _physical_obstacle_polygons,
    _pusher_arena_overflow, evaluate_candidate, level_init_policy,
)


def pose_vector(init):
    return np.asarray(init["object_pose"] + init["goal_pose"], dtype=np.float64)


def pose_matches(vector, training, tolerance):
    if not len(training): return False
    delta = np.asarray(training, dtype=np.float64) - vector
    delta[:, [2, 5]] = (delta[:, [2, 5]] + math.pi) % (2 * math.pi) - math.pi
    return bool(np.any(np.max(np.abs(delta), axis=1) <= tolerance))


def physically_valid(env, tolerance):
    shapes = list(env.agent.physics_shapes(env))
    values = [env._shapes_static_penetration_depth(env._pusher_body, shapes),
              env._pusher_object_penetration_depth(), env._object_static_penetration_depth(),
              env._object_arena_metrics()[0], _pusher_arena_overflow(env)]
    return all(math.isfinite(float(v)) and float(v) <= tolerance for v in values)


def extra_goal_checks(env, contract):
    polygon = env._build_object_polygon(env.goal_pose[:2], env.goal_pose[2])
    arena = _arena_clearance(env, polygon)
    wall = _obstacle_clearance(polygon, _physical_obstacle_polygons(env.obstacle_level))
    if arena < contract["goal_arena_clearance"] or wall < contract["goal_obstacle_clearance"]:
        return None
    portal = level_init_policy(env.obstacle_level).gate_portal
    depth = None
    if portal is not None:
        a, b = np.asarray(portal, dtype=np.float64)
        edge = b - a
        relative = np.asarray(env.goal_pose[:2]) - a
        depth = float(abs(edge[0] * relative[1] - edge[1] * relative[0]) / np.linalg.norm(edge))
        if depth < contract["gate_goal_depth"]: return None
    return {"goal_arena_clearance": float(arena), "goal_obstacle_clearance": float(wall),
            "goal_portal_depth": depth}


def paired_candidate(chain, socket, seed, contract, training_poses):
    level = chain.obstacle_level
    if level:
        candidate = evaluate_candidate(chain, seed)
        if candidate is None: return None
        extra = extra_goal_checks(chain, contract)
        if extra is None: return None
        candidate.update(extra)
    else:
        chain.reset(seed=seed)
        candidate = {"level": 0, "seed": int(seed), "route_type": "unobstructed"}
    if not physically_valid(chain, contract["geometry_tolerance"]): return None
    chain_init = chain.get_episode_init()
    if pose_matches(pose_vector(chain_init), training_poses, contract["initial_pose_tolerance"]):
        return None
    socket.reset(seed=seed)
    if not physically_valid(socket, contract["geometry_tolerance"]): return None
    socket_init = socket.get_episode_init()
    if not np.allclose(pose_vector(chain_init), pose_vector(socket_init),
                       atol=contract["reset_replay_tolerance"], rtol=0):
        raise ValueError("The two embodiments received different object/goal resets")
    candidate["initial_states"] = {"chaingripper": chain_init, "usocket": socket_init}
    candidate["training_pose_overlap"] = False
    return candidate


def assert_reset_matches(env, expected, contract):
    actual = env.get_episode_init()
    for key in ["agent_pos", "object_pose", "goal_pose"]:
        if not np.allclose(actual[key], expected[key], rtol=0,
                           atol=contract["reset_replay_tolerance"]):
            raise ValueError("Frozen reset mismatch: " + key)
    if not math.isclose(actual["agent_angle"], expected["agent_angle"], rel_tol=0,
                        abs_tol=contract["reset_replay_tolerance"]):
        raise ValueError("Frozen agent angle mismatch")
    if not physically_valid(env, contract["geometry_tolerance"]):
        raise ValueError("Invalid initial robot/object geometry")
    if env.obstacle_level and extra_goal_checks(env, contract) is None:
        raise ValueError("Initial goal fails frozen clearance/depth constraints")
