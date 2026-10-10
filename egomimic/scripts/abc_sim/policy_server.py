"""Serve a graph checkpoint (time / arcdur / arcvel E1 policy) over HTTP for the abc_sim client.

abc_sim's Python (amazon-far/abc: py3.12, MuJoCo 3.8, torch 2.11) cannot share the training
venv, so the policy runs here and the simulator talks to it over plain HTTP + JSON (stdlib on
both ends; ``sim_client.py`` is the other half).

    python -m egomimic.scripts.abc_sim.policy_server --run <run dir> [--ckpt <.ckpt>] \
        --model <abc>/abc_sim/models/yam_bimanual_empty.xml --port 8765

POST /infer  {"state": [14], "prompt": str, "images": {"top": {"shape": [H, W, 3], "b64": raw RGB}, ...}}
         ->  {"actions": [[14] x 45], "ik_failures": int}
GET  /health ->  {"ckpt", "cameras", "lag_frames"}

The model is the station's :class:`~egomimic.robot.graph_policy.GraphRobotPolicy` with the
run's own exported ``checkpoints/inference-config.yaml`` (so arc checkpoints decode through
the same ``BimanualArcDecoder`` the station uses). The sim records joints, the policy speaks
Cartesian: the server does FK and IK through abc_sim's own bimanual MJCF at
``<side>_grasp_site`` in the ``left_arm`` frame -- exactly how ``convert_sim_to_zarr.py``
produced the training poses -- and returns absolute joint + gripper commands, one per 30 Hz
step. An unreachable IK target holds that arm's previous command.

``--selftest <episode dir>`` runs one inference on an extracted sim episode's first frame
(no HTTP) and checks shapes, finiteness and the FK/IK round trip.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from egomimic.robot.graph_policy import CartesianGraphAdapter
from egomimic.robot.interface import ARM_OFFSET, pose_matrix, pose_vector
from egomimic.robot.yam.kinematics import MujocoArmKinematics
from egomimic.scripts.abc_sim.convert_sim_to_zarr import BASE_BODY, CAMERAS, SITE

log = logging.getLogger("abc_sim.policy_server")

GRAPH_KEYS = {
    "top": "observations.images.front_img_1",
    "left": "observations.images.left_wrist_img",
    "right": "observations.images.right_wrist_img",
}
HORIZON = 45  # commands per served chunk: 1.5 s at 30 Hz, what sim_client expects
YAM_BIMANUAL = 7


def decode_image(spec: dict) -> np.ndarray:
    """{"shape": [H, W, 3], "b64": raw uint8 RGB} -> (H, W, 3) uint8 BGR (the adapter's input)."""
    rgb = np.frombuffer(base64.b64decode(spec["b64"]), dtype=np.uint8).reshape(spec["shape"])
    return np.ascontiguousarray(rgb[..., ::-1])


class SimAdapter(CartesianGraphAdapter):
    """The sim is a benchmark: clip a gripper outside [0, 1] instead of rejecting the plan."""

    def actions(self, native, obs):
        if self.decoder is not None:
            native = self.decoder(native)
        native = np.array(native.detach().cpu().numpy() if torch.is_tensor(native) else native, dtype=float)
        native[..., [6, 13]] = np.clip(native[..., [6, 13]], 0.0, 1.0)
        decoder, self.decoder = self.decoder, None  # already applied above
        try:
            return super().actions(native, obs)
        finally:
            self.decoder = decoder


class SimKinematics:
    """FK/IK for both arms of abc_sim's bimanual YAM, poses in the ``left_arm`` body frame."""

    def __init__(self, model_xml: str | Path):
        self.arms = {
            side: MujocoArmKinematics(
                model_xml, [f"{side}_joint{i}" for i in range(1, 7)], site_name=f"{side}_{SITE}"
            )
            for side in ARM_OFFSET
        }
        left = self.arms["left"]
        left.mj.mj_forward(left.model, left.data)
        base = left.model.body(BASE_BODY).id
        self.T_world_model = np.eye(4)
        self.T_world_model[:3, :3] = left.data.xmat[base].reshape(3, 3)
        self.T_world_model[:3, 3] = left.data.xpos[base]
        self.T_model_world = np.linalg.inv(self.T_world_model)

    def ee_poses(self, joints: np.ndarray) -> np.ndarray:
        """(14,) sim joints -> (14,) [xyz, ZYX euler, gripper] per arm in the model frame."""
        out = np.zeros(14)
        for side, off in ARM_OFFSET.items():
            pose = self.T_model_world @ self.arms[side].fk(joints[off : off + 6])
            out[off : off + 7] = np.r_[pose_vector(pose), joints[off + 6]]
        return out

    def joints(self, actions: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, int]:
        """(H, 14) model-frame Cartesian rows -> (H, 14) sim joints, warm-started row to row."""
        out, failures, prev = np.empty_like(actions), 0, np.asarray(seed, dtype=float).copy()
        for t, row in enumerate(actions):
            for side, off in ARM_OFFSET.items():
                target = self.T_world_model @ pose_matrix(row[off : off + 6])
                try:
                    prev[off : off + 6] = self.arms[side].ik(target, prev[off : off + 6])
                except ValueError:
                    failures += 1  # hold the previous command for this arm
                prev[off + 6] = float(np.clip(row[off + 6], 0.0, 1.0))
            out[t] = prev
        return out, failures


class GraphJointPolicy:
    """GraphRobotPolicy from a run dir, plus the sim's joint <-> Cartesian conversion."""

    def __init__(self, run: str | Path, ckpt: str | Path | None, model_xml: str, device: str = "cuda:0"):
        from egomimic.robot.graph_policy import load_graph_policy

        run = Path(run)
        self.ckpt_path = str(Path(ckpt) if ckpt else run / "checkpoints" / "last.ckpt")
        training = OmegaConf.load(run / ".hydra" / "config.yaml")
        # ${hydra:runtime.output_dir} only resolves inside a Hydra app; pin it to the run.
        training.paths.output_dir = str(run)
        self._training_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        OmegaConf.save(training, self._training_file.name)
        self.kin = SimKinematics(model_xml)
        self.policy = load_graph_policy(
            {
                "training_config": self._training_file.name,
                "checkpoint": self.ckpt_path,
                "normalizer_path": str(run / "checkpoints" / "data-context.json"),
                "device": device,
                "use_ema": False,
                "adapter": {
                    "_target_": "egomimic.robot.graph_policy.CartesianGraphAdapter",
                    "embodiment_id": YAM_BIMANUAL,
                    # ee_poses are handed over already in the model (left_arm) frame.
                    "base_T_model": {side: np.eye(4).tolist() for side in ARM_OFFSET},
                    "camera_keys": dict(GRAPH_KEYS),
                    "rotation_mode": "euler",
                    "action_frame": "eef_frame",
                    "image_hw": list(self._image_hw(training)),
                    "proprio_key": "observations.state.ee_pose",
                    "action_key": "actions_cartesian",
                    "prompt": "",
                    "decoder": None,
                },
            }
        )
        # The exported inference profile injects its own adapter _target_; swap the class after
        # loading so the gripper is clipped, not rejected (SimAdapter adds no state).
        self.policy.adapter.__class__ = SimAdapter
        self.cameras = list(CAMERAS)
        self.lag_frames = 0

    @staticmethod
    def _image_hw(training) -> tuple[int, int]:
        hw = OmegaConf.select(training, "data.train_datasets.yam_bimanual.resolver.image_hw")
        return (int(hw[0]), int(hw[1])) if hw else (168, 224)

    def infer(self, state, images: dict[str, np.ndarray], prompt: str = "") -> tuple[np.ndarray, int]:
        """(45, 14) absolute joint + gripper commands, one per 30 Hz step."""
        state = np.asarray(state, dtype=float).reshape(14)
        obs = {"joint_positions": state, "ee_poses": self.kin.ee_poses(state), **images}
        self.policy.reset()  # one chunk per request; the client owns the episode
        actions = np.asarray(self.policy.predict(obs), dtype=float)[:HORIZON]
        return self.kin.joints(actions, state)


def make_handler(policy: GraphJointPolicy):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": self.path})
            self._send(200, {"ckpt": policy.ckpt_path, "cameras": policy.cameras, "lag_frames": policy.lag_frames})

        def do_POST(self):
            if self.path != "/infer":
                return self._send(404, {"error": self.path})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                images = {k: decode_image(v) for k, v in req["images"].items() if k in GRAPH_KEYS}
                actions, failures = policy.infer(req["state"], images, req.get("prompt", ""))
                if failures:
                    log.warning("%d IK targets unreachable in this chunk", failures)
                self._send(200, {"actions": actions.tolist(), "ik_failures": failures})
            except Exception as e:  # the client must see the reason, not a hang
                log.exception("request failed")
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return Handler


def selftest(policy: GraphJointPolicy, episode_dir: Path) -> None:
    """One inference on an extracted sim episode's first frame; checks the FK/IK round trip."""
    from egomimic.scripts.abc_sim.convert_sim_to_zarr import STATE_DIM, decode_cameras

    table = np.fromfile(episode_dir / "states_actions.bin", dtype=np.float64).reshape(-1, 2 * STATE_DIM)
    frames = decode_cameras(episode_dir)
    state = table[0, :STATE_DIM].astype(float)
    images = {c: np.ascontiguousarray(frames[c][0][..., ::-1]) for c in CAMERAS}  # RGB -> BGR
    # FK/IK round trip on the observed pose itself: must come back within IK tolerance.
    ee = policy.kin.ee_poses(state)
    back, failures = policy.kin.joints(ee[None], state + 0.05)
    assert failures == 0, "IK could not reproduce the observed pose"
    for side, off in ARM_OFFSET.items():
        err = np.linalg.norm(policy.kin.ee_poses(back[0])[off : off + 3] - ee[off : off + 3])
        assert err < 2e-3, f"{side} FK/IK round trip {err * 1e3:.1f} mm"
    actions, failures = policy.infer(state, images)
    assert actions.shape == (HORIZON, 14), actions.shape
    assert np.isfinite(actions).all()
    assert (0 <= actions[:, [6, 13]]).all() and (actions[:, [6, 13]] <= 1).all()
    step = np.abs(np.diff(np.r_[state[None], actions], axis=0))[:, [*range(6), *range(7, 13)]].max()
    print(
        f"selftest ok: {HORIZON}x14 joints, ik_failures={failures}, "
        f"first-row |dq| max {np.abs(actions[0] - state).max():.3f} rad, "
        f"max per-step |dq| {step:.3f} rad, grippers {actions[:, 6].min():.2f}-{actions[:, 6].max():.2f} / "
        f"{actions[:, 13].min():.2f}-{actions[:, 13].max():.2f}"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, help="training run dir (.hydra/config.yaml, checkpoints/)")
    p.add_argument("--ckpt", default=None, help="default: <run>/checkpoints/last.ckpt")
    p.add_argument("--model", required=True, help="abc_sim/models/yam_bimanual_empty.xml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--selftest", type=Path, default=None, help="extracted sim episode dir; infer once and exit")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    policy = GraphJointPolicy(a.run, a.ckpt, a.model, a.device)
    if a.selftest is not None:
        selftest(policy, a.selftest)
        return 0
    log.info("serving %s on port %d", policy.ckpt_path, a.port)
    HTTPServer(("127.0.0.1", a.port), make_handler(policy)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
