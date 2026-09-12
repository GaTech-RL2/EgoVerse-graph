"""Serve graph robot predictions to rl2_yam's PolicyClient over its existing RPC protocol."""

import argparse
from concurrent import futures
from pathlib import Path
import threading
import time

import cv2
import grpc
import numpy as np
from omegaconf import OmegaConf

from egomimic.robot.yam import policy_inference_pb2 as pb
from egomimic.robot.yam.policy import load_policy

CAMERAS = ("top", "left_wrist", "right_wrist")
STATE_NAMES = (
    *(f"left_arm_joint_{i}" for i in range(1, 7)),
    "left_gripper",
    *(f"right_arm_joint_{i}" for i in range(1, 7)),
    "right_gripper",
)
ACTION_NAMES = (
    *(f"left_arm_action_{i}" for i in range(1, 7)),
    "left_gripper_action",
    *(f"right_arm_action_{i}" for i in range(1, 7)),
    "right_gripper_action",
)


def decode_observations(request):
    if (
        not request.session_id
        or not request.request_id
        or len(request.observations) != 2
    ):
        raise ValueError("Expected a session, positive request ID and two observations")
    samples = []
    for observation in request.observations:
        if observation.depths:
            raise ValueError("This Cartesian graph interface advertises RGB only")
        sample = {}
        for side in ("left", "right"):
            q = np.asarray(
                getattr(observation, side + "_joint_positions"), dtype=np.float32
            )
            if q.shape != (7,) or not np.isfinite(q).all() or not 0 <= q[6] <= 1:
                raise ValueError(
                    "Each arm requires six finite joint positions and a gripper in [0,1]"
                )
            sample[side] = q
        names = [image.name for image in observation.images]
        if len(names) != 3 or set(names) != set(CAMERAS):
            raise ValueError(
                "Expected exactly top, left_wrist and right_wrist RGB cameras"
            )
        images = {}
        for image in observation.images:
            if (
                image.encoding != "jpeg"
                or not image.data
                or len(image.data) > 8 * 1024 * 1024
                or not 0 < image.width <= 4096
                or not 0 < image.height <= 4096
            ):
                raise ValueError("Invalid JPEG camera payload")
            rgb = cv2.imdecode(
                np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if rgb is None or rgb.shape[:2] != (image.height, image.width):
                raise ValueError("Decoded image dimensions differ from the payload")
            images[image.name] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        sample.update(
            images=images,
            sequence=int(observation.sequence),
            capture_monotonic_ns=int(observation.capture_monotonic_ns),
        )
        samples.append(sample)
    if (
        samples[1]["sequence"] <= samples[0]["sequence"]
        or samples[1]["capture_monotonic_ns"] <= samples[0]["capture_monotonic_ns"]
    ):
        raise ValueError("Observation history must be in increasing capture order")
    return samples


class PolicyService:
    def __init__(self, policy, model_id, task):
        self.policy = policy
        self.lock = threading.Lock()
        self.info = pb.PolicyInfo(
            protocol_version=1,
            model_id=model_id,
            task=task,
            fps=30,
            observation_history=2,
            action_chunk_size=24,
            state_names=STATE_NAMES,
            action_names=ACTION_NAMES,
            camera_names=CAMERAS,
        )

    def GetPolicyInfo(self, request, context):
        return self.info

    def Predict(self, request, context):
        try:
            samples = decode_observations(request)
        except (ValueError, cv2.error) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        started = time.monotonic()
        with self.lock:
            if not context.is_active():
                context.abort(
                    grpc.StatusCode.CANCELLED, "Request expired before inference"
                )
            try:
                actions = np.asarray(self.policy.predict(samples), dtype=np.float32)
                if actions.shape != (24, 14) or not np.isfinite(actions).all():
                    raise ValueError(
                        "Policy must return 24 finite bimanual joint targets"
                    )
            except (ValueError, RuntimeError, KeyError) as exc:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return pb.ActionChunk(
            session_id=request.session_id,
            request_id=request.request_id,
            source_observation_sequence=samples[-1]["sequence"],
            start_action_step=request.next_action_step,
            rows=24,
            cols=14,
            actions=actions.reshape(-1),
            inference_ms=(time.monotonic() - started) * 1000,
        )


def create_server(policy, *, model_id, task, address="127.0.0.1:18080"):
    service = PolicyService(policy, model_id, task)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=[
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ],
    )
    handlers = {
        "GetPolicyInfo": grpc.unary_unary_rpc_method_handler(
            service.GetPolicyInfo,
            request_deserializer=pb.Empty.FromString,
            response_serializer=pb.PolicyInfo.SerializeToString,
        ),
        "Predict": grpc.unary_unary_rpc_method_handler(
            service.Predict,
            request_deserializer=pb.PredictRequest.FromString,
            response_serializer=pb.ActionChunk.SerializeToString,
        ),
    }
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "lerobot.inference.v1.PolicyInference", handlers
            ),
        )
    )
    port = server.add_insecure_port(address)
    if not port:
        raise RuntimeError(f"Could not bind policy server: {address}")
    return server, port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    policy = load_policy(config.policy)
    server, port = create_server(
        policy, model_id=config.model_id, task=config.task, address=config.address
    )
    server.start()
    print(f"Yam graph policy listening on {config.address} (port {port})", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2).wait()


if __name__ == "__main__":
    main()
