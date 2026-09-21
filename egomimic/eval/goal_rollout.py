"""Goal-conditioned environment evaluation through PipelineAlgo outputs."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch


def evaluate_goals(pipeline, env, output_dir, episodes=50, seed_start=2000000,
                   task_ids=None, video_episodes=0, video_stride=3):
    """Keep every scored reset/action trace; render only fixed preview seeds.

    The native environment enforces termination and its published time limit.
    The entire decoded chunk is executed, as in the DQC reference protocol.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    task_ids = list(task_ids or range(1, len(env.unwrapped.task_infos) + 1))
    records = []
    prior_mode = pipeline.nets.training
    pipeline.nets.eval()
    try:
        for task_id in task_ids:
            for episode in range(episodes):
                seed = int(seed_start + 10000 * task_id + episode)
                np.random.seed(seed)
                observation, info = env.reset(seed=seed, options={"task_id": task_id, "render_goal": False})
                goals = info["goal"]
                base = env.unwrapped
                physics = base._data if hasattr(base, "_data") else base.data
                initial_qpos, initial_qvel = physics.qpos.copy(), physics.qvel.copy()
                reset_hash = hashlib.sha256(initial_qpos.tobytes() + initial_qvel.tobytes()
                    + np.asarray(goals).tobytes()).hexdigest()
                generator = torch.Generator(device=pipeline.device).manual_seed(seed)
                done, success, total_reward = False, False, 0.
                actions, lengths, frames = [], [], []
                with torch.inference_mode():
                    while not done:
                        batch = {"ogbench": {
                            "observations": torch.as_tensor(observation[None], dtype=torch.float32, device=pipeline.device),
                            "goals": torch.as_tensor(goals[None], dtype=torch.float32, device=pipeline.device),
                            "generator": generator,
                        }}
                        output = pipeline.forward_eval(batch)["ogbench"]
                        length = int(output["action_lengths"][0])
                        chunk = output["pred_action"][0].cpu().numpy()
                        if not (1 <= length <= len(chunk)) or not np.isfinite(chunk).all():
                            raise ValueError("invalid graph action chunk")
                        lengths.append(length)
                        for action in chunk[:length]:
                            observation, reward, terminated, truncated, info = env.step(action)
                            done = bool(terminated or truncated)
                            success = bool(info.get("success", False))
                            total_reward += float(reward)
                            actions.append(action.copy())
                            if episode < video_episodes and len(actions) % video_stride == 0:
                                frames.append(env.render())
                            if done:
                                break
                row = {"task_id": task_id, "episode": episode, "seed": seed,
                       "success": int(success), "return": total_reward, "reset_sha256": reset_hash,
                       "native_steps": len(actions), "replans": len(lengths),
                       "mean_chunk_steps": float(np.mean(lengths))}
                records.append(row)
                stem = f"task{task_id}-seed{seed}"
                np.savez_compressed(directory / (stem + ".npz"), actions=actions,
                                    chunk_lengths=lengths, seed=seed, task_id=task_id,
                                    initial_qpos=initial_qpos, initial_qvel=initial_qvel, goal=goals)
                if frames:
                    import imageio.v2 as imageio
                    imageio.mimsave(directory / (stem + ".mp4"), frames,
                                    fps=env.metadata.get("render_fps", 30) / video_stride)
                with (directory / "rollouts.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
        summary = {"success": float(np.mean([r["success"] for r in records])),
                   "episodes": len(records), "tasks": {
                       str(t): float(np.mean([r["success"] for r in records if r["task_id"] == t]))
                       for t in task_ids}}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        pipeline.nets.train(prior_mode)
