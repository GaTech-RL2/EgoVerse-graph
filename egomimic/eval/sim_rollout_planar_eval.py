"""Closed-loop PushShapes rollout scoring for planar policies.

NON-PROTOCOL. This is a repo-local harness, not the canonical
``bf_eval_par.sbatch``. It follows the SUBSTANCE of the sim_v2 eval protocol
(2026-09-08) -- data-derived p99 budget, PEAK coverage, SR@0.80 and SR@0.95,
40 level-0 rollouts on seeds 0-39, sim_v2 physics -- so its numbers answer
"which tokenizer yields a better policy". They must never be pooled with, or
reported as, canonical protocol results.

It exists because the two halves of a rollout live in different repos:
EgoVerse-graph owns the ARC tokenizer, the detokenizer and the native decoder
but has no rollout evaluator and no simulator; EgoVerse owns eval_sim.py and
Tsimulation but has none of the ARC modules. Tsimulation is imported here off
PYTHONPATH, the same way the codec replay grid does it.

Plugged in as ``cfg.evaluator`` under ``mode=eval`` so trainHydra performs the
model construction, strict checkpoint load and norm-stats binding; hand-rolling
those is what the norm_mode=quantile trap punishes.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from egomimic.eval.eval import Eval

LABEL = "NON-PROTOCOL_REPO_LOCAL_ROLLOUT_NOT_COMPARABLE"


def _env_to_zarr_oriented(obs_env: dict) -> dict:
    """PushShapes obs -> dataset keys for a controlled-angle agent.

    U-Socket stores [agent_x, agent_y, agent_angle, obj_x, obj_y, obj_angle];
    the encoder slices [0,3). Mirrors EgoVerse's _env_to_zarr_pushshapes_oriented.
    """
    parts = [
        np.asarray(obs_env["agent_pos"], dtype=np.float32).reshape(-1),
        np.asarray(obs_env["agent_angle"], dtype=np.float32).reshape(-1),
        np.asarray(obs_env["object_pose"], dtype=np.float32).reshape(-1),
    ]
    # reshape(-1) on each part, so a (1,2) agent_pos cannot silently make the
    # state 2-D and push an extra axis all the way into the encoder.
    state = np.concatenate(parts, axis=0).astype(np.float32)
    image = np.asarray(obs_env["image"], dtype=np.float32)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected HWC or CHW image, got {image.shape}")
    if image.shape[-1] in (1, 3, 4):  # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
    image = image / 255.0
    if state.ndim != 1:
        raise ValueError(f"state must be 1-D per frame, got {state.shape}")
    return {"state_agent_obj": state, "front_img_1": image.astype(np.float32)}


class SimRolloutPlanarEval(Eval):
    def __init__(
        self,
        dataset_dir: str,
        budget_path: str,
        embodiment_name: str = "pushshapes_sim_u_socket",
        action_key: str = "actions",
        native_decoder=None,
        n_episodes: int = 40,
        seed_base: int = 0,
        level: int = 0,
        replan_every: int = 8,
        chunk_start: int = 0,
        expected_sampler_steps: int | None = None,
        results_path: str | None = None,
    ):
        # 0 is a sentinel: execute the ENTIRE decoded chunk before replanning,
        # i.e. fully open loop within a chunk. Any positive value executes that
        # many decoded actions and then re-observes.
        if replan_every < 0:
            raise ValueError("replan_every must be non-negative (0 = full chunk)")
        if chunk_start < 0:
            raise ValueError("chunk_start must be non-negative")
        self.dataset_dir = Path(dataset_dir)
        self.budget_path = Path(budget_path)
        self.embodiment_name = str(embodiment_name)
        self.action_key = str(action_key)
        self.native_decoder = native_decoder
        self.n_episodes = int(n_episodes)
        self.seed_base = int(seed_base)
        self.level = int(level)
        self.replan_every = int(replan_every)
        self.chunk_start = int(chunk_start)
        self.expected_sampler_steps = expected_sampler_steps
        self.results_path = results_path
        self.normalizer = None
        self._done = False
        self._logged_shapes = False
        self._n_obs = 1
        # trainHydra's eval mode copies this straight onto cfg.trainer before
        # building the trainer. Every Eval implementation must supply it.
        # The rollouts run in on_validation_start, so one val batch is only
        # needed to make Lightning enter the validation loop at all.
        self.override_dict = {
            "limit_train_batches": 0,
            "limit_val_batches": 1,
            "check_val_every_n_epoch": 1,
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def bind_data_context(self, *, normalizer):
        self.normalizer = normalizer

    # ---------------------------------------------------------------- helpers
    def _budget(self) -> int:
        payload = json.loads(self.budget_path.read_text())
        if payload.get("statistic") != "p99":
            raise ValueError(
                f"protocol revision 3 requires statistic=p99, got "
                f"{payload.get('statistic')!r}"
            )
        row = payload["by_level"][str(self.level)]
        return int(row["budget"]), payload

    def _env_args(self) -> dict:
        """Take env_args from the corpus so the eval env matches training."""
        import zarr

        episodes = sorted(self.dataset_dir.glob("episode_*.zarr"))
        if not episodes:
            raise FileNotFoundError(f"no episodes under {self.dataset_dir}")
        store = zarr.open_group(str(episodes[0]), mode="r")
        return json.loads(dict(store.attrs)["task_description"])["env_args"]

    def _make_env(self, env_args: dict):
        from Tsimulation.pushshapes import get_env

        env = get_env("v2")(
            object_shape=env_args["object_shape"],
            pusher_shape=env_args["pusher_shape"],
            obstacle_level=self.level,
            image_size=env_args.get("image_size", 96),
            render_mode=None,
        )
        env._skip_obs_render = False  # the policy needs the image
        return env

    def _emb_id(self) -> int:
        from egomimic.rldb.embodiment.embodiment import get_embodiment_id

        return get_embodiment_id(self.embodiment_name)

    def _predict_chunk(self, obs: dict, emb_id: int, device) -> np.ndarray:
        """obs (unnormalized env frame) -> native action chunk (H, 3)."""
        raw = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obs.items()}
        normalized = self.normalizer.normalize(raw, emb_id)
        # FusedObsEncoder wants exactly (batch, n_obs, *per_frame). Reshape to
        # the per-frame shape rather than unsqueezing twice: normalize() is
        # keyed through an identity zarr_keys map and returns whatever it was
        # given, so a stray leading axis would otherwise reach the encoder as
        # (1,1,1,D) and surface as "output must have shape (1, feature_dim)".
        inner = {}
        for key, value in normalized.items():
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            per_frame = tuple(obs[key].shape)
            # FusedObsEncoder.forward is asymmetric: when n_obs_steps == 1 it
            # only checks the batch dim and does NOT collapse an obs axis, so
            # the batch must be (B, *per_frame) with no obs axis at all. Adding
            # one leaves it in place and the encoder returns (1, 1, 67) instead
            # of (1, 67). For n_obs > 1 it does reshape (B, T, ...) itself.
            shape = (1, *per_frame) if self._n_obs == 1 else (1, self._n_obs, *per_frame)
            inner[key] = tensor.reshape(*shape).to(device=device, dtype=torch.float32)
        if not self._logged_shapes:
            self._logged_shapes = True
            for key in sorted(inner):
                delta = float(
                    (normalized[key].float() - raw[key].float()).abs().max()
                )
                print(
                    f"[sim] obs {key}: env={tuple(obs[key].shape)} "
                    f"norm={tuple(normalized[key].shape)} batch={tuple(inner[key].shape)} "
                    f"norm_delta={delta:.6f}"
                )
            # normalize() resolves keys through zarr_keys; if that map is not
            # the identity the proprio silently passes through unnormalized and
            # the policy sees inputs it was never trained on.
            if float(
                (normalized["state_agent_obj"].float()
                 - raw["state_agent_obj"].float()).abs().max()
            ) == 0.0:
                raise ValueError(
                    "state_agent_obj was NOT normalized (delta 0). normalize() "
                    "resolves through zarr_keys; check the key convention."
                )
        batch = {self.embodiment_name: inner}
        batch[self.embodiment_name]["embodiment"] = torch.tensor([emb_id], device=device)
        with torch.no_grad():
            out = self.model.forward_eval(batch)
        token = out[self.embodiment_name]["pred_action"].detach()
        token = self.normalizer.unnormalize({self.action_key: token}, emb_id)[
            self.action_key
        ]
        native = self.native_decoder.decode(token)
        native = np.asarray(native.squeeze(0).cpu(), dtype=np.float32)
        return native

    # ------------------------------------------------------------- Eval hooks
    def on_validation_start(self):
        if self._done:
            return
        self._done = True
        t_start = time.time()
        # Read the observation horizon off the built graph rather than assuming
        # it, since it decides the batch layout above.
        self._n_obs = 1
        for stage in getattr(self.model, "stages", None) or []:
            if hasattr(stage, "n_obs_steps"):
                self._n_obs = int(stage.n_obs_steps)
                break
        if self._n_obs != 1:
            raise NotImplementedError(
                f"n_obs_steps={self._n_obs} needs an observation history; this "
                "harness only builds single-frame observations."
            )
        budget, budget_payload = self._budget()
        env_args = self._env_args()
        emb_id = self._emb_id()
        device = self.trainer.lightning_module.device
        env = self._make_env(env_args)
        print(f"[sim] setup_s={time.time() - t_start:.1f} device={device}")

        print(f"[sim] {LABEL}")
        print(
            f"[sim] budget={budget} level={self.level} "
            f"statistic={budget_payload.get('statistic')} "
            f"multiplier={budget_payload.get('multiplier')} "
            f"dataset={budget_payload.get('dataset')} "
            f"content_sha256={budget_payload.get('content_sha256')}"
        )
        print(
            f"[sim] replan_every={self.replan_every}"
            f"{' (0=full chunk, open loop)' if self.replan_every == 0 else ''} "
            f"chunk_start={self.chunk_start} "
            f"sampler_steps={self.expected_sampler_steps} "
            f"episodes={self.n_episodes} seed_base={self.seed_base}"
        )
        print(f"[sim] env_args={json.dumps(env_args, sort_keys=True)}")

        peaks: list[float] = []
        for ep in range(self.n_episodes):
            seed = self.seed_base + ep
            env.reset(seed=seed)
            peak = 0.0
            chunk: np.ndarray | None = None
            cursor = 0
            aborted = False
            t_ep = time.time()
            calls = 0
            policy_s = 0.0
            steps = 0
            for t in range(budget):
                if chunk is None or cursor >= len(chunk):
                    obs = _env_to_zarr_oriented(env._get_obs())
                    t_call = time.time()
                    native = self._predict_chunk(obs, emb_id, device)
                    policy_s += time.time() - t_call
                    calls += 1
                    span = (
                        self.replan_every
                        if self.replan_every > 0
                        else len(native) - self.chunk_start
                    )
                    chunk = native[self.chunk_start : self.chunk_start + span]
                    if len(chunk) == 0:
                        raise ValueError(
                            f"empty execution chunk: decoded {native.shape} with "
                            f"start={self.chunk_start} replan={self.replan_every}"
                        )
                    cursor = 0
                action = np.asarray(chunk[cursor], dtype=np.float32).reshape(-1)
                cursor += 1
                if not np.all(np.isfinite(action)):
                    print(
                        f"[sim] WARNING: non-finite action t={t} emb{emb_id} "
                        f"ep{ep} (raw={action.tolist()}); abort 0-cov."
                    )
                    peak = 0.0
                    aborted = True
                    break
                _, _, term, trunc, info = env.step(action)
                steps += 1
                peak = max(peak, float(info.get("coverage", 0.0)))
                if term or trunc:
                    break
            peaks.append(float(peak))
            ep_s = time.time() - t_ep
            print(
                f"[sim] ep{ep} seed={seed} peak={peak:.4f} aborted={aborted} "
                f"steps={steps} calls={calls} ep_s={ep_s:.1f} "
                f"policy_s={policy_s:.1f} "
                f"s_per_call={(policy_s / calls if calls else float('nan')):.3f}"
            )
        env.close()

        arr = np.asarray(peaks, dtype=np.float64)
        summary = {
            "label": LABEL,
            "embodiment": self.embodiment_name,
            "level": self.level,
            "episodes": int(arr.size),
            "budget": budget,
            "budget_provenance": budget_payload,
            "replan_every": self.replan_every,
            "replan_mode": "full_chunk_open_loop" if self.replan_every == 0 else "fixed",
            "chunk_start": self.chunk_start,
            "sampler_steps": self.expected_sampler_steps,
            "seed_base": self.seed_base,
            "peak_coverage_mean": float(arr.mean()),
            "peak_coverage_median": float(np.median(arr)),
            "SR@0.80": float((arr >= 0.80).mean()),
            "SR@0.95": float((arr >= 0.95).mean()),
            "ep_coverages": [round(float(v), 4) for v in arr],
        }
        print(
            f"[sim] emb{emb_id} ep_coverages: "
            + " ".join(f"{v:.4f}" for v in arr)
        )
        print("[sim] SUMMARY " + json.dumps(summary, sort_keys=True))
        if self.results_path:
            Path(self.results_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.results_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        self._summary = summary

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        return {}

    def on_validation_end(self):
        return {}
