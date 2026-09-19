"""Group-balanced frame sampling for a MultiDataset that pools several sources.

The default loader shuffles frames uniformly, so a pooled dataset is seen in
proportion to its frame counts: with 2090 ABC + 198 RL2 stationery episodes the
in-domain RL2 station data is under 10 % of what the model sees. This sampler
fixes the *exposure* instead: each named group receives a chosen share of the
draws (Aidan, 2026-09-19: RL2 50 % / ABC 50 %), and frames are uniform within a
group, exactly as the shuffle they replace.

Group membership is an explicit episode-name list per group (one name per
line); every episode not listed falls into ``default_group``. Draws are with
replacement, as in ``e1_anchor_sampler``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


def group_balance_weights(index_map, group_episode_files, group_fractions, default_group):
    fractions = {str(k): float(v) for k, v in dict(group_fractions).items()}
    if default_group not in fractions:
        raise ValueError(f"default_group {default_group!r} has no entry in group_fractions")
    if any(v <= 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"group_fractions must be positive and sum to 1, got {fractions}")
    member = {}
    for group, path in dict(group_episode_files).items():
        if group not in fractions:
            raise ValueError(f"group {group!r} has an episode file but no fraction")
        names = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        if not names:
            raise ValueError(f"episode file for group {group!r} is empty: {path}")
        for n in names:
            if member.setdefault(n, group) != group:
                raise ValueError(f"episode {n} is listed in two groups")
    groups = sorted(fractions)
    gid = {g: i for i, g in enumerate(groups)}
    frame_group = np.fromiter((gid[member.get(str(name), default_group)] for name, _ in index_map),
                              dtype=np.int64, count=len(index_map))
    counts = np.bincount(frame_group, minlength=len(groups))
    episodes = {g: len({str(n) for (n, _), k in zip(index_map, frame_group) if k == gid[g]}) for g in groups}
    for g in groups:
        if counts[gid[g]] == 0:
            raise ValueError(f"group {g!r} has no frames in this dataset split; refusing to train unbalanced")
    per_frame = np.array([fractions[g] / counts[gid[g]] for g in groups])
    w = per_frame[frame_group]
    ess = float(w.sum() ** 2 / np.sum(w**2))
    summary = " ".join(
        f"{g}:episodes={episodes[g]},frames={counts[gid[g]]},natural={counts[gid[g]] / len(w):.3f},target={fractions[g]:.3f}"
        for g in groups)
    logger.info("group balance sampler: %s ESS %.0f (%.0f%%)", summary, ess, 100 * ess / len(w))
    print(f"GROUP_BALANCE_SAMPLER {summary} ess_frac={ess / len(w):.2f}", flush=True)
    return w


def build_group_balance_sampler(dataset, group_episode_files, group_fractions, default_group,
                                num_samples: int | None = None, seed: int = 0) -> WeightedRandomSampler:
    w = group_balance_weights(dataset.index_map, group_episode_files, group_fractions, str(default_group))
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=int(num_samples or len(w)),
                                 replacement=True, generator=gen)
