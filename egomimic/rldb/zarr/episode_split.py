"""Shared native episode split; no dataset traversal or runtime imports."""
import random

def split_dataset_names(dataset_names, valid_ratio=0.2, seed=42):
    """
    Split a list of dataset names into train/valid sets.
    Args:
        dataset_names (Iterable[str])
        valid_ratio (float): fraction of datasets to put in valid.
        seed (int): for deterministic shuffling.


    Returns:
        train_set (set[str]), valid_set (set[str])
    """
    names = sorted(dataset_names)
    if not names:
        return set(), set()

    rng = random.Random(seed)
    rng.shuffle(names)

    if not (0.0 <= valid_ratio <= 1.0):
        raise ValueError(f"valid_ratio must be in [0,1], got {valid_ratio}")

    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)

    valid = set(names[:n_valid])
    train = set(names[n_valid:])
    return train, valid



def complete_window_count(total_frames, raw_horizon, sample_views=1):
    if any(type(x) is not int or x <= 0 for x in (total_frames, raw_horizon, sample_views)):
        raise ValueError("frame/horizon/view counts must be positive integers")
    return max(0, total_frames - raw_horizon + 1) * sample_views
