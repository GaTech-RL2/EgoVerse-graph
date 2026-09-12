# Source: aidan/abc-stationery-pi @ d5f72068. Imports relocated for graph isolation.
import copy
import math

import torch

from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment
from egomimic.utils.action_encoding import _reconstruct_R_from_cols, _ypr_to_matrix
from egomimic.eval.distribution_metrics import (
    dtw_distance,
    frechet_gaussian_over_time,
    reverse_kl_from_samples,
)
from egomimic.utils.pose_utils import bimanual_cartesian_layout


def _paired_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Plain elementwise MSE, stateless (no torchmetrics accumulator)."""
    return (pred.float() - gt.float()).pow(2).mean()


def _split_mse(pred_t: torch.Tensor, gt_t: torch.Tensor):
    """(translation MSE, rotation MSE) over a bimanual cartesian vector, so a
    translation problem reads apart from a rotation one. Handles all four
    native widths via ``bimanual_cartesian_layout``:
      - native model output: 18D (human) / 20D (robot) continuous 6D cols —
        clean (6D has no ±π wrap).
      - reverted cam-frame output: 12D (human) / 14D (robot) ypr — keeps the
        ±π caveat (the headline cam MSE is wrap-corrected; this split is a
        secondary diagnostic; see ``_rot_geodesic_error`` for the wrap- and
        gimbal-free rotation error).
    Returns (None, None) for an unknown width.
    """
    layout = bimanual_cartesian_layout(pred_t.shape[-1])
    if layout is None:
        return None, None
    xyz_idx = list(layout["xyz"])
    rot_idx = list(layout["rot"])
    xyz = _paired_mse(pred_t[..., xyz_idx], gt_t[..., xyz_idx])
    rot = _paired_mse(pred_t[..., rot_idx], gt_t[..., rot_idx])
    return xyz, rot


def _rot_geodesic_error(pred: torch.Tensor, gt: torch.Tensor):
    """Mean geodesic rotation error in radians over batch / time / both arms.

    Euler-free: per arm, builds proper rotation matrices (``_ypr_to_matrix``
    for the 12/14-dim ypr widths; Gram-Schmidt on the two 6D columns for the
    18/20-dim widths — the same reconstruction the model decode uses) and
    takes ``arccos((tr(R_pred^T R_gt) - 1) / 2)``. Unlike the ypr MSE this is
    immune to the ±π wrap AND to the yaw/roll degeneracy at pitch ≈ ±π/2,
    where two nearly identical orientations can differ by ~π in both yaw and
    roll. Returns None for an unknown width.
    """
    layout = bimanual_cartesian_layout(pred.shape[-1])
    if layout is None:
        return None
    rot = list(layout["rot"])
    per_arm = len(rot) // 2
    # float64: arccos near 1 loses ~sqrt(eps), which in float32 puts a
    # ~1e-4 rad floor under identical rotations; double makes it ~1e-8.
    pred = pred.double()
    gt = gt.double()
    errs = []
    for arm in (rot[:per_arm], rot[per_arm:]):
        p, g = pred[..., arm], gt[..., arm]
        if per_arm == 3:
            Rp, Rg = _ypr_to_matrix(p), _ypr_to_matrix(g)
        else:
            Rp = _reconstruct_R_from_cols(p[..., 0:3], p[..., 3:6])
            Rg = _reconstruct_R_from_cols(g[..., 0:3], g[..., 3:6])
        tr = (Rp.transpose(-1, -2) @ Rg).diagonal(dim1=-2, dim2=-1).sum(-1)
        errs.append(torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))
    return torch.stack(errs, dim=-1).mean().float()


def _wrap_aware_mse(pred: torch.Tensor, gt: torch.Tensor):
    """(wrapped, unwrapped) MSE over a bimanual cartesian YPR vector.

    Euler angles wrap at ±π: a prediction of +π-ε against a target of -π+ε is
    physically near-perfect but scores ~(2π)² per dim unwrapped, and a handful
    of wrap events dominates the batch average. Wrap the rotation-dim errors
    to (-π, π] before squaring; positions/grippers are untouched. Falls back
    to plain MSE when the trailing width has no known layout.
    """
    diff = (pred - gt).float()
    nowrap = diff.pow(2).mean()
    layout = bimanual_cartesian_layout(diff.shape[-1])
    # 6D-rotation layouts (18/20) have no angle dims — only wrap YPR widths.
    if layout is None or len(layout["rot"]) != 6:
        return nowrap, nowrap
    rot = list(layout["rot"])
    diff[..., rot] = torch.remainder(diff[..., rot] + math.pi, 2 * math.pi) - math.pi
    return diff.pow(2).mean(), nowrap


def cartesian_metrics(pred, target, *, distribution=True):
    """Native/camera pose scores independent of policy architecture."""
    pred, target = torch.as_tensor(pred).cpu(), torch.as_tensor(target).cpu()
    paired, nowrap = _wrap_aware_mse(pred, target)
    final, final_nowrap = _wrap_aware_mse(pred[:, -1], target[:, -1])
    if distribution:
        paired, final = nowrap, final_nowrap
    values = {
        "paired_mse_avg": paired,
        "final_mse_avg": final,
        "paired_mse_nowrap": nowrap,
        "final_mse_nowrap": final_nowrap,
    }
    xyz, rot = _split_mse(pred, target)
    if xyz is not None:
        values.update(
            xyz_paired_mse_avg=xyz,
            ypr_paired_mse_avg=rot,
            rot_geodesic_avg=_rot_geodesic_error(pred, target),
        )
    if distribution:
        fd = frechet_gaussian_over_time(pred, target)
        values.update(
            frechet_gauss_avg=fd.mean(),
            frechet_gauss_min=fd.min(),
            frechet_gauss_max=fd.max(),
            dtw_avg=dtw_distance(pred, target).mean(),
        )
    return values


def sample_metrics(samples, target):
    """Coverage and reverse KL over independent graph predictions (M, B, T, D)."""
    m = samples.shape[0]
    mse = (samples - target.unsqueeze(0)).square().flatten(start_dim=2).mean(dim=2)
    return {
        f"reverse_kl_M{m}": reverse_kl_from_samples(samples, target),
        f"bestof{m}_paired_mse": mse.min(dim=0).values.mean(),
        f"mean{m}_paired_mse": mse.mean(),
        f"worstof{m}_paired_mse": mse.max(dim=0).values.mean(),
        f"sample_diversity_M{m}": samples.std(dim=0).mean(),
    }
