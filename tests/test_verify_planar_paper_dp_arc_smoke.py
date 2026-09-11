from pathlib import Path

from scripts.train.verify_planar_paper_dp_arc_smoke import (
    ROWS,
    _latest_checkpoint,
    _metric_value,
)


def test_verifier_approves_exactly_the_matched_arc_rows():
    assert set(ROWS) == {
        "pusht/planar_v2_usocket_arc_paper_uniform_D40_M16_R24deg",
        "pusht/planar_v2_usocket_arc_paper_curvature_D40_M16_R24deg",
    }
    assert {sampling for _, sampling in ROWS.values()} == {"uniform", "curvature"}


def test_latest_checkpoint_uses_global_step_not_filename(monkeypatch, tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    older = checkpoint_dir / "z.ckpt"
    newer = checkpoint_dir / "a.ckpt"
    older.touch()
    newer.touch()
    payloads = {older: {"global_step": 1}, newer: {"global_step": 2}}
    monkeypatch.setattr(
        "scripts.train.verify_planar_paper_dp_arc_smoke.torch.load",
        lambda path, **_: payloads[path],
    )
    path, payload = _latest_checkpoint(tmp_path)
    assert path == newer
    assert payload["global_step"] == 2


def test_metric_value_accepts_lightning_step_suffix():
    assert _metric_value({"Train/MSE_step": 0.5}, "Train/MSE") == 0.5
    assert _metric_value(
        {"Train/MSE": 0.4, "Train/MSE_step": 0.5}, "Train/MSE"
    ) == 0.4
