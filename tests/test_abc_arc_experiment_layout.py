from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "egomimic/hydra_configs/experiment/abc_arc"

EXPECTED_YAMLS = {
    "robot_bc": {
        "abc_visual_hpt300_base.yaml",
        "abc_multitask4_hpt300_baseline_visual_openloop.yaml",
        "abc_multitask4_hpt300_hybrid_visual_openloop.yaml",
        "abc_towels_hpt180_baseline_visual_openloop.yaml",
        "abc_towels_hpt180_hybrid_visual_openloop.yaml",
        "stationery_rl2_hpt300_visual_baseline_openloop.yaml",
        "stationery_rl2_hpt300_visual_hybrid_openloop.yaml",
    },
    "human_bc": {
        "mecka_fold_clothes_40h_human_visual_baseline_openloop.yaml",
        "mecka_fold_clothes_40h_human_visual_hybrid_openloop.yaml",
    },
    "cotrain": set(),
}


def test_abc_arc_experiments_are_grouped_by_training_population():
    assert not list(EXPERIMENT_ROOT.glob("*.yaml"))
    assert {path.name for path in EXPERIMENT_ROOT.iterdir() if path.is_dir()} == set(
        EXPECTED_YAMLS
    )

    for folder, expected in EXPECTED_YAMLS.items():
        actual = {path.name for path in (EXPERIMENT_ROOT / folder).glob("*.yaml")}
        assert actual == expected


def test_retained_experiments_are_visual_only_and_qwen_free():
    for path in EXPERIMENT_ROOT.rglob("*.yaml"):
        text = path.read_text().lower()
        assert "qwen" not in text, path


def test_every_arc_recipe_has_a_matching_baseline_recipe():
    expected_pairs = {
        "robot_bc/abc_multitask4_hpt300_hybrid_visual_openloop.yaml":
            "robot_bc/abc_multitask4_hpt300_baseline_visual_openloop.yaml",
        "robot_bc/abc_towels_hpt180_hybrid_visual_openloop.yaml":
            "robot_bc/abc_towels_hpt180_baseline_visual_openloop.yaml",
        "robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop.yaml":
            "robot_bc/stationery_rl2_hpt300_visual_baseline_openloop.yaml",
        "human_bc/mecka_fold_clothes_40h_human_visual_hybrid_openloop.yaml":
            "human_bc/mecka_fold_clothes_40h_human_visual_baseline_openloop.yaml",
    }

    for arc_path, baseline_path in expected_pairs.items():
        assert (EXPERIMENT_ROOT / arc_path).is_file()
        assert (EXPERIMENT_ROOT / baseline_path).is_file()
