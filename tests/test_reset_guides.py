"""Reset templates are static view data; these checks never open hardware."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_reset_guides", ROOT / "egomimic/scripts/build_reset_guides.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_checked_in_catalog_matches_yaml():
    for index in BUILDER.INDEXES:
        assert BUILDER.render_catalog(ROOT, index) == (ROOT / index).read_text()
    assert BUILDER.render_renderer(ROOT) == (ROOT / BUILDER.ROLLOUT_APP).read_text()


def test_renderer_generation_rejects_ambiguous_markers(tmp_path):
    marker = "// BEGIN SHARED RESET GUIDE RENDERER\n// END SHARED RESET GUIDE RENDERER"
    for path in (BUILDER.TELEOP_APP, BUILDER.ROLLOUT_APP):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(marker)
    (tmp_path / BUILDER.TELEOP_APP).write_text(marker * 2)
    with pytest.raises(ValueError, match="exactly one"):
        BUILDER.render_renderer(tmp_path)


def test_confirmed_regions_keep_shared_edges_and_perspective():
    data = yaml.safe_load((ROOT / BUILDER.CATALOG).read_text())
    template = next(
        t for t in data["templates"] if t["id"] == "rl2-yam-earbuds-20261010"
    )
    assert template["camera"] == "front_img_1"
    assert template["image_size"] == [640, 480]
    left, case, right = [r["points"] for r in template["regions"]]
    assert left[1] == case[0] and left[2] == case[3]
    assert case[1] == right[0] and case[2] == right[3]
    assert left[0][0] != left[3][0]  # not camera-aligned pixel squares


def test_builder_escapes_html_and_rejects_ambiguous_marker(tmp_path):
    catalog_path = tmp_path / BUILDER.CATALOG
    index_path = tmp_path / BUILDER.INDEX
    catalog_path.parent.mkdir(parents=True)
    index_path.parent.mkdir(parents=True)
    catalog_path.write_text(yaml.safe_dump({"label": "</script><img src=x> &"}))
    marker = '<script id="reset-guide-catalog" type="application/json">{}</script>'
    index_path.write_text(marker)
    rendered = BUILDER.render_catalog(tmp_path)
    assert rendered.count("</script>") == 1
    payload = rendered.split(">", 1)[1].rsplit("</script>", 1)[0]
    assert json.loads(payload)["label"] == "</script><img src=x> &"
    index_path.write_text(marker * 2)
    with pytest.raises(ValueError, match="exactly one"):
        BUILDER.render_catalog(tmp_path)
