"""Build view-only reset templates: python -m egomimic.scripts.build_reset_guides.

Run in the activated project environment (requires PyYAML), then --check before
Node UI tests; a generic bundled Python need not have project dependencies.
Embedding declarative YAML as JSON in the existing
index lets an already running dashboard serve new guides on browser refresh,
without new backend routes or a hardware-session restart.
"""

import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path("egomimic/hydra_configs/robot/reset_overlay_templates.yaml")
INDEX = Path("egomimic/robot/teleop_dashboard_static/index.html")
ROLLOUT_INDEX = Path("egomimic/robot/rollout_dashboard_static/index.html")
TELEOP_APP = Path("egomimic/robot/teleop_dashboard_static/app.js")
ROLLOUT_APP = Path("egomimic/robot/rollout_dashboard_static/app.js")
INDEXES = (INDEX, ROLLOUT_INDEX)
PATTERN = r'(<script id="reset-guide-catalog" type="application/json">)(.*?)(</script>)'
RENDERER_PATTERN = (
    r"// BEGIN SHARED RESET GUIDE RENDERER\n.*?// END SHARED RESET GUIDE RENDERER"
)


def render_catalog(root: Path, index: Path = INDEX) -> str:
    """Keep task data out of generic UI code; prevent executable HTML injection."""
    catalog = yaml.safe_load((root / CATALOG).read_text())
    payload = json.dumps(
        catalog, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    payload = (
        payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    source = (root / index).read_text()
    matches = list(re.finditer(PATTERN, source, flags=re.DOTALL))
    if len(matches) != 1:
        raise ValueError("Expected exactly one reset-guide-catalog script element")
    return re.sub(PATTERN, lambda m: m[1] + payload + m[3], source, flags=re.DOTALL)


def render_renderer(root: Path) -> str:
    """Generate both UIs from one renderer, without new live-server routes.

    Keep this block free of robot/policy writes and task-specific coordinates.
    Static generation lets an existing hardware session load it by page refresh.
    """
    source = (root / TELEOP_APP).read_text()
    shared = list(re.finditer(RENDERER_PATTERN, source, flags=re.DOTALL))
    destination = (root / ROLLOUT_APP).read_text()
    if (
        len(shared) != 1
        or len(list(re.finditer(RENDERER_PATTERN, destination, flags=re.DOTALL))) != 1
    ):
        raise ValueError("Expected exactly one shared reset-guide renderer in each app")
    return re.sub(
        RENDERER_PATTERN, lambda _: shared[0][0], destination, flags=re.DOTALL
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = {index: render_catalog(ROOT, index) for index in INDEXES}
    expected[ROLLOUT_APP] = render_renderer(ROOT)
    if args.check:
        if any(
            (ROOT / path).read_text() != source for path, source in expected.items()
        ):
            raise SystemExit(
                "Reset guide catalog is stale; run python -m egomimic.scripts.build_reset_guides"
            )
        print("Teleop and rollout reset guides match shared YAML and renderer")
    else:
        for path, source in expected.items():
            (ROOT / path).write_text(source)


if __name__ == "__main__":
    main()
