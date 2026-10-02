"""Refresh the companion branch's file inventory before updating a runtime pin."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# These are consumed by package builds or runtime code, not documentation-only.
RUNTIME_MARKDOWN = {
    "external/lerobot/README.md",
    "external/rpl_vision_utils/README.md",
    "egomimic/robot/oculus_reader/README.md",
    "external/lerobot/lerobot/common/datasets/card_template.md",
}
VALIDATION_FILES = {
    "scripts/audit_components.py",
    "scripts/audit_hydra_configs.py",
    "scripts/benchmark_arc_tokenizer.py",
    "scripts/integration_port_recipes.py",
    "scripts/e1/test_group_balance.py",
    "egomimic/rldb/zarr/test_dataset_filter.py",
    "egomimic/robot/eva/test_mink_solver.py",
    "egomimic/robot/test.py",
    "egomimic/utils/aws/test_s3_write.py",
    "egomimic/utils/aws/test_sql_access.py",
    "egomimic/robot/eva/eva_ws/src/resources/ARX_Model/X5A/export.log",
    "convention.png",
    "mano_keypoints.png",
}


def is_companion_file(name: str) -> bool:
    path = Path(name)
    if path.name == "AGENTS.md" or name in RUNTIME_MARKDOWN:
        return False
    if path.name.upper().startswith(("LICENSE", "NOTICE", "COPYING")):
        return False
    return (
        "tests" in path.parts
        or name.startswith(("docs/", "assets/", "scripts/integration/"))
        or path.suffix in {".md", ".ipynb"}
        or name in VALIDATION_FILES
    )


def main():
    paths = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ]
        )
        .decode()
        .split("\0")
    )
    manifest = {}
    for name in sorted(set(paths)):
        if not name or not is_companion_file(name):
            continue
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Companion entry must be a regular file: {name}")
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / ".github/validation-paths.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    size = sum((ROOT / name).stat().st_size for name in manifest)
    print(f"Pinned {len(manifest)} companion files ({size:,} bytes)")


if __name__ == "__main__":
    main()
