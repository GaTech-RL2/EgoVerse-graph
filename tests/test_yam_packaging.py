import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _extra_conflict_pairs(conflicts: list[list[dict[str, str]]]) -> set[frozenset[str]]:
    pairs = set()
    for group in conflicts:
        selected = [requirement["extra"] for requirement in group]
        if len(selected) == 2:
            pairs.add(frozenset(selected))
    return pairs


def test_yam_dependencies_are_isolated_from_aria_and_pi05() -> None:
    with PYPROJECT.open("rb") as file:
        project = tomllib.load(file)

    base_dependencies = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]
    conflicts = project["tool"]["uv"]["conflicts"]

    assert not any(
        dependency.startswith("projectaria-tools") for dependency in base_dependencies
    )
    assert extras["aria"] == ["projectaria-tools[all]==2.0.0"]

    conflict_pairs = _extra_conflict_pairs(conflicts)
    assert frozenset(("aria", "yam")) in conflict_pairs
    assert frozenset(("pi05", "yam")) in conflict_pairs
