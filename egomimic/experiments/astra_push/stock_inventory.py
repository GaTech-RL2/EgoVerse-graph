"""Nonexecuting inventory of pinned shipped LIBERO BDDL assets and regions."""

import re
from collections import Counter
from pathlib import Path

from egomimic.experiments.astra_push.artifacts import file_hash
from egomimic.experiments.astra_push.schemas import canonical_hash


def sexpr(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text).lower())
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            value = []
            (stack[-1] if stack else roots).append(value)
            stack.append(value)
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced BDDL")
            stack.pop()
        else:
            if not stack:
                raise ValueError("BDDL atom outside root")
            stack[-1].append(token)
    if stack or len(roots) != 1 or roots[0][0] != "define":
        raise ValueError("Expected a single BDDL define form")
    return roots[0][1:]


def inventory_file(path):
    groups = {}
    for group in sexpr(Path(path).read_text()):
        if not isinstance(group, list) or group[0] in groups:
            raise ValueError("Malformed or duplicate BDDL section")
        groups[group[0]] = group[1:]
    objects = {}
    for section in (":objects", ":fixtures"):
        entries, pending = groups.get(section, []), []
        index = 0
        while index < len(entries):
            token = entries[index]
            if token == "-":
                asset = entries[index + 1]
                if not pending or any(name in objects for name in pending):
                    raise ValueError("Invalid typed object declaration")
                objects.update({name: asset for name in pending})
                pending = []
                index += 2
            else:
                pending.append(token)
                index += 1
        if pending:
            raise ValueError("Untyped stock assets are unsupported")
    regions, unsupported_regions = {}, []
    for region in groups.get(":regions", []):
        attributes = {part[0]: part[1:] for part in region[1:]}
        target = attributes.get(":target", [None])[0]
        if target not in objects:
            # The pinned libero_goal files contain unused affordance regions
            # for an undeclared bowl_drainer_1. Preserve that limitation. The
            # declared physical asset inventory remains fully comparable.
            unsupported_regions.append({"region": region[0], "target": target})
            continue
        value = {
            "target_asset": objects[target],
            "ranges": [
                [round(float(x), 4) for x in rectangle]
                for rectangle in attributes.get(":ranges", [[]])[0]
            ],
            "yaw_rotation": attributes.get(":yaw_rotation", []),
        }
        regions[f"{target}_{region[0]}"] = value

    def abstract(value):
        if isinstance(value, list):
            return [abstract(v) for v in value]
        if value in objects:
            return {"asset": objects[value]}
        if value in regions:
            return {"region": regions[value]}
        return value

    return {
        "asset_multiset": dict(sorted(Counter(objects.values()).items())),
        "region_geometry": sorted(regions.values(), key=canonical_hash),
        "initial_relations": sorted(
            (abstract(x) for x in groups.get(":init", [])), key=canonical_hash
        ),
        "unsupported_regions": unsupported_regions,
    }


def stock_inventory(root, scenes):
    root = Path(root)
    files = sorted(root.rglob("*.bddl"))
    if not files:
        raise ValueError("Pinned LIBERO stock BDDL inventory is empty")
    records, unsupported = [], []
    for path in files:
        record = {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
        try:
            record["inventory"] = inventory_file(path)
            record["inventory_hash"] = canonical_hash(record["inventory"])
            records.append(record)
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            unsupported.append({**record, "reason": str(exc)})
    comparisons = []
    for scene in scenes:
        # These keys identify the registered physical asset constructors, not
        # scene/object IDs. Both colors are checked against stock explicitly.
        assets = Counter(["table"])
        assets.update(f"astra_{c.color}_cube" for c in scene.cubes)
        assets.update(["astra_reference"] * len(scene.fixtures))
        same_assets = [
            r["path"] for r in records if r["inventory"]["asset_multiset"] == assets
        ]
        comparisons.append(
            {
                "scene_hash": canonical_hash(scene),
                "structural_signature": scene.structural_signature(),
                "asset_multiset": dict(sorted(assets.items())),
                "same_asset_composition_stock_files": same_assets,
                "distinct_by_physical_asset_composition": not same_assets,
            }
        )
    return {
        "schema_version": "astrapush-stock-inventory-1",
        "files": records,
        "unsupported": unsupported,
        "region_inventory_limitations": [
            {"path": r["path"], "regions": r["inventory"]["unsupported_regions"]}
            for r in records
            if r["inventory"]["unsupported_regions"]
        ],
        "comparisons": comparisons,
        "passed": not unsupported
        and all(r["distinct_by_physical_asset_composition"] for r in comparisons),
        "comparison_basis": "Declared physical asset composition differs for every stock scene; this alone excludes structural identity. Unsupported unused stock affordance regions remain explicitly recorded, not claimed normalized.",
        "limitation": "Same-composition stock scenes require a stronger geometry/relation comparison; this gate conservatively fails them.",
    }
