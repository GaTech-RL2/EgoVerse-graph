#!/usr/bin/env python3
"""Review or merge processed episode metadata into app.episodes.

Dry-run is the default; only --apply writes rows. Database-managed timestamps
always use destination defaults. Lab defaults to the first source_folder segment
(lowercase); --lab supplies an explicit vendor label. Preview links are omitted
by default. --preview-policy verify retains only nonempty R2 objects proven by
exact HEAD requests; missing objects keep the destination's empty-path value.

Both dry-run and apply validate the actual INSERT projection and destination
defaults, rather than treating non-null/conflict counts as sufficient.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re

from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine

DEFAULT_REQUIRED = [
    "episode_hash",
    "embodiment",
    "task",
    "num_frames",
    "zarr_processed_path",
]
_MERGE_COLUMNS = [
    "episode_hash",
    "embodiment",
    "task",
    "task_description",
    "num_frames",
    "zarr_processed_path",
]
_FLAGS = ["is_deleted", "is_eval", "eval_score", "eval_success"]
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LAB = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_R2_ENDPOINT = "https://1beb594fb475d71c4420f7b693524e19.r2.cloudflarestorage.com"


def _ident_ok(name):
    return isinstance(name, str) and bool(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", name)
    )


def _positive_int(value):
    try:
        n = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _normalize_lab(source_folder):
    label = str(source_folder or "").strip(" /").split("/", 1)[0].lower()
    if not _LAB.fullmatch(label):
        raise ValueError("Lab must be a nonempty lowercase vendor label.")
    return label


def _lab_expression(lab_override=None):
    if lab_override is not None:
        return "CAST(:lab_override AS TEXT)"
    return "lower(split_part(btrim(s.source_folder, '/ '), '/', 1))"


def _insert_columns(include_previews=False):
    return (
        _MERGE_COLUMNS
        + ["lab"]
        + (["zarr_mp4_path"] if include_previews else [])
        + _FLAGS
    )


def _projection(lab_override=None, include_previews=False):
    values = [f"s.{c}" for c in _MERGE_COLUMNS]
    values.append(_lab_expression(lab_override))
    if include_previews:
        # A single JSON parameter avoids per-row bind limits for large batches.
        values.append(
            "COALESCE(CAST(:verified_previews AS JSONB) ->> s.episode_hash, '')"
        )
    return values + ["FALSE", "FALSE", "-1", "TRUE"]


def _predicates(required, lab_override=None):
    suff = " AND ".join(f"s.{c} IS NOT NULL" for c in required)
    lab = _lab_expression(lab_override)
    suff += f" AND ({lab}) IS NOT NULL AND length({lab}) > 0"
    not_present = (
        "NOT EXISTS (SELECT 1 FROM app.episodes e "
        "WHERE e.episode_hash = s.episode_hash)"
    )
    return suff, not_present


def _build_insert(
    staging, required, *, lab_override=None, limit=None, include_previews=False
):
    if (
        not _ident_ok(staging)
        or not required
        or not all(_IDENTIFIER.fullmatch(c) for c in required)
    ):
        raise ValueError("Invalid table or required-column identifier.")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise ValueError("Limit must be positive.")
    suff, absent = _predicates(required, lab_override)
    limit_sql = "LIMIT :merge_limit" if limit is not None else ""
    sql = text(f"""
        INSERT INTO { "app.episodes" } ({", ".join(_insert_columns(include_previews))})
        SELECT {", ".join(_projection(lab_override, include_previews))}
        FROM {staging} s
        WHERE ({suff}) AND {absent}
        ORDER BY s.episode_hash
        {limit_sql}
        ON CONFLICT (episode_hash) DO NOTHING
    """)
    params = {}
    if lab_override is not None:
        params["lab_override"] = lab_override
    if limit is not None:
        params["merge_limit"] = limit
    if include_previews:
        params["verified_previews"] = "{}"
    return sql, params


def _validate_schema(
    source_cols, dest_cols, required, lab_override=None, include_previews=False
):
    if not required:
        raise ValueError("At least one required column is necessary.")
    projected_source = set(_MERGE_COLUMNS) | set(required)
    if lab_override is None:
        projected_source.add("source_folder")
    missing = projected_source - set(source_cols)
    if missing:
        raise ValueError(f"Staging table lacks projected columns: {sorted(missing)}")
    dest = {row["column_name"]: row for row in dest_cols}
    inserted = set(_insert_columns(include_previews))
    missing_target = inserted - set(dest)
    if missing_target:
        raise ValueError(
            f"Destination lacks inserted columns: {sorted(missing_target)}"
        )
    for name, column in dest.items():
        if name in inserted:
            continue
        generated = column.get("is_generated", "NEVER") not in ("NEVER", None)
        identity = column.get("is_identity", "NO") == "YES"
        if (
            column["is_nullable"] == "NO"
            and column.get("column_default") is None
            and not generated
            and not identity
        ):
            raise ValueError(
                f"Destination {name} is omitted but has no non-null default."
            )


def _preview_uri(path):
    if not isinstance(path, str):
        return None
    path = path.rstrip("/")
    if (
        not path.startswith("s3://rldb/")
        or not path.endswith(".zarr")
        or "?" in path
        or "#" in path
    ):
        return None
    return path[:-5] + ".mp4"


def _check_preview(s3, uri):
    try:
        result = s3.head_object(Bucket="rldb", Key=uri[len("s3://rldb/") :])
    except Exception as exc:
        response = getattr(exc, "response", {})
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            return ""
        raise RuntimeError(
            f"Preview verification failed ({type(exc).__name__}); merge refused."
        ) from None
    if not isinstance(result.get("ContentLength"), int) or result["ContentLength"] <= 0:
        return ""
    return uri


def _preview_client():
    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("R2_ENDPOINT_URL")
    key = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if endpoint != _R2_ENDPOINT or len(key) != 32 or not secret:
        raise ValueError(
            "Valid host-local R2 configuration is required to verify previews."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        aws_session_token=None,
        config=Config(
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 3},
            max_pool_connections=8,
        ),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staging-table", required=True)
    ap.add_argument("--required-columns", default=",".join(DEFAULT_REQUIRED))
    ap.add_argument("--lab", default=None, help="Explicit short vendor label.")
    ap.add_argument(
        "--preview-policy",
        choices=("omit", "verify"),
        default="omit",
        help="Omit unverified links (default), or verify sibling R2 previews.",
    )
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=_positive_int, default=None)
    args = ap.parse_args(argv)
    if (
        not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", args.staging_table
        )
        or args.staging_table == "app.episodes"
    ):
        ap.error("--staging-table must be a separate schema-qualified staging table.")
    required = list(
        dict.fromkeys(c.strip() for c in args.required_columns.split(",") if c.strip())
    )
    if not required or not all(_IDENTIFIER.fullmatch(c) for c in required):
        ap.error("Required columns must be nonempty, simple column names.")
    lab = None
    if args.lab is not None:
        lab = args.lab.strip().lower()
        if not _LAB.fullmatch(lab):
            ap.error("--lab must be a short lowercase vendor label.")
    include_previews = args.preview_policy == "verify"
    sql, params = _build_insert(
        args.staging_table,
        required,
        lab_override=lab,
        limit=args.limit,
        include_previews=include_previews,
    )
    suff, absent = _predicates(required, lab)
    engine = create_default_engine()
    try:
        with engine.connect() as conn:
            schema, table = args.staging_table.split(".")
            source_cols = {
                r[0]
                for r in conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema=:schema AND table_name=:table"
                    ),
                    {"schema": schema, "table": table},
                )
            }
            dest_cols = [
                dict(r)
                for r in conn.execute(
                    text(
                        "SELECT column_name,is_nullable,column_default,is_identity,is_generated "
                        "FROM information_schema.columns "
                        "WHERE table_schema='app' AND table_name='episodes'"
                    )
                ).mappings()
            ]
            _validate_schema(source_cols, dest_cols, required, lab, include_previews)
            invalid_labs = conn.execute(
                text(
                    f"SELECT COUNT(*) FROM {args.staging_table} s "
                    f"WHERE ({suff}) AND ({_lab_expression(lab)}) !~ '^[a-z0-9][a-z0-9_-]*$'"
                ),
                params,
            ).scalar()
            if invalid_labs:
                raise ValueError(
                    f"Invalid vendor lab labels in {invalid_labs} rows; use --lab."
                )
            total = conn.execute(
                text(f"SELECT COUNT(*) FROM {args.staging_table}")
            ).scalar()
            insufficient = conn.execute(
                text(f"SELECT COUNT(*) FROM {args.staging_table} s WHERE NOT ({suff})"),
                params,
            ).scalar()
            conflicts = conn.execute(
                text(
                    f"SELECT COUNT(*) FROM {args.staging_table} s WHERE ({suff}) AND NOT ({absent})"
                ),
                params,
            ).scalar()
            eligible = conn.execute(
                text(
                    f"SELECT COUNT(*) FROM {args.staging_table} s WHERE ({suff}) AND {absent}"
                ),
                params,
            ).scalar()
            # Validate actual copied expressions against destination nullability.
            dest = {r["column_name"]: r for r in dest_cols}
            for name, expr in zip(
                _insert_columns(include_previews), _projection(lab, include_previews)
            ):
                if dest[name]["is_nullable"] == "NO":
                    invalid = conn.execute(
                        text(
                            f"SELECT COUNT(*) FROM {args.staging_table} s "
                            f"WHERE ({suff}) AND {absent} AND ({expr}) IS NULL"
                        ),
                        params,
                    ).scalar()
                    if invalid:
                        raise ValueError(
                            f"Projected {name} is NULL for {invalid} eligible rows."
                        )
            labs = conn.execute(
                text(
                    f"SELECT {_lab_expression(lab)} AS lab,COUNT(*) AS rows "
                    f"FROM {args.staging_table} s WHERE ({suff}) AND {absent} "
                    f"GROUP BY 1 ORDER BY 1 LIMIT 20"
                ),
                params,
            ).all()
            if include_previews:
                limit_sql = "LIMIT :merge_limit" if args.limit is not None else ""
                paths = conn.execute(
                    text(
                        f"SELECT s.episode_hash,s.zarr_processed_path FROM {args.staging_table} s "
                        f"WHERE ({suff}) AND {absent} ORDER BY s.episode_hash {limit_sql}"
                    ),
                    params,
                ).all()
                s3 = _preview_client()

                def check(row):
                    uri = _preview_uri(row.zarr_processed_path)
                    return row.episode_hash, _check_preview(s3, uri) if uri else ""

                with ThreadPoolExecutor(max_workers=8) as pool:
                    verified = dict(pool.map(check, paths))
                params["verified_previews"] = json.dumps(verified)
                preview_count = sum(bool(v) for v in verified.values())
            else:
                preview_count = 0

        print(f"Merge preview: {args.staging_table} -> app.episodes")
        print(f"  total staged rows      : {total}")
        print(f"  INSUFFICIENT (skipped) : {insufficient}")
        print(f"  CONFLICT already exists: {conflicts}")
        print(f"  ELIGIBLE to merge      : {eligible}")
        print(
            f"  selected by limit      : {min(eligible, args.limit) if args.limit is not None else eligible}"
        )
        print("  timestamps             : destination database defaults")
        print(f"  resolved labs (up to20) : {[(r.lab, r.rows) for r in labs]}")
        print(
            f"  preview policy         : {args.preview_policy}; verified links={preview_count}"
        )
        print("  projection/defaults    : PASS")
        if not args.apply:
            print("DRY RUN: nothing written.")
            return 0
        with engine.begin() as conn:
            result = conn.execute(sql, params)
        print(f"APPLIED: inserted {result.rowcount} rows into app.episodes.")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
