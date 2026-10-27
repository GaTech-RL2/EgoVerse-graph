"""Regression coverage for safe staging-to-episodes SQL projection.

Run with unittest and the ingestion host's existing SQLAlchemy/botocore runtime.
No credentials, S3 requests or production database access are used.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import Mock

from botocore.exceptions import ClientError
from sqlalchemy import create_engine, event, text


def _load_helper():
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "source" / "merge_staging_to_episodes.py",
        root
        / "egomimic"
        / "scripts"
        / "backfill_scripts"
        / "merge_staging_to_episodes.py",
    ]
    if os.environ.get("MERGE_HELPER_SOURCE"):
        candidates.insert(0, Path(os.environ["MERGE_HELPER_SOURCE"]))
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise RuntimeError(
            "The merge helper source must exist before running these tests"
        )
    spec = importlib.util.spec_from_file_location(
        "_merge_staging_helper_under_test", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_helper()
REQUIRED = list(helper.DEFAULT_REQUIRED)


def _destination_column(
    name, nullable="NO", default=None, identity="NO", generated="NEVER"
):
    return {
        "column_name": name,
        "is_nullable": nullable,
        "column_default": default,
        "is_identity": identity,
        "is_generated": generated,
    }


def _destination_schema():
    return [
        _destination_column("episode_hash"),
        _destination_column("embodiment"),
        _destination_column("task"),
        _destination_column("task_description", "YES"),
        _destination_column("num_frames"),
        _destination_column("zarr_processed_path"),
        _destination_column("lab", "YES"),
        _destination_column("created_at", default="now()"),
        _destination_column("updated_at", default="now()"),
        _destination_column("zarr_mp4_path", default="''::text"),
        _destination_column("rig_name", default="''::text"),
        _destination_column("zarr_processing_error", default="''::text"),
        _destination_column("flow_3d_path", default="''::text"),
        _destination_column("is_deleted", default="false"),
        _destination_column("is_eval", default="false"),
        _destination_column("eval_score", default="-1"),
        _destination_column("eval_success", default="true"),
    ]


SOURCE_COLUMNS = {
    "episode_hash",
    "embodiment",
    "task",
    "task_description",
    "num_frames",
    "zarr_processed_path",
    "source_folder",
}


class LabAndArgumentTests(unittest.TestCase):
    def test_nested_source_resolves_stable_lowercase_vendor(self):
        self.assertEqual(
            "archebase", helper._normalize_lab(" /ArcheBase/ego_data_20261022/ ")
        )
        self.assertEqual("lightwheel", helper._normalize_lab("Lightwheel/test10-9"))
        self.assertEqual("microagi", helper._normalize_lab("microagi"))

    def test_lab_override_uses_parameter_instead_of_sql_interpolation(self):
        expression = helper._lab_expression("archebase")
        self.assertIn(":lab_override", expression)
        self.assertNotIn("'archebase'", expression)

    def test_positive_integer_parser(self):
        self.assertEqual(1, helper._positive_int("1"))
        self.assertEqual(33124, helper._positive_int("33124"))
        for invalid in ("0", "-1", "1.5", "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    helper._positive_int(invalid)


class SchemaAdmissionTests(unittest.TestCase):
    def test_valid_projection_accepts_no_staged_timestamp_columns(self):
        helper._validate_schema(set(SOURCE_COLUMNS), _destination_schema(), REQUIRED)

    def test_missing_projected_description_fails_even_when_not_required_filter(self):
        self.assertNotIn("task_description", REQUIRED)
        with self.assertRaisesRegex(ValueError, "task_description"):
            helper._validate_schema(
                SOURCE_COLUMNS - {"task_description"}, _destination_schema(), REQUIRED
            )

    def test_missing_lab_source_fails_without_override(self):
        with self.assertRaisesRegex(ValueError, "source_folder"):
            helper._validate_schema(
                SOURCE_COLUMNS - {"source_folder"}, _destination_schema(), REQUIRED
            )

    def test_lab_override_does_not_require_source_folder_column(self):
        helper._validate_schema(
            SOURCE_COLUMNS - {"source_folder"},
            _destination_schema(),
            REQUIRED,
            lab_override="archebase",
        )

    def test_omitted_nonnullable_without_default_blocks_before_apply(self):
        schema = _destination_schema() + [_destination_column("new_required_field")]
        with self.assertRaisesRegex(ValueError, "new_required_field"):
            helper._validate_schema(set(SOURCE_COLUMNS), schema, REQUIRED)

    def test_db_managed_timestamp_still_requires_a_destination_default(self):
        schema = _destination_schema()
        next(c for c in schema if c["column_name"] == "created_at")[
            "column_default"
        ] = None
        with self.assertRaisesRegex(ValueError, "created_at"):
            helper._validate_schema(set(SOURCE_COLUMNS), schema, REQUIRED)

    def test_omitted_nullable_identity_and_generated_columns_are_allowed(self):
        schema = _destination_schema() + [
            _destination_column("optional_field", nullable="YES"),
            _destination_column("identity_field", identity="YES"),
            _destination_column("generated_field", generated="ALWAYS"),
        ]
        helper._validate_schema(set(SOURCE_COLUMNS), schema, REQUIRED)

    def test_missing_destination_projected_column_blocks(self):
        schema = [c for c in _destination_schema() if c["column_name"] != "num_frames"]
        with self.assertRaisesRegex(ValueError, "num_frames"):
            helper._validate_schema(set(SOURCE_COLUMNS), schema, REQUIRED)

    def test_schema_validation_does_not_mutate_metadata_inputs(self):
        source = set(SOURCE_COLUMNS)
        destination = _destination_schema()
        saved = copy.deepcopy(destination)
        helper._validate_schema(source, destination, REQUIRED)
        self.assertEqual(SOURCE_COLUMNS, source)
        self.assertEqual(saved, destination)


class PreviewTests(unittest.TestCase):
    def test_preview_uri_requires_s3_zarr_and_handles_trailing_slash(self):
        self.assertEqual(
            "s3://rldb/processed_v3/archebase/10157.mp4",
            helper._preview_uri("s3://rldb/processed_v3/archebase/10157.zarr/"),
        )
        for invalid in (
            "https://example.org/episode.zarr",
            "/tmp/episode.zarr",
            "s3://rldb/episode.zip",
        ):
            with self.subTest(invalid=invalid):
                self.assertIsNone(helper._preview_uri(invalid))

    def test_existing_nonempty_preview_is_returned_only_after_exact_head(self):
        s3 = Mock()
        s3.head_object.return_value = {
            "ContentLength": 128,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
        uri = "s3://rldb/processed_v3/archebase/10157.mp4"
        self.assertEqual(uri, helper._check_preview(s3, uri))
        s3.head_object.assert_called_once_with(
            Bucket="rldb", Key="processed_v3/archebase/10157.mp4"
        )

    def test_404_produces_empty_path(self):
        s3 = Mock()
        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "HeadObject",
        )
        self.assertEqual("", helper._check_preview(s3, "s3://rldb/episode.mp4"))

    def test_authentication_failure_is_not_reported_as_missing(self):
        s3 = Mock()
        s3.head_object.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "HeadObject",
        )
        with self.assertRaises(RuntimeError):
            helper._check_preview(s3, "s3://rldb/episode.mp4")

    def test_misleading_missing_code_with_http403_is_failure(self):
        s3 = Mock()
        s3.head_object.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "HeadObject",
        )
        with self.assertRaises(RuntimeError):
            helper._check_preview(s3, "s3://rldb/episode.mp4")

    def test_missing_http_status_does_not_prove_absence(self):
        s3 = Mock()
        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "HeadObject"
        )
        with self.assertRaises(RuntimeError):
            helper._check_preview(s3, "s3://rldb/episode.mp4")

    def test_zero_length_preview_is_not_linked(self):
        s3 = Mock()
        s3.head_object.return_value = {
            "ContentLength": 0,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
        self.assertEqual("", helper._check_preview(s3, "s3://rldb/episode.mp4"))

    def test_network_failure_is_not_reported_as_missing(self):
        s3 = Mock()
        s3.head_object.side_effect = RuntimeError("simulated network failure")
        with self.assertRaises(RuntimeError):
            helper._check_preview(s3, "s3://rldb/episode.mp4")


class InsertProjectionExecutionTests(unittest.TestCase):
    """Execute the generated INSERT against real NOT NULL/default constraints."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")

        @event.listens_for(self.engine, "connect")
        def functions(connection, record):
            self.assertIsInstance(connection, sqlite3.Connection)
            connection.create_function(
                "split_part",
                3,
                lambda value, sep, part: None
                if value is None
                else (
                    value.split(sep)[int(part) - 1]
                    if 0 < int(part) <= len(value.split(sep))
                    else ""
                ),
            )
            connection.create_function(
                "btrim",
                2,
                lambda value, chars: None if value is None else value.strip(chars),
            )

        with self.engine.begin() as connection:
            connection.exec_driver_sql("ATTACH DATABASE ':memory:' AS app")
            connection.exec_driver_sql("""
                CREATE TABLE app.staging_test (
                    episode_hash TEXT PRIMARY KEY,
                    embodiment TEXT,
                    task TEXT,
                    task_description TEXT,
                    num_frames INTEGER,
                    zarr_processed_path TEXT,
                    source_folder TEXT,
                    created_at TEXT
                )
            """)
            connection.exec_driver_sql("""
                CREATE TABLE app.episodes (
                    episode_hash TEXT PRIMARY KEY,
                    embodiment TEXT NOT NULL,
                    task TEXT NOT NULL,
                    task_description TEXT,
                    num_frames INTEGER NOT NULL,
                    zarr_processed_path TEXT NOT NULL,
                    lab TEXT,
                    created_at TEXT NOT NULL DEFAULT '__db_created__',
                    updated_at TEXT NOT NULL DEFAULT '__db_updated__',
                    zarr_mp4_path TEXT NOT NULL DEFAULT '',
                    rig_name TEXT NOT NULL DEFAULT '',
                    zarr_processing_error TEXT NOT NULL DEFAULT '',
                    flow_3d_path TEXT NOT NULL DEFAULT '',
                    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                    is_eval BOOLEAN NOT NULL DEFAULT FALSE,
                    eval_score REAL NOT NULL DEFAULT -1,
                    eval_success BOOLEAN NOT NULL DEFAULT TRUE
                )
            """)

    def tearDown(self):
        self.engine.dispose()

    def seed_staging(
        self,
        episode_hash="10157",
        created_at=None,
        source_folder="ArcheBase/ego_data_20261022",
    ):
        row = {
            "episode_hash": episode_hash,
            "embodiment": "human_bimanual",
            "task": "unnamed_task",
            "task_description": "continuous recording",
            "num_frames": 5466,
            "zarr_processed_path": "s3://rldb/processed_v3/archebase/"
            + episode_hash
            + ".zarr",
            "source_folder": source_folder,
            "created_at": created_at,
        }
        with self.engine.begin() as connection:
            connection.execute(
                text("""
                INSERT INTO app.staging_test
                    (episode_hash, embodiment, task, task_description, num_frames,
                     zarr_processed_path, source_folder, created_at)
                VALUES (:episode_hash, :embodiment, :task, :task_description,
                        :num_frames, :zarr_processed_path, :source_folder, :created_at)
            """),
                row,
            )

    def apply_projection(self, **kwargs):
        statement, params = helper._build_insert("app.staging_test", REQUIRED, **kwargs)
        self.assertNotIn("created_at", str(statement).lower())
        self.assertNotIn("updated_at", str(statement).lower())
        if not kwargs.get("include_previews"):
            self.assertNotIn("zarr_mp4_path", str(statement).lower())
        with self.engine.begin() as connection:
            connection.execute(statement, params)

    def rows(self):
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text("SELECT * FROM app.episodes ORDER BY episode_hash")
                ).mappings()
            ]

    def test_numeric_null_timestamp_uses_db_defaults_and_stable_vendor(self):
        self.seed_staging()
        self.apply_projection()
        row = self.rows()[0]
        self.assertEqual("__db_created__", row["created_at"])
        self.assertEqual("__db_updated__", row["updated_at"])
        self.assertEqual("archebase", row["lab"])
        self.assertEqual("", row["zarr_mp4_path"])
        self.assertEqual("", row["rig_name"])

    def test_recording_timestamp_does_not_override_db_owned_created_at(self):
        self.seed_staging(
            "2026-07-17-02-50-51-788999",
            created_at="2026-07-17T02:50:51.788999+00:00",
        )
        self.apply_projection()
        self.assertEqual("__db_created__", self.rows()[0]["created_at"])

    def test_explicit_lab_override_is_used(self):
        self.seed_staging(source_folder="incorrect/old_batch")
        self.apply_projection(lab_override="archebase")
        self.assertEqual("archebase", self.rows()[0]["lab"])

    def test_conflicting_hash_is_preserved(self):
        self.seed_staging()
        with self.engine.begin() as connection:
            connection.execute(
                text("""
                INSERT INTO app.episodes
                    (episode_hash, embodiment, task, num_frames, zarr_processed_path, lab)
                VALUES ('10157', 'human_bimanual', 'existing_task', 10, 'existing.zarr', 'existing_lab')
            """)
            )
        self.apply_projection()
        rows = self.rows()
        self.assertEqual(1, len(rows))
        self.assertEqual("existing_task", rows[0]["task"])
        self.assertEqual("existing.zarr", rows[0]["zarr_processed_path"])
        self.assertEqual("existing_lab", rows[0]["lab"])

    def test_limit_is_deterministic(self):
        self.seed_staging("303")
        self.seed_staging("101")
        self.seed_staging("202")
        self.apply_projection(limit=1)
        self.assertEqual(["101"], [row["episode_hash"] for row in self.rows()])


if __name__ == "__main__":
    unittest.main()
