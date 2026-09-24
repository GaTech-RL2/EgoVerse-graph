"""Durable experiment artifacts with content hashes and optional S3/R2 output."""
from __future__ import annotations

import json
import os
from pathlib import Path

from egomimic.rldb.goal_replay import sha256_file


class ArtifactWriter:
    def __init__(self, directory, bucket=None, prefix=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.bucket, self.prefix = bucket, (prefix or "").rstrip("/")
        self.client = None
        if bucket:
            import boto3
            from botocore.config import Config
            self.client = boto3.client(
                "s3", endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
                region_name="auto", config=Config(retries={"max_attempts": 8}))

    def publish(self, path):
        path = Path(path)
        digest = sha256_file(path)
        key = self.prefix + "/" + str(path.relative_to(self.directory))
        if self.client:
            self.client.upload_file(str(path), self.bucket, key,
                                    ExtraArgs={"Metadata": {"sha256": digest}})
            head = self.client.head_object(Bucket=self.bucket, Key=key)
            if head["ContentLength"] != path.stat().st_size or head["Metadata"].get("sha256") != digest:
                raise RuntimeError("artifact upload verification failed")
        return {"uri": f"s3://{self.bucket}/{key}" if self.client else str(path),
                "sha256": digest, "bytes": path.stat().st_size}

    def json(self, name, value):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
        return self.publish(path)
