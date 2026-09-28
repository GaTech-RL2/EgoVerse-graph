"""Concurrent conditional publication of a stopped full-run artifact tree."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from egomimic.experiments.astra_push.artifacts import file_hash


def preserve(root, prefix):
    if not prefix.startswith("experiments/astra-hpt-libero-") or ".." in prefix.split(
        "/"
    ):
        raise ValueError("Only the authorized Astra experiment prefix is supported")
    root = Path(root)
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 8}, max_pool_connections=20),
    )

    def upload(path):
        if path.is_symlink():
            raise ValueError("Artifact symlinks are forbidden")
        digest = file_hash(path)
        key = prefix.rstrip("/") + "/" + path.relative_to(root).as_posix()
        record = {"key": key, "bytes": path.stat().st_size, "sha256": digest}
        try:
            existing = client.head_object(Bucket="rldb", Key=key)
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise
        else:
            if existing["Metadata"].get("sha256") != digest:
                raise ValueError(f"Refusing to replace an existing artifact: {key}")
            return record
        with path.open("rb") as stream:
            if record["bytes"] < 64 * 1024 * 1024:
                client.put_object(
                    Bucket="rldb",
                    Key=key,
                    Body=stream,
                    Metadata={"sha256": digest},
                    IfNoneMatch="*",
                )
            else:
                tx = client.create_multipart_upload(
                    Bucket="rldb", Key=key, Metadata={"sha256": digest}
                )
                parts = []
                try:
                    while block := stream.read(64 * 1024 * 1024):
                        part = client.upload_part(
                            Bucket="rldb",
                            Key=key,
                            UploadId=tx["UploadId"],
                            PartNumber=len(parts) + 1,
                            Body=block,
                        )
                        parts.append(
                            {"PartNumber": len(parts) + 1, "ETag": part["ETag"]}
                        )
                    client.complete_multipart_upload(
                        Bucket="rldb",
                        Key=key,
                        UploadId=tx["UploadId"],
                        MultipartUpload={"Parts": parts},
                        IfNoneMatch="*",
                    )
                except BaseException:
                    client.abort_multipart_upload(
                        Bucket="rldb", Key=key, UploadId=tx["UploadId"]
                    )
                    raise
        print("PRESERVE_ARTIFACT", key, record["bytes"], digest, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=16) as pool:
        records = list(
            pool.map(upload, sorted(p for p in root.rglob("*") if p.is_file()))
        )
    body = json.dumps(
        {"status": "preserved", "files": records}, sort_keys=True
    ).encode()
    client.put_object(
        Bucket="rldb",
        Key=prefix.rstrip("/") + "/artifact-receipt.json",
        Body=body,
        IfNoneMatch="*",
    )
    print(
        "ARTIFACT_RECEIPT",
        json.dumps(
            {
                "prefix": "s3://rldb/" + prefix,
                "files": len(records),
                "bytes": sum(r["bytes"] for r in records),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    preserve(**vars(parser.parse_args()))
