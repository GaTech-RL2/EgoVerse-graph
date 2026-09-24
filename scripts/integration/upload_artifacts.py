"""Preserve gate outputs at a new R2 prefix, with conditional writes and hashes."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import boto3
from botocore.config import Config


def upload(root, prefix):
    if not prefix.startswith("experiments/graph-integration-") or ".." in prefix.split(
        "/"
    ):
        raise ValueError(
            "Artifact uploads must use a graph-integration experiment prefix"
        )
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 8}),
    )
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"Artifact symlink is not supported: {path}")
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        key = prefix.rstrip("/") + "/" + path.relative_to(root).as_posix()
        print("PRESERVE_ARTIFACT", key, path.stat().st_size, sha, flush=True)
        metadata = {"sha256": sha}
        with path.open("rb") as stream:
            if path.stat().st_size < 64 * 1024 * 1024:
                client.put_object(
                    Bucket="rldb",
                    Key=key,
                    Body=stream,
                    Metadata=metadata,
                    IfNoneMatch="*",
                )
            else:
                transaction = client.create_multipart_upload(
                    Bucket="rldb", Key=key, Metadata=metadata
                )
                upload_id = transaction["UploadId"]
                parts = []
                try:
                    while block := stream.read(64 * 1024 * 1024):
                        part = client.upload_part(
                            Bucket="rldb",
                            Key=key,
                            UploadId=upload_id,
                            PartNumber=len(parts) + 1,
                            Body=block,
                        )
                        parts.append(
                            {"PartNumber": len(parts) + 1, "ETag": part["ETag"]}
                        )
                    client.complete_multipart_upload(
                        Bucket="rldb",
                        Key=key,
                        UploadId=upload_id,
                        MultipartUpload={"Parts": parts},
                        IfNoneMatch="*",
                    )
                except BaseException:
                    # This aborts only the incomplete upload created above.
                    client.abort_multipart_upload(
                        Bucket="rldb", Key=key, UploadId=upload_id
                    )
                    raise
        records.append({"key": key, "bytes": path.stat().st_size, "sha256": sha})
    body = json.dumps({"status": "preserved", "files": records}, indent=2).encode()
    client.put_object(
        Bucket="rldb",
        Key=prefix.rstrip("/") + "/artifact-receipt.json",
        Body=body,
        IfNoneMatch="*",
    )
    print(
        "ARTIFACT_RECEIPT",
        json.dumps({"prefix": "s3://rldb/" + prefix, "files": len(records)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    upload(**vars(parser.parse_args()))
