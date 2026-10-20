"""Assemble hash-bound runtime/data/source archives without filesystem extraction.

Standard-library CPU work only. Each regular member is checked against its
component manifest while streamed into a fresh canonical workspace archive.
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def bound(binding):
    path = Path(binding["path"])
    need(
        path.is_absolute() and path.is_file() and not path.is_symlink(), "Invalid input"
    )
    need(sha(path) == binding["sha256"], "Input digest changed")
    return path


def normalized(name):
    path = PurePosixPath(name)
    need(
        name
        and not path.is_absolute()
        and ".." not in path.parts
        and not any(c in name for c in "\n\r\t\0"),
        "Unsafe member path",
    )
    return str(path)


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.hash.update(block)
        return block


def assemble(request, output):
    need(
        request["schema"] == "workspace-payload-components-v1", "Wrong component schema"
    )
    output = Path(output)
    output.mkdir()
    archive, manifest = output / "payload.tar.gz", output / "payload.files.sha256"
    files, directories, links, actual_hashes = set(), set(), {}, {}
    regular_bytes = 0
    components = []
    with tarfile.open(archive, "x:gz", compresslevel=1) as destination:
        for component in request["components"]:
            source = bound(component["archive"])
            expected = {}
            for line in bound(component["manifest"]).read_text().splitlines():
                digest, name = line.split("  ", 1)
                need(
                    len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                    "Invalid manifest digest",
                )
                name = normalized(name)
                need(name not in expected, "Duplicate component manifest entry")
                expected[name] = digest
            prefix = component.get("prefix", "")
            need(
                not prefix or normalized(prefix).startswith("workspace/"),
                "Unsafe component prefix",
            )
            prefix = prefix.rstrip("/")
            consumed = set()
            count, component_bytes = 0, 0
            with gzip.open(source, "rb") as compressed:
                with tarfile.open(fileobj=compressed, mode="r|") as members:
                    for original in members:
                        name = normalized(original.name)
                        if name == ".":
                            need(original.isdir(), "Invalid archive root")
                            continue
                        target = normalized(prefix + "/" + name if prefix else name)
                        need(
                            target.startswith("workspace/"),
                            "Payload must use workspace paths",
                        )
                        need(
                            not target.endswith(".ckpt"),
                            "Training checkpoint in fresh payload",
                        )
                        parents = [
                            str(p)
                            for p in PurePosixPath(target).parents
                            if str(p) != "."
                        ]
                        need(
                            not any(
                                parent in files or parent in links for parent in parents
                            ),
                            "Payload traverses a file/link",
                        )
                        item = copy.copy(original)
                        item.name = target
                        if item.isdir():
                            need(
                                target not in files and target not in links,
                                "Directory collision",
                            )
                            if target not in directories:
                                destination.addfile(item)
                            directories.add(target)
                            continue
                        need(
                            target not in files
                            and target not in links
                            and target not in directories,
                            "Duplicate payload member",
                        )
                        directories.update(parents)
                        if item.isreg():
                            need(
                                name in expected and name not in consumed,
                                "Unmanifested regular member",
                            )
                            need(
                                component_bytes + item.size
                                <= component["regular_bytes"],
                                "Component exceeds its immutable regular-byte budget",
                            )
                            reader = HashingReader(members.extractfile(original))
                            destination.addfile(item, reader)
                            need(
                                reader.hash.hexdigest() == expected[name],
                                "Regular member digest changed",
                            )
                            actual_hashes[target] = expected[name]
                            files.add(target)
                            consumed.add(name)
                            component_bytes += item.size
                        elif item.islnk():
                            link = normalized(
                                prefix + "/" + item.linkname
                                if prefix
                                else item.linkname
                            )
                            need(
                                link in actual_hashes
                                and name in expected
                                and expected[name] == actual_hashes[link],
                                "Invalid hardlink identity",
                            )
                            item.linkname = link
                            destination.addfile(item)
                            actual_hashes[target] = expected[name]
                            files.add(target)
                            consumed.add(name)
                        elif item.issym():
                            need(
                                name not in expected,
                                "Symlink substituted for a regular file",
                            )
                            destination.addfile(item)
                            links[target] = item.linkname
                        else:
                            raise ValueError("Special payload member")
                        count += 1
                while compressed.read(1 << 20):
                    pass
            need(consumed == set(expected), "Manifest member missing from archive")
            regular_bytes += component_bytes
            components.append(
                dict(
                    archive_sha256=component["archive"]["sha256"],
                    manifest_sha256=component["manifest"]["sha256"],
                    prefix=prefix,
                    members=count,
                    regular_bytes=component_bytes,
                )
            )
        for name, literal in links.items():
            need(not any(c in literal for c in "\n\r\t\0"), "Unsafe symlink literal")
            logical = "/" + name
            resolved = os.path.normpath(
                os.path.join(str(Path(logical).parent), literal)
            )
            need(
                resolved
                in {"/usr/bin/python", "/usr/bin/python3", "/usr/bin/python3.12"}
                or resolved.lstrip("/") in files | directories | set(links),
                "Symlink leaves the complete selected closure",
            )
    with manifest.open("x") as stream:
        for name, digest in sorted(actual_hashes.items()):
            stream.write(digest + "  " + name + "\n")
    receipt = dict(
        schema="workspace-payload-v1",
        status="PASS",
        components=components,
        regular_files=len(files),
        symlinks=len(links),
        regular_bytes=regular_bytes,
        archive=dict(
            path=str(archive.resolve()),
            bytes=archive.stat().st_size,
            sha256=sha(archive),
            compression="gzip",
        ),
        manifest=dict(path=str(manifest.resolve()), sha256=sha(manifest)),
    )
    with (output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(
        bound(
            dict(path=str(args.request.resolve()), sha256=args.request_sha256)
        ).read_text()
    )
    receipt = assemble(request, args.output)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
