"""Transfer explicit experiment artifacts through authenticated OSMO execution.

No credentials are exported. Uploads are confined to the current task's output
tree and never replace existing content. Generation READY markers are uploaded
separately after their complete, validated request/response archives.
"""

import argparse
import base64
import errno
import fcntl
import hashlib
import io
import json
import os
import pty
import re
import shlex
import struct
import subprocess
import tarfile
import termios
from pathlib import Path, PurePosixPath


def remote(workflow, task, code, raw_path):
    entry = "bash -lc " + shlex.quote(
        "source /tmp/egoverse/emimic/bin/activate; python -c " + shlex.quote(code)
    )
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 160, 0, 0))
    process = subprocess.Popen(
        [
            "osmo",
            "workflow",
            "exec",
            workflow,
            task,
            "--entry",
            entry,
            "--connect-timeout",
            "30",
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    with Path(raw_path).open("xb") as stream:
        while True:
            try:
                block = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not block:
                break
            stream.write(block)
    os.close(master)
    if process.wait():
        raise RuntimeError("OSMO artifact bridge failed; diagnostics retained")
    return Path(raw_path).read_bytes()


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact path escapes the task tree")
    return path


def download(workflow, task, paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in paths:
        safe_member(name)
    code = """import base64,hashlib,io,json,os,tarfile,time
from pathlib import Path
root=Path(os.environ["GATE_OUTPUT"])
files=[root/p for p in PATHS]
if any(not p.is_file() or p.is_symlink() for p in files): raise ValueError("Only explicit regular artifacts may be exported")
if sum(p.stat().st_size for p in files)>200*1024*1024: raise ValueError("Selected artifacts exceed the bounded export")
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode="w:gz") as archive:
 for p in files: archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
 data=json.dumps(manifest,sort_keys=True).encode();meta=tarfile.TarInfo("bridge-hashes.json");meta.size=len(data);archive.addfile(meta,io.BytesIO(data))
payload=base64.b64encode(buf.getvalue()).decode()
print("ASTRA_PAYLOAD_BEGIN",flush=True)
for offset in range(0,len(payload),4096):
 print(payload[offset:offset+4096],flush=True);time.sleep(.02)
print("ASTRA_PAYLOAD_END",flush=True);time.sleep(2)
""".replace("PATHS", repr(paths))
    raw = remote(workflow, task, code, output / "transfer.raw")
    match = re.search(
        rb"ASTRA_PAYLOAD_BEGIN\r?\n(.*?)\r?\nASTRA_PAYLOAD_END", raw, re.S
    )
    if not match:
        raise RuntimeError("Export was incomplete; retained diagnostics")
    blob = base64.b64decode(b"".join(match[1].split()), validate=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                raise ValueError("Only regular artifacts may be imported")
            path = output / Path(*safe_member(member.name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(archive.extractfile(member).read())
    manifest = json.loads((output / "bridge-hashes.json").read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Export hash differs")
    print(json.dumps({"verified_files": len(manifest), "output": str(output)}))


def upload(workflow, task, source, destination, receipt):
    safe_member(destination)
    source = Path(source)
    files = sorted(source.rglob("*")) if source.is_dir() else [source]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError("Symlink uploads are forbidden")
            if path.is_file():
                archive.add(
                    path,
                    arcname=str(path.relative_to(source))
                    if source.is_dir()
                    else path.name,
                    recursive=False,
                )
    payload = base64.b64encode(buffer.getvalue()).decode()
    if len(payload) > 100000:
        raise ValueError("Use bounded request/response archives for this bridge")
    code = """import base64,hashlib,io,json,os,tarfile,time,uuid
from pathlib import Path,PurePosixPath
root=Path(os.environ["GATE_OUTPUT"])/DESTINATION
records={}
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode="r:gz") as archive:
 for m in archive:
  p=PurePosixPath(m.name)
  if not m.isfile() or p.is_absolute() or ".." in p.parts:raise ValueError("Unsafe artifact member")
  dest=root/Path(*p.parts);dest.parent.mkdir(parents=True,exist_ok=True)
  data=archive.extractfile(m).read();digest=hashlib.sha256(data).hexdigest()
  if dest.exists():
   if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:raise ValueError("Existing experiment artifact differs")
  else:
   temporary=dest.with_name("."+dest.name+"."+uuid.uuid4().hex+".partial")
   with temporary.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
   try:os.link(temporary,dest)
   finally:temporary.unlink()
  records[str(dest)]=digest
print("ASTRA_UPLOAD "+json.dumps(records,sort_keys=True),flush=True);time.sleep(2)
""".replace("DESTINATION", repr(destination)).replace("PAYLOAD", repr(payload))
    raw = remote(workflow, task, code, receipt)
    lines = [
        line.split(b"ASTRA_UPLOAD ", 1)[1]
        for line in raw.splitlines()
        if b"ASTRA_UPLOAD " in line
    ]
    if len(lines) != 1:
        raise RuntimeError("Upload did not return a complete receipt")
    records = json.loads(lines[0])
    print(json.dumps({"uploaded_files": len(records), "receipt": str(receipt)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--task", default="learner-curriculum")
    sub = parser.add_subparsers(dest="command", required=True)
    down = sub.add_parser("download")
    down.add_argument("--paths", nargs="+", required=True)
    down.add_argument("--output", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--source", required=True)
    up.add_argument("--destination", required=True)
    up.add_argument("--receipt", required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    (download if command == "download" else upload)(**arguments)
