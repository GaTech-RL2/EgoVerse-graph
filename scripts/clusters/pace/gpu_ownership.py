"""Read-only four-Hopper ownership proof across Slurm and filtered NVML namespaces.

Numeric Slurm GPU indices are validated against native GRES IDX, never used as
NVML ordinals. Physical UUID/minor ownership comes from the current job cgroup
and kernel device-open permissions, cross-checked against NVML, XML and /proc.
"""

import csv
import glob
import os
import re
import stat
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indices(value):
    result = []
    for token in value.split(","):
        token = token.strip()
        if re.fullmatch(r"\d+-\d+", token):
            lo, hi = map(int, token.split("-"))
            require(0 <= lo <= hi < 64, "Invalid Slurm range")
            result.extend(str(x) for x in range(lo, hi + 1))
        else:
            require(
                re.fullmatch(r"\d+|GPU-[0-9a-fA-F-]{36}", token) is not None,
                "Invalid Slurm GPU token",
            )
            result.append(token)
    require(
        len(result) == len(set(result)) == 4,
        "Need exactly four distinct Slurm assignments",
    )
    require(
        all(x.isdigit() for x in result) or all(x.startswith("GPU-") for x in result),
        "Mixed Slurm identity namespaces",
    )
    return result


def bdf(value):
    domain, bus, device = value.strip().lower().split(":")
    return f"{int(domain,16):04x}:{int(bus,16):02x}:{device}"


def verify(evidence, expected):
    cluster_lines = re.findall(
        r"^\s*ClusterName\s*=\s*(\S+)\s*$", evidence["scheduler_config"], re.M
    )
    require(cluster_lines == [expected["cluster"]], "Native cluster mismatch")
    fields = dict(x.split("=", 1) for x in evidence["native"].split() if "=" in x)
    env = evidence["environment"]
    job = env.get("SLURM_JOB_ID", "")
    node = env.get("SLURMD_NODENAME", "")
    require(
        job.isdigit()
        and fields.get("JobId") == job
        and fields.get("JobState") == "RUNNING",
        "Wrong current native job",
    )
    require(
        fields.get("UserId") == f"{expected['user']}({expected['uid']})",
        "Native owner mismatch",
    )
    require(
        fields.get("Account") == expected["account"]
        and fields.get("QOS") == expected["qos"],
        "Association mismatch",
    )
    require(
        fields.get("NodeList") == fields.get("BatchHost") == node
        and fields.get("NumNodes") == "1",
        "Expected one exact node",
    )
    require(
        fields.get("Restarts") == "0" and env.get("SLURM_RESTART_COUNT", "0") == "0",
        "Fresh allocation required",
    )
    allocations = indices(env.get("SLURM_JOB_GPUS", ""))
    native = re.findall(
        r"GRES=gpu:(?:h100|h200):4\(IDX:([0-9,-]+)\)", evidence["native"]
    )
    require(len(native) == 1, "Need one native four-Hopper GRES allocation")
    native_ids = indices(native[0])
    require(all(x.isdigit() for x in native_ids), "Native IDX malformed")
    if allocations[0].isdigit():
        require(set(native_ids) == set(allocations), "Slurm env and native IDX differ")
    cgroups = [
        line.split(":", 2)[2].split("/") for line in evidence["cgroup"].splitlines()
    ]
    sluid = fields.get("SLUID")
    if expected["cluster"] == "phoenix-slurm":
        require(
            bool(sluid) and any(sluid in p for p in cgroups),
            "Native SLUID missing from current cgroup",
        )
    else:
        require(
            (sluid and any(sluid in p for p in cgroups))
            or any("job_" + job in p or "job" + job in p for p in cgroups),
            "Current job missing from cgroup",
        )
    inventory = []
    for row in csv.reader(evidence["inventory"].splitlines(), skipinitialspace=True):
        require(len(row) == 5, "Inventory must have original five supported fields")
        ordinal, uuid, name, sm, memory = [x.strip() for x in row]
        require(
            ordinal.isdigit() and re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid),
            "Bad NVML identity",
        )
        require(
            re.search(r"\bH(?:100|200)\b", name)
            and float(sm) >= 9
            and int(memory) >= 75000,
            "Not a full Hopper GPU",
        )
        inventory.append(
            dict(
                index=int(ordinal),
                uuid=uuid,
                name=name,
                sm=float(sm),
                memory_mib=int(memory),
            )
        )
    require(
        len(inventory)
        == len({r["uuid"] for r in inventory})
        == len({r["index"] for r in inventory})
        == 4,
        "Require four bijective filtered NVML GPUs",
    )
    physical = {}
    for path, text in evidence["proc"].items():
        values = dict(x.split(":", 1) for x in text.splitlines() if ":" in x)
        values = {k.strip(): v.strip() for k, v in values.items()}
        uuid = values.get("GPU UUID")
        minor = int(values["Device Minor"])
        bus = bdf(values["Bus Location"])
        require(
            uuid
            and uuid not in physical
            and minor not in {r["minor"] for r in physical.values()},
            "Ambiguous /proc GPU identity",
        )
        physical[uuid] = dict(minor=minor, bdf=bus)
    require(len(physical) >= 4, "Incomplete physical /proc inventory")
    devices = evidence["devices"]
    require(
        len(devices) == len({r["minor"] for r in devices}) == len(physical),
        "Incomplete kernel device inventory",
    )
    require(
        {r["minor"] for r in devices} == {r["minor"] for r in physical.values()},
        "Physical/device minor sets differ",
    )
    require(
        all(r["character_device"] and r["major"] == 195 for r in devices),
        "Invalid GPU character device",
    )
    allowed = {r["minor"] for r in devices if r["read_open_permitted"]}
    require(len(allowed) == 4, "Kernel must grant exactly four physical GPU minors")
    uuids = {r["uuid"] for r in inventory}
    require(
        {u for u, p in physical.items() if p["minor"] in allowed} == uuids,
        "Kernel ownership and NVML UUIDs differ",
    )
    if allocations[0].startswith("GPU-"):
        require(
            set(allocations) == uuids,
            "Slurm UUID assignments differ from kernel ownership",
        )
    xml = ET.fromstring(evidence["xml"])
    xml_nodes = xml.findall("gpu")
    require(
        len(xml_nodes) == 4 and {n.findtext("uuid") for n in xml_nodes} == uuids,
        "XML/NVML UUID inventories differ",
    )
    for n in xml_nodes:
        identity = physical[n.findtext("uuid")]
        require(
            int(n.findtext("minor_number")) == identity["minor"]
            and bdf(n.findtext("pci/pci_bus_id")) == identity["bdf"],
            "XML and /proc physical identities differ",
        )
    selected = [{**r, **physical[r["uuid"]]} for r in inventory]
    return dict(
        schema="slurm-kernel-gpu-ownership-v3",
        status="PASS",
        job_id=job,
        node=node,
        native_resources={
            k: fields.get(k)
            for k in ("NumCPUs", "MinMemoryNode", "Requeue", "Partition", "TimeLimit")
        },
        sluid=sluid,
        cluster=expected["cluster"],
        user=expected["user"],
        uid=expected["uid"],
        account=expected["account"],
        qos=expected["qos"],
        native_assignment_tokens=allocations,
        native_indices=native_ids,
        mapping_basis="current native job+cgroup+kernel permissions+UUID/BDF/minor bijection;no ordinal substitution",
        gpus=selected,
    )


def command(argv):
    return subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=30
    ).stdout


def collect(scontrol="/opt/slurm/current/bin/scontrol"):
    keys = [
        "SLURM_JOB_ID",
        "SLURMD_NODENAME",
        "SLURM_JOB_GPUS",
        "SLURM_RESTART_COUNT",
        "CUDA_VISIBLE_DEVICES",
    ]
    env = {k: os.environ[k] for k in keys if k in os.environ}
    evidence = dict(
        environment=env,
        scheduler_config=command([scontrol, "show", "config"]),
        native=command([scontrol, "show", "job", "-dd", env["SLURM_JOB_ID"]]),
        inventory=command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        xml=command(["nvidia-smi", "-q", "-x"]),
        cgroup=Path("/proc/self/cgroup").read_text(),
        proc={
            p: Path(p).read_text()
            for p in glob.glob("/proc/driver/nvidia/gpus/*/information")
        },
        devices=[],
    )
    for path in sorted(glob.glob("/dev/nvidia[0-9]*")):
        if not re.fullmatch(r"/dev/nvidia\d+", path):
            continue
        s = os.stat(path)
        row = dict(
            path=path,
            major=os.major(s.st_rdev),
            minor=os.minor(s.st_rdev),
            character_device=stat.S_ISCHR(s.st_mode),
            read_open_permitted=False,
        )
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            os.close(fd)
            row["read_open_permitted"] = True
        except PermissionError:
            pass
        evidence["devices"].append(row)
    return evidence
