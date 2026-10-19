"""Read-only exact-run API check. Never print credentials or raw parser/network errors."""

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path


def inspect_run(run_id):
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        import netrc

        try:
            values = netrc.netrc(
                os.environ.get("PIXEL_WANDB_NETRC", str(Path.home() / ".netrc"))
            ).authenticators("api.wandb.ai")
            key = None if values is None else values[2]
        except Exception:
            raise ValueError("W&B credential lookup failed") from None
    if not key:
        raise ValueError("Missing W&B credentials")
    query = "query($entity:String!,$project:String!,$run:String!){project(name:$project,entityName:$entity){name run(name:$run){name state}}}"
    payload = dict(
        query=query,
        variables=dict(entity="rl2-group", project="mot-world-model", run=run_id),
    )
    request = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic "
            + base64.b64encode(("api:" + key).encode()).decode(),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except Exception:
        raise ValueError("W&B read-only identity query failed") from None
    if result.get("errors") or not result.get("data", {}).get("project"):
        raise ValueError("W&B identity query did not return an unambiguous project")
    run = result["data"]["project"]["run"]
    return dict(
        run_id=run_id,
        entity="rl2-group",
        project="mot-world-model",
        exists=run is not None,
        state=None if run is None else run["state"],
    )


if __name__ == "__main__":
    for run_id in sys.argv[1:]:
        print(json.dumps(inspect_run(run_id)), flush=True)
