"""Audit fingerprints on the provena-openvid volume across many Modal machines, then combine.

    modal run --detach scripts/modal_audit.py                     # /vol/openvid_send -> /vol/openvid_audit/pairs.csv
    modal run scripts/modal_audit.py --machines 20 --send /vol/openvid_send
    modal volume get provena-openvid openvid_audit/pairs.csv data/openvid_pairs.csv

Each machine loads every fingerprint and runs `provena-audit audit --part i --of N`; `combine` joins the parts.
Run from the repo root.
"""

import subprocess

import modal

app = modal.App("provena-openvid-audit")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .add_local_dir("provena-audit", "/pkg", copy=True, ignore=[".venv", "build", "dist", "**/__pycache__"])
    .run_commands("pip install /pkg")
)
volume = modal.Volume.from_name("provena-openvid", create_if_missing=True)
OUT = "/vol/openvid_audit"


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-5:])
    if proc.returncode:
        raise RuntimeError(tail)
    return tail


@app.function(image=image, cpu=32, memory=32768, volumes={"/vol": volume}, timeout=6 * 3600, retries=1)
def audit_part(part: int, of: int, send: str, min_votes: int) -> str:
    import os
    os.makedirs(OUT, exist_ok=True)
    tail = run(["provena-audit", "audit", send, "--part", str(part), "--of", str(of), "--min-votes", str(min_votes),
                "--out", f"{OUT}/pairs_{part:03d}.csv"])
    volume.commit()
    return tail


@app.function(image=image, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=3600)
def combine(of: int, send: str) -> str:
    volume.reload()
    tail = run(["provena-audit", "combine", send, *[f"{OUT}/pairs_{i:03d}.csv" for i in range(of)],
                "--out", f"{OUT}/pairs.csv"])
    volume.commit()
    return tail


@app.local_entrypoint()
def main(machines: int = 10, send: str = "/vol/openvid_send", min_votes: int = 8):
    for result in audit_part.map(range(machines), kwargs={"of": machines, "send": send, "min_votes": min_votes},
                                 order_outputs=False):
        print(result, flush=True)
    print(combine.remote(machines, send))
