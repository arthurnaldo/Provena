"""Fingerprint OpenVid-1M zips on Modal (4 CPUs, one machine per zip), writing provena-audit shards to a volume.

    modal run scripts/modal_openvid.py --test                  # one zip, 3,000 clips: speed and where time goes
    modal run scripts/modal_openvid.py --zips OpenVid_part1.zip,OpenVid_part102
    modal run --detach scripts/modal_openvid.py --all          # every zip (whole and uploaded-in-pieces)

Run from the repo root.
"""

import subprocess

import modal

app = modal.App("provena-openvid-fp")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .pip_install("requests", "stream-unzip")
    .add_local_dir("provena-audit", "/pkg", copy=True, ignore=[".venv", "build", "dist", "**/__pycache__"])
    .run_commands("pip install /pkg")
    .add_local_file("scripts/openvid_fingerprint.py", "/root/openvid_fingerprint.py")
)
volume = modal.Volume.from_name("provena-openvid", create_if_missing=True)


@app.function(image=image, cpu=4, memory=8192, volumes={"/vol": volume}, timeout=3 * 3600, max_containers=100,
              retries=2)
def run_zip(zip_name: str, out: str, limit: int = 0, connections: int = 8) -> str:
    cmd = ["python", "/root/openvid_fingerprint.py", zip_name, "--out", out, "--workers", "4",
           "--connections", str(connections)]
    if limit:
        cmd += ["--limit", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    volume.commit()
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-6:])
    if proc.returncode:
        raise RuntimeError(f"{zip_name} failed:\n{tail}")
    return f"{zip_name}:\n{tail}"


@app.local_entrypoint()
def main(test: bool = False, all: bool = False, zips: str = "", limit: int = 0, connections: int = 8):
    if all:
        import json
        import re
        import urllib.request
        with urllib.request.urlopen("https://huggingface.co/api/datasets/nkp37/OpenVid-1M/tree/main") as r:
            files = [f["path"] for f in json.load(r) if f["type"] == "file"]
        zips = ",".join(sorted({f if f.endswith(".zip") else re.sub(r"_part[a-z]{2}$", "", f) for f in files
                                if re.match(r"OpenVid_part\d+(\.zip|_part[a-z]{2})$", f)}))
    if test:
        names, out, limit = ["OpenVid_part1.zip"], "/vol/fp_test", limit or 3000
    else:
        names, out = zips.split(","), "/vol/openvid_send"
    print(f"{len(names)} zips")
    for result in run_zip.map(names, kwargs={"out": out, "limit": limit, "connections": connections}, order_outputs=False, return_exceptions=True):
        print(result)
