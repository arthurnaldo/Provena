"""Run the one-pass OpenVid fingerprinting on Modal: one cloud machine per zip, many in parallel.

    modal run eval/modal_openvid.py --test           # one zip, to measure speed and cost
    modal run eval/modal_openvid.py                  # all stock zips (~10 at a time)
    modal volume get provena-openvid onepass data/eval/openvid/onepass_cloud   # fetch fingerprints

Each machine streams its zip at datacenter speed, decodes + PDQ-hashes on CPU workers, embeds SSCD on a
GPU, and writes fingerprint shards to the `provena-openvid` volume. Run from the repo root.
"""

import json
from pathlib import Path

import modal

app = modal.App("provena-openvid")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "pkg-config", "build-essential", "libavcodec-dev", "libavformat-dev", "libavutil-dev",
                 "libswscale-dev", "libavdevice-dev", "libavfilter-dev", "libswresample-dev")
    .pip_install("torch", "numpy", "pdqhash", "requests", "stream-unzip", "vpdq")
    .add_local_dir("packages/core/src/core", "/root/core")
    .add_local_dir("packages/fingerprint/src/fingerprint", "/root/fingerprint")
    .add_local_file("eval/openvid_onepass.py", "/root/openvid_onepass.py")
)
volume = modal.Volume.from_name("provena-openvid", create_if_missing=True)
STOCK = ("pixabay", "pexels", "mixkit")


@app.function(image=image, gpu="L4", cpu=16, memory=32768, volumes={"/vol": volume}, timeout=4 * 3600, retries=2, max_containers=10)
def run_part(part: str, limit: int = 0) -> str:
    import os
    import subprocess
    import time
    out = f"/vol/onepass/{part.removesuffix('.zip')}"
    env = {**os.environ, "PYTHONPATH": "/root", "PROVENA_ONEPASS_OUT": out, "PROVENA_MODEL": "/tmp/sscd.pt"}
    cmd = ["python", "/root/openvid_onepass.py", "--parts", part, "--prefix", ",".join(STOCK),
           "--streams", "1", "--workers", "14", "--shard-size", "1000"]
    if limit:
        cmd += ["--limit", str(limit)]
    start = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    volume.commit()
    tail = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines() if "warn" not in l.lower())[-1500:]
    if proc.returncode:
        raise RuntimeError(f"{part} failed:\n{tail}")
    return f"{part}: {(time.time() - start) / 60:.1f} min\n{tail}"


@app.local_entrypoint()
def main(test: bool = False, limit: int = 0):
    index = json.loads(Path("data/eval/openvid/zip_index.json").read_text())
    parts = sorted({p for n, (p, _) in index.items() if n.startswith(STOCK)},
                   key=lambda p: int(p.removeprefix("OpenVid_part").removesuffix(".zip")))
    if test:
        parts, limit = parts[:1], limit or 2000
    print(f"running {len(parts)} parts, up to 10 machines at a time")
    for result in run_part.map(parts, kwargs={"limit": limit}, order_outputs=False, return_exceptions=True):
        print(result if not isinstance(result, Exception) else f"ERROR: {result}")
