"""Run the OpenVid duplicate audit across several Modal machines, against the shards already on the volume.

Each machine loads every fingerprint, but searches only with the clips in its slice (clip id % N == i);
a final step merges the slices.

    modal run eval/modal_audit.py              # 10 machines
    modal volume get provena-openvid audit data/eval/openvid/audit_cloud    # fetch results
"""

import modal

app = modal.App("provena-audit")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("faiss-cpu", "numpy")
    .add_local_file("eval/openvid_full_audit.py", "/root/openvid_full_audit.py")
)
volume = modal.Volume.from_name("provena-openvid")
SLICES = 10


def _run(args: list[str]) -> str:
    import os
    import subprocess
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(["python", "/root/openvid_full_audit.py", "/vol/onepass", "--out", "/vol/audit", *args],
                          env=env, capture_output=True, text=True)
    out = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines() if "searched" not in l)
    if proc.returncode:
        raise RuntimeError(out[-3000:])
    return out


@app.function(image=image, cpu=32, memory=65536, volumes={"/vol": volume}, timeout=3 * 3600, retries=2)
def audit_slice(i: int) -> str:
    out = _run(["--part", str(i), "--of", str(SLICES)])
    volume.commit()
    return out


@app.function(image=image, cpu=4, memory=32768, volumes={"/vol": volume}, timeout=3600)
def combine() -> str:
    volume.reload()
    out = _run(["--combine"])
    with open("/vol/audit/summary.txt", "w") as f:
        f.write(out)
    volume.commit()
    return out


@app.local_entrypoint()
def main():
    for r in audit_slice.map(range(SLICES), order_outputs=False):
        print(r.splitlines()[-1] if r else r, flush=True)
    print(combine.remote(), flush=True)
