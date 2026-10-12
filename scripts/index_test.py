"""Search edited random clips from every dataset against the full 5-dataset index."""
import glob, io, json, random, resource, subprocess, tarfile, tempfile, time
from pathlib import Path
import pyarrow.parquet as pq
import requests
from remotezip import RemoteZip
from provena_audit.audit import Index

ROOT = "/Users/arthur/Provena/data/index"
DATASETS = ["openvid-1m", "lmms-lab__LLaVA-Video-178K", "ShareGPT4Video__ShareGPT4Video", "Mutonix__Vript",
            "MCG-NJU__VideoChat3-LV116k"]
t = time.time()
ix = Index(*[f"{ROOT}/{d}" for d in DATASETS])
print(f"index: {len(ix.ids):,} clips, {len(ix.video):,} frames, loaded in {time.time() - t:.0f} s, "
      f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB", flush=True)

HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
random.seed(12)
ids = {d: pq.read_table(f"{ROOT}/{d}/clips.parquet", columns=["id"]).column(0).to_pylist() for d in DATASETS}
openvid_zip = json.load(open("/Users/arthur/Provena/data/eval/openvid/zip_index.json"))


def fetch(dataset, cid, out):
    if dataset == "openvid-1m":
        z, m = openvid_zip[cid]
        with RemoteZip("https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/" + z) as zz:
            out.write_bytes(zz.read(m))
        return
    repo = dataset.replace("__", "/", 1)
    file, _, member = cid.partition("::")
    if not member:
        out.write_bytes(requests.get(HF.format(repo, file), timeout=300).content)
    elif file.endswith(".zip"):
        with RemoteZip(HF.format(repo, file)) as zz:
            out.write_bytes(zz.read(member))
    else:  # tar(.gz): stream from the start up to the member
        with requests.get(HF.format(repo, file), stream=True, timeout=300) as r:
            r.raw.decode_content = True
            with tarfile.open(fileobj=r.raw, mode="r|*") as tar:
                for m in tar:
                    if m.name == member:
                        out.write_bytes(tar.extractfile(m).read())
                        return
        raise FileNotFoundError(member)


def pick(dataset, n):
    pool = ids[dataset]
    if dataset == "openvid-1m":
        pool = [c for c in random.sample(pool, 2000) if c in openvid_zip]
    elif dataset == "lmms-lab__LLaVA-Video-178K":  # first clip of a tar (cheap to stream)
        firsts, seen = [], set()
        for c in pool:
            f = c.partition("::")[0]
            if f not in seen:
                seen.add(f); firsts.append(c)
        pool = firsts
    elif dataset == "MCG-NJU__VideoChat3-LV116k":
        pool = [c for c in pool if "::" not in c and "SciVideo_Short" in c]
    return random.sample(pool, n)


queries = [(d, c) for d, n in zip(DATASETS, [3, 2, 2, 3, 3]) for c in pick(d, n)]
results = []
with tempfile.TemporaryDirectory() as tmp:
    for dataset, cid in queries:
        src, q = Path(tmp) / "src.mp4", Path(tmp) / "query.mp4"
        try:
            fetch(dataset, cid, src)
        except Exception as e:
            print(f"SKIP {dataset}: could not fetch ({e})", flush=True); continue
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(src)], capture_output=True, text=True).stdout or 0)
        start = max(0.0, dur / 2 - 3)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", str(src), "-t", "6",
                        "-vf", "scale=iw/2:-2", "-crf", "32", "-an", str(q)], check=True)
        t = time.time(); hits = ix.search(q, min_votes=5); dt = time.time() - t
        own = [h for h in hits if h["id"] == cid]
        others = sorted({h["dataset"] for h in hits if h["id"] != cid})
        ok = bool(own)
        where = f"{own[0]['b_start']:.1f}-{own[0]['b_end']:.1f}s (cut at {start:.1f}s)" if own else "-"
        results.append((dataset, ok, dt))
        print(f"{'OK  ' if ok else 'MISS'} {dataset:32s} {cid[-48:]:48s} {where:28s} {len(hits):3d} hits "
              f"{dt:.2f} s; also in: {', '.join(others) or '-'}", flush=True)
    for f in sorted(glob.glob("/Users/arthur/Provena/data/eval/sources/*.mp4"))[:3]:
        hits = ix.search(f, min_votes=5)
        print(f"NEG  unrelated news video -> {len(hits)} matches {[h['dataset'] for h in hits][:5]}", flush=True)
found = sum(ok for _, ok, _ in results)
print(f"found {found}/{len(results)}; median search {sorted(dt for *_, dt in results)[len(results) // 2]:.2f} s")
