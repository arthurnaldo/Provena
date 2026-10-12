"""List Hugging Face video datasets that host actual video, with size, format, licence and gating.

    python3 scripts/hf_video_discovery.py [N] > data/discovery/hf_video.csv
"""
import csv, json, sys, time, urllib.request
from collections import Counter

API = "https://huggingface.co/api"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
VIDEO = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v")
ARCHIVE = (".zip", ".tar", ".tar.gz", ".tgz")

def get(url):
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(2 ** i); err = e
    raise err

found = []
url = f"{API}/datasets?filter=modality:video&sort=downloads&direction=-1&limit=100"
while url and len(found) < N:
    req = urllib.request.urlopen(url, timeout=60)
    found += json.load(req)
    link = req.headers.get("Link", "")
    url = link.split(";")[0].strip("<>") if 'rel="next"' in link else None
w = csv.writer(sys.stdout)
w.writerow(["id", "downloads", "gated", "license", "video_files", "video_gb", "archive_files", "archive_gb",
            "parquet_gb", "total_gb", "kind"])
for d in found[:N]:
    try:
        info = get(f"{API}/datasets/{d['id']}?blobs=true")
    except Exception:
        continue
    sib = info.get("siblings", [])
    ext = Counter(); size = Counter()
    for s in sib:
        f = s["rfilename"].lower(); b = s.get("size") or 0
        k = "video" if f.endswith(VIDEO) else "archive" if f.endswith(ARCHIVE) or ".tar." in f or "_part" in f.rsplit("/", 1)[-1] else "parquet" if f.endswith(".parquet") else "other"
        ext[k] += 1; size[k] += b
    total = sum(size.values())
    kind = ("video files" if ext["video"] else "archives" if ext["archive"] else "parquet" if ext["parquet"] else "links/metadata only")
    lic = (info.get("cardData") or {}).get("license")
    w.writerow([d["id"], info.get("downloads", 0), info.get("gated", False), lic if not isinstance(lic, list) else "|".join(lic),
                ext["video"], round(size["video"] / 1e9, 1), ext["archive"], round(size["archive"] / 1e9, 1),
                round(size["parquet"] / 1e9, 1), round(total / 1e9, 1), kind])
    sys.stdout.flush()
