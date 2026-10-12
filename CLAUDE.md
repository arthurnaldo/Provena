# Provena

Provena finds where a video came from: given a clip, it finds earlier copies, which part of which video it
matches, and what changed. It matches whole clips (not single images), so trimmed, re-encoded, resized
and (with SSCD) cropped, mirrored or overlaid copies still match. It stores fingerprints and metadata,
never video.

This branch (`restart`) is a fresh start. `main` holds an older design (and a stash of earlier vPDQ work).

## Status (Oct 2026)

- Built and tested: vPDQ fingerprinting, SQLite + FAISS store, Hough-voting matcher, Bluesky connector,
  ingest/query CLI, two evals (real news video, OpenVid-1M dataset audit), PDQ vs SSCD comparison,
  one-pass PDQ + SSCD ingest pipeline, Modal cloud runner.
- Done: full OpenVid-1M stock fingerprinting + all-vs-all audit on Modal (see "Full OpenVid stock audit");
  deduplicated keep/drop lists published on Hugging Face (Oct 9, 2026); customer audit tool `provena-audit`
  built (local folders + S3/GCS, many machines, container image). See "Published" and "Audit tool".
- Outreach started Oct 9, 2026 (see "Outreach log"). No customer conversations yet.
- Direction (Oct 9 evening): leaning toward treating Provena as a portfolio project; see "Direction".
- Cal Hacks: Oct 26, 2026.
- Oct 11, 2026: SSCD dropped for now; PDQ only (CPU machines, no GPU). Planned audit speed-ups: run all zips at
  once on CPU spot machines; later a C++ "vote while searching" kernel (no top-100 cap, no match list, each
  pair once). Estimates only, not measured.
- Business angle chosen to validate (not yet validated): duplicate / edited-copy / fraud checks for video
  data vendors (paid collection, marketplaces, stock libraries). See "Strategy" and "Go-to-market".

## Repo layout (branch `provena-audit`, Oct 11 2026)

Everything below except `data/` was removed on this branch; the old code is on `restart` (commit bac2a9e).
The repo is now one pip package, `provena-audit/`:

```
pyproject.toml, CMakeLists.txt   scikit-build-core + nanobind; deps numpy, pdqhash, fsspec ([cloud]: s3fs, gcsfs)
src/provena_audit/
  fingerprint.py   ffmpeg stream -> PDQ at 4/s + quality -> shards (--part/--of, resumable, temp copies for s3/gs)
  shards.py        send/shard_*.npz (random IDs, durations, per-frame owner/offset/pdq/quality), private/names_*.csv
  storage.py       local / s3:// / gs:// via fsspec
  audit.py         shards -> _kernel.find_pairs -> pairs.csv + printed duplicate time (lesser copy, merged)
  kernel.cpp       vote-while-searching (see "Planned: C++ audit kernel", now built): no FAISS, no top-k cap,
                   3-bin voting once per frame (honest vote counts), every bin >= min votes with ±2 blocking
  cli.py           provena-audit fingerprint <src> | audit <send>
tests/test_openvid.py + pairs.csv   downloads 20 known OpenVid pairs (10 hidden, 10 listed twice)
```
Run: `cd provena-audit && uv sync && uv run pytest -s`. Oct 11: 20/20 pairs found, 0 false matches, min 8 votes
(~2.5 min, mostly downloading).
**Full stock run (Oct 11, Mac, 10 cores, min 8, both directions):** 359,513 clips / 3.92M frames in 1,975 s (33 min),
peak 1.87 GB (incl. loading). 1.10M pairs; vs old audit_pairs.csv PDQ >= 8: 329,841 both, 13,378 old only (96.1%
kept; the losses had 8-15 old votes = double-count inflation), 771,490 new only, of which 647k are segments of the
same listing (old audit never compared those) and 124k cross-listing (no top-100 cap). 10% slice (36k clips): 20 s,
0.66 GB; min 6 keeps 99.3% of old pairs, min 7 97.7%. Eye check (thumbnails, not aligned): new different-ID pairs at
8 votes 8/10 same footage (incl. 1 recolour), 1 template variant, 1 same-shoot-different-moment; min-6-only new
pairs weak (1 copy, 1 likely not, 3 unclear); old pairs recovered at 6: 5/5 copies. Default stays 8 until a
static-footage / clear-winner check exists. Head-to-head on the same 10% slice (Mac): old FAISS top-100 + numpy
vote_pairs 38.5 s, peak 1.06 GB (0.55 GB of it loading; held 1.9M frame matches); new kernel 20 s, 0.66 GB, so
~1.9x faster with ~5x less working memory at this size (old list grows with matches, new doesn't).
**Full-size aligned check of new different-ID pairs (Oct 11):** of 15, ~8 literally the same footage; 4 same tripod
view at a different moment (ship/person/waves differ); 3 variants of one AI-made scene (birds/fog differ). All are
~3 s clips (11 frames max, so 11 votes = every frame). Aligned PDQ distances don't separate them: new "different
moment" 16-30, "copies" 20-35, and pairs the old audit also found (incl. the 60/60-type) 0-30. **Decision (user):
these are redundant for training data and count.** The tool finds duplicate or near-duplicate (redundant) footage;
say that, not "the same footage". The top-100 cap was hiding many of these. An exact-vs-near label is optional.
Side-by-side videos were made in the session scratchpad only (not kept).
Each pair once (Oct 11): A only vs later videos; candidates need min_votes-4 A frames; support() counts distinct
frames on both sides, first copy kept if either side >= min, repeats must pass on A's frames. 10% slice: 9.3 s vs
20 s (2.15x), 10,216 vs 10,305 pairs (-0.9%, 9 of them old-audit pairs); OpenVid test 20/20. Wheel builds
(`uv build --wheel`). Next speed-ups: tighter inner loop (~1.5x), 1/s queries (~4x, changes results).
Fingerprinting decodes in-process with PyAV (Oct 11): frame picking copies ffmpeg's fps filter (slot round(t*4),
last frame per slot, gaps repeat, video ends one frame after its last frame), so hashes are identical to
`ffmpeg -vf fps=4` (51/51 videos, same counts/hashes/quality). 2.5x faster on 3 s clips (no process startup),
~1.07x on long videos. No ffmpeg install needed. Grayscale was tried: only ~80% identical, barely faster; dropped.
OpenVid fingerprinting runner (Oct 11): scripts/openvid_fingerprint.py (zip fetched as 8 parallel 64 MB range
requests, read in order with stream-unzip, clips fingerprinted by a process pool, provena-audit shards with clip names
as IDs, resumable) + scripts/modal_openvid.py (one machine per zip, volume provena-openvid, /vol/openvid_send).
Measured on Modal, 3,000 mixkit clips (2.24 h, ~10 GB) per test: 16 CPU 17.5 clips/s; 8 CPU 20.4/s; 4 CPU 18.5/s
(workers limit ~40% of the time). Each machine tops out at ~50-70 MB/s (16 connections no faster than 8).
4 CPU ~= 48 h of footage per machine-hour, ~$0.005/h of footage (rough Modal rates) vs ~$0.10/h for the old GPU+SSCD
run (~20x cheaper); all of OpenVid-1M ~$10, ~15 min if ~200 machines run at once (max_containers is 10 now).
Full OpenVid-1M run on Modal (Oct 11, 100 machines): stopped by Modal's spend cap ($42.50/month, max settable $20 on
this plan) after 12 zips fully done (154k clips, 0 errors, ~9-11 min/zip at 65-70 MB/s) and ~330 shards saved in
volume provena-openvid:/openvid_send (resumable). AWS alternative (scripts/aws_openvid.sh: S3 bucket
provena-openvid-696865460743, role provena-ec2, Graviton spot machines that install, fingerprint to S3, mark
done/<zip>, terminate): tested on 3,000 clips, c7g.xlarge 8.6 clips/s (CPU-bound; Modal's 4 CPUs = ~8 vCPUs), spot
~$0.07-0.08/h -> ~$0.003/h of footage (~1.6x cheaper than Modal). Account spot quota is 5 vCPUs; increase to 400
requested Oct 11 (case opened). AWS CLI logged in as the root user: create a limited IAM user later.
Oct 11 17:18 PT: quota was already raised to 96 vCPUs; scripts/aws_wait_and_launch.sh launched 24 c7g.xlarge spot
machines for all 186 zips -> s3://provena-openvid-696865460743/openvid_send (done markers in done/, logs in logs/).
Expected ~3-4 h, ~$6. Check with `scripts/aws_openvid.sh status`; rerun `launch N` for any zips without a done marker.
AWS audit: `scripts/aws_openvid.sh audit N [PART ...]` (c7g.2xlarge spot; each copies openvid_send, runs its part,
uploads openvid_audit/pairs_i.csv; machine 0 waits for all parts, combines -> openvid_audit/pairs.csv + summary.txt;
relaunch a lost part by number). scripts/aws_after_fp.sh (running in the background since Oct 11, log
data/aws_after_fp.log) relaunches unfinished zips up to 2x, then starts the audit on quota/8 machines (12 at 96 vCPUs,
est. ~3-4 h, ~$6). Not tested end to end yet; first real run is the OpenVid-1M audit.
GPU quota (Oct 11): G/VT spot was 0; requested 16 vCPUs (request f778c182d60f4468a2cd00cdccdc69aaKZ5ZUNfw) for SSCD.
Spot prices us-east-1: g4dn.xlarge (T4) ~$0.30/h, g6.xlarge (L4) ~$0.56/h. Measured SSCD on the Mac: PyTorch CPU
224-445 ms/frame (no thread scaling), Mac GPU 4.8 ms (224 px) / 11.2 ms (320 px); PDQ 1.2 ms/frame (640x360).
Split audit (Oct 11): `audit --part i --of N` (machine i queries chunks i, i+N, ...; every machine loads all
fingerprints) writes a part file; `combine send/ pairs_*.csv` joins them and prints the summary. 10% stock slice:
3 and 7 parts give the same 10,288 rows as one run; OpenVid test also checks 3 parts + combine == one run.
Lookup (Oct 12): kernel `lookup` searches only a clip's frames against an index (threads take blocks of whole
index videos, separate scoreboards); kernel refactored so audit and lookup share vote/pick/support (audit output
unchanged: 10% slice still 10,288 rows). `Index(send).search(clip)` + `provena-audit lookup`. Against all of
OpenVid-1M on the Mac: ~0.3-0.6 s per search incl. fingerprinting the clip (old first-chunk shortcut ~13 s), same
hits. Edited-clip test (10 random clips): trim 8-9/10, half size+crf35 9/10, caption bar 7/10, mirrored 0/10,
unrelated videos 0 false matches; min 5 vs 8 votes barely differs (misses are flat stock clips and mirrors).
Packaging: MIT, macOS 11+ arm64 wheels cp311-cp314 + sdist. **Published to TestPyPI Oct 12:
https://test.pypi.org/project/provena-audit/ (0.1.0, 5 files)**; installed from TestPyPI in a clean venv and ran
fingerprint/audit/lookup OK. The TestPyPI token was pasted in chat: delete it on test.pypi.org. Not done: Linux/Windows wheels (cibuildwheel), real PyPI, mirror (pdq_dihedral) query flips, SSCD. Dropped: reports, evidence, SSCD,
store/SQLite/FAISS, Bluesky ingest, eval/Modal scripts, minidedup.

Old layout (on `restart`):

```
packages/core/         dataclasses only: Video, Frame, Hit, Match
packages/fingerprint/  vpdq.py: fingerprint_video(path) via Meta's vPDQ
                       frames.py: decode_frames(path) -> (full-size frames, 320x320 frames), one decode;
                       iter_frames(path): streaming full-size frames (long videos)
                       pdq.py: pdq_frames / pdq_dihedral via pdqhash (identical to vPDQ on the same frame)
                       sscd.py: SSCD embeddings (needs torch; never in a process with FAISS)
packages/store/        SQLite (videos, frames) + in-memory FAISS binary index; the only I/O code;
                       pdq_neighbours(): all-against-all PDQ search
packages/matcher/      match(hits): Hough voting on time shift; vote_pairs(): same, vectorised for all-pairs audits
services/ingest/       connectors: bluesky.py, common.py (download -> fingerprint -> store -> delete)
services/audit/        customer audit tool `provena-audit` (fingerprint / audit / evidence), see "Audit tool"
scripts/provena.py     CLI: ingest <file>, query <clip>, bluesky <accounts>
scripts/clip_to_source.py   early experiment: map a clip's frames to source timestamps
eval/run_bluesky.py    AP + Reuters indexed, edited clips queried, NYT as negatives
eval/openvid_audit.py  OpenVid-1M duplicate audit (index / ingest / stream / dedup)
eval/sscd_compare.py   PDQ vs SSCD on the Bluesky eval clips (embed step + compare step)
eval/openvid_sscd.py   SSCD on the 9,191-clip OpenVid sample, compared with PDQ
eval/openvid_onepass.py  one-pass ingest: stream zips (resumable range requests) -> decode once ->
                       PDQ (full-size, 4/s) + SSCD (2/s) -> fingerprint shards (npz); resumable
eval/modal_openvid.py  runs openvid_onepass on Modal: one machine per zip (L4 GPU + 16 CPU), 10 at once
eval/openvid_full_audit.py  all-vs-all PDQ + SSCD audit over the shards (--part/--of slices, --combine)
eval/modal_audit.py    runs openvid_full_audit on 10 Modal machines (cpu=32, 64 GB) + combine
eval/openvid_release.py  builds the published keep/drop lists from audit_pairs.csv
tests/                 end-to-end test
data/                  gitignored: databases, cached eval videos, OpenVid index, SSCD model
```

Run with uv: `uv sync && uv run pytest packages services tests` (25 tests pass, Oct 9). Needs ffmpeg +
pkg-config (vPDQ builds from source; it's the `fingerprint[vpdq]` extra, so the audit client doesn't need it).
PyTorch is not a project dependency; SSCD steps use `uv run --with torch ...`. Secrets go in `.env`
(gitignored), never in chat.

### Rules
- `core` depends on nothing. `fingerprint` and `matcher` are pure: no DB, network, or platform code.
- `store` does all I/O; nothing else knows about SQLite, FAISS, or integer IDs.
- Never store video. Download, fingerprint, delete.
- Clean sourcing only: no scraping that breaks platform terms. Our own provenance must be beyond question.

## Audit tool (`services/audit`, CLI `provena-audit`)

- `fingerprint <folder|s3://|gs://>`: runs on the customer's machines; PDQ at 4/s via streaming decode
  (`fingerprint.iter_frames`, so hour-long videos don't fill memory). Writes `send/` (shards: random video IDs,
  durations, PDQ + quality + offsets; ~0.5 MB per hour of footage; no names, images or video) and `private/`
  (names_<part>.csv: ID -> path). Many machines: `--part i --of N` (stable md5 split of paths; defaults from
  Cloud Run / AWS Batch / k8s indexed-job env vars); each part resumes on its own. Remote videos go to a temp
  file and are deleted. Paths via fsspec (`audit[cloud]` extra adds s3fs, gcsfs).
- `audit <send>`: our side, IDs only. PDQ all-pairs (`store.pdq_neighbours`) -> `matcher.vote_pairs` (vectorised
  Hough voting with matched ranges) -> min 8 votes -> duplicate time charged to the lesser (shorter) copy with
  intervals merged -> groups -> report.json + report.html (cost at $60-240/h by default).
- `evidence <report.json> <source>`: customer side; real names + side-by-side frames rendered from their files.
- Container: `services/audit/Dockerfile` (build from repo root; verified building and running on Modal).
- Tested: 49/50 known OpenVid duplicate pairs found in a 200-clip folder (the miss had 7 votes vs 8 needed);
  fake-S3 (moto) run with 3 parts; `services/audit/tests`.
- Not built: SSCD (edited copies), loop/repeat detection inside one video, upload service.
- `fingerprint` no longer needs vPDQ: it's the `fingerprint[vpdq]` extra, imported lazily.
- **Known issue (found Oct 11): `vote_pairs` double-counts frames.** It counts distinct query frames per bin,
  then scores a bin as own + the ±1 bins' counts, so a frame that lands in two neighbouring bins (common: stored
  frames are 0.25 s apart, so a shift near x.5 matches at e.g. 5.05 and 5.55) is counted twice. Audit votes can
  be up to ~2x inflated ("8 votes" may be ~4-5 frames). `matcher.match` (query path) is correct: it counts
  distinct frames across all three bins. Precision still held (60/60), but fixing it may need a lower minimum
  (~5-6) to keep recall; re-test on the 200-clip folder.
- Other audit gaps: top-100 cap per frame (look-alikes/static shots can push a real copy out); only the best bin
  per pair is kept (a clip used twice in one video counts once); no static-footage check (frozen clips can fill
  every bin past the minimum); every pair is searched both ways.

### Planned: C++ audit kernel (not built; plan agreed Oct 11)
Replaces `pdq_neighbours` + `vote_pairs` in `audit.find_pairs`; no FAISS, no fallback.
- `packages/votekernel/`: C++17 + nanobind, built with scikit-build-core; one function
  `find_pairs(pdq, owner, offset, quality, radius=31, min_votes=8)` -> a, b, votes, shift, a/b start/end.
- "Vote while searching": frames sorted by video; each thread takes query videos. Per video A, a scoreboard
  (B, shift bin) -> votes + first/last times. For each A frame (quality >= 50) vs every stored frame: popcount of
  4 XORed 64-bit words; skip > 31 bits and same video; bin = floor(shift + 0.5); **vote into bins bin-1, bin,
  bin+1, each at most once per A frame** (fixes the boundary split and the double count; score = the bin's own
  count, no neighbour adding afterwards; try 2 bins later, real copies align within ~0.02 s). Keep every bin
  >= min votes, best first, blocking bins within ±2 of each accepted one (with 3-bin voting one copy spreads
  over ~4 bins, so ±1 would report it twice; `match()` also uses ±2); repeats further apart still count, median shift from supporting matches, clear,
  next video. No top-k cap and no match list, so memory is one small scoreboard per thread.
- std::thread, not OpenMP (OpenMP clash); portable flags (x86-64-v2 / plain ARM), no `-march=native` in wheels.
- Each pair once (A vs later videos only) only if tests show identical results.
- Checks: hand-built cases (copy, repeat at two shifts, scattered look-alike, same video) vs `match()`;
  existing end-to-end test; 200-clip folder (>= 49/50); full stock shards vs `audit_pairs.csv` (same pairs +
  a few from the removed cap, extras checked by eye); log time and memory.
- Estimates (not measured): ~2x faster audit (each pair once, no sort), much less memory, enables low-memory
  spot machines. All of OpenVid-1M PDQ-only: ~1 h fingerprinting on ~200 CPU spot machines + ~10-25 min audit,
  ~$35-60, vs ~16 h / ~$220 before. Mac audit ~27 h -> ~12-13 h. Assumes ~1e9 comparisons/s per core.
- Then: PDQ-only switch in `eval/openvid_onepass.py`, time one zip; cibuildwheel wheels if the full tool goes
  on PyPI (open decision; customer-side-only means the kernel stays in our container).

## Data model (`packages/core`)

- `Video(platform, native_id, url, channel_id, posted_at, deleted_at=None, video_id=None)`.
  `video_id` is an internal int; `(platform, native_id)` is the unique natural key. `channel_id` is a
  stable platform ID, never a handle. Timestamps are tz-aware UTC.
- `Frame(offset, pdq, quality, video_id=None, frame_id=None)`: IDs are None until stored and for query clips.
- `Hit(query_offset, video_id, offset, distance)`, `Match(video_id, clip_range, video_range, shift, votes)`.
  `shift` is the median of (stored − query offset) over supporting hits, so it's precise (7.25 s, not 7).

## How matching works

1. **Fingerprint:** vPDQ samples frames and returns a 32-byte PDQ hash + quality (0–100) + timestamp.
   - Stored videos: every 0.25 s (`STORED_SECONDS_PER_HASH`). Queries: every 1 s (`QUERY_SECONDS_PER_HASH`).
   - vPDQ rounds the interval down to whole frames (0.25 s at 30 fps = every 7 frames = 0.233 s).
2. **Search:** FAISS `IndexBinaryIDMap(IndexBinaryFlat(256))`, k-nearest (k=100, which also caps hits per
   frame), keep Hamming distance ≤ 31. Only frames with quality ≥ 50 are indexed or queried; all frames are
   kept in SQLite. The index is rebuilt from SQLite on open (~40 ms for 100k frames).
3. **Match (Hough voting):** bin `stored_offset − query_offset` to 1 s; score each bin by distinct query
   frames in it and its ±1 neighbours; keep ≥ 5 votes; suppress neighbouring bins of an accepted match; ties
   go to the bin with more votes of its own.

### Why these choices (learned the hard way)
- **Sampling phase:** at 1 frame/s stored, a clip cut 0.5 s off the grid matched 0/20 frames. Stored at
  0.25 s fixed it at every offset tried. Queries can stay at 1/s.
- **Bin edges:** a true shift of 20.5 s splits votes between bins 20 and 21; neighbour-bin scoring fixes it.
- **Python `round()` is banker's rounding**; use `floor(x + 0.5)` for shift bins.
- **vPDQ's own matcher is not used:** it treats videos as an unordered bag of hashes, can't give time
  ranges, and its percentage thresholds reject subclips. Our matcher is temporal Hough voting (the standard
  technique, e.g. VCSL's "HV", Shazam's offset histogram).
- **PyTorch and FAISS can't load in one process** (each bundles OpenMP and they crash). Keep SSCD embedding
  in its own process; ingest workers write shards, a separate loader builds the indexes.
- **PDQ must see the full-size frame.** PDQ on a pre-shrunk 320x320 frame differs from vPDQ by ~20 bits
  (median); on the full-size frame it's identical (0 bits on the same frame, median 6 when sampling moments
  differ slightly). Decode once, keep a full-size copy for PDQ and a small copy for SSCD.
- **SSCD needs a bigger shortlist than PDQ** (k≈400, or drop same-source neighbours before the cut):
  with k=100 it missed 1,998 of PDQ's pairs on OpenVid because near-identical same-scene frames crowd out
  the real copy; with k=400 it finds 99.4% of PDQ's pairs.
- **Single long downloads get cut.** Stream 40 GB zips with resumable HTTP range requests, not one
  connection; per-clip range requests are latency-bound (~3–4 clips/s).

## Results

### Real news video (eval/run_bluesky.py)
20 AP + Reuters Bluesky videos indexed; 12 s clips at random off-grid starts; 9 edit types; 10 NYT clips
never indexed as negatives. Same clips, PDQ vs SSCD (eval/sscd_compare.py):

| Edit | PDQ | SSCD |
|---|---|---|
| re-encode | 100% | 100% |
| half size / heavy compression / brightness | 91% | 100% |
| crop 10% | 0% | 100% |
| logo box | 36% | 100% |
| caption bar (bottom 12%) | 73% | 100% |
| mirrored | 0% | 100% |
| letterbox (vertical video in 16:9) | 0% | 0% |
| letterbox after cutting black bars (ffmpeg cropdetect) | 91% | 100% |
| NYT negatives | 0 false matches | 0 false matches |

- Median shift error 0.02 s. Small test (11 clips per edit); false matches at scale not yet measured.
- SSCD: Meta's `sscd_disc_mixup` TorchScript (MIT), 320×320 input, ImageNet normalisation, 512-d, cosine
  ≥ 0.75 = copy (Meta: ~90% precision on DISC). Same voting as PDQ.
- Planned fix for both: detect and cut black bars before fingerprinting (done query-side).

### OpenVid-1M audit (eval/openvid_audit.py)
OpenVid-1M (ICLR 2025, Hugging Face `nkp37/OpenVid-1M`, labelled CC BY 4.0): ~1.02M clips, 2,051 hours,
7.5 TB in ~200 zips. Clips are pulled with HTTP range requests (remotezip) and deleted after hashing.

From metadata alone:
- **84% of the footage (1,724 h, 61.5% of clips) is cut from 332,087 YouTube videos**, despite the
  CC BY 4.0 label. Rest: Pixabay 211 h, Pexels 50 h, CelebV (celebrity faces) 56 h, Mixkit 5 h.
- **54,096 clips (5.3% of clips, 2.0% of footage, 15% of stock clips) are the same stock source listed
  under 2+ category names.** Confirmed as identical footage by fingerprinting.
- Category labels are sometimes wrong (a "rice terraces" clip shows a mountain over flooded fields).

Fingerprint audit of 9,191 stock clips (6.3k targeted at multi-listed sources + 3k random; biased sample):
- Recall on known copies: 2,974 / 3,047 flagged. Of the 73 "misses", 37 weren't really copies (listings
  segment the source differently), so real recall ≈ 98.8%. Real misses: 31 flat/low-detail clips (sky)
  with < 5 frames above quality 50; 5 static shots crowded out by the k=100 hit cap.
- Found what metadata can't: 34 pairs of the same footage under different IDs. Every pair checked by eye
  was a true copy; no false matches seen.
- **Correction:** pairs first reported as "same footage on two stock sites" (e.g. `pexels_..._3089895` vs
  `pixabay_..._3089895`) share the same numeric ID: OpenVid files some Pexels videos under "pixabay_..."
  folder names. They are one video listed twice (a labelling duplicate), not footage found on two sites.
  The audit now treats a matching ID as "same id, listed twice" regardless of the site prefix.
- Verified side-by-sides: `data/eval/openvid/verified_pairs/`.

SSCD on the same 9,191 clips (eval/openvid_sscd.py):
- PDQ or SSCD catches 99.6% of the 3,047 known copies (SSCD recovered 61 of PDQ's 73 misses: flat skies,
  crowded static shots).
- SSCD-only matches are mostly **same scene, not same footage**: ~21k pairs of different moments of the same
  shot, 47 pairs of the same shoot / different take, and only 2–3 genuinely new hidden copies.
- SSCD similarity (time-aligned, mean over frames): unedited copy ~1.00; **edited copy ~0.77–0.84** (min
  ~0.64; letterbox 0.38 before bar removal); same shot, different moment ~0.81 (0.36–0.97); unrelated ~0.05.
  At 0.75, 46% of same-shot-different-moment pairs pass. **No threshold separates edited copies from
  same-scene look-alikes**: SSCD finds candidates, other checks must confirm (see match flow).

### Full OpenVid stock audit (eval/openvid_full_audit.py on Modal, Oct 9 2026)
All 359,513 stock clips (Pixabay, Pexels, Mixkit; ~266 h), PDQ + SSCD, all against all, 10 machines.
Output: `data/eval/openvid/audit_cloud/audit_pairs.csv` (1.09M pairs; raw "any match" 47% is NOT a
duplicate rate: it includes different segments of one source and SSCD look-alikes).
- **Duplicate footage (conservative): ~82,800 extra clips = 23% of stock clips (~61 of 266 h).**
  - Labelling duplicates (same source ID + segment under several names/site prefixes): 72,365 extra clips
    (20.1%), fingerprint-confirmed (name count predicts 73,409).
  - **Hidden duplicates (different IDs, same footage; only fingerprints find these):** 61,903 PDQ pairs
    with >=8 votes; **10,433 extra clips (2.9%) beyond the labelling duplicates** (eval/openvid_release.py:
    each pair drops at most one clip, never one whose partner is already dropped). Union-find gives more
    but over-counts by chaining through static footage (groups of 200+ clips from 2 IDs).
  - Clean release (keep/drop lists + README): `data/eval/openvid/release/`, published (see "Published").
- **Precision: 60/60 random strong hidden pairs were the same footage by eye** (>=95% at 95% confidence).
  Includes colour-graded copies, time-shifted cuts, partial overlaps. Many are near-sequential IDs (same
  uploader posting twice), so say "duplicate footage", not "stolen".
- SSCD-only different-ID pairs (196k): 50 sampled. At full size nearly all are the **same shoot at a
  different moment** (camera moved, animals moved) or **template variants** (same stock animation with a
  different colour/background); minimum PDQ distance mostly 40-130. Not copies, so not counted. Small
  3-frame thumbnails make them look identical: always check full-size frames.
- Not yet counted: weaker PDQ pairs (3-7 votes). The edit-undoing confirmation step (SSCD -> undo edits ->
  PDQ) is designed but not built; the SSCD sample suggests it would add little on stock footage.
- How the precision check was done: random.seed(42) 10 pairs + seed(7) 50 pairs of different-ID PDQ pairs with
  >=8 votes; clips fetched with remotezip; 3-frame contact sheets per clip plus full-size frames for doubtful
  ones. Side-by-side videos of the non-exact matches (colour-changed elephant/sunset, time-shifted zebras,
  girl, beach ball, forest, fountain partial overlap): `data/eval/openvid/borderline/`. The colour-changed
  ones are the best demo/LinkedIn material.
- Run time and cost: fingerprinting took 24–43 min per zip (28 zips, 10 machines at a time, ~2 h total);
  the audit took ~26 min on 10 machines. Fastest possible ≈ 1 h end to end (28 machines, one per zip).
  Cost estimate ~$40–60 total (check the Modal dashboard for the real number).

### Speed and size (measured)
- Query: 0.6 ms median per clip (search + match) on 100k frames; all-vs-all audit of 9,191 clips: 23 s.
- Storage: 78 bytes per frame in SQLite; FAISS ~40 bytes per frame in RAM.
- Ingest: ~3.5–4 clips/s via range requests, network-bound (CPU mostly idle). `stream` mode (one
  sequential download per zip) is built but not yet benchmarked; needs retry on dropped connections.
- SSCD on the Mac GPU (MPS): ~100 frames/s at batch 64–128 (batch 256 collapses to ~15/s).
- One-pass decode + PDQ: ~0.2 s CPU per 2.7 s stock clip (ffprobe 0.03, decode 0.12, PDQ 0.04).
- Home internet from Hugging Face: ~2–4 MB/s total regardless of parallel streams, so a Mac run of the stock
  part would take ~27 h. On Modal: ~8 clips/s per machine (test: 2,011 clips in 4.3 min, 0 errors); the
  stock part on 10 machines ≈ 1.5–2 h, est. ~$20–30 (~$1.8 per machine-hour; check the Modal dashboard).

### Projections (estimates, not measured)
- All of OpenVid at 4 frames/s: ~30M frames, ~2.4 GB on disk, ~1.2 GB RAM (~5 GB with MultiHash).
- Per 1M hours of video: ~1.1 TB disk, ~0.6 TB RAM (flat) or ~2.4 TB (MultiHash).
- Brute-force PDQ search is fine to ~50–100M frames; switch to `IndexBinaryMultiHash` beyond that (exact
  for radius 31 if chunks/flips are set right; ~4× RAM). Ingest, not search, is the bottleneck.
- SSCD: ~2 KB/frame raw, ~64 B compressed; needs approximate search (IVF-PQ/HNSW) past a few million
  frames; ~$0.001 per hour of footage on a rented GPU at 1 frame/s.

### Scaling the all-vs-all audit (learned planning FineVideo)
- All-vs-all search grows with the square of frames: 13x the footage ≈ 170x the comparisons. FineVideo at
  4/s ≈ 50M frames vs ~4M for OpenVid stock. This is infrastructure, not matching quality. Fixes before a
  run that size: query at 1/s (stored stays 4/s; the Bluesky eval already queried at 1/s), fewer SSCD frames,
  `IndexBinaryMultiHash` for PDQ (exact, faster). A reverse lookup (one clip vs an index) stays cheap.

### Lookup index with metadata (Oct 12 2026)
Goal: a public page where people check whether their footage is in AI training datasets. Per dataset, a folder
data/index/<dataset>/ with send/ (fingerprint shards; openvid-1m links to data/openvid_send), manifest.json
(dataset, HF source + revision d8a63bd, stated licence, counts), clips.parquet (24 MB; one row per clip in shard load
order: id, zip, source_platform, source_id, source_url, segment, duration) and captions.sqlite (~1 GB; OpenVid's CSV
keyed by id). Built by scripts/openvid_metadata.py from the shards + OpenVid-1M.csv, no re-fingerprinting.
`Index(*folders)` loads several datasets (metadata looked up only for hits; pyarrow = `[index]` extra) and
`provena-audit lookup --index A B` labels hits by dataset. OpenVid-1M: load 4.9 s, search 0.46-0.88 s, hits carry
licence, zip, YouTube/stock source URL. OpenVid names give the YouTube ID but not the exact time in the source video.
Data/ is gitignored: back up data/index to S3 if needed.

### Full OpenVid-1M audit (Oct 11 2026, AWS, provena-audit C++ kernel, PDQ only, min 8 votes)
Fingerprinting: 186 zips on 24 c7g.xlarge spot machines, ~4.4 h; audit: 37 c7g.2xlarge, ~1.2 h incl. setup; total
roughly ~$10-12 (check the AWS bill). Files: data/openvid_send/ (1.3 GB shards, 1,069 files), data/openvid_audit/pairs.csv
(1,220,522 rows) + summary.txt; also in s3://provena-openvid-696865460743/.
- **1,019,956 clips, 2,050.8 h: 155.6 h duplicate or near-duplicate footage = 7.6%** (charged to the shorter copy,
  overlaps merged). By part: Pixabay 106.1 of 211 h (50.4%), Pexels 10.1 of 50 h (20%), Mixkit 1.9 of 5 h (35%),
  stock total ~118 of 266 h (~44%); YouTube 36.2 of 1,728 h (2.1%); CelebV 1.3 of 56 h (2.4%).
- By kind (overlapping): segments of the same listing / same source video 83.7 h (4.1%; stock 75.8 h; the old audit
  never compared these); same video listed twice 53.8 h (2.6%); different source videos 41.4 h (2.0%: YouTube 29.5 h
  over 106k pairs, Pixabay 10.8 h). Excluding segments, stock ~60-65 h, consistent with the published ~61 h / 23%.
- Eye check (aligned full-size frames, random seed 11): YouTube different-video pairs: 10/14 real reused footage
  (cartoon compilations, the same segment uploaded twice, sports, cooking, theme park), 2 identical channel end-screens
  (MSNBC/NBC), 1 recurring promo segment, 1 not a copy (same studio, different recording). CelebV different-video pairs
  5/5 the same interview footage. Small sample; say "duplicate or near-duplicate footage".
- Source keys: YouTube clips are named <11-char video ID>_<n>_<start>to<end>.mp4 (IDs can contain "_"); CelebV
  celebv_<11-char ID>_<n>.mp4. Pre-flight check of 30,688 AWS clips: 97% of old stock pairs re-found.



- **Match flow:** PDQ search → (if no confirmed copy, or full provenance wanted) SSCD search → verify each
  SSCD-only candidate by re-running PDQ on the query after undoing likely edits (8 flips/rotations via
  `pdq_dihedral`, black-bar removal, blurred side panels, picture-in-picture crop; speed via re-voting at
  0.5–2×) + SSCD timing sharpness → tiers. Ingest stores PDQ + SSCD only; region hashes, crop-scale hashes
  and feature points are deliberately not stored for now (overlays and crops land in likely-copy/review).

- **Keep PDQ alongside SSCD.** PDQ: exact, cheap, stable forever (no model upgrades), industry-standard
  format for sharing (GIFCT, ThreatExchange), precise timing. SSCD: robust to crops, mirrors, overlays.
- **Match tiers:** PDQ+SSCD = confirmed copy. SSCD only + PDQ matches after mirror/bar-crop/mask = confirmed
  edited copy (edit named). SSCD only, strong similarity, frame-exact, has motion = likely copy. SSCD only,
  moderate, static, or several shifts match = human review.
- **SSCD will run on nearly everything in a submission gate** (a PDQ miss can't tell new footage from an
  edited copy). Fine on GPUs; the cascade (PDQ first, SSCD on misses) only pays off for reverse search.
- **False matches at scale come from continuously similar footage** (static shots, same viewpoint, shared
  intros, slow pans), not chance; voting kills coincidences. Counter with similarity strength, frame-exact
  alignment, clear-winner check, motion check, PDQ+SSCD agreement. Don't just raise the vote minimum: it
  kills short clips (OpenVid stock clips have 11 fingerprints) and partial reuse.
- **Gate design:** query first; add only footage that's new. Flagged copies become pointers, not frames
  (no crowding, no wasted space).
- **Test design:** edited copies query an index containing their originals (gate-style, the stricter
  number); audit-style "either side" recall is a secondary number.

## Direction (Oct 9, 2026, evening)

The user's reasoning (from a separate conversation) for wrapping up:
- Tech is a commodity: Content ID (2007), Meta's open PDQ/TMK/vPDQ/HMA, Pex/Vobile, reverse video search
  tools launched in 2026 (FrameTrace, reverse-videosearch.com), Hough voting is published.
- It's been technology-first: each pivot (AI detection -> news provenance -> training-data dedup ->
  robotics) was the same engine looking for a buyer; nobody has said "I need this" or paid.
- Video dedup for training data is crowded (Mixpeek ranks tools; Cleanlab, Voxel51, Encord; Claru sells
  dedup for robot data itself).
- Robotics labs say they filter/dedupe in-house and pay for diversity; Figure built its own collection app
  over vendor quality issues; for robot action data, dedup is better on trajectories than video.
- Egocentric contributor pay is low ($1–10/h), so duplicates cost vendors little in payouts; the real cost
  is delivery quality, which is broader than duplicates.
- EU AI Act training-data summaries are inventory/paperwork, not frame-level matching.
- Considered next: robotics diversity/quality audit (environments, hand visibility, blur, task coverage via
  DINOv2/AnyLoc, hand detectors, VLMs): judged the same trap (off-the-shelf, Voxel51/Encord could add it,
  labs do it in-house); only gap is an independent auditor buyers trust.
- Plan: portfolio project (heads-up to OpenVid authors, HF list, GitHub code, LinkedIn/X post), Cal Hacks to
  practise finding a problem fast, serious job search, look for a pain-first idea.

Claude's take (given to the user):
- Agree the tech is a commodity, but commodity tech doesn't kill a business; lack of pull does, and that
  hasn't been tested: outreach started the same day (a couple of messages), so "no responses" is no
  evidence. Recommended: stop building, but run the already-defined test (~20 targeted conversations,
  ~2 h/day for 2 weeks) in parallel with the portfolio plan; close it with evidence, not a guess.
- If the robotics points hold, marketplaces/libraries are the better remaining test than egocentric vendors.
- No clearly paying unclaimed use found: piracy (Vobile, Pex), likeness/scam ads (Loti, Vermillio: proof
  buyers exist there), stock theft (Pex) are taken. Unverified niches worth a curiosity call only: video
  e-discovery dedup (body-cam/deposition footage), insurance-claim media reuse.
- Diversity audit: don't build; in robotics conversations ask "what's hardest about buying/evaluating
  data?" and only act if diversity comes up unprompted.
- More valuable than it seems: the OpenVid audit as a short arXiv/workshop technical report; blog posts on
  the measured lessons (sampling phase 0/20, full-size PDQ, SSCD can't separate edits from same-scene,
  thumbnails mislead); `provena-audit` as open source; the Community Notes trend (4.3k -> 32k misleading-video
  notes) as a data story; job-search evidence of end-to-end infra ownership and verification discipline.
- The user's summary said "~4% hidden duplicates": the correct figure is ~3% (10,433 clips).

## Packaging (PyPI)

- **Teaching version `minidedup`** (`practice/minidedup/`, not part of Provena): ~90-line brute-force CLI
  (ffmpeg pipe -> pdqhash -> all-frames comparison -> shift voting, min 8 votes; `minidedup scan <folder>`).
  Packaged with hatchling + `[project.scripts] minidedup = "minidedup.cli:main"`; wheel built with
  `uv build`, installed into a clean venv and verified (found the 3 copies of one clip, ignored an unrelated
  clip). Name free on PyPI and TestPyPI. Next: the user uploads to TestPyPI with their own token
  (`read -s` + `UV_PUBLISH_TOKEN=... uv publish --publish-url https://test.pypi.org/legacy/`).
- **`provena-audit` on PyPI (planned, not done):** name `provena-audit` is free (`provena` is taken).
  Plan: rename the audit project to `provena-audit` (import name stays `audit`); bundle the internal
  packages core/fingerprint/matcher/store into the wheel (hatch `force-include`), list only public deps
  (numpy, pdqhash, faiss-cpu, fsspec; `cloud` extra: s3fs, gcsfs); clean `__pycache__` before building;
  test fingerprint -> audit -> evidence in a clean venv; README says install ffmpeg first (or use the
  container); recommend `pipx install provena-audit` (generic module names like `core`/`store` could clash
  in shared environments; later rename to `provena_*`). Open decision for the user: publish the full tool
  (matching code becomes public) or only the customer side (fingerprint + evidence).
- Hugging Face's parquet-converter bot opened an automatic discussion on the dataset: harmless, can be
  closed. Optional: add `configs` to the dataset README so the viewer shows "keep" and "drop" as two tabs.

## Published (Oct 9, 2026)

- **Hugging Face dataset (public):** https://huggingface.co/datasets/arthur-nn/openvid-stock-dedup
  (README with numbers + method, keep_list.txt 276,715 clips, drop_list.csv 82,798 rows; file names only,
  CC BY 4.0). Account `arthur-nn`. The write token was pasted in chat: rotate it; it's in `.env`.
- **Note for the OpenVid discussion tab** drafted (friendly, links the dataset); posting is up to the user.
- **Google Docs:**
  - v1 explainer (wrong "two stock sites" claim; delete):
    https://docs.google.com/document/d/1qEW_6Br06fFW6-5rod2573qHjwrIJXvi7WhsaQyZ9sM/edit
  - v2 explainer: https://docs.google.com/document/d/1g0lZYeLTW78P48BIjvDR6jrtgAbHrekyGUm0cbdITEI/edit
  - One-pager draft: https://docs.google.com/document/d/1efnZvMsf_1D8YW01K4xg_G5FifQ3prkppA6BOxy4XOc/edit
  - Both v2 and the one-pager still say hidden duplicates are "3.6–4% / 13,000–14,500 clips"; the correct
    figure is **2.9% / 10,433 clips** beyond the labelling duplicates. The Drive tool can't edit doc bodies:
    make corrected copies. The one-pager's contact name "Atharva" was a guess from the email; confirm.
- **Claims that are safe to make publicly** (checked against the user's LinkedIn drafts, which repeatedly
  overclaimed): "audited all 359,513 stock clips (~266 h) of OpenVid-1M (1M+ clips, ~2,050 h)"; "~23% are
  duplicate footage"; "~20% the same video under different names"; "~3% (10,000+ clips) hidden duplicates
  under different IDs: recoloured copies, shifted cuts, partial overlaps"; "60/60 checked by eye"; "a few
  hours on rented cloud machines". NOT safe: "entire dataset", "5,000 hours", "13/17 minutes", "24%",
  "4% hidden / 13,000+", "stolen". Privacy line: "We never see or store your video" / "Your footage never
  leaves your servers" (true only with the client tool running on their machines).

## Research findings

### Community Notes (X's public data, 3.1M notes, downloaded Oct 2026)
- Misleading-video notes (recycled, wrong place, game, AI): ~4.3k (2023) → 9.8k (2024) → 12.5k (2025) →
  32k (2026 to early Oct).
- Last 12 months: AI-generated 22.2k, old/recycled 11.3k, wrong place 3.5k, video game 1.8k.
- Originals cited in recycled-video notes: X 3,400, YouTube ~1,950, Instagram 503, Facebook 393, TikTok 282,
  Reddit 159, Telegram 15, Bluesky 1. Most recycled footage is 0–3 years old.
- ~4.6k notes/year defend real footage as real; only ~15% cite major news outlets; most cite the original
  uploader (full podcast episode, influencer reel, other angles).
- X source accounts are concentrated: top 1,000 accounts ≈ 59% of X-cited originals. YouTube is a long tail.
- Of AI-video notes: ~10% cite a disclosure (label, watermark, creator tag), ~8% name the tool, ~2% find
  the original creator. Reposts often crop watermarks.

### Platforms and access
- **Bluesky:** open public API and firehose (Jetstream), no key; ~35k video posts/day (1-minute sample).
  Use `indexedAt`, not `createdAt` (client-set, can be backdated). Deletions arrive as events.
- **X:** pay-per-use API since Feb 2026: $0.005/post read, $0.015/post create, $0.20 if it contains a
  link, 3M reads/month cap, then Enterprise (~$42k+/mo). Check the developer terms on keeping deleted posts.
- **Reddit:** self-serve API closed (Responsible Builder Policy); manual approval, 2–4 weeks; new requests
  reportedly closing Oct 31, 2026; terms require deleting content deleted on Reddit. Dropped.
- **Telegram:** user-account API, history backfill, forward metadata = free origin ground truth. Not built.
- **TikTok / Instagram:** effectively closed. **YouTube:** downloading is against its terms.
- **C2PA** already defines "soft binding" (watermark/fingerprint lookup) to recover stripped credentials;
  it's image-first and whole-file. Provena's edge is clip-level matching of edited video and unsigned content.

### Existing tools
- **The Atlantic "AI Watchdog"** (Sept 2025): searchable lookup of 15M+ YouTube videos across a dozen+ AI
  training datasets (Microsoft, Meta, Snap, Tencent, Runway, ByteDance), by creator/channel, via YouTube IDs.
  **Proof News** (2024): lookup for the YouTube-subtitles dataset (173k videos). So ID-based "is my video in
  an AI dataset?" is taken. Gap Provena fills: footage-level matching (re-uploads, non-YouTube footage,
  edited copies, exact time ranges, hosted datasets they don't cover).
- **YouTube** Content ID / Copyright Match Tool / likeness detection only look at YouTube itself.
- **Lightly** (dataset curation/dedup), **ProvenanceAI** (whole-file hashes in your own library), **Pex/Vobile**
  (cross-platform matching), **DATA Foundation Trace** (on-chain paperwork registry, June 2026),
  **Fairly Trained** (nonprofit certification of licensed-only models; small uptake, no big labs).

### Strategy (current conclusion)
- Provena is **not an AI detector**. It answers "what's already known about this video": origin, spread,
  what was cut, existing labels. Brand-new AI video from a random account can't be caught by matching.
- Misinformation provenance: real, growing demand, but grant-sized money and an access wall (originals
  live on X/YouTube/TikTok). Best as the mission/demo side: known-fakes database, "what was cut" (full
  speeches, podcasts via open RSS), Community Notes support, Bluesky labeler, Berkeley Human Rights Center.
- **Business angle to validate: video training-data provenance / fraud gate.**
  - Tailwinds: Bartz v. Anthropic ($1.5B; liability on acquisition, not training), YouTube creators suing
    Runway, OpenAI, Nvidia, Apple, Amazon, ByteDance, Meta over scraped video; 140 US AI copyright suits by
    Sept 2026; EU AI Act training-data rules enforced from Aug 2026; insurers adding AI exclusions.
  - Evidence the problem exists: Adobe Firefly's "licensed" training set was ~5% AI images; stolen footage
    resold on stock sites; marketplace provenance is paperwork (Troveo), not content checks.
  - Evidence against: Firefly contamination had no commercial consequence; buyers rely on indemnities;
    labs have an incentive not to know; no named buyer of technical verification yet.
  - Sharpest form: **paid video collection** (robotics/egocentric, world models, screen recordings), where
    vendors pay per accepted hour and lose money to duplicates, split recordings, re-uploads, looped/padded
    footage, AI clips, and footage sold to several vendors. They want to find it. Pricing idea: share of
    payouts saved; free audit first.
  - Competitors/adjacent: ProvenanceAI (whole-file hashes within your library), Pex/Vobile (cross-platform
    matching, strongest threat), Vermillio and Loti (likeness, scam ads), Meedan (fact-checker matching).
  - Targets: Moonvalley, Adobe Firefly, Runway, Luma, Pika; Troveo (Marty Pesis), Protege (Bobby Samuels),
    Wirestock (Mikayel Khachatryan), Versos AI (Chris Keevill); experts Pamela Samuelson (Berkeley Law),
    Shayne Longpre (Data Provenance Initiative).
  - Kill criteria after ~8 conversations: if everyone says warranties cover it or Pex does it, drop it.
- **Within-dataset dedup is commodity** (open tools; big labs build it). Sell to small/mid vendors,
  marketplaces and stock libraries without strong ML teams. Edge: edited-copy detection with evidence,
  calibration (copy vs look-alike), keeping up with cheaters, third-party reports for contributor disputes.
- **Moat comes from customers, not grabbed data.** Public datasets and public AI sets are not a moat
  (anyone can index them). Moats: (1) opt-in pooled fingerprints from customers (cross-vendor conflicts,
  contributors selling twice), (2) independence/trust (auditor model: labs can't certify themselves to
  buyers, insurers, regulators), (3) workflow integration, (4) calibration. Before customers, speed is the
  moat. Write opt-in pooling into contracts from day one; don't make the first sale depend on it.
- **Exclusivity can't be proven early** (a few members prove nothing; resale to non-members is invisible).
  Say "not found anywhere we can see" with coverage stated; per-buyer invisible watermarks for traceability;
  contracts enforce. A buyer-driven consortium (like insurance/banking fraud sharing, GIFCT) is the
  long-term vision: buyers require suppliers to register; blind matching; no prices or strategy shared;
  antitrust review needed.
- **Demand evidence:** exclusive vs non-exclusive licences are priced as different products; vendors reselling
  one dataset to 5 buyers reach 80–90% margins; Surge/Mercor/Turing reportedly sell ~$500M/yr to Chinese labs
  too; Korea's $1.1B AI-data programme found widespread duplicates; vendor fraud cases exist; EU AI Act and
  California AB 2013 require training-data source disclosure. Against: Fairly Trained's small uptake.
- **Other uses of the engine** (same matching, different data): benchmark-contamination checks (test sets
  in training sets), model-output memorisation checks, stolen-footage screening for stock libraries,
  scam-ad networks via public ad libraries, coordinated fake-account networks, evidence integrity,
  footage clearance. An AI-output registry needs big providers to join; not a startup's first product.

### Money and market sizing (estimates)
- Reported price of AI training video: ~$1–4 per minute ($60–240/h; Bloomberg 2024). Savings examples:
  OpenVid stock's 61 duplicate hours ≈ $4k–15k if bought; a buyer taking 10,000 h at 23% ≈ $140k–550k, at
  the ~3–4% hidden rate ≈ $24k–96k. One audit run costs ~$50 of compute. These assume OpenVid-like data;
  only a real audit shows a customer's rate.
- Fingerprinting costs ~$0.10 per hour of footage (measured on OpenVid stock): 1k h ≈ $100, 10k h ≈ $1k,
  100k h ≈ $10k.
- Paid egocentric collection: Human Archive pays ~$1/h base and can collect up to 8,000 h/day; Claru pays
  $6 per accepted hour; investors expect robotics firms to buy hundreds of millions of hours.

### Honest read on demand (Oct 9)
- For: real money lost per duplicate in paid collection; edited/hidden copies are hard with naive tools;
  provenance pressure (lawsuits, EU AI Act). Against: duplicates in datasets haven't visibly cost anyone;
  contracts/indemnities; in-house dedup; Pex/Vobile could move in. Nobody has said yes yet; the test is ~20
  tier-1 DMs + free audits. A free OpenVid finding is a credibility card, not proof of demand.

### Publicity angles (separate from the business)
- **Duplicates (business pitch):** insider story; LinkedIn is the best channel for vendor founders; use the
  post as proof attached to direct messages, not as outreach itself (expect few inbound DMs).
- **"84% of a 'CC BY' dataset is cut from YouTube"** (have it from metadata): press angle tied to creator
  lawsuits. Check what the OpenVid paper says about sources before claiming anything; neutral tone.
- **"Is your footage in AI datasets?" lookup:** biggest viral potential; people care mostly once (The
  Atlantic, Have I Been Trained); money comes indirectly (lawyers needing evidence, licensing marketplaces
  recruiting creators, alerts when footage appears in new datasets). Edge over The Atlantic: matches the
  footage itself (re-uploads, edits, exact seconds). Use upload-a-clip, not YouTube links (downloading from
  YouTube breaks its terms). State coverage ("checked against N public datasets"); "not found" ≠ not used;
  private lab data (Sora etc.) is never covered. Build plan: index OpenVid YouTube part (~$250–400) + clip ->
  source YouTube ID/timestamp mapping from OpenVid metadata -> Modal search endpoint (fingerprint upload,
  search, vote, delete upload) -> web page with timeline + side-by-sides -> rate limits -> launch.
- **Model memorisation:** generate videos with open models trained on indexed data and match them; a hit is
  a headline, a miss is still a result. Check which open models trained on OpenVid first.
- **Live "magic" demo** (crop + mirror + recolour + caption + speed, still found): best Cal Hacks moment.
- **AI-video angle:** not an AI detector. Can recover the original upload of a viral clip (with its AI label
  or watermark, which reposts crop), catch recycled real footage, show edits. Bottleneck is access to the
  platforms where originals live.

### Public video datasets (indexing)
- Host actual files (indexable, ~8k h, ~$1–2k total): OpenVid-1M YouTube part (~1,800 h), FineVideo
  (HuggingFaceFV/finevideo: 43,751 YouTube CC-BY videos, 3,425 h, avg 4.7 min, 673 GB in 1,357 parquet
  files, gated with auto-approval, needs HF token; est. $300–400 fingerprinting + audit fixes above; run a
  10% sample ~$40–60 first), Kinetics-700 via a hosted mirror (~1,800 h), smaller benchmarks. Confirm hosting
  and licence before each run.
- Link-only (YouTube IDs; don't download): InternVid (~760k h), HD-VILA-100M (~370k h), Panda-70M
  (~167k h), HowTo100M (~134k h). Can be covered by ID matching only. WebVid was taken down.
- Indexing public datasets is a door-opener and a sales feature, not a moat (anyone can copy it for ~$2k).
  Useful to vendors: catches contributors who download public-dataset clips and submit them as their own;
  buyers learn if footage is freely available or has a messy licence; labs get contamination checks.

### Go-to-market
- **Targets, tier 1 (pay contributors; fraud hits margins):** Luel (~3M contributors), Awign (1.5M gig
  workers, 1,000+ h/day), Appen (1M+), Wirestock (700k creators, ~$40M run rate), Troveo (7k rights holders),
  Newsflare (UGC to media and AI; distributed a fake AI video). Tier 2 (aggregate libraries): Protege, Versos
  AI, Calliope Networks (~17k h), Avail Corpus, Storyblocks, Defined.ai, Claru, Datarade. Tier 3 (stolen
  uploads): Pond5, Filmsupply, Artlist, Storyblocks. Skip: self-filming vendors, Scale/Surge/Mercor (text).
- **Motion:** ~40 targets, ~10 messages/day, two follow-ups; Berkeley/SkyDeck/Cal Hacks warm intros first.
  Lead with the OpenVid finding ("found X% hidden duplicates"), offer a free audit (customer runs the client
  tool; only fingerprints leave), review the report together, convert to a paid pilot (~$2–10k, or a share of
  payouts stopped). Two-week target: ~6 calls, 2–3 audits, 1 pilot. No accepted audits = pivot signal.
- **New tier-1 targets (paid egocentric/robotics collection, Oct 2026):** Human Archive (YC W26, $8.2M seed
  led by Wing VC; Indian gig workers with camera headsets, gloves, mocap; 1,000+ headsets; up to 8,000 h/day;
  founders Samay Maini (Berkeley EECS), Rushil Agarwal, Shloke Patel, Raj Patel (CEO)), Luel (YC W26, $31M,
  3M+ contributors; ask Samay for an intro), Claru ($6 per accepted hour), EgoVista (Europe, €7–20/h),
  Egolab.AI (India, founded Jan 2026). Tier 2 additions: Kled (creator video licensing), Troveo (8M+ h,
  "largely exclusive" catalog), Protege ($70M, mostly healthcare: lower priority).
- **Pitch by tier:** collection vendors: looped, padded, re-submitted or duplicate contributor sessions
  (they control the hardware, so internet re-uploads are less likely). Marketplaces: the same footage
  sold/uploaded twice undermines exclusivity.
- **Message shape (≤100 words):** Berkeley connection first; the OpenVid number + a visual example link;
  one line on their specific pain; free check on a sample; "only fingerprints leave your side"; ask for
  15 min ("even just for advice" works with YC founders). Follow up after 3–4 days, then once more a week
  later. On calls mostly ask: how they catch duplicates today, rejection rate and reasons, buyer complaints,
  what they'd do with an X% finding.
- **Distribution order:** evidence page / HF dataset -> LinkedIn (+ X, HN "Show HN", HF discussion) -> DMs
  to ~20 targets with the post link -> Cal Hacks demo -> second dataset -> journalists (404 Media, Proof
  News, The Atlantic, The Verge, TechCrunch). Post Tue–Thu morning PT; method + dataset link as first comment.
- **Trust building:** never overclaim; publish method, precision check and list; open-source the client;
  stay neutral (contact people before publishing about them); clean sourcing; get a credible reviewer
  (Pamela Samuelson, Shayne Longpre). Moat later: opt-in pooled customer fingerprints in contracts from day
  one, independent-auditor role, workflow integration, calibration data.
- **Positioning:** lead as the person who found something, not a vendor. Precise wording ("cut from YouTube",
  "at least X%"), never attack dataset authors, never claim AI detection or proof of exclusivity.

### Outreach log
- Oct 9: Tim Sprecher (CEO, "Hub") messaged; suggested improvements given (fix 21% -> 23%, add the edge,
  proof link, Hub-specific pain, ask for a call, privacy line). Outcome unknown.
- Oct 9: Samay Maini (Human Archive co-founder) message drafted with a Human Archive-specific pitch;
  subject suggestion "Berkeley EECS → duplicate footage in robotics data". To send.
- LinkedIn profile: suggested "Founder · Provena (or Stealth) · Provenance and duplicate checks for video
  training data". LinkedIn post drafted; must use the safe claims above before posting.

## Next steps

Done: Modal stock run + audit + 60-pair verification; HF release; audit tool (local + cloud + container);
`minidedup` practice package.

0. Decide: wrap up building vs. run the 2-week conversation test in parallel (see "Direction").

1. User: rotate the HF token; post the OpenVid discussion note; fix + post LinkedIn; DM Samay + ~20 targets.
2. Commit all uncommitted work (nothing from the OpenVid pipeline, audit, release, audit tool or
   practice/minidedup is committed).
2b. Publish: minidedup to TestPyPI (user); provena-audit packaging per "Packaging"; code on GitHub;
   technical report / blog posts.
3. Corrected copies of the v2 doc and one-pager (2.9% / 10,433 hidden).
4. Audit tool next: C++ audit kernel (see "Planned: C++ audit kernel"; fixes the vote double count); loop/repeat detection inside one video (key for paid collection; beware static shots),
   a way for customers to send fingerprints (upload link or shared bucket), SSCD + edit-undoing for edited
   copies.
5. Cal Hacks (Oct 26) demo: upload an edited clip -> original + matching seconds (search endpoint + page).
6. Index the OpenVid YouTube part (~$300) for the "is your footage in AI datasets?" lookup; then ID lists of
   link-only datasets; then FineVideo (10% sample first) after the scaling fixes.
7. Experiments: model memorisation; benchmark leakage; label conflicts among duplicates.
