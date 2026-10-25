# Provena Gate: system design

Status: built at Cal Hacks 13.0 (Oct 23-25, 2026). Branch `provena-audit`, folder `provena-gate/`.

## 1. What it does

A buyer of video training data receives a delivery (a folder of clips plus an invoice). Before paying, Provena Gate
finds four kinds of footage that should not be paid for:

| Finding | Meaning |
|---|---|
| `repeat_in_delivery` | the same footage appears twice in this delivery |
| `in_own_library` | the buyer already owns this footage |
| `sold_to_other_buyer` | another buyer already has it (only when both buyers opted in to sharing fingerprints) |
| `public_dataset` | the footage is in a public AI training dataset |

Each finding gets matched seconds, a strength, a dollar value and an action (hold, or a person decides). Fixed rules
set the limits; Claude reviews inside them, explains every decision, answers disputes, and asks a person when unsure.
A person makes the final call on anything disputed or uncertain.

## 2. Architecture at a glance

```
                 Browser (gate/static/index.html, one page)
                                  |
                                  v  HTTP / JSON
  +--------------------------- gate/app.py (FastAPI) ----------------------------+
  |                                                                               |
  |  engine.py      findings.py        rules.py        agent.py      evidence.py  |
  |  fingerprint -> check delivery -> fixed limits -> Claude review   frames      |
  |  + thumbnails   (4 kinds)          + money split   + disputes                 |
  |       |               |                                 |              |      |
  +-------|---------------|---------------------------------|--------------|------+
          |               |                                 |              |
          v               v                                 v              v
   provena-audit     account libraries (shards)       Anthropic API    Hugging Face
   (C++ matcher)     public index (1.6M clips)        Claude Opus 5.5  (public clip
                     gate.sqlite (results only)                         frames, cached)
```

It is a modular monolith: one process, one SQLite file, plain function calls between modules. The matching engine
is the part that would split out first (it holds the 2.5 GB index in memory and scales differently).

## 3. Components

### 3.1 Matching engine: `provena-audit` (built before the event)

Our own library (`provena-audit`), unchanged by this project.

- **Fingerprints:** a 256-bit PDQ hash (Meta's open perceptual image hash) of the full-size frame every 0.25 s
  (4 per second), plus a quality score. Frames below quality 50 (flat, featureless) are ignored for matching.
  Decoded in-process with PyAV; hashes match `ffmpeg -vf fps=4` exactly.
- **Shards:** fingerprints are stored in `.npz` files (random clip IDs, durations, per-frame hash/offset/quality).
  No video, images or audio. About 48 bytes per indexed frame in memory.
- **Search (`Index.search`):** a C++ kernel (nanobind) compares the clip's frames against every indexed frame
  (Hamming distance <= 31 of 256 bits) and votes on the time offset between the two videos (Hough voting). A match
  needs at least 8 frames agreeing on one offset. It returns the matching clip, the matched seconds on both sides
  and the vote count. Brute force, multi-threaded, no approximate index.
- **Audit (`find_pairs`):** the same kernel, all clips against all others in one set. Used for duplicates inside a
  delivery.
- What it survives: re-encoding, resizing, heavy compression, trimming, small overlays, colour changes (partly).
  What it does not: crops of ~10% or more, mirroring, letterboxing (PDQ limits).

### 3.2 Public index (built before the event)

`data/index/<dataset>/`: shards + `manifest.json` (name, source, stated licence) + `clips.parquet` (per-clip
metadata such as YouTube source URL or zip location).

| Dataset | Clips | Notes |
|---|---|---|
| OpenVid-1M | 1,019,956 | complete |
| LLaVA-Video-178K | 165,969 | 220 of 250 files |
| Vript | 370,854 | 1,917 of 2,274 files |
| ShareGPT4Video | 40,303 | 91 of 92 files |
| VideoChat3-LV116k | 33,436 | 7,903 of 9,236 files (lectures) |

Total 1,630,518 clips, about 11,500 hours, 162.9M indexed frames. Fingerprinted on AWS spot machines (Graviton),
shards in S3. Loads in about 19 s and 2.5 GB of RAM. A finishing run for LLaVA, Vript and ShareGPT4Video was
started on Oct 23 (results not yet merged into the index).

### 3.3 The app: `provena-gate` (built at Cal Hacks)

| Module | Job |
|---|---|
| `gate/db.py` | SQLite schema and helpers. Results only, never fingerprints. |
| `gate/engine.py` | Glue to the engine: account library folders, fingerprinting into shards, thumbnails. |
| `gate/findings.py` | `check_delivery()`: fingerprint a delivery, find the four kinds of findings, value them; `report()` reads a delivery back. |
| `gate/rules.py` | Fixed limits: allowed actions, default actions, the invoice split, overbilling, the auto-hold cap. |
| `gate/agent.py` | Claude: the review loop (tools) and dispute replies. |
| `gate/evidence.py` | Side-by-side frames for a finding, with privacy rules. |
| `gate/app.py` | HTTP API and the dashboard route. |
| `gate/static/index.html` | The single-page dashboard (plain HTML/JS, no framework). |

## 4. Data

### 4.1 Files (`GATE_DATA`, default `provena-gate/data/`, gitignored)

```
gate.sqlite                         results and the decision log
accounts/<id>/library/send/*.npz    each account's library fingerprints; manifest says "acct:<id>"
deliveries/<id>/send/*.npz          each delivery's fingerprints
deliveries/<id>/evidence/<finding>/ cached public-dataset frames
thumbs/<clip id>/<n>.jpg            one 240 px thumbnail per second of each clip, for evidence
```

Each account library is its own engine "dataset folder", so the engine labels hits `acct:<id>` with no engine changes.

### 4.2 Tables

| Table | Key columns |
|---|---|
| `accounts` | name, `share_fingerprints` (opt-in) |
| `deliveries` | account, vendor, `rate_per_hour`, `invoice_total`, status (received, checked, ruled, decided), errors |
| `clips` | id (= fingerprint ID), account, delivery (NULL = library), contributor, file name, duration |
| `findings` | delivery, clip, kind, match source and ref, clip seconds, match seconds, votes, static, amount, details (licence, links) |
| `decisions` | the log: delivery, finding (or NULL), actor (rules, claude, human), action, amount, reason, time |
| `disputes` | finding, author, message, Claude's reply, outcome (uphold or escalate) |

The current action of a finding is always the latest hold / escalate / release in `decisions`; nothing overwrites
history.

## 5. Flows

### 5.1 Add clips to a library

`POST /api/accounts/{id}/library` → uploads saved to a temp folder → `engine.add_to_library()` fingerprints them in
parallel (all cores), writes a shard, saves thumbnails, inserts `clips` rows → temp folder deleted.

### 5.2 Check a delivery

`POST /api/deliveries` (files, vendor, rate, invoice, contributor per file):

1. `findings.check_delivery()`
   - fingerprint the clips into `deliveries/<id>/send/` (+ thumbnails)
   - repeats: `find_pairs` on just this delivery; the later clip is the repeat
   - libraries: one engine `Index` over the buyer's library plus every other opted-in library (only if the buyer
     also opted in), rebuilt per check (small)
   - public: the public `Index`, loaded once at startup
   - per clip: `search()` against both; each hit becomes a finding
   - static check: if the clip's frames in the matched range barely change (median change < 6 bits between frames
     0.25 s apart), the finding is marked static; for static clips only the strongest finding per kind is kept
     (a colour-bar test pattern otherwise matched 73 public clips)
   - amount = matched seconds / 3600 x rate
2. `rules.apply_rules()`: default action per finding, the split, all logged (actor `rules`), status `ruled`.
3. `agent.run_agent()` (if an API key is set and `GATE_AGENT` is not `off`): Claude reviews, status `decided`.
4. The response is the full delivery view (findings with actions and reasons, split, report, notes, log).

Temp uploads are deleted at the end of the request.

### 5.3 Money rules (`rules.py`)

- `allowed_actions`: strong match on moving footage → hold or escalate; weak (< 12 matching frames) or static →
  escalate only. A person can also `release` (pay).
- `split`: per clip, seconds covered by held findings count once (several findings on the same seconds never
  double-charge), then seconds only covered by escalated findings → review; the rest is approved.
- Overbilling: anything billed beyond (delivered seconds x rate) is held.
- Cap: if held > 30% of the invoice, nothing is approved automatically; the whole delivery goes to a person.
- `siblings`: a dispute or a person's decision covers every finding of the same kind on the same clip (a dataset can
  hold both a video and a scene cut from it).

### 5.4 Claude review (`agent.run_agent`)

- Model `claude-opus-5-5`, effort `medium`, manual tool loop, at most 20 turns, server-side refusal fallbacks on.
- Tools: `get_delivery`, `get_findings` (with allowed actions), `get_evidence`, `decide`, `escalate`, `write_report`.
- `decide` goes through `rules.check()`: a disallowed action is refused, logged, and Claude is told why.
- Claude never computes money: the final split is `rules.split()` on its chosen actions; it is told to quote it.
- Every tool call is logged (actor `claude`), including what it looked at and refused attempts.
- If the API fails, Claude refuses, or it does not finish: the rules' decision stands and `unavailable` is logged.
- Measured: about 40-50 s and a few cents per 10-clip delivery.

### 5.5 Evidence (`evidence.py`)

Three moments across the matched range, the delivered clip next to the same moments in the match:
- repeat / own library: both sides from saved thumbnails
- other buyer: only the buyer's own side; the other buyer's frames are never sent
- public dataset: fetched from Hugging Face on first request (zip members via HTTP range requests, plain video files
  by seeking with ffmpeg, OpenVid via its zip index), then cached; tar archives are not supported

### 5.6 Disputes and human decisions

- `POST .../findings/{fid}/disputes` (author, message): Claude gets the finding, its facts and the evidence frames
  as images, and returns a structured answer: a reply and `uphold` or `escalate`. It cannot release payment.
  Escalate moves held siblings to review and recalculates the split. If Claude is unavailable, the dispute is
  escalated with a polite holding reply.
- `POST .../findings/{fid}/decision` (hold or release, reason): a person's final call, logged as `human`, applied to
  siblings, split recalculated.

## 6. API

| Method and path | Purpose |
|---|---|
| `GET /api/status` | public datasets loaded, whether Claude review is on |
| `GET/POST /api/accounts` | list / create (name, `share_fingerprints`) |
| `GET/POST /api/accounts/{id}/library` | list / add clips (multipart) |
| `GET /api/deliveries?account_id=` | list deliveries |
| `POST /api/deliveries` | check a delivery (multipart: files, vendor, rate_per_hour, invoice_total, contributors JSON) |
| `GET /api/deliveries/{id}` | the delivery view |
| `GET /api/deliveries/{id}/findings/{fid}/evidence` | frames |
| `POST /api/deliveries/{id}/findings/{fid}/disputes` | dispute → Claude reply |
| `POST /api/deliveries/{id}/findings/{fid}/decision` | person: hold or release |

Settings: `GATE_DATA`, `GATE_PUBLIC` (":"-separated folders, or `none`), `GATE_AGENT=off`, `ANTHROPIC_API_KEY`,
`HF_TOKEN` (gated datasets).

## 7. Privacy and trust

- Video is never stored. Uploads live in a temp folder for the length of one request.
- Fingerprints (hashes) are stored per account. Other buyers' fingerprints are compared only when both opted in, and
  their clip IDs, timestamps and names never leave the app; a match shows only "another buyer" and a first-seen date.
- Thumbnails (240 px, one per second) of each account's own clips are stored for evidence and shown only to that
  account. This is the one place the app keeps images; the alternative (frames only during the check) would make
  evidence impossible to reopen.
- Claude sees findings, facts and frames of the buyer's own footage and public datasets, never another buyer's.
- Wording rules for Claude: only tool facts; "found in this public dataset", never "a model was trained on it".

## 8. Performance (measured on a MacBook)

| Step | Time |
|---|---|
| Public index load (1.63M clips, 163M frames) | 16-19 s, 2.5 GB RAM |
| One clip search against the public index | about 1-2.5 s (the clip is fingerprinted again inside `search`) |
| 10-clip delivery end to end incl. Claude | about 48 s |
| Index test, edited clips (6 s cut, half size, crf 32) | 13/13 found, 0 false matches on unrelated videos (twice) |

## 9. Testing

`cd provena-gate && uv run pytest` (16 tests, about 30 s). Synthetic videos made with ffmpeg cover: library ingest
and search of an edited copy; all four finding kinds, privacy, sharing off; the rules (holds, escalation, split,
overbilling, cap); the agent with a scripted fake Claude (refused action, API down); the HTTP API end to end;
evidence (both sides, other buyer hidden); disputes (escalate, uphold, Claude down, siblings). Live checks against
real Claude and the real public index were run by hand.

## 10. Known limitations

- **Edits PDQ can't see:** crops of ~10%+, mirrors, letterboxing are missed (SSCD embeddings would fix this; dropped
  for now).
- **Static threshold is a guess** (`STATIC_BITS = 6`), tuned only on synthetic footage.
- **Weak threshold** (`STRONG_VOTES = 12`) and **cap** (30%) are policy guesses.
- **No login:** any visitor can act as any account. Out of scope for the weekend.
- **Synchronous requests:** a large delivery holds one HTTP request for minutes; needs a job queue.
- **Double fingerprinting:** `Index.search` takes a file and fingerprints it again; a small engine change would let
  it take existing fingerprints.
- **Public evidence:** tar archives unsupported; OpenVid needs the local zip index file.
- **"First seen"** uses the date a clip entered a library, so it says nothing about who filmed it first.
- **Single machine:** the public index must fit in one process's memory.

## 11. Built before vs at Cal Hacks

- Before: the matching engine `provena-audit` (fingerprinting, C++ kernel, `Index`), the public index and its
  metadata, the OpenVid-1M audit (~8% duplicate or near-duplicate footage).
- At Cal Hacks: everything in `provena-gate/`: accounts and libraries, the delivery check, the rules, the Claude
  review and dispute agent, evidence frames, the API and the dashboard.

## 12. Running it

```
cd provena-gate && uv sync
ANTHROPIC_API_KEY=... uv run uvicorn gate.app:app --port 8000   # then open http://127.0.0.1:8000
```

`GATE_PUBLIC=none` starts without the public index (fast, for development).
