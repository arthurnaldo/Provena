# Provena

Provena records where videos first appear. It fingerprints videos from sources where news footage originates (Telegram, YouTube), keeps the record after originals are deleted, and lets users submit a clip, even an edited one, to find its first appearance, who posted it, how it spread, and what was cut.

We never store video, only fingerprints and metadata.

## Current scope (v0)

- **PDQ hashes only.** No SSCD, audio, or transforms yet.
- **Each video is its own asset.** No asset merging. Reposts are stored as separate videos; the earliest match by `posted_at` is the origin.
- **Collisions in scope:** exact copy, subclip, superclip, partial overlap, plus generic-frame filtering.
- Python throughout. SQLite for metadata, in-memory index for PDQ lookup.

## Monorepo structure

```
provena/
├── packages/
│   ├── core/          # data model only; depends on nothing
│   ├── fingerprint/   # video → frame fingerprints
│   ├── matcher/       # CCI stages + temporal voting
│   └── store/         # DB + index access
├── services/
│   ├── ingest/        # connectors (telegram/, youtube/), workers
│   ├── recheck/       # deletion tracking
│   └── api/           # clip in → QueryResult out
├── web/               # upload clip, show origin + timeline
├── eval/              # evaluation harness
├── scripts/
├── infra/
└── pyproject.toml     # single workspace
```

### Rules
- `core` depends on nothing; everything else depends on `core`.
- `fingerprint` and `matcher` never touch databases, networks, or platforms. Frames in, matches out.
- Services are thin wiring. No matching logic inside them.
- `eval` imports the same packages as production.

## Data model (`packages/core`)

```python
@dataclass(frozen=True, slots=True)
class Video:
    platform: str                    # "youtube", "telegram", ...
    native_id: str                   # identifies one video on the platform; unique with platform
    url: str
    channel_id: str                  # stable platform ID, never a handle
    channel_name: str                # display name as last seen; may change
    posted_at: datetime              # platform-reported, tz-aware UTC
    ingested_at: datetime            # when we first recorded it, tz-aware UTC
    last_seen_live_at: datetime      # tz-aware UTC
    deleted_at: datetime | None = None   # None = still live
    video_id: int | None = None          # internal key; None until stored

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.native_id}"

@dataclass
class Frame:
    offset: float               # seconds into the video
    pdq: bytes                  # 32-byte PDQ hash
    quality: int                # PDQ quality, 0-100; low = flat/blurry frame
    video_id: int | None = None # FK → Video.video_id; None until stored, or for query clips
    frame_id: int | None = None # primary key; None until stored

@dataclass
class Match:                    # returned, not stored
    video_id: int
    clip_range: tuple[float, float]   # seconds in the query clip that match
    video_range: tuple[float, float]  # corresponding seconds in the stored video
    confidence: float

@dataclass
class QueryResult:
    clip_id: str
    matches: list[Match]        # empty = no match
```

- `clip_range` is not always the whole clip (added intros, reaction videos, compilations).
- A query can return several matches: different parts matching different videos, or the same part matching the original and its stored reposts.
- `video_id` is an internal integer key, used for all internal references (frames, index, matches). `(platform, native_id)` is the natural key and is unique, so re-ingesting the same video is a no-op.
- `native_id` identifies one video using stable platform IDs, never handles. YouTube: video ID. Telegram: `{channel_id}/{message_id}` with the numeric channel ID. Posts with several videos (X, Instagram) append `#{index}`. Each connector documents its format.
- Eval labels and anything outside the DB reference `(platform, native_id)`, never `video_id`, since internal IDs change when the DB is rebuilt.
- All timestamps are tz-aware UTC. `posted_at` is what the platform reports; `ingested_at` is our own evidence of when the video existed.
- Records are immutable; updates go through `dataclasses.replace`.

## Fingerprint package (`packages/fingerprint`)

```
fingerprint/
├── src/fingerprint/
│   ├── __init__.py   # public API
│   ├── types.py
│   ├── sample.py     # video → frames: 1/sec + keyframes, low-res, skip black/static
│   ├── pdq.py        # frame → 32-byte PDQ hash
│   ├── sscd.py       # later
│   ├── audio.py      # later
│   └── generic.py    # later
└── tests/
    ├── fixtures/     # tiny test videos
    ├── test_sample.py
    ├── test_pdq.py
    └── test_api.py
```

Public API (the only thing other code imports):
```python
fingerprint_video(path) -> list[Frame]
fingerprint_frame(image) -> tuple[bytes, int]   # (32-byte PDQ hash, quality); image is RGB uint8 (H, W, 3)
```

- PDQ comes from `pdqhash`, which wraps Meta's C++ reference implementation. Hash bytes are in reference order, so `.hex()` matches other PDQ tools.
- Every hash has a quality score (0–100). Meta suggests discarding hashes with quality ≤ 49 (flat or blurry frames). Treat it as a starting threshold to tune on the eval set, which is why `Frame` stores `quality`.

Adding SSCD or audio later must not change this API, only add fields.

## Matching: Cascading Clip Identification (CCI)

Cheapest stage first; stop as soon as one is confident. All stages return the same output.

| Stage | Method | Status |
|---|---|---|
| 0 | Metadata: forward headers, known URL, SHA-256 of file | later |
| 1 | PDQ lookup, Hamming distance ≤ 31 | **v0** |
| 2 | SSCD embeddings (cosine ≥ 0.75) + audio | later |
| 3 | Transforms: flip, crop to inner video, speed 0.5–2x | later |
| 4 | No match → new asset | **v0** |

**Temporal voting (shared by all stages):**
1. For each query frame, find stored frames within threshold.
2. For each hit, compute `offset_diff = stored_offset - query_offset`, binned to 1s.
3. Group by `(video_id, offset_diff)`. The group with the most agreeing frames wins.
4. Require ≥ 5 agreeing frames. Report `clip_range` / `video_range` from the supporting frames.
5. Ignore generic frames (frames matching more than ~10 distinct videos).

Thresholds are starting values. Tune them on the eval set, never by hand-picking examples.

## Evaluation harness (`eval/`)

```
eval/
├── data/        # gitignored: originals, generated queries, negatives
├── generate.py  # ffmpeg edits, writes labels
├── run.py       # index originals, run queries, score
└── reports/     # results per matcher version
```

- **Labels:** `{"query", "source", "source_range", "edits", "speed"}`. Negatives have `source: null`.
- **Edit types:** re-encode, trim, resize, crop, filter, text overlay, flip, speed, screen recording, and combinations.
- **Negatives:** held-out clips never indexed, shared channel intros, look-alike scenes.
- **Metrics per edit type:** recall@1, precision, false-positive rate on negatives, range error, latency.
- Split tuning and reporting sets. Every real-world failure becomes a permanent test case.
- Any change to `fingerprint` or `matcher` should be checked with `eval/run.py` and compared against the previous report.

## Deletion tracking (later, `services/recheck`)

- Telegram: listen for live `MessageDeleted` events; recheck as backup.
- YouTube: batch lookups of up to 50 IDs per call.
- Recheck often for new posts, rarely for old ones.
- Store a window (`last_seen_live_at`, `deleted_at`), never a fake exact time.

## Privacy constraints

- Never store video files, face data, or profiles of posters.
- Search is by clip only, never by person.
- Index organization and news channels, not private individuals.

## Out of scope for now

SSCD, audio, transform stage, asset merging, per-segment first-seen, compilations testing, sharding, Reddit/X/TikTok connectors, AI-generated content detection.
