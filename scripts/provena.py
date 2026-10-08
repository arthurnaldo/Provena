"""Minimal Provena CLI: ingest videos, query a clip.

    uv run python scripts/provena.py ingest video.mp4 --platform reddit --native-id abc123 \
        --channel t5_2qh13 --posted-at 2026-10-01T10:00:00Z [--url URL]
    uv run python scripts/provena.py query clip.mp4
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from core import Video
from fingerprint import QUERY_SECONDS_PER_HASH, fingerprint_video
from matcher import match
from store import Store

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "provena.db"


def utc(value: str) -> datetime:
    t = datetime.fromisoformat(value)
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def ingest(store: Store, args: argparse.Namespace) -> None:
    video = Video(platform=args.platform, native_id=args.native_id, url=args.url or str(args.video),
                  channel_id=args.channel, posted_at=utc(args.posted_at))
    frames = fingerprint_video(args.video)
    stored = store.add_video(video, frames)
    print(f"video {stored.video_id} ({video.platform}:{video.native_id}): {len(frames)} frames")


def query(store: Store, args: argparse.Namespace) -> None:
    frames = fingerprint_video(args.clip, QUERY_SECONDS_PER_HASH)
    matches = match(store.search(frames))
    if not matches:
        print(f"no match ({len(frames)} clip frames)")
        return
    videos = {v.video_id: v for v in store.get_videos([m.video_id for m in matches])}
    matches.sort(key=lambda m: videos[m.video_id].posted_at)
    for i, m in enumerate(matches):
        v = videos[m.video_id]
        tag = "ORIGIN" if i == 0 else "      "
        print(f"{tag} {v.posted_at:%Y-%m-%d %H:%M} {v.platform}:{v.native_id} ({v.channel_id})"
              f"  clip {m.clip_range[0]:.1f}-{m.clip_range[1]:.1f}s = video {m.video_range[0]:.1f}-{m.video_range[1]:.1f}s"
              f"  shift {m.shift}s, {m.votes}/{len(frames)} votes")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("ingest")
    i.add_argument("video", type=Path)
    i.add_argument("--platform", required=True)
    i.add_argument("--native-id", required=True)
    i.add_argument("--channel", required=True)
    i.add_argument("--posted-at", required=True, help="ISO 8601; no timezone = UTC")
    i.add_argument("--url")

    q = sub.add_parser("query")
    q.add_argument("clip", type=Path)

    args = p.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = Store(args.db)
    try:
        (ingest if args.command == "ingest" else query)(store, args)
    finally:
        store.close()


if __name__ == "__main__":
    main()
