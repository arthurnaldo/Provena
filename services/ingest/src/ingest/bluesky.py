"""Bluesky connector: an account's video posts → fingerprints → store.

Uses the public AppView API; no account or key needed. Index news and organization accounts only.
"""

import json
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from core import Video
from ingest.common import ingest_videos
from store import Store

PLATFORM = "bluesky"
API = "https://public.api.bsky.app/xrpc"


def parse_post(post: dict) -> tuple[Video, str] | None:
    """One post view (`feed[i].post`) → (Video, HLS playlist URL), or None if it has no video.

    native_id is "{did}/{rkey}" and channel_id the author's DID, which never changes (handles can).
    posted_at is indexedAt, when Bluesky's server first saw the post: createdAt is set by the
    poster's app and can be backdated, which would let anyone fake being first.
    """
    embed = post.get("embed") or {}
    if embed.get("$type") == "app.bsky.embed.recordWithMedia#view":  # quote post with its own video
        embed = embed.get("media") or {}
    if embed.get("$type") != "app.bsky.embed.video#view" or not embed.get("playlist"):
        return None
    did, rkey = post["uri"].removeprefix("at://").split("/app.bsky.feed.post/")
    video = Video(
        platform=PLATFORM,
        native_id=f"{did}/{rkey}",
        url=f"https://bsky.app/profile/{did}/post/{rkey}",
        channel_id=did,
        posted_at=datetime.fromisoformat(post["indexedAt"]),
    )
    return video, embed["playlist"]


def author_video_posts(actor: str, limit: int = 100) -> list[dict]:
    """Newest video posts by `actor` (handle or DID), skipping reposts of other accounts' posts."""
    posts, cursor = [], None
    while len(posts) < limit:
        params = {"actor": actor, "filter": "posts_with_video", "limit": min(100, limit - len(posts))}
        if cursor:
            params["cursor"] = cursor
        with urllib.request.urlopen(f"{API}/app.bsky.feed.getAuthorFeed?{urllib.parse.urlencode(params)}") as r:
            page = json.load(r)
        posts += [item["post"] for item in page["feed"] if not item.get("reason")]
        cursor = page.get("cursor")
        if not cursor or not page["feed"]:
            break
    return posts[:limit]


def download(playlist: str, dest: Path) -> None:
    """First (lowest-resolution) video rendition, no audio. PDQ downsamples anyway."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", playlist, "-map", "0:v:0", "-an", "-c", "copy", str(dest)],
        check=True,
    )


def ingest_posts(store: Store, posts: list[dict]) -> list[Video]:
    return ingest_videos(store, [item for item in map(parse_post, posts) if item], download)
