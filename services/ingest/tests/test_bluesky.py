from datetime import datetime, timezone
from pathlib import Path

from ingest.bluesky import ingest_posts, parse_post
from store import Store

FIXTURE = Path(__file__).parents[3] / "packages/fingerprint/tests/fixtures/testsrc_5s.mp4"
DID = "did:plc:a67zdrt4nl2tv2qojpngogbq"


def video_embed(playlist: str = str(FIXTURE)) -> dict:
    return {"$type": "app.bsky.embed.video#view", "cid": "bafk", "playlist": playlist}


def post(rkey: str = "3mxdb7rtuti2w", embed: dict | None = None) -> dict:
    # Shape of a real app.bsky.feed.getAuthorFeed post view, trimmed.
    return {
        "uri": f"at://{DID}/app.bsky.feed.post/{rkey}",
        "author": {"did": DID, "handle": "apnews.com"},
        "record": {"createdAt": "2020-01-01T00:00:00Z"},  # client-set; deliberately not used
        "indexedAt": "2026-10-08T01:03:10.367Z",
        "embed": video_embed() if embed is None else embed,
    }


def test_parse_video_post():
    video, playlist = parse_post(post())
    assert video.native_id == f"{DID}/3mxdb7rtuti2w"
    assert video.channel_id == DID
    assert video.url == f"https://bsky.app/profile/{DID}/post/3mxdb7rtuti2w"
    assert video.posted_at == datetime(2026, 10, 8, 1, 3, 10, 367000, tzinfo=timezone.utc)
    assert playlist == str(FIXTURE)


def test_quote_post_with_video():
    embed = {"$type": "app.bsky.embed.recordWithMedia#view", "record": {}, "media": video_embed("x.m3u8")}
    assert parse_post(post(embed=embed))[1] == "x.m3u8"


def test_non_video_posts_skipped():
    assert parse_post(post(embed={"$type": "app.bsky.embed.images#view", "images": []})) is None
    assert parse_post(post(embed={})) is None


def test_ingest_posts_stores_once(tmp_path):
    s = Store(tmp_path / "db")
    posts = [post("a"), post("b"), post("c", embed={})]
    assert [v.native_id for v in ingest_posts(s, posts)] == [f"{DID}/a", f"{DID}/b"]
    assert ingest_posts(s, posts) == []
