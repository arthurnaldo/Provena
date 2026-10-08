import tempfile
from collections.abc import Callable
from pathlib import Path

from core import Video
from fingerprint import fingerprint_video
from store import Store


def ingest_videos(store: Store, items: list[tuple[Video, str]], download: Callable[[str, Path], None]) -> list[Video]:
    """Fingerprint and store each (video, source URL) not already stored. The video file is deleted after."""
    added = []
    for video, url in items:
        if store.has_video(video.platform, video.native_id):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            download(url, path)
            added.append(store.add_video(video, fingerprint_video(path)))
    return added
