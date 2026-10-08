import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

from core import Frame, Hit, Video

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id   INTEGER PRIMARY KEY,
    platform   TEXT NOT NULL,
    native_id  TEXT NOT NULL,
    url        TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    posted_at  TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE (platform, native_id)
);
CREATE TABLE IF NOT EXISTS frames (
    frame_id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos (video_id),
    offset   REAL NOT NULL,
    quality  INTEGER NOT NULL,
    pdq      BLOB NOT NULL
);
"""

SQL_VARS = 900  # stay under SQLite's bound-parameter limit


class Store:
    """SQLite holds every video and frame; an in-memory FAISS index holds searchable PDQ hashes.

    The index only contains frames with quality >= min_quality and is rebuilt from SQLite on open.
    """

    def __init__(self, path: str | Path, min_quality: int = 50):
        self.min_quality = min_quality
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.index = faiss.IndexBinaryIDMap(faiss.IndexBinaryFlat(256))
        rows = self.db.execute("SELECT frame_id, pdq FROM frames WHERE quality >= ?", (min_quality,)).fetchall()
        self._index_add([r[0] for r in rows], [r[1] for r in rows])

    def close(self) -> None:
        self.db.close()

    def add_video(self, video: Video, frames: list[Frame]) -> Video:
        """Store a video and all its frames. Re-adding the same (platform, native_id) is a no-op."""
        existing = self.db.execute(
            "SELECT video_id FROM videos WHERE platform = ? AND native_id = ?", (video.platform, video.native_id)
        ).fetchone()
        if existing:
            return replace(video, video_id=existing[0])

        with self.db:
            video_id = self.db.execute(
                "INSERT INTO videos (platform, native_id, url, channel_id, posted_at, deleted_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (video.platform, video.native_id, video.url, video.channel_id,
                 video.posted_at.isoformat(), video.deleted_at.isoformat() if video.deleted_at else None),
            ).lastrowid
            frame_ids = [
                self.db.execute(
                    "INSERT INTO frames (video_id, offset, quality, pdq) VALUES (?, ?, ?, ?)",
                    (video_id, f.offset, f.quality, f.pdq),
                ).lastrowid
                for f in frames
            ]
        good = [(i, f.pdq) for i, f in zip(frame_ids, frames) if f.quality >= self.min_quality]
        self._index_add([i for i, _ in good], [p for _, p in good])
        return replace(video, video_id=video_id)

    def search(self, frames: list[Frame], max_distance: int = 31, k: int = 100) -> list[Hit]:
        """For each query frame, up to k stored frames within max_distance bits.

        Query frames below min_quality are skipped. k caps hits per frame (generic frames hit thousands).
        """
        frames = [f for f in frames if f.quality >= self.min_quality]
        if not frames or self.index.ntotal == 0:
            return []
        queries = np.frombuffer(b"".join(f.pdq for f in frames), np.uint8).reshape(-1, 32)
        distances, ids = self.index.search(queries, min(k, self.index.ntotal))

        found = [
            (f.offset, int(i), int(d))
            for f, row_d, row_i in zip(frames, distances, ids)
            for d, i in zip(row_d, row_i)
            if i != -1 and d <= max_distance
        ]
        where = self._frame_locations({i for _, i, _ in found})
        return [Hit(q, *where[i], d) for q, i, d in found]

    def get_videos(self, video_ids: list[int]) -> list[Video]:
        rows = self._select_in(
            "SELECT video_id, platform, native_id, url, channel_id, posted_at, deleted_at FROM videos"
            " WHERE video_id IN ({})",
            list(video_ids),
        )
        return [
            Video(platform=p, native_id=n, url=u, channel_id=c, posted_at=datetime.fromisoformat(pa),
                  deleted_at=datetime.fromisoformat(da) if da else None, video_id=vid)
            for vid, p, n, u, c, pa, da in rows
        ]

    def _index_add(self, frame_ids: list[int], hashes: list[bytes]) -> None:
        if frame_ids:
            self.index.add_with_ids(
                np.frombuffer(b"".join(hashes), np.uint8).reshape(-1, 32), np.array(frame_ids, np.int64)
            )

    def _frame_locations(self, frame_ids: set[int]) -> dict[int, tuple[int, float]]:
        rows = self._select_in("SELECT frame_id, video_id, offset FROM frames WHERE frame_id IN ({})", list(frame_ids))
        return {fid: (vid, off) for fid, vid, off in rows}

    def _select_in(self, sql: str, values: list) -> list[tuple]:
        rows = []
        for i in range(0, len(values), SQL_VARS):
            chunk = values[i : i + SQL_VARS]
            rows += self.db.execute(sql.format(",".join("?" * len(chunk))), chunk).fetchall()
        return rows
