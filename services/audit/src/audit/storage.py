"""One interface for a local folder, s3://bucket/path or gs://bucket/path (via fsspec).

Remote videos are copied to a temp file for decoding and deleted straight after; nothing is kept.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import fsspec


class Location:
    def __init__(self, url: str | Path):
        self.url = str(url)
        self.fs, root = fsspec.core.url_to_fs(self.url)
        self.root = root.rstrip("/")
        self.local = "file" in (self.fs.protocol if isinstance(self.fs.protocol, tuple) else (self.fs.protocol,))

    def path(self, rel: str) -> str:
        return f"{self.root}/{rel}"

    def files(self, suffixes: set[str] | None = None) -> list[str]:
        """Relative paths of all files below the root, sorted."""
        if not self.fs.exists(self.root):
            return []
        out = []
        for p in self.fs.find(self.root):
            rel = str(PurePosixPath(p).relative_to(PurePosixPath(self.root)))
            if suffixes is None or PurePosixPath(rel).suffix.lower() in suffixes:
                out.append(rel)
        return sorted(out)

    def size(self, rel: str) -> int:
        return int(self.fs.size(self.path(rel)))

    @contextmanager
    def local_file(self, rel: str):
        """A local path to the file: the file itself, or a temp copy that's deleted afterwards."""
        if self.local:
            yield Path(self.path(rel))
            return
        fd, tmp = tempfile.mkstemp(suffix=PurePosixPath(rel).suffix)
        os.close(fd)
        try:
            self.fs.get(self.path(rel), tmp)
            yield Path(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def exists(self, rel: str) -> bool:
        return self.fs.exists(self.path(rel))

    def read_bytes(self, rel: str) -> bytes:
        with self.fs.open(self.path(rel), "rb") as f:
            return f.read()

    def write_bytes(self, rel: str, data: bytes) -> None:
        self.fs.makedirs(str(PurePosixPath(self.path(rel)).parent), exist_ok=True)
        with self.fs.open(self.path(rel), "wb") as f:
            f.write(data)
