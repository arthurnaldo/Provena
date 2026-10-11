import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FRAMES_PER_SECOND = 4.0
SSCD_SIZE = 320  # SSCD's square input size


def _video_size(path: str | Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", str(path)], capture_output=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]


def decode_frames(path: str | Path, fps: float = FRAMES_PER_SECOND) -> tuple[np.ndarray, np.ndarray]:
    """Decode a video once into evenly spaced RGB frames, two copies of each:

    - full size (n, h, w, 3): for PDQ, which must see the original frame to match standard PDQ / vPDQ
      (PDQ on a pre-shrunk frame differs by ~20 bits),
    - SSCD size (n, 320, 320, 3).

    Frame i is at i / fps seconds.
    """
    w, h = _video_size(path)
    with tempfile.NamedTemporaryFile(suffix=".rgb") as small:
        full = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-filter_complex",
             f"[0:v]fps={fps},split=2[a][b];[b]scale={SSCD_SIZE}:{SSCD_SIZE}[s]",
             "-map", "[a]", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
             "-map", "[s]", "-f", "rawvideo", "-pix_fmt", "rgb24", small.name],
            capture_output=True, check=True,
        ).stdout
        small_frames = np.fromfile(small.name, np.uint8).reshape(-1, SSCD_SIZE, SSCD_SIZE, 3)
    full_frames = np.frombuffer(full, np.uint8).reshape(-1, h, w, 3)
    n = min(len(full_frames), len(small_frames))
    return full_frames[:n], small_frames[:n]


def iter_frames(path: str | Path, fps: float = FRAMES_PER_SECOND):
    """Yield full-size RGB frames one at a time (frame i at i / fps seconds), for videos too long to hold in memory."""
    w, h = _video_size(path)
    size = w * h * 3
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps}",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finished = False
    try:
        while len(buf := proc.stdout.read(size)) == size:
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        finished = True
    finally:
        proc.stdout.close()
        err = proc.stderr.read()
        if proc.wait() and finished:  # a non-zero exit after an early stop is just the closed pipe
            raise RuntimeError(f"ffmpeg failed on {path}: {err.decode(errors='replace')[-300:]}")
