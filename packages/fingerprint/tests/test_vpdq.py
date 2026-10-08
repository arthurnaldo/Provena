from pathlib import Path

import pytest

from fingerprint import fingerprint_video

VIDEO = Path(__file__).parent / "fixtures" / "testsrc_5s.mp4"  # 160x120, 25 fps, 5 s


def test_default_samples_four_per_second():
    # 0.25 s × 25 fps = 6.25 frames → every 6 frames (0.24 s).
    offsets = [f.offset for f in fingerprint_video(VIDEO)]
    assert offsets == pytest.approx([i * 6 / 25 for i in range(21)])


def test_one_frame_per_second():
    frames = fingerprint_video(VIDEO, seconds_per_hash=1.0)
    assert [f.offset for f in frames] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_frames_have_hash_and_quality_but_no_ids():
    for f in fingerprint_video(VIDEO):
        assert len(f.pdq) == 32
        assert 0 <= f.quality <= 100
        assert f.video_id is None and f.frame_id is None


def test_seconds_per_hash_rounds_down_to_whole_frames():
    # 0.5 s × 25 fps = 12.5 frames → vPDQ samples every 12 frames (0.48 s).
    offsets = [f.offset for f in fingerprint_video(VIDEO, seconds_per_hash=0.5)]
    assert offsets == pytest.approx([i * 12 / 25 for i in range(11)])


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        fingerprint_video("does_not_exist.mp4")
