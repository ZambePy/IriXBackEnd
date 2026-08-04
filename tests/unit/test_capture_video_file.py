"""Sprint 3 — file-backed frame source over a small generated fixture."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from irisflow.capture.video_file import VideoFileSource
from irisflow.core.exceptions import CaptureError


def _write_video(path: Path, *, frames: int = 6, width: int = 64, height: int = 48) -> Path:
    """Write a tiny MJPG .avi so tests never depend on an external fixture.

    MJPG in .avi is the format most consistently decodable by OpenCV on
    every platform (mp4/H264 requires codecs that are absent on some CI
    runners).
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    if not writer.isOpened():
        pytest.skip("MJPG writer unavailable on this platform")
    try:
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = (i * 40) % 255
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_video_file_source_reads_all_frames_then_returns_none(tmp_path: Path) -> None:
    path = _write_video(tmp_path / "clip.avi", frames=4)
    source = VideoFileSource(path)
    with source:
        frames = []
        while True:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)
    assert len(frames) >= 1
    ids = [f.frame_id for f in frames]
    assert ids == list(range(len(frames)))


def test_video_file_source_loop_never_exhausts(tmp_path: Path) -> None:
    path = _write_video(tmp_path / "loop.avi", frames=3)
    source = VideoFileSource(path, loop=True)
    with source:
        # Read many more frames than the file contains — loop wraps.
        frames = [source.read() for _ in range(10)]
    assert all(f is not None for f in frames)


def test_video_file_source_raises_when_path_missing(tmp_path: Path) -> None:
    source = VideoFileSource(tmp_path / "missing.avi")
    with pytest.raises(CaptureError, match="not found"):
        source.open()


def test_video_file_source_read_before_open_returns_none(tmp_path: Path) -> None:
    path = _write_video(tmp_path / "clip.avi", frames=2)
    source = VideoFileSource(path)
    assert source.read() is None


def test_video_file_source_closes_cleanly(tmp_path: Path) -> None:
    path = _write_video(tmp_path / "clip.avi", frames=2)
    source = VideoFileSource(path)
    source.open()
    source.close()
    source.close()  # idempotent
    assert not source.is_open
