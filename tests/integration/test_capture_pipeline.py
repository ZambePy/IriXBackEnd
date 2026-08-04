"""Sprint 3 integration — capture layer talks to the rest of the system.

Verifies that:

* Concrete sources (:class:`SyntheticFrameSource`, :class:`WebcamSource`
  with a fake capture) conform to the :class:`FrameSource` Protocol
  structurally — the pipeline can consume them interchangeably.
* An extended run over a fake webcam produces coherent frame ids and
  timestamps (proxy for the "60s without leaks" DoD criterion, which we
  cannot run in wall-clock CI time).
"""

from __future__ import annotations

import time
from itertools import pairwise

from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.capture.video_file import VideoFileSource
from irisflow.capture.webcam import WebcamSource
from irisflow.core.interfaces import FrameSource
from tests.fixtures.fake_capture import make_factory


def test_all_sources_satisfy_frame_source_protocol() -> None:
    assert isinstance(SyntheticFrameSource(), FrameSource)
    assert isinstance(
        WebcamSource(capture_factory=make_factory(), reconnect_backoff_ms=10),
        FrameSource,
    )


def test_video_file_source_satisfies_frame_source_protocol(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # VideoFileSource is only structurally checkable — no need to open it here.
    source = VideoFileSource(tmp_path / "unused.avi")
    assert isinstance(source, FrameSource)


def test_extended_webcam_capture_produces_coherent_stream() -> None:
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(),
        reconnect_backoff_ms=10,
    )
    ids: list[int] = []
    timestamps: list[float] = []
    with source:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            frame = source.read()
            if frame is not None:
                ids.append(frame.frame_id)
                timestamps.append(frame.timestamp)
            time.sleep(0.005)
    assert len(ids) > 5
    assert ids == sorted(ids)
    assert timestamps == sorted(timestamps)
    # Drop-oldest means we won't see every id — but consecutive reads must
    # never move backwards.
    for a, b in pairwise(ids):
        assert b > a
