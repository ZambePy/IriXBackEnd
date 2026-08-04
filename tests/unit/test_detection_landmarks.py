"""Sprint 4 — landmark index constants and normalization helper."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.detection.landmarks import (
    FACE_LANDMARK_COUNT_LEGACY,
    FACE_LANDMARK_COUNT_WITH_IRIS,
    FACE_OVAL_INDICES,
    LEFT_EYE_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_IRIS_INDICES,
    landmarks_to_pixel_array,
)


def test_iris_indices_are_in_range_only_for_refined_model() -> None:
    for idx in (*LEFT_IRIS_INDICES, *RIGHT_IRIS_INDICES):
        assert FACE_LANDMARK_COUNT_LEGACY <= idx < FACE_LANDMARK_COUNT_WITH_IRIS


def test_eye_and_face_indices_within_legacy_range() -> None:
    for idx in (*LEFT_EYE_INDICES, *RIGHT_EYE_INDICES, *FACE_OVAL_INDICES):
        assert 0 <= idx < FACE_LANDMARK_COUNT_LEGACY


def test_left_and_right_eye_indices_are_disjoint() -> None:
    assert set(LEFT_EYE_INDICES).isdisjoint(RIGHT_EYE_INDICES)


def test_iris_index_sets_are_disjoint() -> None:
    assert set(LEFT_IRIS_INDICES).isdisjoint(RIGHT_IRIS_INDICES)


def test_landmarks_to_pixel_array_scales_x_and_y_only() -> None:
    normalized = np.array(
        [
            [0.0, 0.0, 0.1],
            [1.0, 1.0, -0.2],
            [0.5, 0.25, 0.0],
        ],
        dtype=np.float32,
    )
    px = landmarks_to_pixel_array(normalized, frame_width=100, frame_height=200)
    assert px[0, 0] == 0
    assert px[1, 0] == 100
    assert px[2, 0] == 50
    assert px[0, 1] == 0
    assert px[1, 1] == 200
    assert px[2, 1] == 50
    # z untouched
    assert px[0, 2] == pytest.approx(0.1)
    assert px[1, 2] == pytest.approx(-0.2)


def test_landmarks_to_pixel_array_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r">=2"):
        landmarks_to_pixel_array(np.zeros((5, 1), dtype=np.float32), 100, 100)
    with pytest.raises(ValueError, match=r">=2"):
        landmarks_to_pixel_array(np.zeros((5,), dtype=np.float32), 100, 100)


def test_landmarks_to_pixel_array_rejects_non_positive_frame() -> None:
    with pytest.raises(ValueError, match="positive"):
        landmarks_to_pixel_array(np.zeros((5, 2), dtype=np.float32), 0, 100)


def test_landmarks_to_pixel_array_does_not_mutate_input() -> None:
    inp = np.array([[0.5, 0.5]], dtype=np.float32)
    _ = landmarks_to_pixel_array(inp, 100, 100)
    assert inp[0, 0] == 0.5
