"""``irisflow preview`` — live overlay of face + eye ROIs on the webcam feed.

Depuration tool, not a production entry point. Opens a window, draws the
current bounding boxes, and updates every frame. Press ``q`` (or close the
window) to exit.

Because it requires an actual display, this command is exercised manually
— the automated suite only asserts the command exists and its ``--help``
text renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cv2
import typer

from irisflow.capture import WebcamSource
from irisflow.config.loader import load_config
from irisflow.core.exceptions import IrisFlowError
from irisflow.core.types import FaceDetection
from irisflow.detection import (
    MediaPipeFaceDetector,
    TrackedFaceDetector,
    build_default_face_mesh,
)

__all__ = ["preview"]


_WINDOW_TITLE = "IrisFlow preview"
_FACE_COLOR = (0, 255, 0)     # BGR — green
_LEFT_EYE_COLOR = (0, 165, 255)  # orange (subject's left)
_RIGHT_EYE_COLOR = (255, 128, 0)  # blue-ish (subject's right)


def preview(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to the YAML config. Defaults to configs/default.yaml.",
        ),
    ] = None,
) -> None:
    """Open a window and overlay face/eye bounding boxes on the camera feed."""
    try:
        cfg = load_config(yaml_path=config) if config is not None else load_config()
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        face_mesh = build_default_face_mesh(
            model_path=cfg.detection.face_model_path,
            refine_landmarks=cfg.detection.refine_landmarks,
            min_detection_confidence=cfg.detection.min_detection_confidence,
            min_tracking_confidence=cfg.detection.min_tracking_confidence,
        )
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    detector = MediaPipeFaceDetector(
        face_mesh, roi_margin=cfg.detection.roi_margin, square_rois=True
    )
    tracked = TrackedFaceDetector(
        detector,
        smoothing_alpha=cfg.detection.smoothing_alpha,
        lost_hysteresis=cfg.detection.lost_hysteresis_frames,
    )
    source = WebcamSource(
        device_id=cfg.camera.device_id,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        reconnect_backoff_ms=cfg.camera.reconnect_backoff_ms,
    )
    typer.echo("Preview running — press 'q' or Esc in the window to exit.")
    try:
        source.open()
        while True:
            frame = source.read()
            if frame is None:
                if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                    break
                continue
            detection = tracked.detect(frame)
            image = frame.data.copy()
            _draw_overlay(image, detection, tracked.is_in_hysteresis)
            cv2.imshow(_WINDOW_TITLE, image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        source.close()
        detector.close()
        cv2.destroyAllWindows()


def _draw_overlay(image, detection: FaceDetection | None, hysteresis: bool) -> None:  # type: ignore[no-untyped-def]
    if detection is None:
        cv2.putText(
            image, "no face", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        return
    face = detection.face_bbox
    left = detection.left_eye_bbox
    right = detection.right_eye_bbox
    cv2.rectangle(image, (face.x, face.y), (face.x2, face.y2), _FACE_COLOR, 2)
    cv2.rectangle(image, (left.x, left.y), (left.x2, left.y2), _LEFT_EYE_COLOR, 2)
    cv2.rectangle(image, (right.x, right.y), (right.x2, right.y2), _RIGHT_EYE_COLOR, 2)
    label = "TRACKING (held)" if hysteresis else "TRACKING"
    cv2.putText(
        image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
