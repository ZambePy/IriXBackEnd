"""Select and construct a :class:`GazeEstimator` from configuration.

The registry is the only place the CLI/pipeline needs to touch to swap
backends — everything else programs against
:class:`irisflow.core.interfaces.GazeEstimator` (SPRINTS.md §7.5). Keeping
the mapping ``config → backend class`` in one file is what makes the
switch a single-line YAML change.

Imports of the heavy backends are lazy so ``irisflow doctor`` /
``preview`` don't drag TensorFlow into the process.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from irisflow.core.exceptions import ConfigError
from irisflow.core.interfaces import GazeEstimator

__all__ = ["BackendName", "build_gaze_estimator", "register_backend"]


BackendName = Literal["keras", "onnx"]


BackendFactory = Callable[..., GazeEstimator]


_BUILTIN_FACTORIES: dict[str, BackendFactory] = {}
_CUSTOM_FACTORIES: dict[str, BackendFactory] = {}


def _keras_factory(
    *, path: Path, channel_order: str, normalization: str
) -> GazeEstimator:
    from irisflow.inference.keras_backend import KerasBackend

    return KerasBackend(path, channel_order=channel_order, normalization=normalization)


def _onnx_factory(
    *, path: Path, channel_order: str, normalization: str
) -> GazeEstimator:
    from irisflow.inference.onnx_backend import OnnxBackend

    return OnnxBackend(path, channel_order=channel_order, normalization=normalization)


_BUILTIN_FACTORIES.update({"keras": _keras_factory, "onnx": _onnx_factory})


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a custom backend factory.

    Useful for tests that want to hand back a fake estimator without
    monkey-patching module attributes.
    """
    _CUSTOM_FACTORIES[name] = factory


def build_gaze_estimator(
    *,
    backend: str,
    model_path: Path,
    channel_order: str = "RGB",
    normalization: str = "unit",
) -> GazeEstimator:
    """Build the estimator declared by ``backend``.

    Raises:
        ConfigError: The backend name is unknown.
    """
    factory = _CUSTOM_FACTORIES.get(backend) or _BUILTIN_FACTORIES.get(backend)
    if factory is None:
        available = sorted(set(_BUILTIN_FACTORIES) | set(_CUSTOM_FACTORIES))
        raise ConfigError(
            f"Unknown inference backend {backend!r}. Known: {available}"
        )
    return factory(
        path=model_path,
        channel_order=channel_order,
        normalization=normalization,
    )
