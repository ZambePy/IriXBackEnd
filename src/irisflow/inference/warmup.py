"""Prime the model so the first real frame doesn't pay the JIT tax.

Both Keras (autograph traces) and ONNX Runtime (kernel selection) do
lazy work on the first call. Running a few dummy inferences at startup
turns a >100 ms first-frame stall into normal steady-state latency —
critical for the "≤ 80 ms fim-a-fim p95" SLA (SPRINTS.md §1.2).
"""

from __future__ import annotations

from irisflow.core.interfaces import GazeEstimator

__all__ = ["warm_up_backend"]


def warm_up_backend(estimator: GazeEstimator, *, iterations: int = 3) -> None:
    """Call :meth:`GazeEstimator.warmup` ``iterations`` times.

    Rationale for ``iterations=3``: TensorFlow's autograph typically
    stabilises after the second call; a third leaves headroom for
    backend-specific quirks (ONNX kernel autotune). More than three is
    just wasted startup time.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    for _ in range(iterations):
        estimator.warmup()
