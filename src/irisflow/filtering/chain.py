"""Compose the filter chain declared in YAML.

The chain is a list of ``FilterName`` strings, resolved into concrete
:class:`~irisflow.filtering.base.SignalFilter` instances by
:func:`build_filter_chain`. ``"fixation"``, if present, must be last:
it produces the ``is_fixation`` flag on the returned
:class:`~irisflow.core.types.SmoothedPoint`. Any occurrence earlier in
the list is a config error.

The chain implements the public
:class:`~irisflow.core.interfaces.Filter` Protocol so the pipeline can
treat it as a single stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from irisflow.core.types import ScreenPoint, SmoothedPoint
from irisflow.filtering.base import SignalFilter, SignalSample
from irisflow.filtering.ema import EmaFilter
from irisflow.filtering.fixation import FixationClassifier
from irisflow.filtering.one_euro import OneEuroFilter
from irisflow.filtering.outlier import OutlierRejector

__all__ = [
    "ChainConfig",
    "EmaParams",
    "FilterChain",
    "FixationParams",
    "OneEuroParams",
    "OutlierParams",
    "build_filter_chain",
]


FilterName = Literal["outlier", "ema", "one_euro", "fixation"]


# ---------------------------------------------------------------------------
# Parameter bundles -- kept as plain dataclasses so this module does not
# depend on irisflow.config (sibling layer).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OutlierParams:
    max_velocity_px_per_s: float = 6000.0


@dataclass(frozen=True, slots=True)
class EmaParams:
    alpha: float = 0.4


@dataclass(frozen=True, slots=True)
class OneEuroParams:
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0


@dataclass(frozen=True, slots=True)
class FixationParams:
    velocity_threshold_px_per_s: float = 40.0


@dataclass(frozen=True, slots=True)
class ChainConfig:
    """Order-preserving list of filter names + their params.

    Only the params for filters actually named in ``chain`` are read --
    the rest are ignored, so a caller can pass one config bundle for
    every possible filter without having to strip unused ones first.
    """

    chain: tuple[FilterName, ...] = ("outlier", "one_euro", "fixation")
    outlier: OutlierParams = field(default_factory=OutlierParams)
    ema: EmaParams = field(default_factory=EmaParams)
    one_euro: OneEuroParams = field(default_factory=OneEuroParams)
    fixation: FixationParams = field(default_factory=FixationParams)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
@dataclass
class FilterChain:
    """Public :class:`~irisflow.core.interfaces.Filter` implementation.

    Args:
        signal_filters: Ordered list of SignalFilter instances the chain
            walks each sample through.
        fixation: Optional terminal classifier. When ``None``, the chain
            reports every sample as non-fixation with zero velocity --
            valid, just not very useful downstream.
    """

    signal_filters: list[SignalFilter]
    fixation: FixationClassifier | None = None

    def apply(self, point: ScreenPoint, timestamp: float) -> SmoothedPoint:
        sample = SignalSample(x=float(point.px), y=float(point.py), timestamp=timestamp)
        for stage in self.signal_filters:
            sample = stage.step(sample)
        if self.fixation is not None:
            decision = self.fixation.classify(sample)
            is_fix = decision.is_fixation
            velocity = decision.velocity_px_per_s
        else:
            is_fix = False
            velocity = 0.0
        return SmoothedPoint(
            px=round(sample.x),
            py=round(sample.y),
            is_fixation=is_fix,
            velocity=velocity,
        )

    def reset(self) -> None:
        for stage in self.signal_filters:
            stage.reset()
        if self.fixation is not None:
            self.fixation.reset()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_filter_chain(config: ChainConfig) -> FilterChain:
    """Turn a declarative chain spec into a wired :class:`FilterChain`.

    Raises :class:`ValueError` when the spec is inconsistent:

    * unknown filter name
    * duplicate filter (except by design each name appears at most once)
    * ``"fixation"`` present anywhere except last
    """
    _validate_chain(config.chain)
    signal_stages: list[SignalFilter] = []
    fixation: FixationClassifier | None = None
    for name in config.chain:
        if name == "outlier":
            signal_stages.append(
                OutlierRejector(max_velocity_px_per_s=config.outlier.max_velocity_px_per_s)
            )
        elif name == "ema":
            signal_stages.append(EmaFilter(alpha=config.ema.alpha))
        elif name == "one_euro":
            signal_stages.append(
                OneEuroFilter(
                    min_cutoff=config.one_euro.min_cutoff,
                    beta=config.one_euro.beta,
                    d_cutoff=config.one_euro.d_cutoff,
                )
            )
        elif name == "fixation":
            fixation = FixationClassifier(
                velocity_threshold_px_per_s=config.fixation.velocity_threshold_px_per_s
            )
    return FilterChain(signal_filters=signal_stages, fixation=fixation)


def _validate_chain(chain: Sequence[FilterName]) -> None:
    if not chain:
        raise ValueError("filter chain must have at least one stage")
    seen: set[str] = set()
    for i, name in enumerate(chain):
        if name in seen:
            raise ValueError(f"filter {name!r} appears more than once in chain")
        seen.add(name)
        if name == "fixation" and i != len(chain) - 1:
            raise ValueError(
                f"'fixation' must be the last stage in the chain, got position {i}/{len(chain) - 1}"
            )
