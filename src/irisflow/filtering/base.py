"""Internal filter building blocks.

The public :class:`~irisflow.core.interfaces.Filter` Protocol takes a
:class:`~irisflow.core.types.ScreenPoint` and returns a
:class:`~irisflow.core.types.SmoothedPoint`. Chaining filters requires a
uniform intermediate signal: this module defines that signal
(:class:`SignalSample`) and the :class:`SignalFilter` Protocol every
individual filter (outlier/EMA/one-euro) implements.

Only :class:`~irisflow.filtering.chain.FilterChain` implements the
public ``Filter`` Protocol -- individual filters are internal building
blocks composed by the chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["SignalFilter", "SignalSample"]


@dataclass(frozen=True, slots=True)
class SignalSample:
    """The signal passed between filters inside a chain.

    Floats, not ints: rounding to pixel coords only happens at the very
    end of the chain. The intermediate arithmetic (EMA, one-euro) needs
    sub-pixel precision to avoid dead zones.
    """

    x: float
    y: float
    timestamp: float


@runtime_checkable
class SignalFilter(Protocol):
    """One stage of the filter chain.

    ``step`` returns the smoothed / passed-through sample. Implementations
    that reject a sample (outlier) return the last accepted sample -- the
    Sprint 10 safety layer prefers a frozen cursor to a jumping one.
    """

    def step(self, sample: SignalSample) -> SignalSample: ...
    def reset(self) -> None: ...
