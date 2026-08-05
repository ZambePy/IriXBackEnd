"""Filtering layer (Sprint 9): outlier / EMA / One Euro / fixation, composed by chain."""

from irisflow.filtering.base import SignalFilter, SignalSample
from irisflow.filtering.chain import (
    ChainConfig,
    EmaParams,
    FilterChain,
    FilterName,
    FixationParams,
    OneEuroParams,
    OutlierParams,
    build_filter_chain,
)
from irisflow.filtering.ema import EmaFilter
from irisflow.filtering.fixation import FixationClassifier, FixationDecision
from irisflow.filtering.one_euro import OneEuroFilter
from irisflow.filtering.outlier import OutlierRejector

__all__ = [
    "ChainConfig",
    "EmaFilter",
    "EmaParams",
    "FilterChain",
    "FilterName",
    "FixationClassifier",
    "FixationDecision",
    "FixationParams",
    "OneEuroFilter",
    "OneEuroParams",
    "OutlierParams",
    "OutlierRejector",
    "SignalFilter",
    "SignalSample",
    "build_filter_chain",
]
