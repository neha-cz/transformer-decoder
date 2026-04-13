"""Graph decoder package for quantum error correction.

Subpackages
-----------
base    Problem definitions (TannerGraph, DetectorErrorModel, code constructors)
core    Runtime computations (Player, Simulator, Decoders, Layers)
task    Pipeline orchestrations (Training, Evaluation, Losses)
"""

# base ── problem definitions
from .base import (
    TannerGraph,
    DetectorErrorModel,
    repetition_code,
    surface_code,
)

# core ── runtime computations
from .core import (
    Simulator,
    Decoder,
    DecoderConfig,
    GraphDecoder,
    MWPMDecoder,
)

# task ── pipeline orchestrations
from .task import (
    evaluate_logical_error_rate,
    scan_logical_error_rate,
    build_decoder,
    Trainer,
    TrainerConfig,
    DecoderLoss,
    GameRecord,
)

__all__ = [
    # base
    "TannerGraph",
    "DetectorErrorModel",
    "repetition_code",
    "surface_code",
    # core
    "Simulator",
    "Decoder",
    "DecoderConfig",
    "GraphDecoder",
    "MWPMDecoder",
    # task
    "evaluate_logical_error_rate",
    "scan_logical_error_rate",
    "build_decoder",
    "Trainer",
    "TrainerConfig",
    "DecoderLoss",
    "GameRecord",
]
