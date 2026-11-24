"""Domain-independent DriftRecon components."""

from .cg import H1CGPseudoinverse, periodic_gradient_gram
from .conditioning import CFGConditionEncoder, FiLM, RMSNorm, sinusoidal_scalar_embedding
from .drift import (
    cfg_drift_from_features,
    driftrecon_objective,
    expand_generations,
    group_generations,
    haar_bands,
    sample_cfg_scales,
    scheduled_null_l1_weight,
)
from .ema import EMA, load_model_weights

__all__ = [
    "CFGConditionEncoder",
    "EMA",
    "FiLM",
    "H1CGPseudoinverse",
    "RMSNorm",
    "cfg_drift_from_features",
    "driftrecon_objective",
    "expand_generations",
    "group_generations",
    "haar_bands",
    "load_model_weights",
    "periodic_gradient_gram",
    "sample_cfg_scales",
    "scheduled_null_l1_weight",
    "sinusoidal_scalar_embedding",
]

# Input null-space noise level used during training.  Validation/test use
# 0.02 x the training level (0.001); it can be overridden on the command line.
TRAIN_NULL_NOISE_SIGMA = 0.05
TEST_NULL_NOISE_SIGMA = 0.001
