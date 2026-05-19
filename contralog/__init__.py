"""
ContraLog AnomalyDetection Package

A package for log anomaly detection using transformer-based models.
Provides tools for training and inference.
"""

# Model components
from contralog.models import (
    MessageEncoder,
    SequenceEncoder,
    AnomalyModel,
)

# Training utilities
from contralog.trainer import (
    Trainer,
    Tokenizer,
    EarlyStopping,
)

# Data handling
from contralog.data_loaders import (
    LogDataset,
    LogDataset_collate,
)

# Inference components
from contralog.inference_scripts import (
    PointAnomalyDetector,
)

# Embedding utilities
from contralog.log_embedder import (
    LogEmbedder,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "MessageEncoder",
    "SequenceEncoder",
    "AnomalyModel",
    # Training
    "Trainer",
    "Tokenizer",
    "EarlyStopping",
    # Data
    "LogDataset",
    "LogDataset_collate",
    # Inference
    "PointAnomalyDetector",
    # Embedding
    "LogEmbedder",
]
