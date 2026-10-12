"""Evaluation guards for leakage-resistant fraud experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def assert_disjoint(fit_indices: np.ndarray, evaluation_indices: np.ndarray) -> None:
    """Fail fast when any evaluated node was used to fit the model."""
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    evaluation_indices = np.asarray(evaluation_indices, dtype=np.int64)
    overlap = np.intersect1d(fit_indices, evaluation_indices, assume_unique=False)
    if overlap.size:
        preview = ", ".join(map(str, overlap[:5]))
        raise ValueError(
            f"Evaluation leakage: {overlap.size} nodes occur in both fit and "
            f"evaluation sets (first: {preview})."
        )


def temporal_holdout(
    indices: np.ndarray,
    timestamps: np.ndarray,
    validation_fraction: float = 0.15,
    purge_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split labeled nodes chronologically, with an optional purged gap."""
    indices = np.asarray(indices, dtype=np.int64)
    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 <= purge_fraction < 1 - validation_fraction:
        raise ValueError("purge_fraction leaves no training data")
    if not np.isfinite(timestamps[indices]).all():
        raise ValueError("timestamps contain missing or infinite values")

    ordered = indices[np.argsort(timestamps[indices], kind="stable")]
    validation_size = max(1, int(np.ceil(len(ordered) * validation_fraction)))
    purge_size = int(np.floor(len(ordered) * purge_fraction))
    validation_start = len(ordered) - validation_size
    train_end = validation_start - purge_size
    if train_end <= 0:
        raise ValueError("temporal split leaves no training observations")
    train_indices = ordered[:train_end]
    validation_indices = ordered[validation_start:]
    assert_disjoint(train_indices, validation_indices)
    return train_indices, validation_indices


def save_prediction_artifact(
    path: str | Path,
    indices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    metadata: dict,
) -> None:
    """Persist predictions with the exact evaluated node IDs and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=np.asarray(indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.float64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
