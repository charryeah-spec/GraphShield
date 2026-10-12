"""Explicit temporal features and quarantined proxy diagnostics.

The original course project inferred time from node IDs and edge row order.
Those values are useful as *negative controls*, but are not timestamps. This
module only promotes columns with documented semantics to production features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalFeatureSpec:
    column: int
    name: str
    semantics: str

    def validate(self) -> None:
        if self.column < 0:
            raise ValueError("timestamp column must be non-negative")
        if not self.name.strip() or not self.semantics.strip():
            raise ValueError("every timestamp needs a name and documented semantics")


class ExplicitTemporalFeatureBuilder:
    """Fit time transformations on training nodes and reuse them unchanged."""

    def __init__(self, specs: list[TemporalFeatureSpec]):
        if not specs:
            raise ValueError(
                "No verified timestamp columns were configured. Keep temporal "
                "features disabled instead of substituting node IDs."
            )
        for spec in specs:
            spec.validate()
        self.specs = specs
        self._bounds: dict[int, tuple[float, float]] = {}

    def fit(self, x: np.ndarray, fit_indices: np.ndarray) -> "ExplicitTemporalFeatureBuilder":
        x = np.asarray(x)
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        if fit_indices.size == 0:
            raise ValueError("fit_indices cannot be empty")
        for spec in self.specs:
            if spec.column >= x.shape[1]:
                raise IndexError(f"timestamp column {spec.column} is out of bounds")
            values = np.asarray(x[fit_indices, spec.column], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"{spec.name} has no finite training values")
            self._bounds[spec.column] = (float(finite.min()), float(finite.max()))
        return self

    def transform(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        if len(self._bounds) != len(self.specs):
            raise RuntimeError("fit must be called before transform")
        x = np.asarray(x)
        columns: list[np.ndarray] = []
        names: list[str] = []
        for spec in self.specs:
            low, high = self._bounds[spec.column]
            raw = np.asarray(x[:, spec.column], dtype=np.float64)
            scale = high - low
            normalized = np.zeros_like(raw) if scale == 0 else (raw - low) / scale
            elapsed = np.maximum(raw - low, 0.0)
            columns.extend([normalized, np.log1p(elapsed)])
            names.extend([f"{spec.name}_normalized", f"{spec.name}_log_elapsed"])
        matrix = np.column_stack(columns).astype(np.float32)
        return np.nan_to_num(matrix), names

    def fit_transform(
        self, x: np.ndarray, fit_indices: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        return self.fit(x, fit_indices).transform(x)


def node_id_proxy(num_nodes: int) -> np.ndarray:
    """Return a negative-control feature; never treat this as event time."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    return np.arange(num_nodes, dtype=np.float64) / max(num_nodes - 1, 1)


def edge_order_proxy(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Construct first/last row positions solely for leakage diagnostics."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edge_index must have shape (E, 2)")
    edge_count = len(edges)
    first_src = np.full(num_nodes, edge_count, dtype=np.float64)
    first_dst = np.full(num_nodes, edge_count, dtype=np.float64)
    last_src = np.zeros(num_nodes, dtype=np.float64)
    last_dst = np.zeros(num_nodes, dtype=np.float64)
    for position, (source, target) in enumerate(edges):
        first_src[source] = min(first_src[source], position)
        first_dst[target] = min(first_dst[target], position)
        last_src[source] = position
        last_dst[target] = position
    denominator = max(edge_count - 1, 1)
    return np.column_stack(
        [first_src, first_dst, last_src, last_dst]
    ).astype(np.float64) / denominator


def edge_order_permutation_change(
    edge_index: np.ndarray, num_nodes: int, seed: int = 42
) -> float:
    """Measure how much an alleged time feature changes under harmless row shuffle."""
    edges = np.asarray(edge_index)
    original = edge_order_proxy(edges, num_nodes)
    rng = np.random.default_rng(seed)
    shuffled = edge_order_proxy(edges[rng.permutation(len(edges))], num_nodes)
    return float(np.mean(~np.isclose(original, shuffled, rtol=0.0, atol=1e-12)))
