"""Dataset-level proxy and split leakage audit with NumPy-only dependencies."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from utils.temporal_features import edge_order_permutation_change


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware binary ROC-AUC, implemented without scikit-learn."""
    labels = np.asarray(y_true) == 1
    values = np.nan_to_num(np.asarray(scores, dtype=np.float64))
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both classes")

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def best_orientation_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    value = roc_auc(y_true, scores)
    return float(max(value, 1.0 - value))


def audit_dataset(path: str, include_permutation_check: bool = True) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {"x", "y", "edge_index", "edge_type", "train_mask", "test_mask"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"dataset is missing keys: {missing}")

    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    edges = np.asarray(data["edge_index"])
    train_indices = np.asarray(data["train_mask"], dtype=np.int64)
    test_indices = np.asarray(data["test_mask"], dtype=np.int64)
    train_labels = y[train_indices]
    binary = (train_labels == 1).astype(np.int64)
    fraud_ids = train_indices[binary == 1]
    normal_ids = train_indices[binary == 0]

    node_auc = best_orientation_auc(binary, train_indices)
    perfectly_block_ordered = bool(
        fraud_ids.size
        and normal_ids.size
        and (
            fraud_ids.max() < normal_ids.min()
            or normal_ids.max() < fraud_ids.min()
        )
    )
    feature_auc = {
        f"x[{column}]": round(best_orientation_auc(binary, x[train_indices, column]), 6)
        for column in range(x.shape[1])
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "nodes": int(x.shape[0]),
            "edges": int(edges.shape[0]),
            "raw_features": int(x.shape[1]),
            "observed_edge_types": int(np.unique(data["edge_type"]).size),
            "train_nodes": int(train_indices.size),
            "test_nodes": int(test_indices.size),
            "normal_train_nodes": int(normal_ids.size),
            "fraud_train_nodes": int(fraud_ids.size),
            "fraud_rate": float(binary.mean()),
            "train_test_overlap": int(np.intersect1d(train_indices, test_indices).size),
        },
        "temporal_semantics": {
            "verified_timestamp_columns": [],
            "status": "no documented timestamp is present in the supplied NPZ schema",
        },
        "proxy_audit": {
            "node_id_best_orientation_auc": round(node_auc, 6),
            "fraud_id_min": int(fraud_ids.min()),
            "fraud_id_max": int(fraud_ids.max()),
            "normal_id_min": int(normal_ids.min()),
            "normal_id_max": int(normal_ids.max()),
            "perfect_class_block_ordering": perfectly_block_ordered,
            "raw_feature_best_orientation_auc": feature_auc,
        },
        "risk_level": "critical" if node_auc >= 0.95 else "review",
        "decision": (
            "exclude node ID, neighbor-ID derivatives, and edge row order from "
            "production features; use them only as leakage negative controls"
        ),
    }
    if include_permutation_check:
        report["proxy_audit"]["edge_order_cells_changed_after_row_shuffle"] = round(
            edge_order_permutation_change(edges, x.shape[0]), 6
        )
    return report
