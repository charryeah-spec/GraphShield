#!/usr/bin/env python3
"""Run leakage-safe nested-holdout GraphSAGE OOF evaluation.

The outer fold is never used for early stopping. Each outer training fold is
split again into fit/early-stop subsets, then evaluated once on the untouched
outer fold. Graph construction remains transductive: node attributes and graph
edges are visible, but outer-fold labels are not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/graphshield-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from stacking import clear_memory, create_model, set_seed, train_one_fold
from utils.data_processor import DataProcessor


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def ranking_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    numerator = 2.0 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(scores))])


def save_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    fold_assignment: np.ndarray,
    output: Path,
) -> None:
    fold_ids = np.unique(fold_assignment)
    mean_fpr = np.linspace(0.0, 1.0, 301)
    mean_recall = np.linspace(0.0, 1.0, 301)
    interpolated_tpr = []
    interpolated_precision = []
    roc_values = []
    pr_values = []
    for fold in fold_ids:
        mask = fold_assignment == fold
        fold_labels = labels[mask]
        fold_probabilities = probabilities[mask]
        fpr, tpr, _ = roc_curve(fold_labels, fold_probabilities)
        precision, recall, _ = precision_recall_curve(
            fold_labels, fold_probabilities
        )
        interpolated_tpr.append(np.interp(mean_fpr, fpr, tpr))
        interpolated_precision.append(
            np.interp(mean_recall, recall[::-1], precision[::-1])
        )
        roc_values.append(roc_auc_score(fold_labels, fold_probabilities))
        pr_values.append(average_precision_score(fold_labels, fold_probabilities))
    tpr_matrix = np.asarray(interpolated_tpr)
    precision_matrix = np.asarray(interpolated_precision)
    mean_tpr, std_tpr = tpr_matrix.mean(axis=0), tpr_matrix.std(axis=0)
    mean_precision = precision_matrix.mean(axis=0)
    prevalence = float(labels.mean())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(
        mean_fpr,
        mean_tpr,
        color="#315efb",
        lw=2,
        label=f"AUC = {np.mean(roc_values):.4f} ± {np.std(roc_values):.4f}",
    )
    axes[0].fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0.0),
        np.minimum(mean_tpr + std_tpr, 1.0),
        color="#315efb",
        alpha=0.14,
        linewidth=0,
    )
    axes[0].plot([0, 1], [0, 1], "--", color="#8a94a6", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="5-fold outer ROC")
    axes[0].legend(frameon=False)

    axes[1].plot(
        mean_recall,
        mean_precision,
        color="#ed6a45",
        lw=2,
        label=f"AP = {np.mean(pr_values):.4f} ± {np.std(pr_values):.4f}",
    )
    axes[1].axhline(prevalence, ls="--", color="#8a94a6", lw=1, label=f"Prevalence = {prevalence:.4f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="5-fold outer precision–recall")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to data.npz")
    parser.add_argument("--output-dir", default="artifacts/graphsage_oof")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-validation", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--full-features",
        action="store_true",
        help="include slow clustering and 3-hop features instead of FAST_MODE",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    config.DATA_PATH = args.data
    config.FAST_MODE = not args.full_features
    config.USE_TIME_FEATURES = False
    config.EPOCHS = args.epochs
    config.PATIENCE = args.patience
    config.HIDDEN_DIM = args.hidden_dim
    config.NUM_LAYERS = args.layers
    config.SEED = args.seed
    config.ENSEMBLE_MODELS = ["GraphSAGE"]

    device = select_device(args.device)
    print(f"Device: {device}", flush=True)
    started = time.time()
    set_seed(config.SEED)

    processor = DataProcessor(config)
    raw = processor.load_data(config.DATA_PATH)
    features = processor.create_enhanced_features(raw)
    data = processor.prepare_pyg_data(raw, features)

    labeled_indices = np.flatnonzero(
        data.train_mask.cpu().numpy() & data.foreground_mask.cpu().numpy()
    )
    labels = (data.y.cpu().numpy()[labeled_indices] == config.FRAUD_CLASS).astype(np.int64)
    test_indices = np.flatnonzero(data.test_mask.cpu().numpy())
    oof_probabilities = np.zeros(len(labeled_indices), dtype=np.float64)
    test_probabilities = np.zeros(len(test_indices), dtype=np.float64)
    fold_assignment = np.full(len(labeled_indices), -1, dtype=np.int64)
    oof_predictions = np.zeros(len(labeled_indices), dtype=np.int64)

    outer = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=config.SEED
    )
    fold_results = []
    for fold, (outer_train_pos, outer_val_pos) in enumerate(
        outer.split(labeled_indices, labels), start=1
    ):
        inner_train_pos, inner_val_pos = train_test_split(
            outer_train_pos,
            test_size=args.inner_validation,
            stratify=labels[outer_train_pos],
            random_state=config.SEED + fold,
        )
        fit_indices = labeled_indices[inner_train_pos]
        early_stop_indices = labeled_indices[inner_val_pos]
        outer_val_indices = labeled_indices[outer_val_pos]
        if np.intersect1d(fit_indices, outer_val_indices).size:
            raise RuntimeError("outer validation nodes leaked into the fit set")
        if np.intersect1d(early_stop_indices, outer_val_indices).size:
            raise RuntimeError("outer validation nodes leaked into early stopping")

        print(
            f"\nOuter fold {fold}/{args.folds}: fit={len(fit_indices)}, "
            f"early_stop={len(early_stop_indices)}, evaluate={len(outer_val_indices)}",
            flush=True,
        )
        set_seed(config.SEED + fold)
        model = create_model(
            "GraphSAGE",
            data.x.size(1),
            config.HIDDEN_DIM,
            config.NUM_CLASSES,
            config,
            int(data.edge_type.max().item()) + 1,
        )
        fold_started = time.time()
        model, all_probabilities, inner_auc = train_one_fold(
            model,
            "GraphSAGE",
            config,
            device,
            data,
            fit_indices,
            early_stop_indices,
            use_edge_type=False,
        )
        outer_probabilities = all_probabilities[outer_val_indices]
        outer_labels = labels[outer_val_pos]
        inner_labels = (
            data.y.cpu().numpy()[early_stop_indices] == config.FRAUD_CLASS
        ).astype(np.int64)
        threshold = select_f1_threshold(
            inner_labels, all_probabilities[early_stop_indices]
        )
        outer_predictions = (outer_probabilities >= threshold).astype(np.int64)
        oof_probabilities[outer_val_pos] = outer_probabilities
        oof_predictions[outer_val_pos] = outer_predictions
        fold_assignment[outer_val_pos] = fold
        test_probabilities += all_probabilities[test_indices] / args.folds
        fold_metrics = ranking_metrics(outer_labels, outer_probabilities)
        fold_metrics.update(classification_metrics(outer_labels, outer_predictions))
        fold_metrics.update(
            {
                "fold": fold,
                "inner_early_stop_roc_auc": float(inner_auc),
                "threshold_selected_on_inner_validation": threshold,
                "fit_nodes": int(len(fit_indices)),
                "early_stop_nodes": int(len(early_stop_indices)),
                "evaluation_nodes": int(len(outer_val_indices)),
                "seconds": round(time.time() - fold_started, 3),
            }
        )
        fold_results.append(fold_metrics)
        print(
            f"Outer ROC-AUC={fold_metrics['roc_auc']:.4f}, "
            f"PR-AUC={fold_metrics['pr_auc']:.4f}",
            flush=True,
        )
        del model, all_probabilities
        clear_memory()

    if np.any(fold_assignment < 0):
        raise RuntimeError("some labeled nodes did not receive an OOF prediction")

    pooled_ranking = ranking_metrics(labels, oof_probabilities)
    cross_fitted_classification = classification_metrics(labels, oof_predictions)
    fold_auc_values = np.array([row["roc_auc"] for row in fold_results])
    fold_pr_values = np.array([row["pr_auc"] for row in fold_results])
    total_seconds = time.time() - started
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": f"transductive_nested_holdout_{args.folds}_fold_oof",
        "outer_fold_is_never_used_for_early_stopping": True,
        "time_features_enabled": False,
        "node_id_or_edge_order_proxies_enabled": False,
        "dataset_sha256": file_sha256(args.data),
        "dataset": {
            "nodes": int(data.num_nodes),
            "edges": int(data.num_edges),
            "features": int(data.x.size(1)),
            "labeled_nodes": int(len(labeled_indices)),
            "fraud_nodes": int(labels.sum()),
            "fraud_rate": float(labels.mean()),
        },
        "configuration": {
            "folds": args.folds,
            "inner_validation_fraction": args.inner_validation,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "fast_mode": config.FAST_MODE,
            "seed": config.SEED,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "primary_metrics": "mean and standard deviation across outer folds",
        "pooled_oof_ranking_diagnostic": pooled_ranking,
        "cross_fitted_threshold_metrics": cross_fitted_classification,
        "fold_summary": {
            "roc_auc_mean": float(fold_auc_values.mean()),
            "roc_auc_std": float(fold_auc_values.std(ddof=0)),
            "pr_auc_mean": float(fold_pr_values.mean()),
            "pr_auc_std": float(fold_pr_values.std(ddof=0)),
            "f1_mean": float(np.mean([row["f1"] for row in fold_results])),
            "f1_std": float(np.std([row["f1"] for row in fold_results])),
            "mcc_mean": float(np.mean([row["mcc"] for row in fold_results])),
            "mcc_std": float(np.std([row["mcc"] for row in fold_results])),
        },
        "folds": fold_results,
        "runtime_seconds": round(total_seconds, 3),
        "confusion_matrix_cross_fitted_thresholds": confusion_matrix(
            labels, oof_predictions
        ).tolist(),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        labeled_indices=labeled_indices,
        labels=labels,
        fold=fold_assignment,
        oof_probability=oof_probabilities,
        oof_prediction=oof_predictions,
        test_indices=test_indices,
        test_probability=test_probabilities,
    )
    save_curves(
        labels, oof_probabilities, fold_assignment, output_dir / "oof_curves.png"
    )
    print(
        "\nFold summary:", json.dumps(metrics["fold_summary"], indent=2), flush=True
    )
    print(
        "Cross-fitted threshold metrics:",
        json.dumps(cross_fitted_classification, indent=2),
        flush=True,
    )
    print(f"Saved to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
