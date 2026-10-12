# utils/evaluator.py
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef
)


class FraudEvaluator:
    def __init__(self):
        self.metrics_history = []

    def evaluate(self, y_true, y_pred_proba, threshold=None):
        if threshold is None:
            try:
                fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
                youden_index = tpr - fpr
                best_idx = np.argmax(youden_index)
                threshold = thresholds_roc[best_idx]
            except:
                threshold = 0.5

        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {}

        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics['auc'] = auc(fpr, tpr)
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc'] = 0.0
            metrics['roc_auc'] = 0.0

        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['pr_auc'] = 0.0

        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['accuracy'] = (y_pred == y_true).mean()
        metrics['threshold'] = threshold

        try:
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics['mcc'] = 0.0

        self.metrics_history.append(metrics)

        return metrics

    def find_best_threshold(self, y_true, y_pred_proba, metric='f1'):
        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred)
            scores.append(score)

        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]

        return best_threshold, best_score
