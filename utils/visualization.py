# utils/visualization.py
"""可视化工具"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def plot_training_history(history_dict, save_path=None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if 'train_loss' in history_dict:
        axes[0].plot(history_dict['train_loss'], label='Train Loss', color='blue')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if 'val_auc' in history_dict:
        axes[1].plot(history_dict['val_auc'], label='Val AUC', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return fig


def plot_roc_curve(y_true, y_pred, save_path=None):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return roc_auc


def plot_confusion_matrix(y_true, y_pred, threshold=None, save_path=None):
    """绘制混淆矩阵"""
    if threshold is None:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        threshold = thresholds_roc[best_idx]

    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Threshold={threshold:.4f})')

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prediction_distribution(y_pred, y_true=None, save_path=None):
    """绘制预测概率分布"""
    # 如果有真实标签，计算最优阈值
    optimal_threshold = 0.5
    if y_true is not None:
        from sklearn.metrics import roc_curve
        try:
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            optimal_threshold = thresholds_roc[best_idx]
        except:
            optimal_threshold = 0.5

    fig, axes = plt.subplots(1, 2 if y_true is not None else 1, figsize=(15, 5))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].hist(y_pred, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Prediction Distribution')
    axes[0].axvline(x=optimal_threshold, color='red', linestyle='--',
                    label=f'Optimal Threshold={optimal_threshold:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if y_true is not None:
        normal_preds = y_pred[y_true == 0]
        fraud_preds = y_pred[y_true == 1]

        axes[1].hist(normal_preds, bins=30, alpha=0.6, label='Normal', color='green')
        axes[1].hist(fraud_preds, bins=30, alpha=0.6, label='Fraud', color='red')
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Prediction Distribution by Class')
        axes[1].axvline(x=optimal_threshold, color='black', linestyle='--',
                       label=f'Optimal Threshold={optimal_threshold:.4f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_model_comparison(model_results, save_path=None):
    """绘制模型对比图"""
    if not model_results:
        print("Warning: No model results provided")
        return None

    model_names = list(model_results.keys())
    metrics = ['auc', 'pr_auc', 'f1', 'mcc']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [model_results[name].get(metric, 0) for name in model_names]

        bars = ax.bar(range(len(model_names)), values,
                     color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return fig


def plot_tsne_embeddings(embeddings, labels=None, save_path=None, n_samples=5000, perplexity=30):
    """绘制t-SNE节点嵌入可视化"""
    if embeddings.shape[0] > n_samples:
        indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
        embeddings_subset = embeddings[indices]
        if labels is not None:
            labels_subset = labels[indices]
        else:
            labels_subset = None
    else:
        embeddings_subset = embeddings
        labels_subset = labels

    print(f"Computing t-SNE for {embeddings_subset.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_subset)

    fig, ax = plt.subplots(figsize=(12, 10))

    if labels_subset is not None:
        unique_labels = np.unique(labels_subset)

        # 使用更清晰的颜色区分Normal和Fraud
        # 假设标签：0=Normal, 1=Fraud, -100=Test
        color_map = {
            0: '#4169E1',      # 蓝色：正常
            1: '#FF4444',      # 红色：欺诈
            -100: '#CCCCCC'    # 灰色：测试集
        }
        label_names = {
            0: 'Normal',
            1: 'Fraud',
            -100: 'Test'
        }

        # 按类别绘制，Normal和Fraud使用更明显的颜色和大小
        for label in unique_labels:
            mask = labels_subset == label
            color = color_map.get(label, '#808080')
            name = label_names.get(label, f'Class {label}')

            # 欺诈节点用更大更明显的标记
            if label == 1:
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=color, label=name, alpha=0.7, s=30, edgecolors='darkred', linewidths=0.5)
            elif label == 0:
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=color, label=name, alpha=0.5, s=15, edgecolors='darkblue', linewidths=0.3)
            else:
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=color, label=name, alpha=0.3, s=10)

        ax.legend(loc='best', fontsize=11)
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)

    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('t-SNE Visualization: Node Embeddings\n(Showing class separation trend)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return fig
