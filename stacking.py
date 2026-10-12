# stacking.py
import os
import json
import gc
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from config import Config
from utils.data_processor import DataProcessor
from utils.trainer import FocalLoss
from models.advanced_models import EnhancedGCN, EnhancedGraphSAGE, EnhancedGIN, EnhancedRGCN, EnhancedGAT

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Using LogisticRegression as meta-learner.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_model(model_type, in_channels, hidden_channels, out_channels, config, num_relations=None):
    dropout = getattr(config, 'MODEL_DROPOUT', {}).get(model_type, config.DROPOUT)

    if model_type == 'GCN':
        return EnhancedGCN(in_channels, hidden_channels, out_channels,
                           num_layers=config.NUM_LAYERS, dropout=dropout)
    elif model_type == 'GraphSAGE':
        return EnhancedGraphSAGE(in_channels, hidden_channels, out_channels,
                                 num_layers=config.NUM_LAYERS, dropout=dropout)
    elif model_type == 'GIN':
        return EnhancedGIN(in_channels, hidden_channels, out_channels,
                           num_layers=config.NUM_LAYERS, dropout=dropout)
    elif model_type == 'RGCN':
        if num_relations is None:
            num_relations = getattr(config, 'NUM_RELATIONS', 11)
        return EnhancedRGCN(in_channels, hidden_channels, out_channels,
                            num_relations=num_relations,
                            num_layers=getattr(config, 'RGCN_NUM_LAYERS', 2),
                            dropout=dropout,
                            num_bases=getattr(config, 'NUM_BASES', 8))
    elif model_type == 'GAT':
        return EnhancedGAT(in_channels, hidden_channels, out_channels,
                           num_layers=config.NUM_LAYERS, dropout=dropout,
                           heads=getattr(config, 'GAT_HEADS', 8))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_fold(model, model_type, config, device, data, train_idx, val_idx, use_edge_type=False):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6)
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA,
                          label_smoothing=config.LABEL_SMOOTHING)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = None
    if use_edge_type and hasattr(data, 'edge_type') and data.edge_type is not None:
        edge_type = data.edge_type.to(device)

    # GCN使用SparseTensor
    adj_t = None
    if hasattr(data, 'adj_t') and data.adj_t is not None and model_type == 'GCN':
        adj_t = data.adj_t.to(device)

    labels = data.y.to(device)

    num_nodes = data.x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask[val_idx] = True

    foreground = (labels == config.NORMAL_CLASS) | (labels == config.FRAUD_CLASS)
    train_valid = train_mask & foreground
    val_valid = val_mask & foreground

    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0

    # 减少epoch数，因为要训练多个fold
    epochs = min(config.EPOCHS, 150)

    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()

        if adj_t is not None:
            logits = model(x, adj_t=adj_t)
        else:
            logits = model(x, edge_index, edge_type)
        logits_fg = logits[train_valid][:, :2]
        binary_labels = (labels[train_valid] == config.FRAUD_CLASS).long()

        loss = criterion(logits_fg, binary_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            if adj_t is not None:
                logits = model(x, adj_t=adj_t)
            else:
                logits = model(x, edge_index, edge_type)
            logits_val = logits[val_valid][:, :2]
            probs = torch.softmax(logits_val, dim=1)[:, 1].cpu().numpy()
            val_labels_np = (labels[val_valid] == config.FRAUD_CLASS).long().cpu().numpy()

            try:
                # 标签和概率格式检查
                if len(np.unique(val_labels_np)) < 2:
                    val_auc = 0.0  # 只有一类标签，无法计算AUC
                else:
                    fpr, tpr, _ = roc_curve(val_labels_np, probs)
                    val_auc = sk_auc(fpr, tpr)
                    # 验证AUC值在合理范围内
                    if np.isnan(val_auc) or val_auc < 0 or val_auc > 1:
                        val_auc = 0.0
            except Exception as e:
                val_auc = 0.0

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            break

    # 加载模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # 获取所有节点的预测概率
    model.eval()
    with torch.no_grad():
        if adj_t is not None:
            logits = model(x, adj_t=adj_t)
        else:
            logits = model(x, edge_index, edge_type)
        all_probs = torch.softmax(logits[:, :2], dim=1)[:, 1].cpu().numpy()

    del x, edge_index, labels
    if edge_type is not None:
        del edge_type
    if adj_t is not None:
        del adj_t
    clear_memory()

    return model, all_probs, best_val_auc


def select_key_features(node_features, n_select=15):
    if node_features.shape[1] <= n_select:
        return node_features, np.arange(node_features.shape[1])

    feature_variance = np.var(node_features, axis=0)
    top_indices = np.argsort(feature_variance)[-n_select:]
    top_indices = np.sort(top_indices)
    return node_features[:, top_indices], top_indices


def generate_oof_predictions(config, device, data, model_types, num_relations, n_folds=5):
    train_mask = data.train_mask.cpu().numpy()
    foreground_mask = data.foreground_mask.cpu().numpy()
    valid_train_mask = train_mask & foreground_mask

    train_indices = np.where(valid_train_mask)[0]
    train_labels = (data.y.cpu().numpy()[train_indices] == config.FRAUD_CLASS).astype(int)

    test_indices = np.where(data.test_mask.cpu().numpy())[0]

    in_channels = data.x.size(1)

    oof_preds = np.zeros((len(train_indices), len(model_types)))
    test_preds = np.zeros((len(test_indices), len(model_types)))

    train_node_features_raw = data.x[train_indices].cpu().numpy()
    test_node_features_raw = data.x[test_indices].cpu().numpy()

    train_node_features, selected_indices = select_key_features(train_node_features_raw, n_select=15)
    test_node_features = test_node_features_raw[:, selected_indices]

    print(f"\n{'=' * 60}")
    print(f"Generating OOF predictions with {n_folds}-Fold CV")
    print(f"{'=' * 60}")
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    print(f"Fraud ratio: {train_labels.mean() * 100:.2f}%")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    fold_aucs_all = {model_type: [] for model_type in model_types}

    for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(train_indices, train_labels)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        set_seed(config.SEED + fold)
        clear_memory()

        train_node_idx = train_indices[train_fold_idx]
        val_node_idx = train_indices[val_fold_idx]

        for model_idx, model_type in enumerate(model_types):
            print(f"\n  Model {model_idx + 1}/{len(model_types)}: {model_type}", end=" ")

            use_edge_type = (model_type == 'RGCN')
            model = create_model(model_type, in_channels, config.HIDDEN_DIM,
                                 config.NUM_CLASSES, config, num_relations)

            model, all_probs, fold_auc = train_one_fold(
                model, model_type, config, device, data,
                train_node_idx, val_node_idx,
                use_edge_type=use_edge_type
            )

            fold_aucs_all[model_type].append(fold_auc)
            print(f"- AUC: {fold_auc:.4f}")

            oof_preds[val_fold_idx, model_idx] = all_probs[val_node_idx]
            test_preds[:, model_idx] += all_probs[test_indices]

            del model
            clear_memory()

    for model_idx in range(len(model_types)):
        test_preds[:, model_idx] /= n_folds

    print(f"\n{'=' * 60}")
    print("Cross-Validation Results")
    print(f"{'=' * 60}")
    for model_idx, model_type in enumerate(model_types):
        mean_auc = np.mean(fold_aucs_all[model_type])
        std_auc = np.std(fold_aucs_all[model_type])
        print(f"  {model_type}: {mean_auc:.4f} (+/- {std_auc:.4f})")

    return oof_preds, test_preds, train_indices, test_indices, train_labels, train_node_features, test_node_features


def create_meta_features(preds, num_models, node_features=None, model_names=None):
    features = []

    if model_names is None:
        model_names = [f"model_{i}" for i in range(num_models)]

    features.append(preds)

    # 统计特征
    pred_mean = np.mean(preds, axis=1, keepdims=True)
    pred_std = np.std(preds, axis=1, keepdims=True)
    pred_max = np.max(preds, axis=1, keepdims=True)
    pred_min = np.min(preds, axis=1, keepdims=True)
    pred_median = np.median(preds, axis=1, keepdims=True)
    pred_range = pred_max - pred_min
    features.extend([pred_mean, pred_std, pred_max, pred_min, pred_median, pred_range])

    # 模型分歧度
    pred_cv = pred_std / (pred_mean + 1e-8)
    features.append(pred_cv)

    # 预测熵
    pred_range_check = pred_range.flatten()
    pred_entropy = np.zeros((preds.shape[0], 1), dtype=np.float32)
    valid_mask = pred_range_check > 1e-6
    if valid_mask.sum() > 0:
        preds_valid = preds[valid_mask]
        pred_shifted = preds_valid - preds_valid.min(axis=1, keepdims=True)
        pred_sum = pred_shifted.sum(axis=1, keepdims=True)
        pred_sum = np.maximum(pred_sum, 1e-8)
        pred_normalized = pred_shifted / pred_sum
        pred_normalized = np.clip(pred_normalized, 1e-10, 1.0)
        entropy_valid = -np.sum(pred_normalized * np.log(pred_normalized), axis=1, keepdims=True)
        pred_entropy[valid_mask, 0] = entropy_valid.flatten()

    features.append(pred_entropy)

    # 模型间差异
    if num_models >= 2:
        for i in range(num_models):
            for j in range(i + 1, num_models):
                diff = (preds[:, i] - preds[:, j]).reshape(-1, 1)
                features.append(diff)

    # 节点特征与交互
    if node_features is not None:
        # 特征已在generate_oof_predictions中选择，直接使用
        key_features = node_features
        n_key_features = key_features.shape[1]
        features.append(key_features)

        # 预测 × 度特征交互
        if n_key_features >= 1:
            degree_like = key_features[:, 0:1]
            for i in range(num_models):
                interaction = preds[:, i:i+1] * degree_like
                features.append(interaction)

            if n_key_features >= 2:
                degree_like2 = key_features[:, 1:2]
                for i in range(num_models):
                    interaction2 = preds[:, i:i+1] * degree_like2
                    features.append(interaction2)

        # 预测 × 聚类特征交互
        if n_key_features >= 5:
            cluster_like = key_features[:, 4:5]
            for i in range(num_models):
                interaction = preds[:, i:i+1] * cluster_like
                features.append(interaction)

        # 预测差异 × 分歧度
        for i in range(num_models):
            pred_deviation = (preds[:, i:i+1] - pred_mean)
            interaction = pred_deviation * pred_std
            features.append(interaction)

    # 排名特征
    pred_ranks = np.argsort(np.argsort(-preds, axis=1), axis=1).astype(float)
    pred_ranks_normalized = pred_ranks / (num_models - 1 + 1e-8)
    features.append(pred_ranks_normalized)

    # 一致性特征
    high_consensus = (preds > 0.5).sum(axis=1, keepdims=True) / num_models
    low_consensus = (preds < 0.5).sum(axis=1, keepdims=True) / num_models
    features.extend([high_consensus, low_consensus])

    # 预测方差
    pred_variance = np.var(preds, axis=1, keepdims=True)
    features.append(pred_variance)

    # 极值特征
    extreme_high = (preds > 0.9).any(axis=1, keepdims=True).astype(float)
    extreme_low = (preds < 0.1).any(axis=1, keepdims=True).astype(float)
    features.extend([extreme_high, extreme_low])

    # 加权平均
    pred_weights = np.exp(preds * 5)
    pred_weights = pred_weights / (pred_weights.sum(axis=1, keepdims=True) + 1e-8)
    weighted_mean = np.sum(preds * pred_weights, axis=1, keepdims=True)
    features.append(weighted_mean)
    result = np.hstack(features)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def train_meta_learner(oof_preds, train_labels, test_preds, config, train_node_features=None, test_node_features=None):
    print("\n" + "=" * 60)
    print("Training Meta-Learner")
    print("=" * 60)

    num_models = oof_preds.shape[1]
    model_names = config.ENSEMBLE_MODELS[:num_models] if hasattr(config, 'ENSEMBLE_MODELS') else [f"model_{i}" for i in range(num_models)]

    train_meta_features = create_meta_features(oof_preds, num_models, train_node_features, model_names)
    test_meta_features = create_meta_features(test_preds, num_models, test_node_features, model_names)

    print(f"Meta-features: {train_meta_features.shape[1]} features")

    if HAS_LIGHTGBM:
        print("Using LightGBM as meta-learner")

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': config.SEED,
            'n_jobs': -1,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'max_depth': 5,
            'min_data_in_leaf': 10
        }

        train_data = lgb.Dataset(train_meta_features, label=train_labels)

        # 使用StratifiedKFold进行CV，获取验证集预测
        skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
        cv_preds = np.zeros(len(train_labels))
        cv_aucs = []
        cv_iters = []

        for train_idx, val_idx in skf_meta.split(train_meta_features, train_labels):
            train_fold = lgb.Dataset(train_meta_features[train_idx], label=train_labels[train_idx])
            val_fold = lgb.Dataset(train_meta_features[val_idx], label=train_labels[val_idx], reference=train_fold)

            model_fold = lgb.train(
                params,
                train_fold,
                num_boost_round=500,
                valid_sets=[val_fold],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )

            best_iter_fold = model_fold.best_iteration if hasattr(model_fold, 'best_iteration') and model_fold.best_iteration > 0 else 500
            cv_iters.append(best_iter_fold)

            cv_preds[val_idx] = model_fold.predict(train_meta_features[val_idx])

            if len(np.unique(train_labels[val_idx])) >= 2:
                fpr, tpr, _ = roc_curve(train_labels[val_idx], cv_preds[val_idx])
                fold_auc = sk_auc(fpr, tpr)
                if not (np.isnan(fold_auc) or fold_auc < 0 or fold_auc > 1):
                    cv_aucs.append(fold_auc)

        best_auc = np.mean(cv_aucs) if cv_aucs else 0.0
        print(f"CV AUC: {best_auc:.4f} (+/- {np.std(cv_aucs) if cv_aucs else 0.0:.4f})")

        # 使用CV平均迭代次数
        best_iter = int(np.mean(cv_iters)) if cv_iters else 500

        meta_model = lgb.train(
            params,
            train_data,
            num_boost_round=best_iter,
            callbacks=[lgb.log_evaluation(0)]
        )

        meta_oof_pred = cv_preds
        meta_test_pred = meta_model.predict(test_meta_features)

        print("\nTop Feature Importance:")
        importance = meta_model.feature_importance(importance_type='gain')
        feature_names = []
        num_models = oof_preds.shape[1]
        model_types = model_names

        for i, model_type in enumerate(model_types):
            feature_names.append(f"{model_type}_pred")

        feature_names.extend(['pred_mean', 'pred_std', 'pred_max', 'pred_min', 'pred_median', 'pred_range'])
        feature_names.append('pred_cv')
        feature_names.append('pred_entropy')

        if num_models >= 2:
            for i in range(num_models):
                for j in range(i + 1, num_models):
                    feature_names.append(f"diff_{model_types[i]}_{model_types[j]}")

        if train_node_features is not None:
            n_key = min(15, train_node_features.shape[1])
            for i in range(n_key):
                feature_names.append(f"node_feat_{i}")

            if n_key >= 1:
                for i, model_type in enumerate(model_types):
                    feature_names.append(f"{model_type}_x_degree1")

            if n_key >= 2:
                for i, model_type in enumerate(model_types):
                    feature_names.append(f"{model_type}_x_degree2")

            if n_key >= 5:
                for i, model_type in enumerate(model_types):
                    feature_names.append(f"{model_type}_x_cluster")

            for i, model_type in enumerate(model_types):
                feature_names.append(f"{model_type}_deviation_weighted")

        for i, model_type in enumerate(model_types):
            feature_names.append(f"rank_{model_type}")

        feature_names.extend(['high_consensus', 'low_consensus'])
        feature_names.append('pred_variance')
        feature_names.extend(['extreme_high', 'extreme_low'])
        feature_names.append('weighted_mean')

        # 验证维度
        expected_dim = len(feature_names)
        actual_dim = train_meta_features.shape[1]
        if expected_dim != actual_dim:
            print(f"  Warning: Feature name count ({expected_dim}) != feature dim ({actual_dim})")
            # 用通用名称填充
            feature_names = [f"feat_{i}" for i in range(actual_dim)]

        top_indices = np.argsort(importance)[-15:][::-1]
        print("\nTop 15 Most Important Features:")
        for idx in top_indices:
            if idx < len(feature_names):
                print(f"  {feature_names[idx]}: {importance[idx]:.2f}")

        # 交互特征贡献度
        interaction_importance = 0
        for idx, name in enumerate(feature_names):
            if '_x_' in name or '_deviation_' in name:
                interaction_importance += importance[idx]

        total_importance = importance.sum()
        if total_importance > 0:
            print(f"\nInteraction features contribution: {interaction_importance/total_importance*100:.1f}%")

    else:
        print("Using LogisticRegression as meta-learner")

        # 使用交叉验证获取无偏的OOF预测
        skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
        cv_preds = np.zeros(len(train_labels))
        cv_aucs = []

        for train_idx, val_idx in skf_meta.split(train_meta_features, train_labels):
            model_fold = LogisticRegression(
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=config.SEED
            )
            model_fold.fit(train_meta_features[train_idx], train_labels[train_idx])
            cv_preds[val_idx] = model_fold.predict_proba(train_meta_features[val_idx])[:, 1]

            if len(np.unique(train_labels[val_idx])) >= 2:
                fpr, tpr, _ = roc_curve(train_labels[val_idx], cv_preds[val_idx])
                fold_auc = sk_auc(fpr, tpr)
                if not (np.isnan(fold_auc) or fold_auc < 0 or fold_auc > 1):
                    cv_aucs.append(fold_auc)

        meta_model = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=config.SEED
        )
        meta_model.fit(train_meta_features, train_labels)

        meta_oof_pred = cv_preds
        meta_test_pred = meta_model.predict_proba(test_meta_features)[:, 1]

        print("\nModel Coefficients:")
        for i, coef in enumerate(meta_model.coef_[0]):
            print(f"  Model {i}: {coef:.4f}")

        best_auc = np.mean(cv_aucs) if cv_aucs else 0.0
        print(f"CV AUC: {best_auc:.4f} (+/- {np.std(cv_aucs) if cv_aucs else 0.0:.4f})")

    return meta_model, meta_oof_pred, meta_test_pred


def main():
    config = Config()
    set_seed(config.SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # ============ 数据加载 ============
    print("\n" + "=" * 60)
    print("Loading and processing data...")
    print("=" * 60)

    processor = DataProcessor(config)
    raw_data = processor.load_data(config.DATA_PATH)

    print("\nEngineering features...")
    enhanced_features = processor.create_enhanced_features(raw_data)

    print("\nPreparing PyG data...")
    data = processor.prepare_pyg_data(raw_data, enhanced_features)

    in_channels = data.x.shape[1]
    print(f"Feature dimension: {in_channels}")

    # 获取边类型数量
    if hasattr(data, 'edge_type') and data.edge_type is not None:
        num_relations = int(data.edge_type.max().item()) + 1
        print(f"Number of edge types: {num_relations}")
    else:
        num_relations = 1

    config.NUM_RELATIONS = num_relations

    # 生成OOF预测
    model_types = config.ENSEMBLE_MODELS
    n_folds = 5  # K折数

    oof_preds, test_preds, train_indices, test_indices, train_labels, train_node_features, test_node_features = generate_oof_predictions(
        config, device, data, model_types, num_relations, n_folds=n_folds
    )

    np.save(os.path.join(config.SAVE_DIR, 'oof_preds.npy'), oof_preds)
    np.save(os.path.join(config.SAVE_DIR, 'test_preds.npy'), test_preds)
    np.save(os.path.join(config.SAVE_DIR, 'train_labels.npy'), train_labels)
    np.save(os.path.join(config.SAVE_DIR, 'test_indices.npy'), test_indices)

    # 评估各模型和简单平均
    print("\n" + "=" * 60)
    print("Base Model Performance (OOF)")
    print("=" * 60)

    model_aucs = []
    for i, model_type in enumerate(model_types):
        # 标签和概率格式检查
        if len(np.unique(train_labels)) < 2:
            model_auc = 0.0
        else:
            fpr, tpr, _ = roc_curve(train_labels, oof_preds[:, i])
            model_auc = sk_auc(fpr, tpr)
            if np.isnan(model_auc) or model_auc < 0 or model_auc > 1:
                model_auc = 0.0
        model_aucs.append(model_auc)
        print(f"  {model_type}: AUC = {model_auc:.4f}")

    # 简单平均
    simple_avg = oof_preds.mean(axis=1)
    if len(np.unique(train_labels)) >= 2:
        fpr, tpr, _ = roc_curve(train_labels, simple_avg)
        avg_auc = sk_auc(fpr, tpr)
        if np.isnan(avg_auc) or avg_auc < 0 or avg_auc > 1:
            avg_auc = 0.0
    else:
        avg_auc = 0.0
    print(f"\n  Simple Average: AUC = {avg_auc:.4f}")

    weights = np.array(model_aucs)
    weights = np.exp((weights - weights.max()) * 10)
    weights = weights / weights.sum()
    weighted_avg = np.average(oof_preds, axis=1, weights=weights)
    if len(np.unique(train_labels)) >= 2:
        fpr, tpr, _ = roc_curve(train_labels, weighted_avg)
        weighted_auc = sk_auc(fpr, tpr)
        if np.isnan(weighted_auc) or weighted_auc < 0 or weighted_auc > 1:
            weighted_auc = 0.0
    else:
        weighted_auc = 0.0
    print(f"  Weighted Average: AUC = {weighted_auc:.4f}")
    print(f"    Weights: {weights}")

    # 训练Stacking元学习器
    meta_model, meta_oof_pred, meta_test_pred = train_meta_learner(
        oof_preds, train_labels, test_preds, config, train_node_features, test_node_features
    )

    # Stacking OOF AUC
    if len(np.unique(train_labels)) >= 2:
        fpr, tpr, _ = roc_curve(train_labels, meta_oof_pred)
        stacking_auc = sk_auc(fpr, tpr)
        if np.isnan(stacking_auc) or stacking_auc < 0 or stacking_auc > 1:
            stacking_auc = 0.0
    else:
        stacking_auc = 0.0
    print(f"\n  Stacking: AUC = {stacking_auc:.4f}")

    # 使用评估器进行详细评估
    from utils.evaluator import FraudEvaluator
    evaluator = FraudEvaluator()
    # Threshold-dependent metrics use a fixed, predeclared threshold. ROC-AUC
    # and PR-AUC remain the primary metrics for the imbalanced task.
    stacking_metrics = evaluator.evaluate(train_labels, meta_oof_pred, threshold=0.5)
    print(f"\n  Stacking Detailed Metrics:")
    print(f"    AUC: {stacking_metrics['auc']:.4f}")
    print(f"    PR AUC: {stacking_metrics['pr_auc']:.4f}")
    print(f"    F1: {stacking_metrics['f1']:.4f}")
    print(f"    Precision: {stacking_metrics['precision']:.4f}")
    print(f"    Recall: {stacking_metrics['recall']:.4f}")
    print(f"    MCC: {stacking_metrics['mcc']:.4f}")

    model_results = {
        model_type: evaluator.evaluate(train_labels, oof_preds[:, i], threshold=0.5)
        for i, model_type in enumerate(model_types)
    }
    model_results['Simple Avg'] = evaluator.evaluate(
        train_labels, simple_avg, threshold=0.5
    )
    model_results['Weighted Avg'] = evaluator.evaluate(
        train_labels, weighted_avg, threshold=0.5
    )
    model_results['Stacking'] = stacking_metrics

    os.makedirs(config.LOG_DIR, exist_ok=True)
    oof_metrics = {
        'protocol': 'stratified_5_fold_oof',
        'seed': config.SEED,
        'folds': n_folds,
        'labeled_nodes': int(len(train_labels)),
        'fraud_rate': float(np.mean(train_labels)),
        'time_features_enabled': bool(config.USE_TIME_FEATURES),
        'proxy_features_enabled': False,
        'threshold_policy': 'fixed_0.5_for_threshold_metrics',
        'metrics': {
            name: {key: float(value) for key, value in values.items()}
            for name, values in model_results.items()
        },
    }
    with open(os.path.join(config.LOG_DIR, 'oof_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(oof_metrics, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        os.path.join(config.LOG_DIR, 'oof_predictions.npz'),
        indices=train_indices,
        labels=train_labels,
        base_predictions=oof_preds,
        stacking_predictions=meta_oof_pred,
        model_names=np.asarray(model_types),
    )

    # 可视化
    try:
        from utils.visualization import (
            plot_roc_curve, plot_confusion_matrix, plot_prediction_distribution,
            plot_model_comparison
        )
        os.makedirs(config.LOG_DIR, exist_ok=True)
        plot_roc_curve(train_labels, meta_oof_pred,
                      save_path=os.path.join(config.LOG_DIR, 'stacking_roc_curve.png'))
        # 使用预先声明的固定阈值，避免在 OOF 预测上调阈值和报分
        plot_confusion_matrix(train_labels, meta_oof_pred, threshold=0.5,
                             save_path=os.path.join(config.LOG_DIR, 'stacking_confusion_matrix.png'))
        plot_prediction_distribution(meta_oof_pred, train_labels,
                                    save_path=os.path.join(config.LOG_DIR, 'stacking_pred_distribution.png'))

        # 模型对比图（只使用上面保存的 OOF 指标）
        plot_model_comparison(model_results,
                             save_path=os.path.join(config.LOG_DIR, 'model_comparison.png'))

        print(f"\n  Visualizations saved to {config.LOG_DIR}")
    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")

    # 选择融合方法并生成提交文件
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Simple Average:   {avg_auc:.4f}")
    print(f"  Weighted Average: {weighted_auc:.4f}")
    print(f"  Stacking:         {stacking_auc:.4f}")

    # 选择融合方法
    methods = {
        'simple_avg': (avg_auc, test_preds.mean(axis=1)),
        'weighted_avg': (weighted_auc, np.average(test_preds, axis=1, weights=weights)),
        'stacking': (stacking_auc, meta_test_pred)
    }

    best_method = max(methods.keys(), key=lambda k: methods[k][0])
    best_auc, best_test_pred = methods[best_method]
    print(f"\n  Best method: {best_method} (AUC = {best_auc:.4f})")

    # ============ 生成提交文件 ============
    print("\n" + "=" * 60)
    print("Generating submission file...")
    print("=" * 60)

    # 保存结果
    submission = pd.DataFrame({
        'index': test_indices.astype(int),
        'predict': best_test_pred.astype(float)
    })

    # 数据类型检查
    submission['index'] = submission['index'].astype(int)
    submission['predict'] = submission['predict'].astype(float)

    # 限制predict在[0,1]范围内
    submission['predict'] = submission['predict'].clip(0.0, 1.0)

    result_path = os.path.join(config.RESULT_DIR, 'result.csv')
    submission.to_csv(result_path, index=False)
    print(f"\nBest method ({best_method}) result saved to: {result_path}")
    print(f"  AUC: {best_auc:.4f}")

    # 预测统计
    print(f"\nPrediction statistics ({best_method}):")
    print(f"  Min:  {best_test_pred.min():.6f}")
    print(f"  Max:  {best_test_pred.max():.6f}")
    print(f"  Mean: {best_test_pred.mean():.6f}")
    print(f"  Std:  {best_test_pred.std():.6f}")

    for threshold in [0.3, 0.5, 0.7]:
        count = (best_test_pred > threshold).sum()
        print(f"  > {threshold}: {count} ({100 * count / len(best_test_pred):.2f}%)")

    print("\n" + "=" * 60)
    print("Stacking completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
