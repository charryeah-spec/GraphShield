# train.py
import os
import json
import gc
import random
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config
from utils.data_processor import DataProcessor
from utils.trainer import FocalLoss
from models.advanced_models import EnhancedGCN, EnhancedGraphSAGE, EnhancedGIN, EnhancedRGCN, EnhancedGAT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_val(train_mask, labels, config, val_ratio=0.15, seed=42):
    train_indices = np.where(train_mask.cpu().numpy())[0]
    foreground_mask = ((labels == config.NORMAL_CLASS) | (labels == config.FRAUD_CLASS)).cpu().numpy()
    valid_train_mask = train_mask.cpu().numpy() & foreground_mask
    valid_train_indices = np.where(valid_train_mask)[0]
    train_labels_binary = (labels[valid_train_indices].cpu().numpy() == config.FRAUD_CLASS).astype(int)
    train_idx, val_idx = train_test_split(
        valid_train_indices,
        test_size=val_ratio,
        stratify=train_labels_binary,
        random_state=seed
    )

    new_train_mask = torch.zeros_like(train_mask)
    new_val_mask = torch.zeros_like(train_mask)
    new_train_mask[train_idx] = True
    new_val_mask[val_idx] = True

    return new_train_mask, new_val_mask


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
            num_relations = getattr(config, 'NUM_RELATIONS', 12)
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


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_single_model(model, model_type, config, device, data, train_mask, val_mask, use_edge_type=False):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)
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

    train_mask_d = train_mask.to(device)
    val_mask_d = val_mask.to(device)

    foreground = (labels == config.NORMAL_CLASS) | (labels == config.FRAUD_CLASS)
    train_valid = train_mask_d & foreground
    val_valid = val_mask_d & foreground

    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_auc': []}

    pbar = tqdm(range(config.EPOCHS), desc='Training')
    for epoch in pbar:
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
        history['train_loss'].append(loss.item())
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'auc': f'{val_auc:.4f}', 'best': f'{best_val_auc:.4f}'})

        if patience_counter >= config.PATIENCE:
            print(f'\nEarly stopping at epoch {epoch}')
            break

        if epoch % 20 == 0:
            clear_memory()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    del x, edge_index, labels
    if edge_type is not None:
        del edge_type
    clear_memory()

    return model, history, best_val_auc


def train_with_full_data(model_type, config, device, data, full_train_mask, best_epoch, in_channels, num_relations):
    set_seed(config.SEED)

    use_edge_type = (model_type == 'RGCN')
    model = create_model(model_type, in_channels, config.HIDDEN_DIM, config.NUM_CLASSES, config, num_relations)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
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
    mask = full_train_mask.to(device)

    foreground = (labels == config.NORMAL_CLASS) | (labels == config.FRAUD_CLASS)
    valid_mask = mask & foreground

    train_epochs = max(int(best_epoch * 1.2), 60)

    print(f"  Training {model_type} for {train_epochs} epochs with full data...")

    pbar = tqdm(range(train_epochs), desc=f'  {model_type}')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        if adj_t is not None:
            logits = model(x, adj_t=adj_t)
        else:
            logits = model(x, edge_index, edge_type)
        logits_fg = logits[valid_mask][:, :2]
        binary_labels = (labels[valid_mask] == config.FRAUD_CLASS).long()

        loss = criterion(logits_fg, binary_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch % 20 == 0:
            clear_memory()

    del x, edge_index, labels
    if edge_type is not None:
        del edge_type
    if adj_t is not None:
        del adj_t
    clear_memory()

    return model


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

    # 数据加载
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
        print(f"Number of edge types (relations): {num_relations}")
    else:
        num_relations = 1
        print("No edge types found, using single relation")

    config.NUM_RELATIONS = num_relations

    # 划分训练/验证集（分层采样）
    train_mask, val_mask = split_train_val(data.train_mask, data.y, config, val_ratio=0.15, seed=config.SEED)

    # 显示划分后的类别分布
    foreground = (data.y == config.NORMAL_CLASS) | (data.y == config.FRAUD_CLASS)
    train_fg = train_mask & foreground
    val_fg = val_mask & foreground

    train_labels = data.y[train_fg]
    val_labels = data.y[val_fg]

    train_fraud_ratio = (train_labels == config.FRAUD_CLASS).float().mean().item() * 100
    val_fraud_ratio = (val_labels == config.FRAUD_CLASS).float().mean().item() * 100

    print(f"\nTrain size: {train_mask.sum().item()} (foreground: {train_fg.sum().item()}, fraud ratio: {train_fraud_ratio:.2f}%)")
    print(f"Val size: {val_mask.sum().item()} (foreground: {val_fg.sum().item()}, fraud ratio: {val_fraud_ratio:.2f}%)")

    print("\n" + "=" * 60)
    print("Training models")
    print("=" * 60)

    model_types = config.ENSEMBLE_MODELS
    best_epochs = {}
    val_aucs = []
    validation_models = []

    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        set_seed(config.SEED)
        clear_memory()

        use_edge_type = (model_type == 'RGCN')
        model = create_model(model_type, in_channels, config.HIDDEN_DIM, config.NUM_CLASSES, config, num_relations)

        model, history, best_auc = train_single_model(model, model_type, config, device, data, train_mask, val_mask, use_edge_type)

        best_epoch = np.argmax(history['val_auc']) + 1
        best_epochs[model_type] = best_epoch
        val_aucs.append(best_auc)

        print(f"  {model_type}: Best epoch = {best_epoch}, Val AUC = {best_auc:.4f}")

        # Preserve the checkpoint that has never seen validation labels. The
        # full-data refit below is for deployment only and is not evaluated.
        validation_models.append((model_type, model.cpu()))
        clear_memory()

    # 计算集成权重
    val_aucs = np.array(val_aucs)
    exp_aucs = np.exp((val_aucs - val_aucs.max()) * 10)
    ensemble_weights = exp_aucs / exp_aucs.sum()

    print(f"\nModel AUCs: {val_aucs}")
    print(f"Ensemble weights: {ensemble_weights}")

    print("\n" + "=" * 60)
    print("Retraining with full data")
    print("=" * 60)

    full_train_mask = data.train_mask
    for i, model_type in enumerate(model_types):
        print(f"\nRetraining {model_type}...")
        clear_memory()

        model = train_with_full_data(
            model_type, config, device, data,
            full_train_mask, best_epochs[model_type], in_channels, num_relations
        )

        save_path = os.path.join(config.SAVE_DIR, f'ensemble_{model_type}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_val_auc': val_aucs[i],
            'best_epoch': best_epochs[model_type],
            'num_relations': num_relations,
            'in_channels': in_channels
        }, save_path)
        print(f"  Saved to {save_path}")

        # Deployment checkpoints are persisted and released immediately. They
        # are deliberately absent from the validation path below.
        del model
        clear_memory()

    # 保存权重和配置
    np.save(os.path.join(config.SAVE_DIR, 'ensemble_weights.npy'), ensemble_weights)
    np.save(os.path.join(config.SAVE_DIR, 'num_relations.npy'), np.array([num_relations]))
    np.save(os.path.join(config.SAVE_DIR, 'in_channels.npy'), np.array([in_channels]))

    # ============ 评估集成效果 ============
    print("\n" + "=" * 60)
    print("Evaluating preserved holdout checkpoints")
    print("=" * 60)

    all_probs = []
    for model_type, model in validation_models:
        model = model.to(device)
        model.eval()
        use_edge_type = (model_type == 'RGCN')
        with torch.no_grad():
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_type = None
            if use_edge_type and hasattr(data, 'edge_type') and data.edge_type is not None:
                edge_type = data.edge_type.to(device)

            logits = model(x, edge_index, edge_type)
            probs = torch.softmax(logits[:, :2], dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

            del x, edge_index, logits
            if edge_type is not None:
                del edge_type
            model.cpu()
            clear_memory()

    all_probs = np.array(all_probs)
    ensemble_probs = np.average(all_probs, axis=0, weights=ensemble_weights)

    # 在验证集上评估
    val_indices = val_mask.cpu().numpy()
    foreground_indices = data.foreground_mask.cpu().numpy()
    valid_indices = val_indices & foreground_indices

    val_labels = (data.y.cpu().numpy()[valid_indices] == config.FRAUD_CLASS).astype(int)
    val_probs = ensemble_probs[valid_indices]

    # 使用sklearn.metrics计算AUC（符合评分标准）
    if len(np.unique(val_labels)) >= 2:
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        ensemble_auc = sk_auc(fpr, tpr)
        if np.isnan(ensemble_auc) or ensemble_auc < 0 or ensemble_auc > 1:
            ensemble_auc = 0.0
    else:
        ensemble_auc = 0.0
    print(f"\nEnsemble Val AUC: {ensemble_auc:.4f}")

    # 使用评估器进行详细评估
    from utils.evaluator import FraudEvaluator
    evaluator = FraudEvaluator()
    metrics = evaluator.evaluate(val_labels, val_probs, threshold=0.5)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    metrics_payload = {
        'protocol': 'single_stratified_holdout_before_full_refit',
        'seed': config.SEED,
        'validation_nodes': int(valid_indices.sum()),
        'time_features_enabled': bool(config.USE_TIME_FEATURES),
        'proxy_features_enabled': False,
        'models': {
            name: {'roc_auc': float(score), 'best_epoch': int(best_epochs[name])}
            for name, score in zip(model_types, val_aucs)
        },
        'ensemble': {key: float(value) for key, value in metrics.items()},
    }
    with open(os.path.join(config.LOG_DIR, 'holdout_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    print(f"\nDetailed Metrics:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  PR AUC: {metrics['pr_auc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  MCC: {metrics['mcc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    # 可视化
    try:
        from utils.visualization import (
            plot_roc_curve, plot_confusion_matrix,
            plot_model_comparison, plot_tsne_embeddings
        )
        os.makedirs(config.LOG_DIR, exist_ok=True)
        plot_roc_curve(val_labels, val_probs,
                      save_path=os.path.join(config.LOG_DIR, 'roc_curve.png'))
        # 使用预先声明的固定阈值，避免在同一验证集上调阈值和报分
        plot_confusion_matrix(val_labels, val_probs, threshold=0.5,
                             save_path=os.path.join(config.LOG_DIR, 'confusion_matrix.png'))

        # 模型对比图
        model_results = {}
        for i, (model_type, model) in enumerate(validation_models):
            model = model.to(device)
            model.eval()
            use_edge_type = (model_type == 'RGCN')
            with torch.no_grad():
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                edge_type = None
                if use_edge_type and hasattr(data, 'edge_type') and data.edge_type is not None:
                    edge_type = data.edge_type.to(device)
                logits = model(x, edge_index, edge_type)
                probs = torch.softmax(logits[:, :2], dim=1)[:, 1].cpu().numpy()
                model_probs = probs[valid_indices]
                model_metrics = evaluator.evaluate(val_labels, model_probs, threshold=0.5)
                model_results[model_type] = model_metrics
                del x, edge_index, logits
                if edge_type is not None:
                    del edge_type
                model.cpu()
                clear_memory()

        model_results['Ensemble'] = metrics
        plot_model_comparison(model_results,
                             save_path=os.path.join(config.LOG_DIR, 'model_comparison.png'))

        # t-SNE可视化（使用最后一个模型的嵌入）
        try:
            if validation_models:
                last_model_type, last_model = validation_models[-1]
                last_model = last_model.to(device)
                last_model.eval()
                use_edge_type = (last_model_type == 'RGCN')
                with torch.no_grad():
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    edge_type = None
                    if use_edge_type and hasattr(data, 'edge_type') and data.edge_type is not None:
                        edge_type = data.edge_type.to(device)

                    # 获取中间层嵌入（使用第一个卷积层后的特征）
                    if hasattr(last_model, 'input_lin'):
                        x_emb = last_model.input_lin(x)
                        if hasattr(last_model, 'input_bn'):
                            x_emb = last_model.input_bn(x_emb)
                        x_emb = torch.relu(x_emb)
                        for i, (conv, bn) in enumerate(zip(last_model.convs, last_model.bns)):
                            if i == 0:
                                x_emb = conv(x_emb, edge_index)
                                x_emb = bn(x_emb)
                                x_emb = torch.relu(x_emb)
                                break
                        embeddings = x_emb.cpu().numpy()

                        # 只对前景节点进行可视化
                        foreground_embeddings = embeddings[data.foreground_mask.cpu().numpy()]
                        foreground_labels = data.y[data.foreground_mask].cpu().numpy()
                        foreground_labels_binary = (foreground_labels == config.FRAUD_CLASS).astype(int)

                        plot_tsne_embeddings(
                            foreground_embeddings,
                            foreground_labels_binary,
                            save_path=os.path.join(config.LOG_DIR, 'tsne_embeddings.png')
                        )
                    del x, edge_index
                    if edge_type is not None:
                        del edge_type
                    last_model.cpu()
                    clear_memory()
        except Exception as e2:
            print(f"  Warning: t-SNE visualization failed: {e2}")

        print(f"\nVisualizations saved to {config.LOG_DIR}")
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")

    # 清理
    for _, model in validation_models:
        del model
    clear_memory()

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Models saved to: {config.SAVE_DIR}")
    print(f"Run 'python inference.py' to generate predictions")


if __name__ == "__main__":
    main()
