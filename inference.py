# inference.py
import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch

from config import Config
from utils.data_processor import DataProcessor
from models.advanced_models import EnhancedGCN, EnhancedGraphSAGE, EnhancedGIN, EnhancedRGCN, EnhancedGAT

warnings.filterwarnings('ignore')


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


def main():
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    processor = DataProcessor(config)
    raw_data = processor.load_data(config.DATA_PATH)

    print("\nEngineering features...")
    enhanced_features = processor.create_enhanced_features(raw_data)

    print("\nPreparing PyG data...")
    data = processor.prepare_pyg_data(raw_data, enhanced_features)

    in_channels = data.x.shape[1]

    # 加载配置
    num_relations_path = os.path.join(config.SAVE_DIR, 'num_relations.npy')
    if os.path.exists(num_relations_path):
        num_relations = int(np.load(num_relations_path)[0])
    else:
        if hasattr(data, 'edge_type') and data.edge_type is not None:
            num_relations = int(data.edge_type.max().item()) + 1
        else:
            num_relations = 1

    print(f"Feature dimension: {in_channels}")
    print(f"Number of relations: {num_relations}")

    # 加载集成权重
    weights_path = os.path.join(config.SAVE_DIR, 'ensemble_weights.npy')
    if os.path.exists(weights_path):
        ensemble_weights = np.load(weights_path)
        print(f"Loaded ensemble weights: {ensemble_weights}")
    else:
        ensemble_weights = None
        print("No weights file found, using equal weights")

    # 加载模型并预测
    print("\n" + "=" * 60)
    print("Loading models and predicting...")
    print("=" * 60)

    model_types = config.ENSEMBLE_MODELS
    all_probs = []
    loaded_weights = []

    for i, model_type in enumerate(model_types):
        model_path = os.path.join(config.SAVE_DIR, f'ensemble_{model_type}.pth')
        if not os.path.exists(model_path):
            print(f"  [WARNING] Model not found: {model_path}")
            continue

        print(f"  Loading {model_type}...")
        clear_memory()

        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 如果checkpoint中有num_relations，使用它
        ckpt_num_relations = checkpoint.get('num_relations', num_relations)

        # 创建模型
        model = create_model(model_type, in_channels, config.HIDDEN_DIM, config.NUM_CLASSES, config, ckpt_num_relations)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # 预测
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

            if ensemble_weights is not None and i < len(ensemble_weights):
                loaded_weights.append(ensemble_weights[i])
            else:
                loaded_weights.append(1.0)

            del x, edge_index, logits, model
            if edge_type is not None:
                del edge_type
            clear_memory()

        print(f"    Val AUC (from training): {checkpoint.get('best_val_auc', 'N/A')}")

    if len(all_probs) == 0:
        raise FileNotFoundError("No trained models found! Run train.py first.")

    # 集成预测
    all_probs = np.array(all_probs)
    loaded_weights = np.array(loaded_weights)
    loaded_weights = loaded_weights / loaded_weights.sum()

    print(f"\nUsing weights: {loaded_weights}")
    ensemble_probs = np.average(all_probs, axis=0, weights=loaded_weights)

    # 提取测试集预测
    test_indices = data.test_mask.cpu().numpy().nonzero()[0]
    test_predictions = ensemble_probs[test_indices]

    # 生成提交文件
    print("\n" + "=" * 60)
    print("Creating submission file...")
    print("=" * 60)

    submission = pd.DataFrame({
        'index': test_indices.astype(int),
        'predict': test_predictions.astype(float)
    })

    # 数据类型检查
    submission['index'] = submission['index'].astype(int)
    submission['predict'] = submission['predict'].astype(float)

    # 限制predict在[0,1]范围内
    submission['predict'] = submission['predict'].clip(0.0, 1.0)

    os.makedirs(config.RESULT_DIR, exist_ok=True)
    output_path = os.path.join(config.RESULT_DIR, 'result.csv')
    submission.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")
    print(f"Total predictions: {len(submission)}")

    # 统计信息
    print(f"\nPrediction statistics:")
    print(f"  Min:  {test_predictions.min():.6f}")
    print(f"  Max:  {test_predictions.max():.6f}")
    print(f"  Mean: {test_predictions.mean():.6f}")
    print(f"  Std:  {test_predictions.std():.6f}")

    # 阈值分布
    for threshold in [0.3, 0.5, 0.7]:
        count = (test_predictions > threshold).sum()
        print(f"  > {threshold}: {count} ({100*count/len(test_predictions):.2f}%)")

    # 可视化预测分布
    try:
        from utils.visualization import plot_prediction_distribution
        os.makedirs(config.LOG_DIR, exist_ok=True)
        plot_prediction_distribution(test_predictions,
                                    save_path=os.path.join(config.LOG_DIR, 'inference_pred_distribution.png'))
        print(f"\nPrediction distribution saved to {config.LOG_DIR}")
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")

    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
