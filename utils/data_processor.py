# utils/data_processor.py
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import RobustScaler
from collections import defaultdict


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = RobustScaler()

    def load_data(self, data_path):
        print(f"Loading data from {data_path}")

        data = np.load(data_path)

        print("\n" + "=" * 60)
        print("Dataset Information:")
        print("=" * 60)
        for key in data.files:
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

        edge_index = data['edge_index']
        if len(edge_index.shape) != 2 or edge_index.shape[1] != 2:
            raise ValueError(f"edge_index must be (E, 2), got {edge_index.shape}")

        edge_type = data['edge_type'] if 'edge_type' in data.files else None
        if edge_type is not None:
            observed = np.unique(edge_type)
            relation_map = {int(value): index for index, value in enumerate(observed)}
            edge_type = np.searchsorted(observed, edge_type).astype(np.int64)
            print(f"  Relation remap: {relation_map}")

        return {
            'x': data['x'],
            'y': data['y'],
            'edge_index': edge_index,
            'edge_type': edge_type,
            'train_mask': data['train_mask'],
            'test_mask': data['test_mask']
        }

    def create_enhanced_features(self, raw_data):
        print("\nCreating enhanced features...")

        x = raw_data['x'].copy()
        num_nodes = x.shape[0]
        edge_index = raw_data['edge_index']
        edge_types = raw_data['edge_type']

        # 只用训练集fit scaler，避免数据泄露
        train_mask_raw = raw_data.get('train_mask', np.array([], dtype=int))
        if len(train_mask_raw) == 0:
            train_mask = np.zeros(num_nodes, dtype=bool)
        elif train_mask_raw.dtype == bool:
            # 已经是布尔掩码
            if len(train_mask_raw) == num_nodes:
                train_mask = train_mask_raw
            else:
                # 长度不匹配，创建新的掩码
                train_mask = np.zeros(num_nodes, dtype=bool)
                train_mask[:len(train_mask_raw)] = train_mask_raw
        else:
            # 是索引数组，转换为布尔掩码
            train_mask = np.zeros(num_nodes, dtype=bool)
            valid_indices = train_mask_raw[(train_mask_raw >= 0) & (train_mask_raw < num_nodes)]
            train_mask[valid_indices] = True

        y = raw_data.get('y', np.zeros(num_nodes, dtype=int))
        foreground_mask = (y == self.config.NORMAL_CLASS) | (y == self.config.FRAUD_CLASS)
        train_foreground_mask = train_mask & foreground_mask

        if train_foreground_mask.sum() > 0:
            train_foreground_x = x[train_foreground_mask]
            self.scaler.fit(train_foreground_x)
            print(f"  Scaler fitted on {train_foreground_mask.sum()} training foreground nodes")
        else:
            print("  Warning: No training foreground nodes, using all nodes for scaler fit")
            self.scaler.fit(x)

        # 对所有节点transform（但scaler只基于训练前景拟合）
        x_normalized = self.scaler.transform(x).astype(np.float32)

        feature_list = [x_normalized]
        feature_names = ['original(17)']

        # 度特征
        print("  - Computing degree features...")
        in_degree = np.zeros(num_nodes, dtype=np.float32)
        out_degree = np.zeros(num_nodes, dtype=np.float32)

        for src, dst in edge_index:
            out_degree[src] += 1
            in_degree[dst] += 1

        total_degree = in_degree + out_degree

        degree_features = np.column_stack([
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(total_degree),
            np.where(total_degree > 0, in_degree / (total_degree + 1e-8), 0.5),
            np.where(total_degree > 0, out_degree / (total_degree + 1e-8), 0.5),
        ]).astype(np.float32)

        feature_list.append(degree_features)
        feature_names.append('degree(5)')

        # 边类型特征
        if edge_types is not None:
            print("  - Computing edge type features...")
            edge_types_int = edge_types.astype(int)
            num_edge_types = int(edge_types_int.max()) + 1
            print(f"    Number of edge types: {num_edge_types}")

            # 入边类型统计
            in_edge_type_count = np.zeros((num_nodes, num_edge_types), dtype=np.float32)
            # 出边类型统计
            out_edge_type_count = np.zeros((num_nodes, num_edge_types), dtype=np.float32)

            for i, (src, dst) in enumerate(edge_index):
                et = edge_types_int[i]
                out_edge_type_count[src, et] += 1
                in_edge_type_count[dst, et] += 1

            # log变换
            in_edge_log = np.log1p(in_edge_type_count)
            out_edge_log = np.log1p(out_edge_type_count)

            # 边类型多样性（有多少种不同的边类型）
            in_type_diversity = (in_edge_type_count > 0).sum(axis=1, keepdims=True).astype(np.float32)
            out_type_diversity = (out_edge_type_count > 0).sum(axis=1, keepdims=True).astype(np.float32)

            # 主导边类型比例
            in_total = in_edge_type_count.sum(axis=1, keepdims=True) + 1e-8
            out_total = out_edge_type_count.sum(axis=1, keepdims=True) + 1e-8
            in_max_ratio = in_edge_type_count.max(axis=1, keepdims=True) / in_total
            out_max_ratio = out_edge_type_count.max(axis=1, keepdims=True) / out_total

            edge_type_features = np.column_stack([
                in_edge_log,
                out_edge_log,
                in_type_diversity,
                out_type_diversity,
                in_max_ratio,
                out_max_ratio
            ]).astype(np.float32)

            feature_list.append(edge_type_features)
            feature_names.append(f'edge_type({edge_type_features.shape[1]})')

        # 邻居度统计
        print("  - Computing neighbor degree statistics...")

        # 构建邻接表
        in_neighbors = defaultdict(list)
        out_neighbors = defaultdict(list)
        for src, dst in edge_index:
            out_neighbors[src].append(dst)
            in_neighbors[dst].append(src)

        # 邻居的度统计
        neighbor_in_deg_mean = np.zeros(num_nodes, dtype=np.float32)
        neighbor_in_deg_std = np.zeros(num_nodes, dtype=np.float32)
        neighbor_out_deg_mean = np.zeros(num_nodes, dtype=np.float32)
        neighbor_out_deg_std = np.zeros(num_nodes, dtype=np.float32)

        for node in range(num_nodes):
            # 入邻居的度
            neighs = in_neighbors[node]
            if len(neighs) > 0:
                neigh_in_degs = in_degree[neighs]
                neigh_out_degs = out_degree[neighs]
                neighbor_in_deg_mean[node] = neigh_in_degs.mean()
                neighbor_in_deg_std[node] = neigh_in_degs.std() if len(neighs) > 1 else 0
                neighbor_out_deg_mean[node] = neigh_out_degs.mean()
                neighbor_out_deg_std[node] = neigh_out_degs.std() if len(neighs) > 1 else 0

        neighbor_deg_features = np.column_stack([
            np.log1p(neighbor_in_deg_mean),
            np.log1p(neighbor_in_deg_std),
            np.log1p(neighbor_out_deg_mean),
            np.log1p(neighbor_out_deg_std),
        ]).astype(np.float32)

        feature_list.append(neighbor_deg_features)
        feature_names.append('neighbor_deg(4)')

        # 无向邻接同时供聚类系数和结构模式使用。FAST_MODE 只跳过昂贵的
        # 聚类计算，不能跳过后续结构特征依赖的数据结构。
        adj_set = defaultdict(set)
        for src, dst in edge_index:
            adj_set[src].add(dst)
            adj_set[dst].add(src)

        # 聚类系数
        fast_mode = getattr(self.config, 'FAST_MODE', False)
        if fast_mode:
            print("  - Skipping clustering coefficient (FAST_MODE)")
            clustering_coef = np.zeros(num_nodes, dtype=np.float32)
            feature_list.append(clustering_coef.reshape(-1, 1))
            feature_names.append('clustering(1)')
        else:
            print("  - Computing clustering coefficient...")
            clustering_coef = np.zeros(num_nodes, dtype=np.float32)

            for node in range(num_nodes):
                neighbors = list(adj_set[node])
                k = len(neighbors)
                if k < 2:
                    continue

                # 计算邻居之间的边数
                link_count = 0
                for i in range(k):
                    for j in range(i + 1, k):
                        if neighbors[j] in adj_set[neighbors[i]]:
                            link_count += 1

                clustering_coef[node] = 2 * link_count / (k * (k - 1))

            feature_list.append(clustering_coef.reshape(-1, 1))
            feature_names.append('clustering(1)')

        # 路径特征
        fast_mode = getattr(self.config, 'FAST_MODE', False)
        if fast_mode:
            print("  - Computing path features (2-hop only, FAST_MODE)...")
            path_features = self._compute_path_features(edge_index, num_nodes, in_neighbors, out_neighbors, fast_mode=True)
            feature_names.append('path(3)')
        else:
            print("  - Computing path features (2-hop, 3-hop)...")
            path_features = self._compute_path_features(edge_index, num_nodes, in_neighbors, out_neighbors, fast_mode=False)
            feature_names.append('path(5)')
        feature_list.append(path_features)

        # 结构模式
        print("  - Computing structure patterns...")
        pattern_features = self._compute_structure_patterns(edge_index, num_nodes, adj_set)
        feature_list.append(pattern_features)
        feature_names.append('pattern(4)')

        # 异常度特征
        print("  - Computing anomaly features...")
        anomaly_features = self._compute_anomaly_features(
            edge_index, edge_types, num_nodes, in_degree, out_degree,
            train_mask=train_mask, foreground_mask=foreground_mask
        )
        feature_list.append(anomaly_features)
        feature_names.append('anomaly(5)')

        # 中心性特征
        print("  - Computing centrality features...")
        centrality_features = self._compute_centrality_features(edge_index, num_nodes)
        feature_list.append(centrality_features)
        feature_names.append('centrality(4)')

        # 时间特征（默认关闭；只接受具有文档语义的真实时间列）
        use_time_features = getattr(self.config, 'USE_TIME_FEATURES', False)
        if use_time_features:
            from utils.temporal_features import (
                ExplicitTemporalFeatureBuilder,
                TemporalFeatureSpec,
            )

            columns = tuple(getattr(self.config, 'TIME_COLUMNS', ()))
            semantics = dict(getattr(self.config, 'TIME_COLUMN_SEMANTICS', {}))
            specs = [
                TemporalFeatureSpec(
                    column=column,
                    name=f'time_col_{column}',
                    semantics=semantics.get(column, ''),
                )
                for column in columns
            ]
            fit_indices = np.flatnonzero(train_mask)
            builder = ExplicitTemporalFeatureBuilder(specs)
            time_features, time_feature_names = builder.fit_transform(x, fit_indices)
            feature_list.append(time_features)
            feature_names.extend(time_feature_names)

        # 合并所有特征
        enhanced_features = np.hstack(feature_list).astype(np.float32)

        # 处理异常值
        enhanced_features = np.nan_to_num(enhanced_features, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"\nFeature summary:")
        for name in feature_names:
            print(f"  - {name}")
        print(f"Total dimension: {enhanced_features.shape[1]}")

        return enhanced_features

    def prepare_pyg_data(self, raw_data, enhanced_features=None):
        if enhanced_features is None:
            enhanced_features = raw_data['x']

        x = torch.as_tensor(enhanced_features, dtype=torch.float32)
        y = torch.as_tensor(raw_data['y'], dtype=torch.long)

        edge_index_array = raw_data['edge_index']
        edge_index = torch.as_tensor(edge_index_array.T, dtype=torch.long)

        edge_type = None
        if raw_data['edge_type'] is not None:
            edge_type = torch.as_tensor(raw_data['edge_type'], dtype=torch.long)

        print(f"  Edge index shape: {edge_index.shape}")

        num_nodes = x.size(0)

        # 处理train_mask
        train_mask_raw = raw_data.get('train_mask', np.array([], dtype=int))
        if len(train_mask_raw) == 0:
            train_mask_np = np.zeros(num_nodes, dtype=bool)
        elif train_mask_raw.dtype == bool:
            # 已经是布尔掩码
            if len(train_mask_raw) == num_nodes:
                train_mask_np = train_mask_raw
            else:
                # 长度不匹配，创建新的掩码
                train_mask_np = np.zeros(num_nodes, dtype=bool)
                train_mask_np[:len(train_mask_raw)] = train_mask_raw
        else:
            # 是索引数组，转换为布尔掩码
            train_mask_np = np.zeros(num_nodes, dtype=bool)
            valid_indices = train_mask_raw[(train_mask_raw >= 0) & (train_mask_raw < num_nodes)]
            train_mask_np[valid_indices] = True
        train_mask = torch.from_numpy(train_mask_np)

        # 处理test_mask
        test_mask_raw = raw_data.get('test_mask', np.array([], dtype=int))
        if len(test_mask_raw) == 0:
            test_mask_np = np.zeros(num_nodes, dtype=bool)
        elif test_mask_raw.dtype == bool:
            # 已经是布尔掩码
            if len(test_mask_raw) == num_nodes:
                test_mask_np = test_mask_raw
            else:
                # 长度不匹配，创建新的掩码
                test_mask_np = np.zeros(num_nodes, dtype=bool)
                test_mask_np[:len(test_mask_raw)] = test_mask_raw
        else:
            # 是索引数组，转换为布尔掩码
            test_mask_np = np.zeros(num_nodes, dtype=bool)
            valid_indices = test_mask_raw[(test_mask_raw >= 0) & (test_mask_raw < num_nodes)]
            test_mask_np[valid_indices] = True
        test_mask = torch.from_numpy(test_mask_np)

        foreground_mask = (y == self.config.NORMAL_CLASS) | (y == self.config.FRAUD_CLASS)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            foreground_mask=foreground_mask
        )

        # SparseTensor
        try:
            from torch_sparse import SparseTensor
            row, col = edge_index
            adj_t = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            data.adj_t = adj_t
        except ImportError:
            data.adj_t = None

        if edge_type is not None:
            data.edge_type = edge_type

        print("\n" + "=" * 60)
        print("PyG Data Summary:")
        print("=" * 60)
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {data.x.shape[1]}")

        n_normal = ((y == 0) & train_mask).sum().item()
        n_fraud = ((y == 1) & train_mask).sum().item()
        print(f"  Train - Normal: {n_normal}, Fraud: {n_fraud}")
        print(f"  Imbalance ratio: {n_normal / max(n_fraud, 1):.1f}:1")

        return data

    def _compute_path_features(self, edge_index, num_nodes, in_neighbors, out_neighbors, fast_mode=False):
        """计算路径特征 - 2-hop、3-hop邻居统计"""
        path_features = []

        # 构建有向邻接表（用于路径计算）
        out_adj = defaultdict(set)
        for src, dst in edge_index:
            out_adj[src].add(dst)

        for node in range(num_nodes):
            # 1-hop邻居
            hop1_neighbors = out_adj[node]
            hop1_count = len(hop1_neighbors)

            # 2-hop邻居（不包含1-hop和自身）
            hop2_neighbors = set()
            for n1 in hop1_neighbors:
                hop2_neighbors.update(out_adj[n1])
            hop2_neighbors -= hop1_neighbors
            hop2_neighbors.discard(node)
            hop2_count = len(hop2_neighbors)

            if fast_mode:
                hop_ratio = hop2_count / (hop1_count + 1e-8)
                path_features.append([
                    np.log1p(hop1_count),
                    np.log1p(hop2_count),
                    np.log1p(hop_ratio + 1e-8),
                ])
            else:
                hop3_neighbors = set()
                for n2 in hop2_neighbors:
                    hop3_neighbors.update(out_adj[n2])
                hop3_neighbors -= hop1_neighbors
                hop3_neighbors -= hop2_neighbors
                hop3_neighbors.discard(node)
                hop3_count = len(hop3_neighbors)

                # 路径多样性：有多少条不同的2-hop路径
                path_diversity = 0
                for n1 in hop1_neighbors:
                    for n2 in out_adj[n1]:
                        if n2 != node and n2 not in hop1_neighbors:
                            path_diversity += 1

                # 2-hop/1-hop比例
                hop_ratio = hop2_count / (hop1_count + 1e-8)

                path_features.append([
                    np.log1p(hop1_count),
                    np.log1p(hop2_count),
                    np.log1p(hop3_count),
                    np.log1p(path_diversity),
                    np.log1p(hop_ratio + 1e-8),
                ])

        return np.array(path_features, dtype=np.float32)

    def _compute_structure_patterns(self, edge_index, num_nodes, adj_set, max_degree=100):
        """计算结构模式特征 - 三角形、四边形等（优化版本，对高度节点采样）"""
        pattern_features = []
        np.random.seed(42)

        for node in range(num_nodes):
            neighbors = list(adj_set[node])
            k = len(neighbors)

            if k < 2:
                pattern_features.append([0.0, 0.0, 0.0, 0.0])
                continue

            # 对高度节点采样，避免O(degree²)复杂度
            if k > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False).tolist()
                k = max_degree
                scale_factor = len(adj_set[node]) / max_degree
            else:
                scale_factor = 1.0

            # 三角形数量
            triangles = 0
            for i in range(k):
                for j in range(i + 1, k):
                    if neighbors[j] in adj_set[neighbors[i]]:
                        triangles += 1

            # 四边形数量
            squares = 0
            for i in range(k):
                for j in range(i + 1, k):
                    n1, n2 = neighbors[i], neighbors[j]
                    common = adj_set[n1] & adj_set[n2]
                    common.discard(node)
                    squares += len(common)

            # 邻居之间的连接密度
            neighbor_links = 0
            for i in range(k):
                for j in range(i + 1, k):
                    if neighbors[j] in adj_set[neighbors[i]]:
                        neighbor_links += 1

            # 缩放回原始规模（对采样节点）
            if scale_factor > 1.0:
                triangles = int(triangles * scale_factor)
                squares = int(squares * scale_factor * scale_factor)
                neighbor_links = int(neighbor_links * scale_factor)

            neighbor_density = 2 * neighbor_links / (k * (k - 1) + 1e-8) if k > 1 else 0.0
            normalized_triangles = triangles / (k * (k - 1) / 2 + 1e-8) if k > 1 else 0.0

            pattern_features.append([
                np.log1p(triangles),
                np.log1p(squares),
                neighbor_density,
                normalized_triangles,
            ])

        return np.array(pattern_features, dtype=np.float32)

    def _compute_anomaly_features(self, edge_index, edge_types, num_nodes, in_degree, out_degree, train_mask=None, foreground_mask=None):
        """计算异常度特征 - 基于度分布、边类型分布"""
        # 只用训练前景节点计算全局统计量（避免数据泄露）
        if train_mask is not None and foreground_mask is not None:
            train_foreground_mask = train_mask & foreground_mask
            if train_foreground_mask.sum() > 0:
                train_in_degree = in_degree[train_foreground_mask]
                train_out_degree = out_degree[train_foreground_mask]
                global_in_deg_mean = train_in_degree.mean()
                global_in_deg_std = train_in_degree.std() + 1e-8
                global_out_deg_mean = train_out_degree.mean()
                global_out_deg_std = train_out_degree.std() + 1e-8
            else:
                global_in_deg_mean = in_degree.mean()
                global_in_deg_std = in_degree.std() + 1e-8
                global_out_deg_mean = out_degree.mean()
                global_out_deg_std = out_degree.std() + 1e-8
        else:
            global_in_deg_mean = in_degree.mean()
            global_in_deg_std = in_degree.std() + 1e-8
            global_out_deg_mean = out_degree.mean()
            global_out_deg_std = out_degree.std() + 1e-8

        anomaly_features = []

        # 构建边类型索引（如果有）
        edge_type_map = {}
        if edge_types is not None:
            for idx, (src, dst) in enumerate(edge_index):
                if src not in edge_type_map:
                    edge_type_map[src] = []
                edge_type_map[src].append(edge_types[idx])

        for node in range(num_nodes):
            # 度异常度（Z-score）
            in_deg_z = (in_degree[node] - global_in_deg_mean) / global_in_deg_std
            out_deg_z = (out_degree[node] - global_out_deg_mean) / global_out_deg_std

            # 度分布异常度（使用绝对值）
            in_deg_anomaly = np.abs(in_deg_z)
            out_deg_anomaly = np.abs(out_deg_z)

            # 边类型分布异常度（如果有边类型）
            if edge_types is not None and node in edge_type_map:
                node_edge_types = np.array(edge_type_map[node], dtype=int)
                if len(node_edge_types) > 0:
                    # 计算边类型的多样性（熵）
                    type_counts = np.bincount(node_edge_types, minlength=int(edge_types.max()) + 1)
                    type_probs = type_counts / (type_counts.sum() + 1e-8)
                    type_entropy = -np.sum(type_probs * np.log(type_probs + 1e-8))
                    type_diversity = len(np.unique(node_edge_types))
                else:
                    type_entropy = 0.0
                    type_diversity = 0.0
            else:
                type_entropy = 0.0
                type_diversity = 0.0

            # 综合异常度
            combined_anomaly = (in_deg_anomaly + out_deg_anomaly) / 2

            anomaly_features.append([
                in_deg_anomaly,
                out_deg_anomaly,
                np.log1p(type_entropy),
                np.log1p(type_diversity),
                combined_anomaly,
            ])

        return np.array(anomaly_features, dtype=np.float32)

    def _compute_centrality_features(self, edge_index, num_nodes):
        """计算中心性特征 - 优化版本"""
        print("    Computing centrality features (optimized)...")

        # 度中心性
        in_degree = np.zeros(num_nodes, dtype=np.float32)
        out_degree = np.zeros(num_nodes, dtype=np.float32)
        for src, dst in edge_index:
            out_degree[src] += 1
            in_degree[dst] += 1

        total_degree = in_degree + out_degree
        degree_centrality = total_degree / (num_nodes - 1 + 1e-8)

        # PageRank
        pagerank = self._fast_pagerank(edge_index, num_nodes, num_iters=20)

        # 2-hop可达节点数近似接近中心性
        hop2_reach = self._compute_2hop_reachability(edge_index, num_nodes)
        approx_closeness = hop2_reach / (num_nodes + 1e-8)

        # HITS hub/authority分数
        fast_mode = getattr(self.config, 'FAST_MODE', False)
        if fast_mode:
            hub_score = np.zeros(num_nodes, dtype=np.float32)
            auth_score = np.zeros(num_nodes, dtype=np.float32)
        else:
            hub_score, auth_score = self._fast_hits(edge_index, num_nodes, num_iters=20)

        centrality_features = np.column_stack([
            np.log1p(degree_centrality),
            np.log1p(pagerank),
            np.log1p(approx_closeness),
            np.log1p(hub_score + auth_score),
        ]).astype(np.float32)

        return centrality_features

    def _fast_pagerank(self, edge_index, num_nodes, num_iters=20, alpha=0.85):
        """快速PageRank - 幂迭代法"""
        # 构建出度
        out_degree = np.zeros(num_nodes, dtype=np.float32)
        for src, dst in edge_index:
            out_degree[src] += 1
        out_degree = np.maximum(out_degree, 1)

        # 构建邻接表
        in_neighbors = [[] for _ in range(num_nodes)]
        for src, dst in edge_index:
            in_neighbors[dst].append(src)

        # 幂迭代
        pr = np.ones(num_nodes, dtype=np.float32) / num_nodes
        for _ in range(num_iters):
            new_pr = np.ones(num_nodes, dtype=np.float32) * (1 - alpha) / num_nodes
            for node in range(num_nodes):
                for neighbor in in_neighbors[node]:
                    new_pr[node] += alpha * pr[neighbor] / out_degree[neighbor]
            pr = new_pr

        return pr

    def _compute_2hop_reachability(self, edge_index, num_nodes):
        """计算2-hop可达节点数"""
        # 构建邻接表
        out_neighbors = [set() for _ in range(num_nodes)]
        for src, dst in edge_index:
            out_neighbors[src].add(dst)

        hop2_count = np.zeros(num_nodes, dtype=np.float32)
        for node in range(num_nodes):
            reached = set(out_neighbors[node])
            for n1 in out_neighbors[node]:
                reached.update(out_neighbors[n1])
            reached.discard(node)
            hop2_count[node] = len(reached)

        return hop2_count

    def _fast_hits(self, edge_index, num_nodes, num_iters=20):
        """快速HITS算法"""
        # 构建邻接表
        in_neighbors = [[] for _ in range(num_nodes)]
        out_neighbors = [[] for _ in range(num_nodes)]
        for src, dst in edge_index:
            out_neighbors[src].append(dst)
            in_neighbors[dst].append(src)

        hub = np.ones(num_nodes, dtype=np.float32)
        auth = np.ones(num_nodes, dtype=np.float32)

        for _ in range(num_iters):
            # Authority更新
            new_auth = np.zeros(num_nodes, dtype=np.float32)
            for node in range(num_nodes):
                for neighbor in in_neighbors[node]:
                    new_auth[node] += hub[neighbor]

            # Hub更新
            new_hub = np.zeros(num_nodes, dtype=np.float32)
            for node in range(num_nodes):
                for neighbor in out_neighbors[node]:
                    new_hub[node] += auth[neighbor]

            # 归一化
            auth_norm = np.linalg.norm(new_auth) + 1e-8
            hub_norm = np.linalg.norm(new_hub) + 1e-8
            auth = new_auth / auth_norm
            hub = new_hub / hub_norm

        return hub, auth
