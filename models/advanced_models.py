# models/advanced_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, FastRGCNConv, GATConv


class AnomalyDetectionModule(nn.Module):
    def __init__(self, hidden_channels, dropout=0.2):
        super().__init__()
        self.anomaly_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.anomaly_net(x)


class EnhancedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, use_anomaly_detection=True, **kwargs):
        super().__init__()

        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.use_anomaly_detection = use_anomaly_detection
        if use_anomaly_detection:
            self.anomaly_detector = AnomalyDetectionModule(hidden_channels, dropout)
            self.feature_adapter = nn.Linear(hidden_channels + 1, hidden_channels)
        else:
            self.anomaly_detector = None
            self.feature_adapter = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_type=None, adj_t=None):
        x = self.input_lin(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if adj_t is not None:
            for conv, bn in zip(self.convs, self.bns):
                x_new = conv(x, adj_t)
                x_new = bn(x_new)
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                x = x + x_new
        else:
            for conv, bn in zip(self.convs, self.bns):
                x_new = conv(x, edge_index)
                x_new = bn(x_new)
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                x = x + x_new

        if self.use_anomaly_detection and self.anomaly_detector is not None:
            anomaly_score = self.anomaly_detector(x)
            x = torch.cat([x, anomaly_score], dim=1)
            x = self.feature_adapter(x)

        return self.classifier(x)


class EnhancedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, **kwargs):
        super().__init__()

        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        x = self.input_lin(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new

        return self.classifier(x)


class EnhancedGIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, **kwargs):
        super().__init__()

        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        x = self.input_lin(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new

        return self.classifier(x)


class EnhancedRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations, num_layers=2, dropout=0.3, num_bases=None, **kwargs):
        super().__init__()

        if num_bases is None:
            num_bases = min(num_relations, 8)

        self.num_relations = num_relations
        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        # R-GCN层 - 使用FastRGCNConv更省显存
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                FastRGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        # 输入投影
        x = self.input_lin(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # R-GCN层
        for conv, bn in zip(self.convs, self.bns):
            if edge_type is not None:
                x_new = conv(x, edge_index, edge_type)
            else:
                # 如果没有edge_type，创建全0类型
                x_new = conv(x, edge_index, torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device))

            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # 残差连接

        return self.classifier(x)


class EnhancedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, heads=8, **kwargs):
        super().__init__()

        if hidden_channels % heads != 0:
            adjusted_hidden = (hidden_channels // heads) * heads
            if adjusted_hidden == 0:
                adjusted_hidden = heads
            hidden_channels = adjusted_hidden

        self.hidden_channels = hidden_channels
        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                      heads=heads, dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        x = self.input_lin(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # 残差连接

        return self.classifier(x)
