"""Central configuration for GraphShield.

Proxy variables are intentionally disabled. A column may only be used as time
after its semantics have been documented in ``TIME_COLUMN_SEMANTICS``.
"""

import os


class Config:
    DATA_PATH = os.getenv("GRAPHSHIELD_DATA", "./data/xydata/raw/data.npz")
    SAVE_DIR = "./checkpoints"
    RESULT_DIR = "./results"
    LOG_DIR = "./artifacts"

    NUM_FEATURES = 17
    NUM_CLASSES = 2
    FRAUD_CLASS = 1
    NORMAL_CLASS = 0

    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    DROPOUT = 0.4
    MODEL_DROPOUT = {
        "GCN": 0.2,
        "GraphSAGE": 0.2,
        "GIN": 0.5,
        "RGCN": 0.4,
        "GAT": 0.2,
    }

    # Raw values 1..11 are remapped to contiguous relation IDs 0..10 on load.
    NUM_RELATIONS = 11
    NUM_BASES = 8
    RGCN_NUM_LAYERS = 2
    GAT_HEADS = 8

    EPOCHS = 300
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 1e-4
    PATIENCE = 30
    BATCH_SIZE = 4096
    NUM_NEIGHBORS = [10, 5]

    ENSEMBLE_MODELS = ["GCN", "GraphSAGE", "GIN", "RGCN", "GAT"]
    ENSEMBLE_WEIGHTS = None

    USE_EDGE_ATTR = True
    USE_UNDIRECTED = False
    FAST_MODE = False

    # Safe-by-default temporal policy. The supplied NPZ has no documented
    # timestamp column. Node IDs and row order are never accepted as time.
    USE_TIME_FEATURES = False
    TIME_COLUMNS = ()
    TIME_COLUMN_SEMANTICS = {}
    ALLOW_NODE_ID_PROXY = False
    ALLOW_EDGE_ORDER_PROXY = False

    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.95
    FOCAL_GAMMA = 2.5
    LABEL_SMOOTHING = 0.1

    SEED = 42
    DEVICE = os.getenv("GRAPHSHIELD_DEVICE", "cuda")
    NUM_WORKERS = 0
