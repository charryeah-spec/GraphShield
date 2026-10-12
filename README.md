# GraphShield

### Leakage-resistant graph neural networks for financial fraud detection

GraphShield is an end-to-end fraud detection study on a sparse, multi-relational
financial graph with **579,157 nodes**, **167,559 directed edges**, and only
**1.16% labeled fraud cases**. It compares five GNN families, engineers
graph-structural signals, and treats suspiciously strong “temporal” features as
an audit target instead of a free performance gain.

> The most important result is methodological: the supplied anonymized file has
> no documented timestamp, while node IDs perfectly separate the labeled
> classes. ID-derived “time” therefore produces AUC 1.0 through storage-order
> leakage and is disabled by default.

[中文说明](README.zh-CN.md) · [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) ·
[Data contract](data/README.md)

## Why this project matters

- **Large graph:** 579K nodes, 168K edges, 11 observed relation types.
- **Severe imbalance:** 4,492 fraud vs. 381,612 normal labeled nodes.
- **Model breadth:** GCN, GraphSAGE, GIN, R-GCN, and GAT.
- **Feature depth:** raw attributes plus degree, relation, neighborhood,
  clustering, path, motif, anomaly, and centrality features.
- **Evaluation discipline:** fold-level OOF predictions, PR-AUC, reproducible
  metric artifacts, disjoint-set guards, and post-selection full-data refits.
- **Leakage forensics:** node-ID diagnostics, edge-order permutation tests, and
  an explicit allowlist for genuine timestamp columns.

## Findings

### 1. Reproduced leakage-safe GraphSAGE result

The public result uses five untouched outer folds. Each outer training fold is
split again: one part fits the model and the other controls early stopping and
selects the classification threshold. Node-ID and edge-order proxies are off.

| Metric | Five-fold mean ± std |
|---|---:|
| ROC-AUC | **0.7544 ± 0.0124** |
| PR-AUC / Average Precision | **0.02718 ± 0.00193** |
| F1, threshold selected on inner validation | **0.0515 ± 0.0076** |
| MCC, threshold selected on inner validation | **0.0438 ± 0.0104** |

The PR-AUC is **2.34×** the fraud prevalence baseline of 0.01163. The pooled
cross-fitted decisions yield F1 0.0526, precision 0.0326, recall 0.1362, and MCC
0.0442. Primary ranking results are the mean and standard deviation across outer
folds because probability scales vary across independently trained models.

![Five-fold GraphSAGE ROC and precision-recall curves](artifacts/graphsage_oof/oof_curves.png)

Machine-readable results:
[`artifacts/graphsage_oof/metrics.json`](artifacts/graphsage_oof/metrics.json).

### 2. Historical graph baseline

The original course report recorded the following single-holdout results before
full-data refitting. They are retained as historical baselines—not presented as
a fresh reproduction.

| Model | ROC-AUC | PR-AUC | F1 | MCC |
|---|---:|---:|---:|---:|
| GCN | 0.7535 | 0.0255 | 0.0461 | 0.0911 |
| **GraphSAGE** | **0.7803** | **0.0300** | 0.0462 | 0.0975 |
| GIN | 0.7696 | 0.0280 | **0.0478** | **0.0988** |
| R-GCN | 0.7666 | 0.0283 | 0.0477 | 0.0963 |
| GAT | 0.7733 | 0.0289 | 0.0469 | 0.0979 |

Machine-readable source: [`artifacts/reported_baseline.json`](artifacts/reported_baseline.json).

### 3. “Temporal” lift was a leakage alarm

The unusually large gain from constructed time features led to a data audit:

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Node ID alone, best-orientation ROC-AUC | **1.0000** | Critical proxy leakage |
| Fraud-node ID range | `0–6784` | All labeled fraud is in the first block |
| Normal-node ID range | `6785–579156` | All labeled normal follows it |
| Documented timestamp columns | `0` | Temporal semantics are unavailable |
| Edge-row order after shuffling | changes | File order is not an invariant graph property |

This is consistent with class-wise serialization, not evidence that node ID is
registration time. GraphShield quarantines node ID, neighbor-ID derivatives, and
edge row positions as **negative controls**. A true temporal ablation is allowed
only after timestamp provenance and prediction-time availability are documented.

Run the audit yourself:

```bash
python scripts/audit_dataset.py /path/to/data.npz \
  --output artifacts/data_audit.json
```

## Evaluation design

```text
raw attributes + graph structure
              │
              ├── 5-fold OOF ──> base GNN probabilities
              │                       │
              │                       └──> stacking / weighted ensemble
              │
verified time ├── feature allowlist ──> temporal ablation
              │
ID / row order└── negative control ──> leakage & invariance report only
```

`stacking.py` saves both `artifacts/oof_metrics.json` and the exact predictions
in `artifacts/oof_predictions.npz`. Full-data refits are used only after metrics
have been produced from unseen folds. `train.py` likewise evaluates the preserved
holdout checkpoints rather than models retrained on the validation nodes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[stacking]'

export GRAPHSHIELD_DATA=/absolute/path/to/data.npz
python scripts/audit_dataset.py "$GRAPHSHIELD_DATA"
python scripts/run_graphsage_oof.py --data "$GRAPHSHIELD_DATA"
python stacking.py
python inference.py
```

Run the synthetic leakage and split-guard tests without the private dataset:

```bash
pip install -e '.[dev]'
pytest -q
```

The dataset is not distributed in this repository. See the
[`data/README.md`](data/README.md) contract.

## Repository map

```text
graphshield/
├── models/advanced_models.py       # GCN, GraphSAGE, GIN, R-GCN, GAT
├── utils/data_processor.py         # structural feature pipeline
├── utils/temporal_features.py      # verified timestamps + proxy quarantine
├── utils/leakage_audit.py          # reproducible NumPy-only audit
├── utils/split_guard.py            # disjoint and forward-time split guards
├── scripts/audit_dataset.py        # audit CLI
├── scripts/run_graphsage_oof.py    # nested-holdout GraphSAGE evaluation
├── stacking.py                     # five-fold OOF + meta learner
├── train.py                        # leakage-safe holdout workflow
├── inference.py                    # hidden-node predictions
├── tests/                          # synthetic leakage regression tests
└── artifacts/                      # compact, versioned JSON evidence
```

## Resume-ready summary

> Built a leakage-resistant financial fraud pipeline over a 579K-node,
> multi-relational graph, benchmarking five GNN architectures under 1.16% fraud
> prevalence; engineered 60+ structural signals and implemented OOF stacking.
> Diagnosed a class-ordered node-ID proxy that inflated “temporal” ROC-AUC to
> 1.00, then introduced proxy permutation tests, timestamp allowlisting, and
> nested holdout guards; reproduced GraphSAGE at **0.7544 ± 0.0124 ROC-AUC** and
> **0.02718 ± 0.00193 PR-AUC** across five untouched outer folds.

## Limitations and next experiment

The anonymized NPZ contains no timestamp semantics, so GraphShield does **not**
claim a validated temporal improvement yet. The next valid experiment requires a
source-backed event or account-creation timestamp, followed by the forward
holdout and purge-gap protocol in `docs/EXPERIMENT_PROTOCOL.md`.
