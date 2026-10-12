# Data contract

Place the competition file at `data/xydata/raw/data.npz`, or set
`GRAPHSHIELD_DATA=/absolute/path/to/data.npz`.

Expected arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `x` | `(579157, 17)` | anonymized node features |
| `y` | `(579157,)` | `0` normal, `1` fraud, `-100` hidden label |
| `edge_index` | `(167559, 2)` | directed edges |
| `edge_type` | `(167559,)` | 11 observed relation values (`1..11`) |
| `train_mask` | `(386104,)` | labeled node indices |
| `test_mask` | `(193053,)` | hidden-label node indices |

The original dataset is not committed because it is large and its redistribution
rights are not established. The anonymized schema does not document any timestamp
column. Do not reinterpret node IDs or edge row positions as timestamps.
