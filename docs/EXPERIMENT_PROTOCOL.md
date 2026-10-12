# Experiment protocol

## Research question

Does verified temporal information improve fraud detection beyond anonymized node
attributes and graph structure, without exploiting storage order?

## Required experiment matrix

| Run | Features | Split | Purpose |
|---|---|---|---|
| A | raw + structural | stratified 5-fold OOF | reproducible baseline |
| B | A + verified timestamp features | same folds | temporal ablation |
| C | B | forward temporal holdout + purge gap | deployment-style estimate |
| D | A + node-ID proxy | same folds | leakage negative control only |
| E | A + edge-row-order proxy | original vs shuffled rows | invariance test only |

Runs D and E must never be used for model selection, reported as production
performance, or included in test predictions.

## Evaluation rules

1. Save every OOF probability together with node ID, label, fold, seed, feature
   policy, and commit hash.
2. Fit scalers and temporal transformations on the fit fold only.
3. Select thresholds inside the fit fold; do not optimize a threshold on the
   same predictions used to report F1 or MCC.
4. Compute ROC-AUC and PR-AUC. For 1.16% fraud prevalence, PR-AUC is essential.
5. Evaluate validation predictions before any full-data refit. A model refit on
   all labeled nodes cannot be evaluated on an old validation fold.
6. Publish mean and standard deviation across folds and seeds. Do not publish
   the legacy near-1.0 chart.

## Temporal feature admission checklist

A field is admitted only when its source, unit, timezone, event semantics, and
availability at prediction time are documented. Node IDs, row offsets, file
order, and neighbor IDs fail this test by default.
