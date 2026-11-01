# ProtoVerify

**ProtoVerify: Dual-Reliability Pseudo-Label Selection with Class-Adaptive Confidence and Prototype Consistency**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` downloads [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html), reserves a class-balanced validation set and preserves the test split. The config selects 250 balanced labeled training images; the remaining training pool is unlabeled. Saved `split.json` records the exact labeled/unlabeled indices.

```bash
python -m protoverify prepare --config configs/cifar10.json
```

## Training and execution

`configs/cifar10.json` fixes seed `2026`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON.

```bash
python -m protoverify train --config configs/cifar10.json --output outputs/protoverify
```

## Evaluation

```bash
python -m protoverify evaluate --config configs/cifar10.json --checkpoint outputs/protoverify/model.pt --output outputs/protoverify
```

## Comparisons and ablations

`compare` runs the configured variants; it includes model fitting where required. Keep comparison outputs separate from the main run.

```bash
python -m protoverify compare --config configs/cifar10.json --output outputs/comparison
python -m protoverify plot --config configs/cifar10.json --output outputs/comparison
```

## Outputs

Run directories contain `model.pt`, `history.csv`, `metrics.json`, `selection.csv`, `class_acceptance.csv` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files.

## Protocol, controls and verification

`confidence_only` is CAT-only. `fixed_prototype` keeps the fixed confidence threshold and the full prototype rule. `similarity_only` removes nearest-prototype agreement while retaining the class-adaptive confidence gate. `prototype_only` drops the confidence gate entirely. `no_prototype_loss` retains the dual selection while removing the labeled prototype objective.

`freematch` is a single-process SAT/SAF adaptation based on [Microsoft USB's FreeMatch implementation](https://github.com/microsoft/Semi-supervised-learning/tree/main/semilearn/algorithms/freematch): mean-confidence EMA, class-probability modulation, prediction-histogram adjustment and the entropy objective (`lambda_e=0.01`). Its own buffers update before training selection; evaluation freezes them. It shares the backbone, augmentation, optimizer and final-checkpoint protocol with the other methods. FixMatch/Pseudo-Label/FreeMatch start unlabeled learning immediately; prototype variants use the configured prototype warm-up.

A 5,000-image validation set is reserved from official training data. The remaining pool supplies 250 balanced labels and unlabeled examples. Test labels are used only for final diagnostics. CIFAR normalization constants are fixed. Config defaults are `gamma=0.2`, `eta=1`, cosine bounds `[0.2,0.95]`, prototype temperature `0.1`, RandAugment `(2,10)`, SGD/cosine and 200 epochs.

`compare` writes per-seed `comparison.csv` and `summary.json`. Means give each independently trained seed equal weight. Accuracy SD is the sample SD across training seeds, unavailable with one seed. PL accuracy is correct accepted predictions / accepted predictions; utilization is accepted / unlabeled pool (both percent); acceptance CV uses the population SD of the K accepted class counts divided by their mean. An empty accepted set has null PL accuracy/CV. Hidden unlabeled labels enter this final-checkpoint diagnostic only.

| Feature or metric | Entry | Command/settings |
|---|---|---|
| Balanced labels, weak/strong views, WRN-28-2 | `cli.split_indices/draw`, `common.FeatureNet` | `prepare`, `train`, `configs/cifar10.json` |
| Delayed CAT/prototype gates and labeled prototype loss | `method.PrototypeVerifier` | `tau0=0.95`, `beta=0.999`, `mu=0.99`, `lambda_p=0.1`, warm-up 10 |
| Baselines and component removals | `VARIANTS`, `FreeMatchSelector`, `cli.train` | `compare`, `--variant fixed_prototype/similarity_only/freematch` |
| Accuracy, PL precision/utilization/CV; seed means/SD | `cli.evaluate`, `metrics` | `evaluate`, `compare` |
| Training/seed visualization | `common.plot_csv` | `plot` |

```bash
python tests/smoke.py
```
