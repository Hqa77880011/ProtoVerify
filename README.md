# ProtoVerify

Implementation of **ProtoVerify: Dual-Reliability Pseudo-Label Selection with Class-Adaptive Confidence and Prototype Consistency**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` downloads [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), reserves a class-balanced validation set and preserves the test split. The config selects 250 balanced labeled training images; the remaining training pool is unlabeled. Saved `split.json` records the exact labeled/unlabeled indices.

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

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
