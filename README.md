# ProtoVerify

Implementation of **ProtoVerify: Dual-Reliability Pseudo-Label Selection with Class-Adaptive Confidence and Prototype Consistency**. The method extends weak-to-strong semi-supervised learning with class-specific confidence thresholds and a nearest-prototype agreement check.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Data and training

`prepare` downloads [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) through torchvision and creates a seeded, class-balanced validation split of 5,000 training images. The remaining 45,000 images form the training pool; the official test split is untouched. The prepared directory contains `train.npz`, `val.npz`, `test.npz` and `meta.json`. Array indices identify the original training examples. A custom image collection can use `dataset="folder"` with matching class directories under `root/train`, `root/val` and `root/test`.

The supplied CIFAR-10 configuration selects 250 labeled training images, balanced across classes, from the prepared training pool. The remaining training examples are unlabeled. `split.json` saves their indices; identical seeds and data produce identical splits.

```bash
python -m protoverify prepare --config configs/cifar10.json
python -m protoverify train --config configs/cifar10.json --output outputs/protoverify
python -m protoverify evaluate --config configs/cifar10.json --checkpoint outputs/protoverify/model.pt --output outputs/protoverify
python -m protoverify plot --config configs/cifar10.json --output outputs/protoverify
```

Hidden training labels are never supplied to unlabeled losses or memory updates. Evaluation reads them after the checkpoint is fixed to calculate pseudo-label diagnostics. Validation is independent of the unlabeled pool and test data.

## Comparisons

```bash
python -m protoverify compare --config configs/cifar10.json --output outputs/comparison
python -m protoverify plot --config configs/cifar10.json --output outputs/comparison
```

| Variant | Selection and objective |
| --- | --- |
| `supervised` | Labeled cross entropy only |
| `pseudo_label` | Fixed-confidence pseudo-labels on weak views |
| `fixmatch` | Fixed-confidence weak-to-strong consistency |
| `confidence_only` | Adaptive confidence and labeled prototype loss |
| `prototype_only` | Prototype selection and labeled prototype loss |
| `no_prototype_loss` | Joint selection without labeled prototype loss |
| `protoverify` | Adaptive confidence, prototype agreement and prototype loss |

FreeMatch is not included in the executable suite. `compare` retains results for every configured seed and variant.

## Configuration and result files

Commands read the supplied JSON configuration. `--data`, `--output`, `--device`, `--seed`, `--batch-size`, `--epochs`, `--max-steps` and `--limit` override individual fields. `--device auto` chooses CUDA when available. `max_steps=0` and `limit=0` mean no limit. `max_steps` stops optimization after the requested number of updates; it does not produce a trained benchmark model. For other fields use `--set KEY=JSON`, for example `--set lr=0.01` or `--set tau0=0.95`. String values require JSON quotes.

The backbone is WideResNet-28-2, with weak crop/flip augmentation and RandAugment for strong views. `num_labels` must divide evenly across classes and leave unlabeled examples. `unlabeled_ratio=7` controls the unlabeled-to-labeled batch ratio. `steps_per_epoch=0` derives the epoch length from the unlabeled pool. Training uses ten warm-up epochs, `tau0=0.95`, `beta=0.999`, `mu=0.99`, `lambda_u=1` and `lambda_p=0.1`.

The explicit conventions for parameters left unspecified are `gamma=0.2`, confidence bounds [0.8, 0.99], `eta=1`, similarity bounds [0.2, 0.95] and prototype temperature 0.1. Optimizer and epoch count are in the JSON configuration.

`history.csv` contains loss, utilization and validation accuracy. `model.pt` saves model and verifier state plus the exact labeled/unlabeled partition. `metrics.json` contains test accuracy, accepted pseudo-label accuracy, utilization and class-wise acceptance CV. `selection.csv` retains the two gate decisions per unlabeled example. `class_acceptance.csv` contains accepted pseudo-label counts. Pseudo-label accuracy and CV are `null` if no example is accepted, making zero utilization distinguishable from a valid precision measurement.

## Update convention

Labeled features initialize all class prototypes before training. During warm-up, the unsupervised loss is disabled and only labeled examples update prototypes. Each iteration evaluates both gates using the preceding buffers. A prediction must agree with the nearest prototype and exceed its class-specific cosine boundary. After optimization, detached features from labeled and accepted unlabeled samples update memory. Confidence EMA uses all weak predictions; similarity statistics use labeled features. The unsupervised loss divides by the complete unlabeled batch size, preserving the FixMatch masking convention. The 5,000-image validation reservation makes the training pool smaller than an experiment using all 50,000 CIFAR training images.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.
