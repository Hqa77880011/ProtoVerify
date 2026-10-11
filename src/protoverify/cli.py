from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms

from .common import (
    Images,
    FeatureNet,
    accuracy,
    device_of,
    seed_everything,
    prepare,
    parse_config,
    save_json,
    save_rows,
    plot_csv,
)
from .method import PrototypeVerifier, VARIANTS


def split_indices(dataset, config):
    rng = np.random.default_rng(config["seed"])
    classes = len(dataset.classes)
    count = config["num_labels"]
    if count < classes or count % classes:
        raise ValueError("num_labels must be a positive multiple of the class count")
    selected = []
    for k in range(classes):
        candidates = np.flatnonzero(dataset.labels == k)
        if len(candidates) <= count // classes:
            raise ValueError(f"Class {k} needs labeled and unlabeled examples")
        selected.extend(rng.permutation(candidates)[: count // classes])
    labeled = np.asarray(selected)
    unlabeled = np.setdiff1d(np.arange(len(dataset)), labeled)
    return labeled, unlabeled


def draw(dataset, indices, count, rng, strong=False):
    chosen = rng.choice(indices, size=count, replace=len(indices) < count)
    weak, strong_views = [], []
    strong_policy = transforms.RandAugment(num_ops=2, magnitude=10)
    for index in chosen:
        image = dataset.image(int(index))
        image = transforms.Resize((dataset.size, dataset.size))(image)
        image = transforms.RandomHorizontalFlip()(
            transforms.RandomCrop(dataset.size, padding=4)(image)
        )
        weak.append(transforms.ToTensor()(image))
        if strong:
            strong_views.append(transforms.ToTensor()(strong_policy(image)))
    return chosen, torch.stack(weak), torch.stack(strong_views) if strong else None


def initialize_verifier(model, dataset, indices, verifier, config):
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(indices), config["batch_size"]):
            batch = indices[start : start + config["batch_size"]]
            images = torch.stack([dataset[int(i)][0] for i in batch]).to(
                device_of(config)
            )
            vectors.append(model(images)[1])
    verifier.initialize(
        torch.cat(vectors),
        torch.as_tensor(dataset.labels[indices], device=device_of(config)),
    )


def train(config):
    if config["variant"] not in VARIANTS:
        raise ValueError(f"Unknown variant: {config['variant']}")
    seed_everything(config["seed"])
    rng = np.random.default_rng(config["seed"])
    dataset = Images(config, "train")
    labeled, unlabeled = split_indices(dataset, config)
    model = FeatureNet(len(dataset.classes), arch=config["arch"]).to(device_of(config))
    verifier = PrototypeVerifier(len(dataset.classes), model.dimension, config).to(
        device_of(config)
    )
    initialize_verifier(model, dataset, labeled, verifier, config)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    history = []
    step = 0
    per_epoch = config["steps_per_epoch"] or max(
        1,
        int(
            np.ceil(len(unlabeled) / (config["batch_size"] * config["unlabeled_ratio"]))
        ),
    )
    for epoch in range(config["epochs"]):
        model.train()
        losses, acceptance = [], []
        warmup = epoch < config["warmup_epochs"]
        for _ in range(per_epoch):
            chosen, x, _ = draw(dataset, labeled, config["batch_size"], rng)
            _, weak, strong = draw(
                dataset,
                unlabeled,
                config["batch_size"] * config["unlabeled_ratio"],
                rng,
                strong=True,
            )
            y = torch.as_tensor(dataset.labels[chosen], device=device_of(config))
            x, weak, strong = [
                value.to(device_of(config)) for value in [x, weak, strong]
            ]
            labeled_logits, labeled_features = model(x)
            weak_logits, weak_features = model(weak)
            pseudo_labels, mask, _, _ = verifier.select(
                weak_logits.detach(), weak_features.detach(), config["variant"]
            )
            if warmup:
                mask.zero_()
            supervised = F.cross_entropy(labeled_logits, y)
            unsupervised = supervised.new_zeros(())
            if config["variant"] != "supervised" and not warmup:
                student_logits = (
                    weak_logits
                    if config["variant"] == "pseudo_label"
                    else model(strong)[0]
                )
                unsupervised = (
                    F.cross_entropy(student_logits, pseudo_labels, reduction="none")
                    * mask
                ).mean()
            prototype = supervised.new_zeros(())
            if config["variant"] in [
                "protoverify",
                "prototype_only",
                "confidence_only",
            ]:
                prototype = verifier.prototype_loss(labeled_features, y)
            loss = (
                supervised
                + config["lambda_u"] * unsupervised
                + config["lambda_p"] * prototype
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            verifier.update(
                labeled_features.detach(),
                y,
                weak_logits.detach(),
                weak_features.detach(),
                pseudo_labels,
                mask,
                warmup,
            )
            losses.append(loss.item())
            acceptance.append(mask.float().mean().item())
            step += 1
            if config["max_steps"] and step >= config["max_steps"]:
                break
        row = {
            "epoch": epoch + 1,
            "variant": config["variant"],
            "seed": config["seed"],
            "loss": float(np.mean(losses)),
            "utilization": float(np.mean(acceptance)),
            "val_accuracy": accuracy(model, Images(config, "val"), config),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        scheduler.step()
        if config["max_steps"] and step >= config["max_steps"]:
            break
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "verifier": verifier.state_dict(),
            "labeled": labeled,
            "unlabeled": unlabeled,
            "classes": dataset.classes,
            "config": config,
        },
        output / "model.pt",
    )
    save_json(
        output / "split.json",
        {
            "labeled": labeled.tolist(),
            "unlabeled": unlabeled.tolist(),
            "seed": config["seed"],
        },
    )
    save_rows(output / "history.csv", history)
    save_json(output / "config.json", config)
    return str(output / "model.pt")


def evaluate(config):
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    trained = state["config"]
    model = FeatureNet(len(state["classes"]), trained["arch"]).to(device_of(config))
    model.load_state_dict(state["model"])
    verifier = PrototypeVerifier(len(state["classes"]), model.dimension, trained).to(
        device_of(config)
    )
    verifier.load_state_dict(state["verifier"])
    model.eval()
    test_accuracy = accuracy(model, Images(config, "test"), config)
    dataset = Images(config, "train")
    indices = state["unlabeled"]
    if len(dataset) <= int(np.max(indices)):
        raise ValueError(
            "Evaluation training pool differs from the saved split; use the same limit and data directory"
        )
    accepted = correct = total = 0
    counts = np.zeros(len(state["classes"]), dtype=int)
    diagnostics = []
    with torch.no_grad():
        for start in range(0, len(indices), config["batch_size"]):
            batch = indices[start : start + config["batch_size"]]
            x = torch.stack([dataset[int(i)][0] for i in batch]).to(device_of(config))
            logits, features = model(x)
            labels, mask, conf, proto = verifier.select(
                logits, features, trained["variant"]
            )
            labels, mask = labels.cpu().numpy(), mask.cpu().numpy()
            ground_truth = dataset.labels[batch]
            accepted += int(mask.sum())
            correct += int(((labels == ground_truth) & mask).sum())
            total += len(batch)
            counts += np.bincount(labels[mask], minlength=len(counts))
            for i, index in enumerate(batch):
                diagnostics.append(
                    {
                        "index": int(index),
                        "prediction": int(labels[i]),
                        "label": int(ground_truth[i]),
                        "accepted": bool(mask[i]),
                        "confidence_gate": bool(conf[i]),
                        "prototype_gate": bool(proto[i]),
                    }
                )
    summary = {
        "variant": trained["variant"],
        "seed": trained["seed"],
        "accuracy": test_accuracy,
        "pseudo_label_accuracy": 100 * correct / accepted if accepted else None,
        "utilization": 100 * accepted / total,
        "acceptance_cv": float(counts.std() / counts.mean()) if accepted else None,
        "accepted": accepted,
    }
    save_json(Path(config["output"]) / "metrics.json", summary)
    save_rows(Path(config["output"]) / "selection.csv", diagnostics)
    save_rows(
        Path(config["output"]) / "class_acceptance.csv",
        [{"class": k, "accepted": int(value)} for k, value in enumerate(counts)],
    )
    print(json.dumps(summary), flush=True)
    return summary


def compare(config):
    rows = []
    for seed in config["seeds"]:
        for variant in config["variants"]:
            current = dict(
                config,
                variant=variant,
                seed=seed,
                output=str(Path(config["output"]) / variant / str(seed)),
            )
            current["checkpoint"] = train(current)
            rows.append(evaluate(current))
    save_rows(Path(config["output"]) / "comparison.csv", rows)


def main():
    command, config = parse_config(["prepare", "train", "evaluate", "compare", "plot"])
    if command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "compare":
        compare(config)
    else:
        root = Path(config["output"])
        if (root / "comparison.csv").exists():
            plot_csv(
                root / "comparison.csv", root / "comparison.pdf", "seed", "accuracy"
            )
        else:
            plot_csv(
                root / "history.csv", root / "training.pdf", "epoch", "val_accuracy"
            )
