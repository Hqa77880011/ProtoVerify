"""Synthetic SSL checks, never benchmark evidence."""

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from protoverify.common import FeatureNet, Images, prepare, seed_everything, plot_csv
from protoverify.cli import train, evaluate, split_indices
from protoverify.method import PrototypeVerifier, FreeMatchSelector
from protoverify.metrics import selection_metrics, summarize_seeds


def main():
    torch.set_num_threads(1)
    seed_everything(71)
    out = Path("outputs/smoke").resolve()
    rng = np.random.default_rng(71)
    for split in ("train", "val", "test"):
        for label in ("a", "b"):
            folder = out / "raw" / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                    folder / f"{i}.png"
                )
    cfg = json.loads(Path("configs/cifar10.json").read_text())
    cfg.update(
        dataset="folder",
        root=str(out / "raw"),
        data=str(out / "data"),
        output=str(out / "train"),
        device="cpu",
        epochs=1,
        max_steps=2,
        steps_per_epoch=2,
        batch_size=2,
        unlabeled_ratio=1,
        num_labels=2,
        warmup_epochs=0,
        threads=1,
    )
    prepare(cfg)
    data = Images(cfg, "train")
    labeled, unlabeled = split_indices(data, cfg)
    assert not set(labeled) & set(unlabeled)
    model = FeatureNet(2, "wrn28_2")
    verifier = PrototypeVerifier(2, model.dimension, cfg)
    x = torch.stack([data[i][0] for i in range(4)])
    y = torch.tensor([0, 0, 1, 1])
    model.eval()
    with torch.no_grad():
        verifier.initialize(model(x)[1], y)
    model.train()
    old = deepcopy(model.state_dict())
    logits, features = model(x)
    before = deepcopy(verifier.state_dict())
    targets, mask, _, _ = verifier.select(logits, features, "protoverify")
    assert all(torch.equal(v, verifier.state_dict()[k]) for k, v in before.items())
    loss = F.cross_entropy(logits, y) + 0.1 * verifier.prototype_loss(features, y)
    assert torch.isfinite(loss)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    optimizer.zero_grad()
    loss.backward()
    for name in ("encoder.0.weight", "classifier.weight"):
        value = dict(model.named_parameters())[name]
        assert (
            value.grad is not None
            and torch.isfinite(value.grad).all()
            and value.grad.abs().sum() > 0
        )
    optimizer.step()
    assert not torch.equal(
        old["encoder.0.weight"], model.state_dict()["encoder.0.weight"]
    )
    verifier.update(
        features.detach(), y, logits.detach(), features.detach(), targets, mask, False
    )
    model.eval()
    saved = out / "roundtrip.pt"
    torch.save({"model": model.state_dict(), "verifier": verifier.state_dict()}, saved)
    other = FeatureNet(2, "wrn28_2").eval()
    state = torch.load(saved, weights_only=True)
    other.load_state_dict(state["model"])
    torch.testing.assert_close(model(x)[0], other(x)[0], rtol=0, atol=0)
    tiny = PrototypeVerifier(2, 2, cfg)
    tiny.initialize(torch.eye(2), torch.tensor([0, 1]))
    cases = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    vectors = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    assert tiny.select(cases, vectors, "protoverify")[1].tolist() == [True, False]
    loose = dict(cfg, delta_max=0.5)
    tiny2 = PrototypeVerifier(2, 2, loose)
    tiny2.initialize(torch.eye(2), torch.tensor([0, 1]))
    assert tiny2.select(cases, vectors, "similarity_only")[1].tolist() == [True, True]
    assert tiny2.select(cases, vectors, "fixed_prototype")[1].tolist() == [True, False]
    free = FreeMatchSelector(2, 0.5)
    _, accepted = free.select(cases, update=True)
    strong = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    fairness = free.fairness_loss(strong, accepted)
    fairness.backward()
    assert (
        torch.isfinite(fairness)
        and torch.isfinite(strong.grad).all()
        and strong.grad.abs().sum() > 0
    )
    hand = selection_metrics([0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0], 2)
    assert (
        np.isclose(hand["pseudo_label_accuracy"], 200 / 3) and hand["utilization"] == 75
    )
    assert np.isclose(hand["acceptance_cv"], 1 / 3)
    rows = [
        dict(variant="v", seed=1, accuracy=50.0, **hand),
        dict(variant="v", seed=2, accuracy=70.0, **hand),
    ]
    summary = summarize_seeds(rows)[0]
    assert summary["accuracy_mean"] == 60 and np.isclose(
        summary["accuracy_sd"], np.sqrt(200)
    )
    cfg["checkpoint"] = train(cfg)
    result = evaluate(cfg)
    assert 0 <= result["accuracy"] <= 100
    free_cfg = dict(
        cfg,
        variant="freematch",
        output=str(out / "freematch"),
        max_steps=1,
        warmup_epochs=10,
    )
    free_cfg["checkpoint"] = train(free_cfg)
    free_result = evaluate(free_cfg)
    assert free_result["accepted"] > 0 and free_result["utilization"] > 0
    plot_csv(out / "train/history.csv", out / "training.pdf", "epoch", "val_accuracy")
    print(
        "PASS: file preparation, two-step SSL, encoder/head gradients and updates, detached delayed memory, SAT/SAF, checkpoint equivalence, gate ablations, hand metrics, evaluation/plot"
    )


if __name__ == "__main__":
    main()
