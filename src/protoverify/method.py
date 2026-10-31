import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = [
    "protoverify",
    "supervised",
    "pseudo_label",
    "fixmatch",
    "freematch",
    "fixed_prototype",
    "similarity_only",
    "confidence_only",
    "prototype_only",
    "no_prototype_loss",
]


class PrototypeVerifier(nn.Module):
    def __init__(self, classes, dimension, config):
        super().__init__()
        self.config = config
        self.register_buffer("prototypes", torch.zeros(classes, dimension))
        self.register_buffer("confidence", torch.full((classes,), config["tau0"]))
        self.register_buffer("similarity_mean", torch.ones(classes))
        self.register_buffer("similarity_std", torch.zeros(classes))
        self.register_buffer("initialized", torch.zeros(classes, dtype=torch.bool))

    @torch.no_grad()
    def initialize(self, features, labels):
        features = F.normalize(features, dim=1)
        for k in range(len(self.prototypes)):
            selected = features[labels == k]
            if not len(selected):
                raise ValueError(f"Labeled split has no examples of class {k}")
            self.prototypes[k] = F.normalize(selected.mean(0), dim=0)
            similarities = selected @ self.prototypes[k]
            self.similarity_mean[k] = similarities.mean()
            self.similarity_std[k] = similarities.std(unbiased=False)
            self.initialized[k] = True

    @torch.no_grad()
    def select(self, logits, features, variant):
        probabilities = logits.softmax(1)
        confidence, labels = probabilities.max(1)
        thresholds = (
            self.config["tau0"]
            + self.config["gamma"] * (self.confidence - self.confidence.mean())
        ).clamp(self.config["tau_min"], self.config["tau_max"])
        if variant in ["fixmatch", "pseudo_label", "fixed_prototype"]:
            thresholds = torch.full_like(thresholds, self.config["tau0"])
        confidence_mask = confidence >= thresholds[labels]
        similarities = F.normalize(features, dim=1) @ self.prototypes.T
        boundary = (
            self.similarity_mean - self.config["eta"] * self.similarity_std
        ).clamp(self.config["delta_min"], self.config["delta_max"])
        similarity_mask = (
            similarities.gather(1, labels[:, None]).squeeze(1) >= boundary[labels]
        )
        prototype_mask = (
            similarity_mask
            if variant == "similarity_only"
            else (similarities.argmax(1) == labels) & similarity_mask
        )
        if variant == "supervised":
            mask = torch.zeros_like(confidence_mask)
        elif variant in ["fixmatch", "pseudo_label", "confidence_only"]:
            mask = confidence_mask
        elif variant == "prototype_only":
            mask = prototype_mask
        else:
            mask = confidence_mask & prototype_mask
        return labels, mask, confidence_mask, prototype_mask

    @torch.no_grad()
    def update(
        self,
        labeled_features,
        labels,
        weak_logits,
        weak_features,
        pseudo_labels,
        mask,
        warmup,
    ):
        probabilities = weak_logits.softmax(1)
        confidence, predicted = probabilities.max(1)
        for k in predicted.unique():
            values = confidence[predicted == k]
            self.confidence[k] = (
                self.config["beta"] * self.confidence[k]
                + (1 - self.config["beta"]) * values.mean()
            )
        labeled = F.normalize(labeled_features, dim=1)
        old_similarities = (labeled * self.prototypes[labels]).sum(1)
        features = labeled
        update_labels = labels
        if not warmup:
            features = torch.cat([features, F.normalize(weak_features[mask], dim=1)])
            update_labels = torch.cat([labels, pseudo_labels[mask]])
        for k in update_labels.unique():
            center = features[update_labels == k].mean(0)
            self.prototypes[k] = F.normalize(
                self.config["mu"] * self.prototypes[k]
                + (1 - self.config["mu"]) * center,
                dim=0,
            )
        for k in labels.unique():
            values = old_similarities[labels == k]
            self.similarity_mean[k] = (
                self.config["mu"] * self.similarity_mean[k]
                + (1 - self.config["mu"]) * values.mean()
            )
            self.similarity_std[k] = self.config["mu"] * self.similarity_std[k] + (
                1 - self.config["mu"]
            ) * values.std(unbiased=False)

    def prototype_loss(self, features, labels):
        logits = (
            F.normalize(features, dim=1)
            @ self.prototypes.detach().T
            / self.config["temperature"]
        )
        return F.cross_entropy(logits, labels)


class FreeMatchSelector(nn.Module):
    """Single-process SAT/SAF baseline following Microsoft USB FreeMatch.

    This state is independent of ProtoVerify's delayed prototype buffers.
    The mean-confidence threshold is updated before training selection.
    """

    def __init__(self, classes, momentum=0.999):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("probability", torch.full((classes,), 1 / classes))
        self.register_buffer("histogram", torch.full((classes,), 1 / classes))
        self.register_buffer("threshold", torch.tensor(1 / classes))

    @torch.no_grad()
    def select(self, logits, update=False):
        probabilities = logits.detach().softmax(1)
        confidence, labels = probabilities.max(1)
        if update:
            m = self.momentum
            self.threshold.mul_(m).add_(confidence.mean(), alpha=1 - m)
            self.probability.mul_(m).add_(probabilities.mean(0), alpha=1 - m)
            histogram = torch.bincount(labels, minlength=len(self.probability)).to(
                probabilities
            )
            self.histogram.mul_(m).add_(histogram / histogram.sum(), alpha=1 - m)
        boundary = self.threshold * self.probability / self.probability.max()
        return labels, confidence >= boundary[labels]

    def fairness_loss(self, logits, mask):
        if not mask.any():
            return logits.sum() * 0
        probabilities = logits[mask].softmax(1)
        histogram = torch.bincount(
            probabilities.argmax(1), minlength=logits.shape[1]
        ).to(logits)
        histogram /= histogram.sum()
        target = self.probability.detach() / self.histogram.detach().clamp_min(1e-12)
        target /= target.sum()
        inverse = torch.where(histogram > 0, 1 / histogram.clamp_min(1e-12), 0)
        adjusted = probabilities.mean(0) * inverse.detach()
        adjusted /= adjusted.sum()
        return (target * (adjusted + 1e-12).log()).sum()
