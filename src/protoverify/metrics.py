"""Final-checkpoint selection counts and equally weighted seed summaries."""

import numpy as np


def selection_metrics(predictions, labels, mask, classes):
    predictions, labels, mask = (
        np.asarray(predictions),
        np.asarray(labels),
        np.asarray(mask, dtype=bool),
    )
    if not len(mask):
        raise ValueError("Selection evaluation pool is empty")
    accepted = int(mask.sum())
    counts = np.bincount(predictions[mask].astype(int), minlength=classes)
    return {
        "pseudo_label_accuracy": 100 * float((predictions[mask] == labels[mask]).mean())
        if accepted
        else None,
        "utilization": 100 * accepted / len(mask),
        "acceptance_cv": float(counts.std(ddof=0) / counts.mean())
        if accepted
        else None,
        "accepted": accepted,
    }


def summarize_seeds(rows):
    summary = []
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        if len({row["seed"] for row in subset}) != len(subset):
            raise ValueError("Duplicate trained seed for variant " + variant)
        item = {"variant": variant, "training_seeds": len(subset)}
        for key in (
            "accuracy",
            "pseudo_label_accuracy",
            "utilization",
            "acceptance_cv",
        ):
            values = [row[key] for row in subset if row[key] is not None]
            item[key + "_mean"] = float(np.mean(values)) if values else None
            item[key + "_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        summary.append(item)
    return summary
