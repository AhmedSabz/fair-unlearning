"""
Test-set metrics used in the paper (Sec. 5.1 and Appendix A.10).

All fairness metrics are absolute differences between the two sensitive
groups (lower is fairer):
    aeod          Absolute Equalized Odds Difference (Eq. 1)
                  = 0.5 * (|TPR_a - TPR_b| + |FPR_a - FPR_b|)
    dp            |P(yhat=1 | a) - P(yhat=1 | b)|        demographic parity
    eopp          |TPR_a - TPR_b|                         equality of opportunity
    subgroup_acc  |acc_a - acc_b|                         subgroup accuracy
"""

import torch


def predict(theta, X):
    """Logistic regression prediction: 1 when X @ theta >= 0 (sigmoid >= 0.5)."""
    with torch.no_grad():
        return (X @ theta >= 0).long()


def _rate(numerator_mask, denominator_mask):
    count = denominator_mask.sum().item()
    return (numerator_mask & denominator_mask).sum().item() / max(count, 1)


def compute_metrics(predictions, y, sensitive):
    """Accuracy and group-fairness metrics from 0/1 predictions."""
    pred = predictions.long()
    labels = y.long()
    correct = pred == labels

    groups = {}
    for g in (0, 1):
        in_g = sensitive == g
        groups[g] = {
            "tpr": _rate(pred == 1, in_g & (labels == 1)),
            "fpr": _rate(pred == 1, in_g & (labels == 0)),
            "positive_rate": _rate(pred == 1, in_g),
            "accuracy": _rate(correct, in_g),
        }
    a, b = groups[0], groups[1]

    return {
        "accuracy": correct.double().mean().item(),
        "aeod": 0.5 * (abs(a["tpr"] - b["tpr"]) + abs(a["fpr"] - b["fpr"])),
        "dp": abs(a["positive_rate"] - b["positive_rate"]),
        "eopp": abs(a["tpr"] - b["tpr"]),
        "subgroup_acc": abs(a["accuracy"] - b["accuracy"]),
    }


def calculate_metrics(theta, X, y, sensitive):
    """Metrics for a single logistic regression model."""
    return compute_metrics(predict(theta, X), y, sensitive)
