import math
from collections import defaultdict

import numpy as np
import torch
from torchmetrics.functional.classification import (
    multiclass_confusion_matrix,
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score,
    matthews_corrcoef,
    multiclass_cohen_kappa,
)
import scipy.stats as stats
from sklearn.metrics import precision_recall_curve, auc


def compute_clf_metrics(
    preds, gt, num_classes: int, prefix: str = "", ignore_index: int = -1
):
    # Perhaps not the most efficient way to do this
    # But guaranteed to be correct...
    metrics = dict()

    # Per-class metrics ====================================================
    precision = multiclass_precision(
        preds=preds,
        target=gt,
        num_classes=num_classes,
        average="none",
        ignore_index=ignore_index,
    )
    recall = multiclass_recall(
        preds=preds,
        target=gt,
        num_classes=num_classes,
        average="none",
        ignore_index=ignore_index,
    )
    f1 = multiclass_f1_score(
        preds=preds,
        target=gt,
        num_classes=num_classes,
        average="none",
        ignore_index=ignore_index,
    )

    metrics.update(
        {
            f"{prefix}precision_{c}": score.squeeze().float()
            for c, score in enumerate(precision.split(1))
        }
    )
    metrics.update(
        {
            f"{prefix}recall_{c}": score.squeeze().float()
            for c, score in enumerate(recall.split(1))
        }
    )
    metrics.update(
        {
            f"{prefix}f1_{c}": score.squeeze().float()
            for c, score in enumerate(f1.split(1))
        }
    )

    class_prevalence = torch.bincount(gt[gt != ignore_index])
    class_prevalence = class_prevalence / class_prevalence.sum()
    f1_gain = torch.clip(
        (f1 - class_prevalence) / ((1 - class_prevalence) * f1),
        min=-1.0,
        max=1.0
    )
    metrics.update(
        {
            f"{prefix}f1_gain_{c}": score.squeeze().float()
            for c, score in enumerate(f1_gain.split(1))
        }
    )

    # Aggregated metrics ===================================================
    accuracy = multiclass_accuracy(
        preds=preds,
        target=gt,
        num_classes=num_classes,
        average="macro",
        ignore_index=ignore_index,
    )
    f1_micro = multiclass_f1_score(
        preds=preds,
        target=gt,
        num_classes=num_classes,
        average="micro",
        ignore_index=ignore_index,
    )
    f1_macro = multiclass_f1_score(
        preds=preds,
        target=gt,
        num_classes=num_classes,
        average="macro",
        ignore_index=ignore_index,
    )
    mcc = matthews_corrcoef(
        preds=preds,
        target=gt,
        num_classes=num_classes,
        task="multiclass",
        ignore_index=ignore_index,
    )
    cohens_kappa = multiclass_cohen_kappa(
        preds=preds,
        target=gt,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    metrics.update(
        {
            f"{prefix}accuracy": accuracy.float(),
            f"{prefix}f1_micro": f1_micro.float(),
            f"{prefix}f1_macro": f1_macro.float(),
            f"{prefix}mcc": mcc.float(),
            f"{prefix}f1_gain_macro": f1_gain.mean().float(),
            f"{prefix}cohens_kappa": cohens_kappa.float(),
        }
    )

    return metrics


def compute_aupr_metrics(
    probs, gt, num_classes: int, prefix: str = "", ignore_index: int = -1
):
    probs = probs[gt != ignore_index].numpy()
    gt = gt[gt != ignore_index].numpy()

    macro_aupr = 0
    macro_auprg = 0
    pr_analysis = dict()
    for l in range(num_classes):
        # Standard PR analysis
        precision, recall, _ = precision_recall_curve(
            y_true=gt, probas_pred=probs[:, l], pos_label=l
        )
        aupr = auc(recall, precision)

        pr_analysis[f"{prefix}aupr_{l}"] = torch.tensor(aupr)
        macro_aupr += aupr

        # Flach & Kull PR Analysis
        prevalence = (gt == l).mean()

        with np.errstate(divide="ignore"):
            precision_gain = (precision - prevalence) / ((1 - prevalence) * precision)
            recall_gain = (recall - prevalence) / ((1 - prevalence) * recall)

        precision_gain = np.clip(precision_gain, a_min=-1, a_max=1)
        recall_gain = np.clip(recall_gain, a_min=-1, a_max=1)

        # AUPRG over 0 to 1 on recall
        above_0 = recall_gain >= 0
        if above_0.sum() > 1:
            auprg = auc(recall_gain[above_0], precision_gain[above_0])
        else:
            auprg = -1.0

        pr_analysis[f"{prefix}auprg_{l}"] = torch.tensor(auprg)
        macro_auprg += auprg

    pr_analysis[f"{prefix}macro_aupr"] = torch.tensor(macro_aupr / num_classes)
    pr_analysis[f"{prefix}macro_auprg"] = torch.tensor(macro_auprg / num_classes)

    return pr_analysis


def ci_multiplier(N: int, alpha: float = 0.10):
    return stats.t.ppf(1 - alpha / 2, N - 1)


def summarize_clf_metrics(metrics_dict, alpha: float = 0.10):
    metrics = {k: [] for k in metrics_dict[0].keys()}

    for fold_metrics in metrics_dict:
        for k, v in fold_metrics.items():
            metrics[k] += [v]

    metrics = {k: torch.stack(v) for k, v in metrics.items()}

    confidence_level = (1 - alpha) * 100

    summary_string = f"{' '*4}{'Metric':<20} {'Mean':^6} ({'Std.':^6}/{'SE':^6}) {int(confidence_level):2d}{'% CI [LB, UB]':^14}\n"
    metrics_summarized = defaultdict(dict)
    for k, v in metrics.items():
        N = v.shape[0]
        m = ci_multiplier(N, alpha)

        mean = torch.mean(v).item()
        std = torch.std(v).item()
        se = std / math.sqrt(v.shape[0])
        ub = mean + m * se
        lb = mean - m * se

        summary_string += (
            f"{' '*4}{k:<20} {mean:.4f} ({std:.4f}/{se:.4f}) [{lb:.4f}, {ub:.4f}]"
        )
        summary_string += "\n"

        metrics_summarized[k]["values"] = metrics[k].tolist()
        metrics_summarized[k]["mean"] = mean
        metrics_summarized[k]["std"] = std
        metrics_summarized[k]["se"] = se
        metrics_summarized[k]["ub"] = ub
        metrics_summarized[k]["lb"] = lb

    metrics_summarized = dict(metrics_summarized)

    return metrics_summarized, summary_string
