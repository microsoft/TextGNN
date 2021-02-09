"""A bunch of helper functions to generate a dictionary for evaluation metrics."""

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import numpy as np


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]

    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2
    }


def auc(preds, labels):
    roc_auc = roc_auc_score(y_true=labels, y_score=preds)
    pr_auc = average_precision_score(y_true=labels, y_score=preds)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def metrics(preds, labels):
    results = acc_and_f1(np.argmax(preds, axis=1), labels)
    results.update(auc(preds[:, 1], labels))
    return results
