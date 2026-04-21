import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from core.models.trainer import FoldResult


def evaluate_folds(fold_results: list[FoldResult]) -> dict[str, float]:
    accuracies, briers, aucs = [], [], []

    for result in fold_results:
        accuracies.append(accuracy_score(result.y_true, result.y_pred))
        briers.append(brier_score_loss(result.y_true, result.y_prob))

        if len(np.unique(result.y_true)) < 2:
            aucs.append(None)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                aucs.append(roc_auc_score(result.y_true, result.y_prob))

    valid_aucs = [a for a in aucs if a is not None]
    skipped_aucs = len(aucs) - len(valid_aucs)

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "brier_score_mean": float(np.mean(briers)),
        "brier_score_std": float(np.std(briers)),
        "roc_auc_mean": float(np.mean(valid_aucs)) if valid_aucs else None,
        "roc_auc_std": float(np.std(valid_aucs)) if valid_aucs else None,
    }
