import numpy as np
import pandas as pd
from dataclasses import dataclass

from core.models.classifier import CryptoClassifier


@dataclass
class FoldResult:
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def walk_forward_train(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
    min_train_size: float = 0.5,
    random_seed: int = 42,
) -> list[FoldResult]:
    n = len(features)
    min_train = int(n * min_train_size)
    remaining = n - min_train
    fold_size = remaining // n_splits

    results: list[FoldResult] = []
    skipped = 0

    for i in range(n_splits):
        train_end = min_train + i * fold_size
        val_start = train_end
        val_end = val_start + fold_size if i < n_splits - 1 else n

        y_train = labels.iloc[:train_end]

        if len(np.unique(y_train)) < 2:
            skipped += 1
            continue

        X_train = features.iloc[:train_end]
        X_val = features.iloc[val_start:val_end]
        y_val = labels.iloc[val_start:val_end]

        clf = CryptoClassifier(random_seed=random_seed)
        clf.fit(X_train, y_train)
        clf.calibrate(X_val, y_val)

        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results.append(FoldResult(
            fold=i,
            y_true=y_val.to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
        ))

    return results
