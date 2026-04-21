import numpy as np
import pandas as pd

from core.models.classifier import CryptoClassifier


class CrossSectionalRanker:
    def __init__(self, random_seed: int = 42):
        self._clf = CryptoClassifier(random_seed=random_seed)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CrossSectionalRanker":
        self._clf.fit(X, y)
        return self

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series) -> "CrossSectionalRanker":
        self._clf.calibrate(X_val, y_val)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1]

    def rank_date(self, feature_rows: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
        coins = list(feature_rows.keys())
        X = pd.concat([feature_rows[c] for c in coins], ignore_index=True)
        scores = self.score(X)
        return sorted(zip(coins, scores), key=lambda x: -x[1])
