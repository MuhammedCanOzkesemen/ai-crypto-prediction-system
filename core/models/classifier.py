import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class CryptoClassifier:
    def __init__(self, random_seed: int = 42):
        self._seed = random_seed
        self._model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        )
        self._calibrator: LogisticRegression | None = None
        self.calibrated: bool = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "CryptoClassifier":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self._model.predict_proba(X)
        if not self.calibrated or self._calibrator is None:
            return raw
        pos = raw[:, 1].reshape(-1, 1)
        calibrated_pos = self._calibrator.predict_proba(pos)[:, 1]
        return np.column_stack([1.0 - calibrated_pos, calibrated_pos])

    def calibrate(self, X_val: pd.DataFrame | np.ndarray, y_val: pd.Series | np.ndarray) -> "CryptoClassifier":
        y_arr = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)

        if len(np.unique(y_arr)) < 2:
            self._calibrator = None
            self.calibrated = False
            return self

        raw_probs = self._model.predict_proba(X_val)[:, 1]
        brier_before = brier_score_loss(y_arr, raw_probs)

        calibrator = LogisticRegression(
            C=1e10,
            solver="lbfgs",
            max_iter=1000,
            random_state=self._seed,
        )
        calibrator.fit(raw_probs.reshape(-1, 1), y_arr)

        calibrated_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        brier_after = brier_score_loss(y_arr, calibrated_probs)

        if brier_after < brier_before:
            self._calibrator = calibrator
            self.calibrated = True
        else:
            self._calibrator = None
            self.calibrated = False

        return self
