import pandas as pd

from core.models.classifier import CryptoClassifier
from core.models.trainer import walk_forward_train
from core.decision.engine import decide


def run_prediction(
    features_df: pd.DataFrame,
    labels: pd.Series,
    probability_threshold: float = 0.58,
    vol_threshold: float = 0.06,
) -> dict:
    walk_forward_train(features_df, labels)

    cal_size = max(1, int(len(features_df) * 0.1))
    train_end = len(features_df) - 1

    X_train = features_df.iloc[:train_end]
    y_train = labels.iloc[:train_end]
    X_cal = features_df.iloc[train_end - cal_size:train_end]
    y_cal = labels.iloc[train_end - cal_size:train_end]

    clf = CryptoClassifier(random_seed=42)
    clf.fit(X_train, y_train)
    clf.calibrate(X_cal, y_cal)

    latest_row = features_df.iloc[[-1]]
    probability = float(clf.predict_proba(latest_row)[0, 1])
    realized_vol_5d = float(features_df["realized_vol_5d"].iloc[-1])

    decision = decide(
        probability=probability,
        realized_vol_5d=realized_vol_5d,
        last_trade_day=None,
        current_day=0,
        probability_threshold=probability_threshold,
    )

    return {
        "probability": decision["probability"],
        "signal": decision["signal"],
        "reasons": decision["reasons"],
    }
