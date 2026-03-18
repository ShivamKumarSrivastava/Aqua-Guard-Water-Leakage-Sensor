"""
modules/ml_models.py
--------------------
Isolation Forest and Random Forest wrappers.
Returns a copy of the DataFrame with updated 'active_label' column.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

FEATURES_BASE = ["flow_rate_lpm", "baseline_mean", "pressure_psi"]

FEATURES_EXTENDED = [
    "pressure_ratio",
    "flow_ratio",
    "pressure_psi",
    "flow_deviation",
    "flow_rate_lpm",
    "baseline_mean",
]


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flow_ratio"]     = df["flow_rate_lpm"] / (df["baseline_mean"] + 1)
    df["pressure_ratio"] = df["pressure_psi"] / 80
    df["flow_deviation"] = abs(df["flow_rate_lpm"] - df["baseline_mean"])
    return df


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Apply Isolation Forest; returns df with updated active_label."""
    df = df.copy()
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(df[FEATURES_BASE])
    df["active_label"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
    return df


def run_random_forest(df: pd.DataFrame) -> pd.DataFrame:
    """Train + predict with Random Forest; returns df with updated active_label."""
    df = _add_engineered_features(df)
    le = LabelEncoder()
    y  = le.fit_transform(df["active_label"])

    rf = RandomForestClassifier(random_state=42)
    rf.fit(df[FEATURES_EXTENDED], y)
    df["active_label"] = le.inverse_transform(rf.predict(df[FEATURES_EXTENDED]))
    return df


def get_feature_importance(df: pd.DataFrame) -> pd.Series:
    """Return a Series of feature importances from a trained Random Forest."""
    df = _add_engineered_features(df)
    le = LabelEncoder()
    y  = le.fit_transform(df["active_label"])

    rf = RandomForestClassifier(random_state=42)
    rf.fit(df[FEATURES_EXTENDED], y)
    return pd.Series(rf.feature_importances_, index=FEATURES_EXTENDED).sort_values()
