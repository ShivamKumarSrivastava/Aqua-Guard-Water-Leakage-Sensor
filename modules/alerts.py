"""
modules/alerts.py
-----------------
Alert filtering and severity helpers.
"""

import pandas as pd

# Labels that are considered non-normal
ALERT_LABELS = {"Major_Burst", "High_Pressure", "Minor_Leak", "Anomaly"}

# Severity ranking (higher = worse)
SEVERITY = {
    "Major_Burst":  3,
    "High_Pressure": 2,
    "Minor_Leak":   1,
    "Anomaly":      1,
    "Normal":       0,
}

ALERT_COLORS = {
    "Major_Burst":   ("90deg, #7f1d1d, #dc2626", "🔴"),
    "High_Pressure": ("90deg, #78350f, #d97706", "🟠"),
    "Minor_Leak":    ("90deg, #1e3a5f, #2563eb", "🔵"),
    "Anomaly":       ("90deg, #3b1f6e, #7c3aed", "🟣"),
}


def get_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose label is not Normal."""
    return df[df["active_label"] != "Normal"].copy()


def get_alerts_sorted(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Return top_n anomalies sorted by severity (worst first)."""
    anomalies = get_anomalies(df)
    anomalies["_severity"] = anomalies["active_label"].map(SEVERITY).fillna(0)
    return (
        anomalies
        .sort_values("_severity", ascending=False)
        .drop(columns="_severity")
        .head(top_n)
    )


def alert_summary(df: pd.DataFrame) -> dict:
    """Return a dict of counts per alert type."""
    anomalies = get_anomalies(df)
    return anomalies["active_label"].value_counts().to_dict()
