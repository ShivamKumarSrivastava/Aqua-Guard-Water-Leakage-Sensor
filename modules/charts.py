"""
modules/charts.py
-----------------
All matplotlib figure builders.
Every function returns a Figure — call st.pyplot(fig) in the app.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

FIG_W, FIG_H = 6, 5   # shared size for all charts

LABEL_COLORS = {
    "Normal":        "#22c55e",
    "High_Pressure": "#f97316",
    "Minor_Leak":    "#38bdf8",
    "Major_Burst":   "#ef4444",
    "Anomaly":       "#eab308",
}


def anomaly_scatter(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Pressure vs Flow scatter coloured by label."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for label, group in df.groupby("active_label"):
        ax.scatter(
            group["flow_rate_lpm"],
            group["pressure_psi"],
            label=label,
            alpha=0.7,
            s=40,
            color=LABEL_COLORS.get(label, "gray"),
        )

    ax.axhline(80, linestyle="--", color="red", label="Threshold 80 psi")
    ax.set_title("Pressure vs Flow — Anomaly Map")
    ax.set_xlabel("Flow Rate (lpm)")
    ax.set_ylabel("Pressure (psi)")
    ax.legend()
    fig.tight_layout()
    return fig


def feature_importance_bar(importance: pd.Series) -> matplotlib.figure.Figure:
    """Horizontal bar chart for RF feature importances."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    importance.plot(kind="barh", ax=ax, color="#38bdf8")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig


def zone_distribution(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Bar chart: anomaly count per location zone."""
    from modules.alerts import get_anomalies
    counts = (
        get_anomalies(df)
        .groupby("location_zone")["active_label"]
        .count()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    counts.plot(kind="bar", ax=ax, color="#f97316", edgecolor="none")
    ax.set_title("Anomalies by Zone")
    ax.set_xlabel("Zone")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def label_pie(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Pie chart of label distribution."""
    counts = df["active_label"].value_counts()
    colors = [LABEL_COLORS.get(l, "gray") for l in counts.index]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.pie(counts, labels=counts.index, colors=colors,
           autopct="%1.1f%%", startangle=140)
    ax.set_title("Label Distribution")
    fig.tight_layout()
    return fig
