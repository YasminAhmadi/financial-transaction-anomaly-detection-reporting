from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLS = [
    "invoice_amount",
    "quantity",
    "unit_price",
    "payment_terms_days",
    "days_past_due",
    "outstanding_amount",
]


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def detect_anomalies(
    ar_df: pd.DataFrame,
    isolation_contamination: float,
    zscore_threshold: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Apply Isolation Forest and z-score anomaly detection to AR transactions."""
    if ar_df.empty:
        raise ValueError("AR dataframe is empty.")

    missing = [c for c in FEATURE_COLS if c not in ar_df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    df = ar_df.copy()
    x = df[FEATURE_COLS].astype(float).fillna(0.0)

    model = IsolationForest(
        n_estimators=300,
        contamination=isolation_contamination,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(x)

    df["iforest_score"] = -model.score_samples(x)
    df["iforest_flag"] = model.predict(x) == -1

    zscore_matrix = np.abs(np.column_stack([_safe_zscore(x[col]).to_numpy() for col in FEATURE_COLS]))
    df["max_abs_zscore"] = zscore_matrix.max(axis=1)
    df["zscore_flag"] = df["max_abs_zscore"] >= zscore_threshold

    df["anomaly_flag"] = df["iforest_flag"] | df["zscore_flag"]
    df["anomaly_reason"] = np.select(
        [
            df["iforest_flag"] & df["zscore_flag"],
            df["iforest_flag"] & ~df["zscore_flag"],
            ~df["iforest_flag"] & df["zscore_flag"],
        ],
        ["iforest+zscore", "iforest_only", "zscore_only"],
        default="normal",
    )

    high_impact_mask = (df["invoice_amount"] > df["invoice_amount"].quantile(0.95)) | (df["days_past_due"] > 60)
    df["risk_level"] = np.select(
        [
            df["anomaly_reason"].eq("iforest+zscore"),
            df["anomaly_flag"] & high_impact_mask,
            df["anomaly_flag"],
        ],
        ["critical", "high", "medium"],
        default="low",
    )

    summary = {
        "total_transactions": float(len(df)),
        "flagged_transactions": float(df["anomaly_flag"].sum()),
        "flag_rate_pct": float(df["anomaly_flag"].mean() * 100),
        "iforest_flags": float(df["iforest_flag"].sum()),
        "zscore_flags": float(df["zscore_flag"].sum()),
        "estimated_exposure": float(df.loc[df["anomaly_flag"], "outstanding_amount"].sum()),
    }

    return df, summary
