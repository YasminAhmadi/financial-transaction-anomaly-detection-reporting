from __future__ import annotations

import numpy as np
import pandas as pd

from financial_anomaly_detection.anomaly_detection import detect_anomalies


def test_detect_anomalies_returns_expected_columns() -> None:
    rng = np.random.default_rng(42)
    size = 200

    df = pd.DataFrame(
        {
            "transaction_id": [f"t-{i}" for i in range(size)],
            "invoice_amount": rng.normal(1000, 150, size).clip(50, None),
            "quantity": rng.integers(1, 10, size),
            "unit_price": rng.normal(120, 20, size).clip(5, None),
            "payment_terms_days": rng.choice([15, 30, 45, 60], size=size),
            "days_past_due": rng.integers(0, 25, size),
            "outstanding_amount": rng.normal(180, 75, size).clip(0, None),
        }
    )

    df.loc[0, "invoice_amount"] = 9000
    df.loc[1, "days_past_due"] = 180

    scored, summary = detect_anomalies(
        ar_df=df,
        isolation_contamination=0.05,
        zscore_threshold=3.0,
        random_seed=42,
    )

    assert "anomaly_flag" in scored.columns
    assert "risk_level" in scored.columns
    assert summary["total_transactions"] == float(size)
