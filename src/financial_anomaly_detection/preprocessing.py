from __future__ import annotations

import numpy as np
import pandas as pd


def build_ar_transactions(
    raw_df: pd.DataFrame,
    random_seed: int,
    max_records: int,
    synthetic_anomaly_ratio: float,
) -> pd.DataFrame:
    """Transform retail-like transactions into simulated AR transactions."""
    if raw_df.empty:
        raise ValueError("Raw dataframe is empty.")

    rng = np.random.default_rng(random_seed)

    required_cols = ["InvoiceNo", "StockCode", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID", "Country"]
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    df = raw_df.copy()
    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice", "CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    if len(df) > max_records:
        df = df.sample(n=max_records, random_state=random_seed)

    df = df.reset_index(drop=True)

    ar_df = pd.DataFrame()
    ar_df["transaction_id"] = df["InvoiceNo"].astype(str) + "-" + (df.index + 1).astype(str)
    ar_df["invoice_number"] = df["InvoiceNo"].astype(str)
    ar_df["customer_id"] = df["CustomerID"].astype(str)
    ar_df["customer_country"] = df["Country"].astype(str)
    ar_df["invoice_date"] = pd.to_datetime(df["InvoiceDate"])
    ar_df["quantity"] = df["Quantity"].astype(float)
    ar_df["unit_price"] = df["UnitPrice"].astype(float)
    ar_df["invoice_amount"] = ar_df["quantity"] * ar_df["unit_price"]

    payment_terms = rng.choice([15, 30, 45, 60], size=len(ar_df), p=[0.1, 0.55, 0.25, 0.1])
    ar_df["payment_terms_days"] = payment_terms
    ar_df["due_date"] = ar_df["invoice_date"] + pd.to_timedelta(ar_df["payment_terms_days"], unit="D")

    unpaid_mask = rng.random(len(ar_df)) < 0.08
    payment_delay_days = np.round(rng.normal(loc=3, scale=12, size=len(ar_df))).astype(int)
    payment_delay_days = np.clip(payment_delay_days, -10, 120)

    payment_date = ar_df["due_date"] + pd.to_timedelta(payment_delay_days, unit="D")
    payment_date = payment_date.mask(unpaid_mask, pd.NaT)
    ar_df["payment_date"] = payment_date

    today = pd.Timestamp.now().normalize()
    paid_days_past_due = (ar_df["payment_date"] - ar_df["due_date"]).dt.days.fillna(0).clip(lower=0)
    unpaid_days_past_due = (today - ar_df["due_date"]).dt.days.clip(lower=0)
    ar_df["days_past_due"] = np.where(unpaid_mask, unpaid_days_past_due, paid_days_past_due).astype(int)

    outstanding_ratio = rng.uniform(0.3, 1.0, size=len(ar_df))
    ar_df["outstanding_amount"] = np.where(unpaid_mask, ar_df["invoice_amount"] * outstanding_ratio, 0.0)
    ar_df["payment_status"] = np.where(
        unpaid_mask,
        "unpaid",
        np.where(ar_df["days_past_due"] > 0, "paid_late", "paid_on_time"),
    )

    anomaly_count = max(1, int(len(ar_df) * synthetic_anomaly_ratio))
    anomaly_idx = rng.choice(ar_df.index.to_numpy(), size=anomaly_count, replace=False)
    spike_multiplier = rng.uniform(4.0, 12.0, size=anomaly_count)

    ar_df.loc[anomaly_idx, "invoice_amount"] = ar_df.loc[anomaly_idx, "invoice_amount"].to_numpy() * spike_multiplier
    ar_df.loc[anomaly_idx, "outstanding_amount"] = np.maximum(
        ar_df.loc[anomaly_idx, "outstanding_amount"].to_numpy(),
        ar_df.loc[anomaly_idx, "invoice_amount"].to_numpy() * rng.uniform(0.4, 1.0, size=anomaly_count),
    )
    ar_df.loc[anomaly_idx, "days_past_due"] = ar_df.loc[anomaly_idx, "days_past_due"].to_numpy() + rng.integers(45, 150, size=anomaly_count)

    ar_df["is_injected_anomaly"] = False
    ar_df.loc[anomaly_idx, "is_injected_anomaly"] = True

    return ar_df.sort_values(by="invoice_date").reset_index(drop=True)
