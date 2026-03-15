from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def build_pdf_report(scored_df: pd.DataFrame, summary: Dict[str, float], output_path: Path) -> None:
    """Create a compact PDF report with KPI and anomaly visuals."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flagged = scored_df[scored_df["anomaly_flag"]].copy()

    with PdfPages(output_path) as pdf:
        fig1, ax1 = plt.subplots(figsize=(11, 8.5))
        ax1.axis("off")

        lines = [
            "Financial Transaction Anomaly Detection Report",
            "",
            f"Total transactions analyzed: {int(summary['total_transactions']):,}",
            f"Flagged transactions: {int(summary['flagged_transactions']):,}",
            f"Flag rate: {summary['flag_rate_pct']:.2f}%",
            f"Isolation Forest flags: {int(summary['iforest_flags']):,}",
            f"Z-score flags: {int(summary['zscore_flags']):,}",
            f"Estimated outstanding exposure: {_format_currency(summary['estimated_exposure'])}",
            "",
            "Interpretation guide:",
            "- Critical: flagged by both methods",
            "- High: flagged and financially impactful",
            "- Medium: flagged by one method",
        ]
        ax1.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=12)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        reason_counts = flagged["anomaly_reason"].value_counts().sort_values(ascending=False)
        if not reason_counts.empty:
            axes[0].bar(reason_counts.index, reason_counts.values, color=["#2E86AB", "#F18F01", "#C73E1D"])
            axes[0].set_title("Flagged Transactions by Method")
            axes[0].set_ylabel("Count")
            axes[0].tick_params(axis="x", rotation=30)
        else:
            axes[0].text(0.5, 0.5, "No flagged transactions", ha="center", va="center")
            axes[0].set_axis_off()

        risk_counts = flagged["risk_level"].value_counts().reindex(["critical", "high", "medium", "low"], fill_value=0)
        axes[1].bar(risk_counts.index, risk_counts.values, color=["#B22222", "#E67E22", "#F4D03F", "#2ECC71"])
        axes[1].set_title("Risk Tier Distribution")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=20)

        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        top_exposure = (
            flagged.groupby("customer_id", dropna=False)["outstanding_amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig3, ax3 = plt.subplots(figsize=(11, 8.5))
        if not top_exposure.empty:
            ax3.barh(top_exposure["customer_id"].astype(str), top_exposure["outstanding_amount"], color="#AF7AC5")
            ax3.invert_yaxis()
            ax3.set_title("Top 10 Customer Exposure from Flagged Transactions")
            ax3.set_xlabel("Outstanding Amount")
        else:
            ax3.text(0.5, 0.5, "No customer exposure to display", ha="center", va="center")
            ax3.set_axis_off()

        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)
