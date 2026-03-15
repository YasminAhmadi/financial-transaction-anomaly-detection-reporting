from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def build_excel_report(scored_df: pd.DataFrame, summary: Dict[str, float], output_path: Path) -> None:
    """Create an analyst-friendly Excel workbook with key anomaly views."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flagged_df = scored_df[scored_df["anomaly_flag"]].copy()
    flagged_df = flagged_df.sort_values(by=["risk_level", "outstanding_amount"], ascending=[True, False])

    method_breakdown = (
        scored_df[scored_df["anomaly_flag"]]
        .groupby("anomaly_reason", dropna=False)
        .agg(
            flagged_transactions=("transaction_id", "count"),
            total_invoice_amount=("invoice_amount", "sum"),
            total_outstanding=("outstanding_amount", "sum"),
        )
        .reset_index()
    )

    customer_exposure = (
        scored_df[scored_df["anomaly_flag"]]
        .groupby(["customer_id", "customer_country"], dropna=False)
        .agg(
            flagged_transactions=("transaction_id", "count"),
            total_invoice_amount=("invoice_amount", "sum"),
            total_outstanding=("outstanding_amount", "sum"),
            max_days_past_due=("days_past_due", "max"),
        )
        .sort_values("total_outstanding", ascending=False)
        .head(25)
        .reset_index()
    )

    summary_df = pd.DataFrame(
        [
            {"metric": "Total Transactions", "value": int(summary["total_transactions"])},
            {"metric": "Flagged Transactions", "value": int(summary["flagged_transactions"])},
            {"metric": "Flag Rate (%)", "value": round(summary["flag_rate_pct"], 2)},
            {"metric": "Isolation Forest Flags", "value": int(summary["iforest_flags"])},
            {"metric": "Z-Score Flags", "value": int(summary["zscore_flags"])},
            {"metric": "Estimated Exposure", "value": round(summary["estimated_exposure"], 2)},
        ]
    )

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        flagged_df.to_excel(writer, sheet_name="Flagged Transactions", index=False)
        method_breakdown.to_excel(writer, sheet_name="Method Breakdown", index=False)
        customer_exposure.to_excel(writer, sheet_name="Top Customer Exposure", index=False)

        workbook = writer.book
        money_format = workbook.add_format({"num_format": "$#,##0.00"})
        pct_format = workbook.add_format({"num_format": "0.00"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#DDEBF7"})
        critical_fmt = workbook.add_format({"bg_color": "#F8CBAD"})
        high_fmt = workbook.add_format({"bg_color": "#FCE4D6"})

        summary_ws = writer.sheets["Summary"]
        summary_ws.set_column("A:A", 30)
        summary_ws.set_column("B:B", 18)
        summary_ws.set_row(0, None, header_fmt)
        summary_ws.set_column("B:B", 18, pct_format)

        flagged_ws = writer.sheets["Flagged Transactions"]
        flagged_ws.set_row(0, None, header_fmt)
        flagged_ws.set_column("A:D", 18)
        flagged_ws.set_column("E:E", 14)
        flagged_ws.set_column("F:G", 14)
        flagged_ws.set_column("H:H", 14, money_format)
        flagged_ws.set_column("I:I", 14, money_format)
        flagged_ws.set_column("J:J", 14)
        flagged_ws.set_column("K:L", 14)
        flagged_ws.set_column("M:O", 16)

        if len(flagged_df) > 0:
            risk_col = flagged_df.columns.get_loc("risk_level")
            row_end = len(flagged_df) + 1
            flagged_ws.conditional_format(
                1,
                risk_col,
                row_end,
                risk_col,
                {"type": "text", "criteria": "containing", "value": "critical", "format": critical_fmt},
            )
            flagged_ws.conditional_format(
                1,
                risk_col,
                row_end,
                risk_col,
                {"type": "text", "criteria": "containing", "value": "high", "format": high_fmt},
            )

        method_ws = writer.sheets["Method Breakdown"]
        method_ws.set_row(0, None, header_fmt)
        method_ws.set_column("A:A", 18)
        method_ws.set_column("B:B", 22)
        method_ws.set_column("C:D", 20, money_format)

        customer_ws = writer.sheets["Top Customer Exposure"]
        customer_ws.set_row(0, None, header_fmt)
        customer_ws.set_column("A:B", 18)
        customer_ws.set_column("C:C", 22)
        customer_ws.set_column("D:E", 20, money_format)
        customer_ws.set_column("F:F", 18)
