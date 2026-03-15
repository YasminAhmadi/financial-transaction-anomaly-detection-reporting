from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from financial_anomaly_detection.anomaly_detection import detect_anomalies
from financial_anomaly_detection.config import PipelineConfig
from financial_anomaly_detection.data_ingestion import download_online_dataset, load_raw_transactions
from financial_anomaly_detection.preprocessing import build_ar_transactions
from financial_anomaly_detection.reporting.excel_report import build_excel_report
from financial_anomaly_detection.reporting.pdf_report import build_pdf_report


def run_pipeline(config: PipelineConfig, force_refresh: bool = False) -> Dict[str, Path]:
    """Run the full pipeline and return key output file paths."""
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = download_online_dataset(
        urls=config.dataset_urls,
        target_path=config.raw_dataset_path,
        force_refresh=force_refresh,
    )
    raw_df = load_raw_transactions(raw_csv)

    ar_df = build_ar_transactions(
        raw_df=raw_df,
        random_seed=config.random_seed,
        max_records=config.max_records,
        synthetic_anomaly_ratio=config.synthetic_anomaly_ratio,
    )

    scored_df, summary = detect_anomalies(
        ar_df=ar_df,
        isolation_contamination=config.isolation_contamination,
        zscore_threshold=config.zscore_threshold,
        random_seed=config.random_seed,
    )

    scored_df.to_csv(config.ar_dataset_path, index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = config.reports_dir / f"anomaly_report_{timestamp}.xlsx"
    pdf_path = config.reports_dir / f"anomaly_report_{timestamp}.pdf"

    build_excel_report(scored_df=scored_df, summary=summary, output_path=excel_path)
    build_pdf_report(scored_df=scored_df, summary=summary, output_path=pdf_path)

    summary_path = config.processed_dir / "summary_metrics.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return {
        "raw_dataset": raw_csv,
        "processed_dataset": config.ar_dataset_path,
        "summary_metrics": summary_path,
        "excel_report": excel_path,
        "pdf_report": pdf_path,
    }
