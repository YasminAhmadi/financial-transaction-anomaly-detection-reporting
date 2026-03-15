# Financial Transaction Anomaly Detection & Reporting

End-to-end Python pipeline to:
- Ingest online transaction data (downloaded in code, no local dataset dependency)
- Simulate accounts-receivable (AR) transaction behavior
- Detect anomalies using both Isolation Forest and z-score techniques
- Produce analyst-ready Excel and PDF reports with at-a-glance insights

## 1. Project Structure

```text
Financial Transaction Anomaly Detection/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   └── reports/
├── src/
│   └── financial_anomaly_detection/
│       ├── reporting/
│       │   ├── excel_report.py
│       │   └── pdf_report.py
│       ├── anomaly_detection.py
│       ├── config.py
│       ├── data_ingestion.py
│       ├── pipeline.py
│       ├── pipeline_cli.py
│       └── preprocessing.py
├── tests/
│   └── test_anomaly_detection.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── run_pipeline.py
```

## 2. What This Pipeline Does

1. Downloads a public online retail dataset from a remote URL.
2. Converts it into simulated AR-style records with fields such as:
- `invoice_amount`
- `payment_terms_days`
- `days_past_due`
- `outstanding_amount`
- `payment_status`
3. Injects a small percentage of synthetic anomalies to stress-test detection.
4. Runs anomaly detection:
- Isolation Forest (`iforest_flag`)
- Z-score thresholding (`zscore_flag`)
5. Combines both methods into a final `anomaly_flag` and assigns `risk_level`.
6. Exports outputs:
- Processed transaction dataset
- Summary metrics CSV
- Excel workbook (multi-sheet)
- PDF management report

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Run

### Option A: root script

```bash
python run_pipeline.py --max-records 50000
```

### Option B: package CLI (after editable install)

```bash
pip install -e .
run-financial-anomaly-pipeline --max-records 50000
```

### Optional arguments

- `--force-refresh`: force redownload of online dataset
- `--max-records`: cap records processed for speed/performance

## 5. Outputs

Generated files include:
- `data/raw/online_retail.csv`
- `data/processed/ar_transactions.csv`
- `data/processed/summary_metrics.csv`
- `outputs/reports/anomaly_report_<timestamp>.xlsx`
- `outputs/reports/anomaly_report_<timestamp>.pdf`

## 6. Reporting Views

### Excel workbook
- `Summary`: KPI snapshot for operations teams
- `Flagged Transactions`: transaction-level queue for review
- `Method Breakdown`: which detector triggered anomalies
- `Top Customer Exposure`: customer-level risk concentration

### PDF report
- Executive summary page
- Method/risk distribution visuals
- Top customer exposure chart

## 7. Notes for GitHub Upload

- Project is structured and self-contained.
- Large generated data and report artifacts are ignored in `.gitignore`.
- Dataset source is downloaded online in code, matching your requirement.

## 8. Test

```bash
PYTHONPATH=src pytest -q
```
