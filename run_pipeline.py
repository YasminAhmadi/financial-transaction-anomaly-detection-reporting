from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from financial_anomaly_detection.config import PipelineConfig
from financial_anomaly_detection.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run financial transaction anomaly detection and reporting pipeline."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Redownload the online dataset even if cached locally.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=50000,
        help="Maximum number of records to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(max_records=args.max_records)
    outputs = run_pipeline(config=config, force_refresh=args.force_refresh)

    print("Pipeline completed. Output files:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
