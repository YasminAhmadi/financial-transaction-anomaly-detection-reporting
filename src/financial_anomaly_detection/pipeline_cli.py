from __future__ import annotations

import argparse

from financial_anomaly_detection.config import PipelineConfig
from financial_anomaly_detection.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run financial transaction anomaly detection and reporting pipeline."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Redownload online dataset even if cached locally.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=50000,
        help="Maximum records to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(max_records=args.max_records)
    outputs = run_pipeline(config=config, force_refresh=args.force_refresh)

    print("Pipeline completed. Output files:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
