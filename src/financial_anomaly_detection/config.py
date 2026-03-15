from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PipelineConfig:
    """Runtime settings for the anomaly detection pipeline."""

    random_seed: int = 42
    max_records: int = 50000
    isolation_contamination: float = 0.02
    zscore_threshold: float = 3.0
    synthetic_anomaly_ratio: float = 0.01
    dataset_urls: List[str] = field(
        default_factory=lambda: [
            "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Exploratory-Data-Analysis-with-Python/master/chapter03/OnlineRetail.csv",
            "https://raw.githubusercontent.com/datagy/data/main/OnlineRetail.csv",
        ]
    )

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "reports"

    @property
    def raw_dataset_path(self) -> Path:
        return self.raw_dir / "online_retail.csv"

    @property
    def ar_dataset_path(self) -> Path:
        return self.processed_dir / "ar_transactions.csv"
