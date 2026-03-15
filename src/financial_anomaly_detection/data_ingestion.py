from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


def _download_file(url: str, target_path: Path, timeout: int = 60) -> None:
    """Download a file from a URL to the target path."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    target_path.write_bytes(response.content)


def download_online_dataset(urls: Iterable[str], target_path: Path, force_refresh: bool = False) -> Path:
    """Download the first reachable dataset URL and return the local file path."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force_refresh:
        return target_path

    last_error: Exception | None = None
    for url in urls:
        try:
            _download_file(url=url, target_path=target_path)
            return target_path
        except Exception as exc:  # pragma: no cover - network failures vary
            last_error = exc

    raise RuntimeError(
        "Failed to download dataset from all configured URLs. "
        "Check internet connectivity or update dataset URLs."
    ) from last_error


def load_raw_transactions(csv_path: Path) -> pd.DataFrame:
    """Load raw online retail transactions."""
    df = pd.read_csv(csv_path, encoding="latin1")
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df
