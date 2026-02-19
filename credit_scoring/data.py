from __future__ import annotations

from pathlib import Path


def download_data(data_dir: str) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
