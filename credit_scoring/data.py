import subprocess
from pathlib import Path


def download_data():
    subprocess.run(["dvc", "pull"], check=True)


def ensure_file(path):
    p = Path(path)
    if not p.exists():
        download_data()
    if not p.exists():
        raise FileNotFoundError(f"File not found after dvc pull: {p}")
