import subprocess
from pathlib import Path
from urllib.request import urlretrieve


def download_data(url, dst_path):
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url} -> {dst_path}")
    urlretrieve(url, dst_path)
    print("download_data: OK")
    return str(dst_path)


def _try_dvc_pull(path):
    try:
        subprocess.run(
            ["dvc", "pull", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def ensure_file(path, url=None):
    path = Path(path)
    if path.exists():
        return

    _try_dvc_pull(path)
    if path.exists():
        return

    if url is None:
        raise FileNotFoundError(f"Missing: {path}")

    download_data(url, path)
