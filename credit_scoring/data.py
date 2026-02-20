import subprocess
from pathlib import Path
from urllib.request import urlretrieve


def _repo_root():
    return Path(__file__).resolve().parents[1]


def download_data(url, dst_path):
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url} -> {dst_path}")
    urlretrieve(url, dst_path)
    print("download_data: OK")
    return str(dst_path)


def _try_dvc_pull(target_abs):
    repo_root = _repo_root()

    try:
        target_rel = target_abs.relative_to(repo_root)
    except ValueError:
        target_rel = target_abs

    try:
        subprocess.run(
            ["dvc", "pull", str(target_rel)],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def ensure_file(path, url=None):

    path = Path(path)
    repo_root = _repo_root()

    target_abs = path if path.is_absolute() else (repo_root / path)

    if target_abs.exists():
        return str(target_abs)

    pulled = _try_dvc_pull(target_abs)
    if pulled and target_abs.exists():
        return str(target_abs)

    if url is None:
        raise FileNotFoundError(f"Missing: {target_abs}")

    download_data(url, target_abs)
    return str(target_abs)
