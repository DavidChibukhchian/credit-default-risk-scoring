from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from credit_scoring.data import download_data


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print("Working dir:", os.getcwd())
    print("Config:\n", cfg)

    artifacts_dir = cfg.paths.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    download_data(cfg.paths.data_dir)
    print(f"Created artifacts dir: {artifacts_dir}")
    print("train: OK (stub)")


if __name__ == "__main__":
    main()
