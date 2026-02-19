from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from credit_scoring.data import download_data


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    download_data(cfg.paths.artifacts_dir)
    print("Working dir:", os.getcwd())
    print("Config:\n", cfg)
    print("infer: OK (stub)")


if __name__ == "__main__":
    main()
