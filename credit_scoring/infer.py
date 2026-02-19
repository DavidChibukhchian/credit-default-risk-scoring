from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    print("Working dir:", os.getcwd())
    print("Config:\n", cfg)
    print("infer: OK (stub)")


if __name__ == "__main__":
    main()
