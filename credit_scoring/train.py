import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from credit_scoring.data import ensure_file
from credit_scoring.dataset import make_loaders_from_application_train
from credit_scoring.lightning_module import CreditRiskLitModule
from credit_scoring.model import MLP, Perceptron


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.seed), workers=True)

    artifacts_dir = Path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_csv = f"{cfg.paths.data_dir}/{cfg.data.train_file}"
    ensure_file(train_csv)

    loaders_out = make_loaders_from_application_train(
        data_dir=cfg.paths.data_dir,
        filename_train=cfg.data.train_file,
        target_col=cfg.data.target_col,
        id_cols=cfg.data.id_cols,
        numeric_only=cfg.preprocess.numeric_only,
        split_train_size=cfg.split.train_size,
        split_val_size=cfg.split.val_size,
        split_test_size=cfg.split.test_size,
        seed=cfg.split.random_state,
        batch_size=cfg.trainer.batch_size,
    )

    train_loader = loaders_out[0]
    val_loader = loaders_out[1]
    test_loader = loaders_out[2]
    num_features = loaders_out[3]
    preprocess_artifacts = loaders_out[4]

    model_name = str(cfg.model.name)
    if model_name == "baseline_perceptron":
        torch_model = Perceptron(num_features=num_features)
    elif model_name == "mlp":
        torch_model = MLP(
            num_features=num_features,
            hidden_sizes=list(cfg.model.hidden_sizes),
            dropout=float(cfg.model.dropout),
        )
    else:
        raise ValueError(f"Unknown model.name: {model_name}")

    lr = float(cfg.trainer.lr)
    lit_model = CreditRiskLitModule(model=torch_model, lr=lr)

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    trainer.validate(lit_model, dataloaders=val_loader, verbose=False)
    trainer.test(lit_model, dataloaders=test_loader, verbose=False)

    weights_path = artifacts_dir / f"{model_name}.pt"
    torch.save(lit_model.state_dict(), weights_path)

    preprocess_artifacts = OmegaConf.to_container(
        OmegaConf.create(preprocess_artifacts),
        resolve=True,
    )

    preprocess_path = artifacts_dir / "preprocess.json"
    preprocess_path.write_text(
        json.dumps(preprocess_artifacts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["num_features"] = int(num_features)

    model_cfg_path = artifacts_dir / "model_config.json"
    model_cfg_path.write_text(
        json.dumps(model_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved model:", weights_path)
    print("Saved preprocess:", preprocess_path)
    print("Saved model config:", model_cfg_path)
    print("train: OK")


if __name__ == "__main__":
    main()
