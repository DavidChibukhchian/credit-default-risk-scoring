import json
import subprocess
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback

from credit_scoring.data import ensure_file
from credit_scoring.dataset import make_loaders_from_application_train
from credit_scoring.lightning_module import CreditRiskLitModule
from credit_scoring.model import MLP, Perceptron


class MetricHistory(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for k, v in metrics.items():
            try:
                val = float(v)
            except Exception:
                continue
            self.history.setdefault(str(k), []).append(val)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for k, v in metrics.items():
            try:
                val = float(v)
            except Exception:
                continue
            self.history.setdefault(str(k), []).append(val)


def get_git_commit_id():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def save_plots(history, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "train/logloss" in history:
        plt.figure()
        plt.plot(history["train/logloss"])
        plt.title("train/logloss")
        plt.xlabel("epoch")
        plt.ylabel("value")
        p = plots_dir / "train_logloss.png"
        plt.savefig(p, bbox_inches="tight")
        plt.close()

    if "val/logloss" in history:
        plt.figure()
        plt.plot(history["val/logloss"])
        plt.title("val/logloss")
        plt.xlabel("epoch")
        plt.ylabel("value")
        p = plots_dir / "val_logloss.png"
        plt.savefig(p, bbox_inches="tight")
        plt.close()

    if "val/roc_auc" in history:
        plt.figure()
        plt.plot(history["val/roc_auc"])
        plt.title("val/roc_auc")
        plt.xlabel("epoch")
        plt.ylabel("value")
        p = plots_dir / "val_roc_auc.png"
        plt.savefig(p, bbox_inches="tight")
        plt.close()

    if "train/roc_auc" in history:
        plt.figure()
        plt.plot(history["train/roc_auc"])
        plt.title("train/roc_auc")
        plt.xlabel("epoch")
        plt.ylabel("value")
        p = plots_dir / "train_roc_auc.png"
        plt.savefig(p, bbox_inches="tight")
        plt.close()


def _flatten_params(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        items = d.items()
    else:
        return {prefix: str(d)}

    for k, v in items:
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_params(v, prefix=key))
        elif isinstance(v, (list, tuple)):
            out[key] = str(v)
        else:
            out[key] = v
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.seed), workers=True)

    artifacts_dir = Path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_csv = f"{cfg.paths.data_dir}/{cfg.data.train_file}"
    ensure_file(train_csv, url=str(cfg.hf_urls.application_train))

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

    git_commit = get_git_commit_id()

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    params = _flatten_params(cfg_dict)

    history_cb = MetricHistory()

    with mlflow.start_run(run_name=str(cfg.mlflow.run_name)):
        mlflow.log_param("git_commit_id", git_commit)
        mlflow.log_params(params)

        trainer = pl.Trainer(
            max_epochs=int(cfg.trainer.max_epochs),
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            callbacks=[history_cb],
        )

        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        trainer.validate(lit_model, dataloaders=val_loader, verbose=False)
        trainer.test(lit_model, dataloaders=test_loader, verbose=False)

        for name, values in history_cb.history.items():
            for step, val in enumerate(values):
                mlflow.log_metric(name, val, step=step)

        save_plots(history_cb.history, plots_dir)

        for p in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(p), artifact_path="plots")

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

        mlflow.log_artifact(str(weights_path), artifact_path="artifacts")
        mlflow.log_artifact(str(preprocess_path), artifact_path="artifacts")
        mlflow.log_artifact(str(model_cfg_path), artifact_path="artifacts")

    print("Saved model:", weights_path)
    print("Saved preprocess:", preprocess_path)
    print("Saved model config:", model_cfg_path)
    print("train: OK")


if __name__ == "__main__":
    main()
