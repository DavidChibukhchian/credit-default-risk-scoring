import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from credit_scoring.data import ensure_file
from credit_scoring.model import MLP, Perceptron


def _prepare_single_row(row_df, preprocess):
    for col in preprocess["id_cols"]:
        if col in row_df.columns:
            row_df = row_df.drop(columns=[col])

    if preprocess["numeric_only"]:
        row_df = row_df.select_dtypes(include=["number"])

    feature_names = preprocess["feature_names"]
    row_df = row_df.reindex(columns=feature_names)

    row_df = row_df.replace([np.inf, -np.inf], np.nan)

    row_df = row_df.fillna(preprocess["medians"])

    x = row_df.to_numpy(dtype=np.float32)

    mean = np.array(preprocess["scaler"]["mean"], dtype=np.float32)
    std = np.array(preprocess["scaler"]["std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)

    x = (x - mean) / std
    return torch.from_numpy(x)


def _load_model_config(artifacts_dir: Path):
    path = artifacts_dir / "model_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    preprocess_path = artifacts_dir / "preprocess.json"
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Missing preprocess.json: {preprocess_path}")

    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    num_features = len(preprocess["feature_names"])

    model_cfg = _load_model_config(artifacts_dir)

    requested_name = str(cfg.model.name)
    if model_cfg is not None:
        model_name = str(model_cfg.get("name", requested_name))
        if model_name != requested_name:
            print(
                f"[warn] config requested model='{requested_name}', "
                f"but artifacts contain model='{model_name}'. Using artifacts."
            )
        default_hs = cfg.model.hidden_sizes
        hidden_sizes = list(model_cfg.get("hidden_sizes", default_hs))
        dropout = float(model_cfg.get("dropout", float(cfg.model.dropout)))
    else:
        model_name = requested_name
        hidden_sizes = list(cfg.model.hidden_sizes)
        dropout = float(cfg.model.dropout)

    if model_name == "baseline_perceptron":
        model = Perceptron(num_features=num_features)
        weights_path = artifacts_dir / "baseline_perceptron.pt"
    elif model_name == "mlp":
        model = MLP(
            num_features=num_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        weights_path = artifacts_dir / "mlp.pt"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")

    if any(k.startswith("model.") for k in state_dict.keys()):
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace("model.", "", 1)] = v
        state_dict = new_state

    model.load_state_dict(state_dict)
    model.eval()

    test_csv = ensure_file(
        f"{cfg.paths.data_dir}/{cfg.data.test_file}",
        url=str(cfg.hf_urls.application_test),
    )
    df_test = pd.read_csv(test_csv)

    id_col = str(cfg.data.id_cols[0])
    if id_col not in df_test.columns:
        raise ValueError(f"ID column '{id_col}' not found in test CSV.")

    request_id = cfg.infer_params.request_id
    if request_id is None:
        row_df = df_test.iloc[[0]]
        request_id_value = int(row_df[id_col].iloc[0])
    else:
        request_id_value = int(request_id)
        row_df = df_test[df_test[id_col] == request_id_value]
        if row_df.shape[0] == 0:
            msg = f"request_id={request_id_value} not found in test CSV."
            raise ValueError(msg)
        row_df = row_df.iloc[[0]]

    features = _prepare_single_row(row_df, preprocess)

    with torch.no_grad():
        logits = model(features).squeeze(1)
        prob = torch.sigmoid(logits).item()

    threshold = float(cfg.infer_params.threshold)
    pred = int(prob >= threshold)

    print(f"id={request_id_value}")
    print(f"model={model_name}")
    print(f"prob_default={prob:.6f}")
    print(f"pred_class={pred} (threshold={threshold})")


if __name__ == "__main__":
    main()
