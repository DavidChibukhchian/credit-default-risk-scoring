import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from credit_scoring.model import MLP, Perceptron


def _load_preprocess(artifacts_dir):
    path = Path(artifacts_dir) / "preprocess.json"
    if not path.exists():
        msg = "Missing preprocess.json in " f"{artifacts_dir}."
        raise FileNotFoundError(msg)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_config(artifacts_dir):
    path = Path(artifacts_dir) / "model_config.json"
    if not path.exists():
        msg = "Missing model_config.json in " f"{artifacts_dir}."
        raise FileNotFoundError(msg)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_model(model_name, num_features, model_cfg):
    if model_name == "baseline_perceptron":
        return Perceptron(num_features=num_features)

    if model_name == "mlp":
        hidden_sizes = list(model_cfg.get("hidden_sizes", [128, 64]))
        dropout = float(model_cfg.get("dropout", 0.1))
        return MLP(
            num_features=num_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    msg = f"Unknown model name: {model_name}"
    raise ValueError(msg)


def _preprocess_single(features, prep):
    feature_names = list(prep["feature_names"])
    medians = prep.get("medians", {})

    scaler = prep.get("scaler", {})
    mean = scaler.get("mean")
    std = scaler.get("std")

    if mean is None or std is None:
        msg = "preprocess.json is missing scaler.mean/scaler.std"
        raise ValueError(msg)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    if len(mean) != len(feature_names) or len(std) != len(feature_names):
        msg = "Scaler params length does not match"
        raise ValueError(msg)

    std = np.where(std == 0, 1.0, std)

    x = np.zeros((1, len(feature_names)), dtype=np.float32)
    for i, name in enumerate(feature_names):
        val = features.get(name, medians.get(name, 0.0))
        if val is None:
            val = medians.get(name, 0.0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = float(medians.get(name, 0.0))
        if np.isnan(val) or np.isinf(val):
            val = float(medians.get(name, 0.0))
        x[0, i] = val

    x = (x - mean) / std
    return torch.from_numpy(x)


@lru_cache(maxsize=8)
def load_predictor(model_name=None, artifacts_dir="artifacts"):
    prep = _load_preprocess(artifacts_dir)
    model_cfg = _load_model_config(artifacts_dir)
    artifact_model = str(model_cfg.get("name"))

    if model_name is None:
        model_name = artifact_model
    elif str(model_name) != artifact_model:
        msg = (
            "[warn] requested model="
            f"'{model_name}', but artifacts contain "
            f"model='{artifact_model}'. Using artifacts."
        )
        print(msg)
        model_name = artifact_model

    n = len(prep["feature_names"])
    num_features = int(model_cfg.get("num_features", n))
    model = _build_model(model_name, num_features, model_cfg)

    weights_path = Path(artifacts_dir) / f"{model_name}.pt"
    if not weights_path.exists():
        msg = f"Missing model weights: {weights_path}"
        raise FileNotFoundError(msg)

    state = torch.load(weights_path, map_location="cpu")

    if isinstance(state, dict) and any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    return model, prep


def predict_proba(features, model_name=None, artifacts_dir="artifacts"):
    model, prep = load_predictor(
        model_name=model_name,
        artifacts_dir=str(artifacts_dir),
    )
    x = _preprocess_single(features, prep)
    with torch.no_grad():
        logits = model(x).reshape(-1)
        prob = torch.sigmoid(logits)[0].item()
    return float(prob)


def predict(
    features,
    threshold=0.5,
    model_name=None,
    artifacts_dir="artifacts",
):
    prob = predict_proba(
        features,
        model_name=model_name,
        artifacts_dir=artifacts_dir,
    )
    return int(prob >= float(threshold))
