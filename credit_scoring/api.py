import json
from pathlib import Path

import numpy as np
import torch

from credit_scoring.model import MLP, Perceptron


def _load_preprocess(artifacts_dir):
    path = Path(artifacts_dir) / "preprocess.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_model(model_name, num_features, hidden_sizes, dropout):
    if model_name == "baseline_perceptron":
        return Perceptron(num_features=num_features)
    if model_name == "mlp":
        return MLP(
            num_features=num_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def _preprocess_single(features, prep):
    feature_names = prep["feature_names"]
    mean = np.array(prep["scaler"]["mean"], dtype=np.float32)
    std = np.array(prep["scaler"]["std"], dtype=np.float32)

    x = np.zeros((1, len(feature_names)), dtype=np.float32)
    for i, name in enumerate(feature_names):
        val = features.get(name, prep["medians"].get(name, 0.0))
        x[0, i] = float(val)

    x = (x - mean) / std
    return torch.from_numpy(x)


def load_predictor(model_name, artifacts_dir="artifacts"):
    prep = _load_preprocess(artifacts_dir)
    num_features = len(prep["feature_names"])

    hidden_sizes = [128, 64]
    dropout = 0.1

    model = _build_model(model_name, num_features, hidden_sizes, dropout)
    state = torch.load(
        Path(artifacts_dir) / f"{model_name}.pt",
        map_location="cpu",
    )

    if any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    return model, prep


def predict_proba(features, model_name="mlp", artifacts_dir="artifacts"):
    model, prep = load_predictor(model_name, artifacts_dir)
    x = _preprocess_single(features, prep)

    with torch.no_grad():
        logits = model(x).reshape(-1)
        prob = torch.sigmoid(logits)[0].item()
    return float(prob)


def predict(
    features,
    threshold=0.5,
    model_name="mlp",
    artifacts_dir="artifacts",
):
    prob = predict_proba(
        features,
        model_name=model_name,
        artifacts_dir=artifacts_dir,
    )
    return int(prob >= float(threshold))
