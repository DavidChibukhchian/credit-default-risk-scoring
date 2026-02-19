import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def _standardize_train_val_test(x_train, x_val, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    x_train_s = (x_train - mean) / std
    x_val_s = (x_val - mean) / std
    x_test_s = (x_test - mean) / std

    scaler = {"mean": mean.tolist(), "std": std.tolist()}
    return x_train_s, x_val_s, x_test_s, scaler


def make_loaders_from_application_train(
    data_dir,
    filename_train,
    target_col,
    id_cols,
    numeric_only,
    split_train_size,
    split_val_size,
    split_test_size,
    seed,
    batch_size,
):
    csv_path = f"{data_dir}/{filename_train}"
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    y = df[target_col].astype(np.float32).to_numpy()

    feature_df = df.drop(columns=[target_col], errors="ignore")
    for col in id_cols:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])

    if numeric_only:
        feature_df = feature_df.select_dtypes(include=["number"])

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    split_sum = split_train_size + split_val_size + split_test_size
    if not np.isclose(split_sum, 1.0):
        raise ValueError("Split sizes must sum to 1.0")

    x_train_df, x_tmp_df, y_train, y_tmp = train_test_split(
        feature_df,
        y,
        train_size=split_train_size,
        random_state=seed,
        stratify=y,
    )

    tmp_total = split_val_size + split_test_size
    val_frac_in_tmp = split_val_size / tmp_total

    x_val_df, x_test_df, y_val, y_test = train_test_split(
        x_tmp_df,
        y_tmp,
        train_size=val_frac_in_tmp,
        random_state=seed,
        stratify=y_tmp,
    )

    medians = x_train_df.median(numeric_only=True)
    x_train_df = x_train_df.fillna(medians)
    x_val_df = x_val_df.fillna(medians)
    x_test_df = x_test_df.fillna(medians)

    feature_names = x_train_df.columns.tolist()

    x_train = x_train_df.to_numpy(dtype=np.float32)
    x_val = x_val_df.to_numpy(dtype=np.float32)
    x_test = x_test_df.to_numpy(dtype=np.float32)

    standardized = _standardize_train_val_test(x_train, x_val, x_test)
    x_train, x_val, x_test, scaler = standardized

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    train_ds = TensorDataset(x_train_t, y_train_t)

    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    val_ds = TensorDataset(x_val_t, y_val_t)

    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)
    test_ds = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_features = x_train.shape[1]

    preprocess_artifacts = {
        "numeric_only": numeric_only,
        "id_cols": id_cols,
        "feature_names": feature_names,
        "medians": medians.to_dict(),
        "scaler": scaler,
        "target_col": target_col,
        "filename_train": filename_train,
    }

    return (
        train_loader,
        val_loader,
        test_loader,
        num_features,
        preprocess_artifacts,
    )
