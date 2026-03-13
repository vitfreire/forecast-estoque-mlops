import os
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_raw(data_raw_dir: str) -> dict:
    train_path = os.path.join(data_raw_dir, "train.csv")
    feat_path = os.path.join(data_raw_dir, "features.csv")
    stores_path = os.path.join(data_raw_dir, "stores.csv")

    train = pd.read_csv(train_path)
    features = pd.read_csv(feat_path)
    stores = pd.read_csv(stores_path)

    return {"train": train, "features": features, "stores": stores}


def write_parquet(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
