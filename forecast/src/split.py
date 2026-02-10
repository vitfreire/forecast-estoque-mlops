import pandas as pd


def temporal_split(df: pd.DataFrame, date_col: str, horizon_days: int):
    df = df.sort_values(date_col)
    last_date = pd.to_datetime(df[date_col]).max()

    cutoff = last_date - pd.Timedelta(days=horizon_days)
    train_df = df[pd.to_datetime(df[date_col]) <= cutoff].copy()
    valid_df = df[pd.to_datetime(df[date_col]) > cutoff].copy()

    return train_df, valid_df, cutoff
