import pandas as pd
import numpy as np
import pytest

from src.split import temporal_split, rolling_splits


def make_weekly_df(n_weeks: int = 104, n_stores: int = 2, n_depts: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
    rows = []
    rng = np.random.default_rng(42)
    for store in range(1, n_stores + 1):
        for dept in range(1, n_depts + 1):
            sales = rng.uniform(100, 5000, n_weeks)
            for i, dt in enumerate(dates):
                rows.append({"Store": store, "Dept": dept, "Date": dt, "Weekly_Sales": sales[i]})
    return pd.DataFrame(rows)


# ── temporal_split ─────────────────────────────────────────────────────────

class TestTemporalSplit:
    def test_all_rows_accounted_for(self):
        df = make_weekly_df()
        train, valid, _ = temporal_split(df, "Date", horizon_days=28)
        assert len(train) + len(valid) == len(df)

    def test_no_data_leakage(self):
        df = make_weekly_df()
        train, valid, cutoff = temporal_split(df, "Date", horizon_days=28)
        assert train["Date"].max() <= cutoff
        assert valid["Date"].min() > cutoff

    def test_valid_not_empty(self):
        df = make_weekly_df(n_weeks=52)
        _, valid, _ = temporal_split(df, "Date", horizon_days=28)
        assert len(valid) > 0

    def test_train_not_empty(self):
        df = make_weekly_df(n_weeks=52)
        train, _, _ = temporal_split(df, "Date", horizon_days=28)
        assert len(train) > 0

    def test_cutoff_is_correct(self):
        df = make_weekly_df(n_weeks=52)
        _, _, cutoff = temporal_split(df, "Date", horizon_days=28)
        expected = pd.to_datetime(df["Date"]).max() - pd.Timedelta(days=28)
        assert cutoff == expected

    def test_longer_horizon_produces_smaller_train(self):
        df = make_weekly_df(n_weeks=104)
        train_short, _, _ = temporal_split(df, "Date", horizon_days=14)
        train_long, _, _ = temporal_split(df, "Date", horizon_days=90)
        assert len(train_short) >= len(train_long)


# ── rolling_splits ─────────────────────────────────────────────────────────

class TestRollingSplits:
    def test_returns_correct_number_of_folds(self):
        df = make_weekly_df(n_weeks=104)
        splits = rolling_splits(df, "Date", horizon_days=28, n_folds=3)
        assert len(splits) == 3

    def test_each_fold_has_train_and_valid(self):
        df = make_weekly_df(n_weeks=104)
        for train, valid, fold in rolling_splits(df, "Date", horizon_days=28, n_folds=3):
            assert len(train) > 0
            assert len(valid) > 0

    def test_no_leakage_in_any_fold(self):
        df = make_weekly_df(n_weeks=104)
        for train, valid, fold in rolling_splits(df, "Date", horizon_days=28, n_folds=3):
            assert train["Date"].max() <= fold.train_end

    def test_fold_numbers_are_unique_and_complete(self):
        n = 4
        df = make_weekly_df(n_weeks=104)
        fold_numbers = [f.fold for _, _, f in rolling_splits(df, "Date", horizon_days=28, n_folds=n)]
        # Todos os folds de 1 a n devem estar presentes (ordem invertida por design)
        assert sorted(fold_numbers) == list(range(1, n + 1))
