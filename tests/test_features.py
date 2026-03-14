import pandas as pd
import numpy as np
import pytest

from src.features.features import add_time_features, add_lag_features


def make_sales_df(n_weeks: int = 60, n_stores: int = 2, n_depts: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
    rng = np.random.default_rng(0)
    rows = []
    for store in range(1, n_stores + 1):
        for dept in range(1, n_depts + 1):
            for dt in dates:
                rows.append({
                    "Store": store,
                    "Dept": dept,
                    "Date": dt,
                    "Weekly_Sales": rng.uniform(100, 5000),
                })
    return pd.DataFrame(rows)


# ── add_time_features ──────────────────────────────────────────────────────

class TestAddTimeFeatures:
    EXPECTED_COLS = ["year", "month", "weekofyear", "dayofweek",
                     "is_month_start", "is_month_end", "week_sin", "week_cos"]

    def test_creates_all_expected_columns(self):
        df = make_sales_df()
        result = add_time_features(df)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Coluna ausente: {col}"

    def test_no_nulls_in_calendar_columns(self):
        df = make_sales_df()
        result = add_time_features(df)
        for col in ["year", "month", "weekofyear", "dayofweek"]:
            assert result[col].isna().sum() == 0, f"NaN encontrado em: {col}"

    def test_original_columns_preserved(self):
        df = make_sales_df()
        result = add_time_features(df)
        assert "Weekly_Sales" in result.columns
        assert "Date" in result.columns

    def test_week_sin_cos_range(self):
        df = make_sales_df()
        result = add_time_features(df)
        assert result["week_sin"].between(-1.0, 1.0).all()
        assert result["week_cos"].between(-1.0, 1.0).all()

    def test_row_count_unchanged(self):
        df = make_sales_df()
        result = add_time_features(df)
        assert len(result) == len(df)

    def test_month_values_valid(self):
        df = make_sales_df()
        result = add_time_features(df)
        assert result["month"].between(1, 12).all()

    def test_weekofyear_values_valid(self):
        df = make_sales_df()
        result = add_time_features(df)
        assert result["weekofyear"].between(1, 53).all()


# ── add_lag_features ───────────────────────────────────────────────────────

class TestAddLagFeatures:
    def test_creates_lag_columns(self):
        df = make_sales_df().sort_values(["Store", "Dept", "Date"])
        result = add_lag_features(df, ["Store", "Dept"], "Weekly_Sales")
        for lag in [1, 2, 3, 4]:
            assert f"lag_{lag}" in result.columns

    def test_creates_rolling_columns(self):
        df = make_sales_df().sort_values(["Store", "Dept", "Date"])
        result = add_lag_features(df, ["Store", "Dept"], "Weekly_Sales")
        for w in [4, 8, 12]:
            assert f"roll_mean_{w}" in result.columns
            assert f"roll_std_{w}" in result.columns

    def test_creates_diff_column(self):
        df = make_sales_df().sort_values(["Store", "Dept", "Date"])
        result = add_lag_features(df, ["Store", "Dept"], "Weekly_Sales")
        assert "diff_1" in result.columns

    def test_first_row_of_group_has_null_lag1(self):
        df = make_sales_df().sort_values(["Store", "Dept", "Date"])
        result = add_lag_features(df, ["Store", "Dept"], "Weekly_Sales")
        # nth(0) retorna a primeira linha real de cada grupo (incluindo NaN)
        first_rows = result.groupby(["Store", "Dept"]).nth(0).reset_index()
        assert first_rows["lag_1"].isna().all()

    def test_no_future_values_in_lag(self):
        # lag_1 na linha i deve ser igual a Weekly_Sales na linha i-1 (por grupo)
        df = make_sales_df(n_weeks=20, n_stores=1, n_depts=1).sort_values("Date").reset_index(drop=True)
        result = add_lag_features(df, ["Store", "Dept"], "Weekly_Sales").reset_index(drop=True)
        # A partir da segunda linha, lag_1 deve ser a venda anterior
        for i in range(1, len(result)):
            expected = result.loc[i - 1, "Weekly_Sales"]
            actual = result.loc[i, "lag_1"]
            assert actual == pytest.approx(expected), f"lag_1 errado na linha {i}"

    def test_no_infinities(self):
        df = make_sales_df().sort_values(["Store", "Dept", "Date"])
        result = add_lag_features(df, ["Store", "Dept"], "Weekly_Sales")
        numeric_cols = result.select_dtypes(include="number")
        assert not np.isinf(numeric_cols.values).any()
