import pandas as pd
import numpy as np
import pytest

from src.models.preprocessing import TabularPreprocessor


def make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "Weekly_Sales": rng.uniform(100, 5000, n).astype(float),
        "Store": rng.integers(1, 4, n),
        "Dept": rng.integers(1, 6, n),
        "Date": pd.date_range("2021-01-01", periods=n, freq="W"),
        "lag_1": rng.uniform(100, 5000, n),
        "roll_mean_4": rng.uniform(100, 5000, n),
        "Type": rng.choice(["A", "B", "C"], n),  # categórica
    })


# ── fit_transform ──────────────────────────────────────────────────────────

class TestFitTransform:
    def test_removes_target_column(self):
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        X, _ = pre.fit_transform(make_df())
        assert "Weekly_Sales" not in X.columns

    def test_removes_date_column(self):
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        X, _ = pre.fit_transform(make_df())
        assert "Date" not in X.columns

    def test_target_array_matches_column(self):
        df = make_df()
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        _, y = pre.fit_transform(df)
        np.testing.assert_array_almost_equal(y, df["Weekly_Sales"].values.astype("float32"))

    def test_output_is_float32(self):
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        X, y = pre.fit_transform(make_df())
        assert X.dtypes.unique().tolist() == [np.float32]
        assert y.dtype == np.float32

    def test_encodes_categorical_columns(self):
        df = make_df()
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        X, _ = pre.fit_transform(df)
        # "Type" é categórica — deve ter sido one-hot encoded
        assert "Type" not in X.columns
        assert any(col.startswith("Type_") for col in X.columns)

    def test_stores_feature_columns(self):
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        X, _ = pre.fit_transform(make_df())
        assert len(pre.feature_columns) == X.shape[1]
        assert list(pre.feature_columns) == list(X.columns)


# ── transform (alinhamento) ────────────────────────────────────────────────

class TestTransform:
    def test_columns_match_training(self):
        df_train = make_df(30)
        df_valid = make_df(10)
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        X_train, _ = pre.fit_transform(df_train)
        X_valid = pre.transform(df_valid)
        assert list(X_train.columns) == list(X_valid.columns)

    def test_missing_category_filled_with_zero(self):
        # treina só com Type=A, valida com Type=B — colunas novas devem ser 0
        df_train = make_df(20)
        df_train["Type"] = "A"

        df_valid = make_df(5)
        df_valid["Type"] = "B"

        pre = TabularPreprocessor(target_col="Weekly_Sales")
        X_train, _ = pre.fit_transform(df_train)
        X_valid = pre.transform(df_valid)

        if "Type_A" in X_valid.columns:
            assert (X_valid["Type_A"] == 0.0).all()

    def test_transform_output_is_float32(self):
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        pre.fit_transform(make_df(20))
        X_valid = pre.transform(make_df(5))
        assert X_valid.dtypes.unique().tolist() == [np.float32]

    def test_to_dict_contains_required_keys(self):
        pre = TabularPreprocessor(target_col="Weekly_Sales")
        pre.fit_transform(make_df())
        d = pre.to_dict()
        assert "target_col" in d
        assert "feature_columns" in d
        assert "categorical_cols" in d
