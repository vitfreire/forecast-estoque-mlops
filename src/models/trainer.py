from __future__ import annotations

import os
import json
import time
import hashlib
import matplotlib
matplotlib.use("Agg")  # para evitar problemas em ambientes sem display (ex: servidores Linux)
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from .registry import build_model_registry, parse_run_models
from .preprocessing import TabularPreprocessor
from ..metrics import mae, rmse, smape
from ..mlflow_utils import setup_mlflow


@dataclass(frozen=True)
class TrainResult:
    leaderboard: pd.DataFrame
    best_key: str
    best_run_id: str
    best_model_uri: str


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _maybe_end_active_runs() -> None:
    """
    Por padrão NÃO encerra runs ativos (evita quebrar runs pais em rolling CV).
    Se você quiser forçar limpeza (após falhas), use:
      export MLFLOW_FORCE_END_ACTIVE_RUNS=1
    """
    force = os.getenv("MLFLOW_FORCE_END_ACTIVE_RUNS", "0").strip() == "1"
    if not force:
        return
    while mlflow.active_run() is not None:
        mlflow.end_run()


def _start_run(run_name: str) -> mlflow.ActiveRun:
    """
    Se existir run ativo, cria nested run.
    Se não existir, cria run normal.
    """
    if mlflow.active_run() is None:
        return mlflow.start_run(run_name=run_name)
    return mlflow.start_run(run_name=run_name, nested=True)


def _safe_log_params(params: Dict[str, Any]) -> None:
    """
    MLflow params precisam ser valores simples (str/int/float/bool).
    Essa função filtra e serializa params para evitar erros de logging.
    """
    clean: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            # fallback: serializa
            try:
                clean[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                clean[k] = str(v)
    if clean:
        mlflow.log_params(clean)


def _dataset_fingerprint(df: pd.DataFrame, cols: Optional[List[str]] = None, max_rows: int = 2000) -> str:
    """
    Hash leve e estável (para rastreabilidade)
    - pega subset de colunas
    - amostra determinística (head/tail)
    - usa pandas hashing
    """
    if df is None or len(df) == 0:
        return "empty"

    if cols is None:
        cols = [c for c in df.columns if c != "Weekly_Sales"]

    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy()

    if "Date" in view.columns:
        view["Date"] = pd.to_datetime(view["Date"], errors="coerce")

    # amostra determinística (evita random)
    if len(view) > max_rows:
        half = max_rows // 2
        view = pd.concat([view.head(half), view.tail(max_rows - half)], axis=0)

    h = pd.util.hash_pandas_object(view, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def _get_model_params(model: Any, model_impl: Any) -> Dict[str, Any]:
    """
    Tenta extrair hiperparâmetros do modelo (de forma genérica).
    """
    # 1) se o impl tem método dedicado
    if hasattr(model_impl, "get_params") and callable(getattr(model_impl, "get_params")):
        try:
            out = model_impl.get_params(model)
            if isinstance(out, dict):
                return out
        except Exception:
            pass

    # 2) sklearn-like
    if hasattr(model, "get_params") and callable(getattr(model, "get_params")):
        try:
            out = model.get_params()
            if isinstance(out, dict):
                return out
        except Exception:
            pass

    # 3) LightGBM Booster / XGBoost Booster podem ter `.params`
    if hasattr(model, "params") and isinstance(getattr(model, "params"), dict):
        try:
            return dict(model.params)
        except Exception:
            pass

    # 4) fallback vazio
    return {}


def _extract_feature_importance(model: Any, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Retorna DF com feature_importance quando possível (LightGBM / XGBoost / sklearn).
    """
    # LightGBM Booster
    if hasattr(model, "feature_importance") and callable(getattr(model, "feature_importance")):
        try:
            imp = model.feature_importance()
            df = pd.DataFrame({"feature": feature_names, "importance": imp})
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    # XGBoost Booster
    if hasattr(model, "get_score") and callable(getattr(model, "get_score")):
        try:
            score = model.get_score(importance_type="gain")
            # score vem como {"f0": ..., "f1": ...}
            rows = []
            for k, v in score.items():
                if k.startswith("f"):
                    idx = int(k[1:])
                    feat = feature_names[idx] if 0 <= idx < len(feature_names) else k
                else:
                    feat = k
                rows.append((feat, float(v)))
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["feature", "importance"])
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    # sklearn tree-based
    if hasattr(model, "feature_importances_"):
        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
            if imp.shape[0] == len(feature_names):
                df = pd.DataFrame({"feature": feature_names, "importance": imp})
                return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    return None


def _save_feature_importance_plot(df_imp: pd.DataFrame, path_png: str, top_k: int = 30) -> None:
    """
    Gera gráfico simples com matplotlib (sem seaborn).
    """
    os.makedirs(os.path.dirname(path_png), exist_ok=True)

    plot_df = df_imp.head(top_k).iloc[::-1]  # inverte para aparecer maior no topo
    plt.figure()
    plt.barh(plot_df["feature"].astype(str), plot_df["importance"].astype(float))
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


def train_compare_and_log(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    fold_info: Optional[Dict[str, Any]] = None,
) -> TrainResult:

    setup_mlflow()
    _maybe_end_active_runs()

    run_models = parse_run_models(os.getenv("RUN_MODELS", "lightgbm,xgboost,random_forest"))
    registry = build_model_registry()

    eval_mode = os.getenv("EVAL_MODE", "holdout").lower().strip()
    horizon_days = int(os.getenv("HORIZON_DAYS", "90"))

    pre = TabularPreprocessor(target_col=target_col)

    X_train, y_train = pre.fit_transform(train_df)
    X_valid = pre.transform(valid_df)
    y_valid = valid_df[target_col].astype("float32").to_numpy()

    input_example = X_train.head(5)
    signature = infer_signature(X_train, np.asarray(y_train))

    results: List[Dict[str, Any]] = []

    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # fingerprint de dataset e features
    feature_names = list(X_train.columns)
    train_fp = _dataset_fingerprint(train_df, cols=["Store", "Dept", "Date"] if "Date" in train_df.columns else None)
    valid_fp = _dataset_fingerprint(valid_df, cols=["Store", "Dept", "Date"] if "Date" in valid_df.columns else None)

    with _start_run(run_name="train_compare") as run:
        # ---------- PARAMS DO RUN PAI (aparecem na UI) ----------
        _safe_log_params(
            {
                "target_col": target_col,
                "eval_mode": eval_mode,
                "horizon_days": horizon_days,
                "models": ",".join(run_models),
                "train_rows": int(len(X_train)),
                "valid_rows": int(len(X_valid)),
                "n_features": int(X_train.shape[1]),
                "dataset_train_fp": train_fp,
                "dataset_valid_fp": valid_fp,
            }
        )

        # tags úteis p/ filtros
        mlflow.set_tag("pipeline", "prefect")
        mlflow.set_tag("problem", "walmart_forecast")
        mlflow.set_tag("artifact_dir", artifacts_dir)

        if fold_info:
            for k, v in fold_info.items():
                mlflow.set_tag(f"fold_{k}", str(v))

        # ---------- ARTIFACTS DE METADATA ----------
        meta_dir = os.path.join(artifacts_dir, "metadata")
        os.makedirs(meta_dir, exist_ok=True)

        # lista de features
        features_path = os.path.join(meta_dir, "features.json")
        _write_json(features_path, {"features": feature_names})
        mlflow.log_artifact(features_path, artifact_path="metadata")

        # amostras
        sample_dir = os.path.join(artifacts_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        train_sample_path = os.path.join(sample_dir, "train_sample.parquet")
        valid_sample_path = os.path.join(sample_dir, "valid_sample.parquet")
        train_df.head(5000).to_parquet(train_sample_path, index=False)
        valid_df.head(5000).to_parquet(valid_sample_path, index=False)
        mlflow.log_artifact(train_sample_path, artifact_path="data")
        mlflow.log_artifact(valid_sample_path, artifact_path="data")

        # preprocessor metadata
        pre_path = os.path.join(meta_dir, "preprocess.json")
        _write_json(pre_path, pre.to_dict())
        mlflow.log_artifact(pre_path, artifact_path="preprocess")

        best_key: Optional[str] = None
        best_rmse: float = float("inf")
        best_model_uri: Optional[str] = None
        best_child_run_id: Optional[str] = None

        # ---------- TREINA E LOGA CADA MODELO ----------
        for key in run_models:
            model_impl = registry[key]

            with _start_run(run_name=key) as child:
                # params do child run
                _safe_log_params(
                    {
                        "model": key,
                        "eval_mode": eval_mode,
                        "horizon_days": horizon_days,
                        "train_rows": int(len(X_train)),
                        "valid_rows": int(len(X_valid)),
                        "n_features": int(X_train.shape[1]),
                        "dataset_train_fp": train_fp,
                        "dataset_valid_fp": valid_fp,
                    }
                )

                # tags de organização
                mlflow.set_tag("run_level", "model")
                if fold_info:
                    mlflow.set_tag("fold_type", str(fold_info.get("type", "")))
                    mlflow.set_tag("fold", str(fold_info.get("fold", "")))

                t0 = time.time()
                model = model_impl.fit(X_train, y_train, X_valid, y_valid)
                train_seconds = float(time.time() - t0)

                preds = model_impl.predict(model, X_valid)

                m_mae = float(mae(y_valid, preds))
                m_rmse = float(rmse(y_valid, preds))
                m_smape = float(smape(y_valid, preds))

                # métricas do modelo
                mlflow.log_metric("mae", m_mae)
                mlflow.log_metric("rmse", m_rmse)
                mlflow.log_metric("smape", m_smape)
                mlflow.log_metric("train_seconds", train_seconds)

                # hiperparams reais do modelo (quando disponível)
                model_params = _get_model_params(model, model_impl)
                if model_params:
                    # prefixa para não colidir com params globais
                    prefixed = {f"hp__{k}": v for k, v in model_params.items()}
                    _safe_log_params(prefixed)

                # log do modelo
                registered_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", None)
                model_impl.log_to_mlflow(
                    model=model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature,
                    registered_model_name=registered_name,
                )

                model_uri = f"runs:/{child.info.run_id}/model"

                # feature importance (se suportado)
                df_imp = _extract_feature_importance(model, feature_names)
                if df_imp is not None and len(df_imp) > 0:
                    fi_dir = os.path.join(artifacts_dir, "feature_importance", key)
                    os.makedirs(fi_dir, exist_ok=True)

                    fi_csv = os.path.join(fi_dir, "feature_importance.csv")
                    df_imp.to_csv(fi_csv, index=False)

                    fi_png = os.path.join(fi_dir, "feature_importance.png")
                    _save_feature_importance_plot(df_imp, fi_png, top_k=int(os.getenv("FI_TOP_K", "30")))

                    mlflow.log_artifact(fi_csv, artifact_path=f"feature_importance/{key}")
                    mlflow.log_artifact(fi_png, artifact_path=f"feature_importance/{key}")

                results.append(
                    {
                        "model": key,
                        "mae": m_mae,
                        "rmse": m_rmse,
                        "smape": m_smape,
                        "train_seconds": train_seconds,
                        "run_id": child.info.run_id,
                        "model_uri": model_uri,
                    }
                )

                if m_rmse < best_rmse:
                    best_rmse = m_rmse
                    best_key = key
                    best_model_uri = model_uri
                    best_child_run_id = child.info.run_id

        leaderboard = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

        # ---------- MÉTRICAS AGREGADAS NO RUN PAI ----------
        best_row = leaderboard.iloc[0]
        mlflow.log_metric("best_rmse", float(best_row["rmse"]))
        mlflow.log_metric("best_mae", float(best_row["mae"]))
        mlflow.log_metric("best_smape", float(best_row["smape"]))

        mlflow.log_metric("avg_rmse", float(leaderboard["rmse"].mean()))
        mlflow.log_metric("avg_mae", float(leaderboard["mae"].mean()))
        mlflow.log_metric("avg_smape", float(leaderboard["smape"].mean()))

        mlflow.log_metric("std_rmse", float(leaderboard["rmse"].std(ddof=0)))
        mlflow.log_metric("std_mae", float(leaderboard["mae"].std(ddof=0)))
        mlflow.log_metric("std_smape", float(leaderboard["smape"].std(ddof=0)))

        # métricas por modelo no run pai (comparação rápida)
        for _, r in leaderboard.iterrows():
            mlflow.log_metric(f"{r['model']}_rmse", float(r["rmse"]))
            mlflow.log_metric(f"{r['model']}_mae", float(r["mae"]))
            mlflow.log_metric(f"{r['model']}_smape", float(r["smape"]))
            mlflow.log_metric(f"{r['model']}_train_seconds", float(r["train_seconds"]))

        # ---------- ARTIFACTS ----------
        reports_dir = os.path.join(artifacts_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        lb_path = os.path.join(reports_dir, "leaderboard.csv")
        _write_csv(lb_path, leaderboard)
        mlflow.log_artifact(lb_path, artifact_path="reports")

        # tags do melhor modelo
        mlflow.set_tag("best_model", str(best_key))
        mlflow.set_tag("best_model_uri", str(best_model_uri))
        mlflow.set_tag("best_child_run_id", str(best_child_run_id))

        # ---------- REGISTRA MELHOR MODELO NO MLFLOW ----------
        registered_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "walmart_forecast_model"
       )

        try:
                result = mlflow.register_model(
                model_uri=best_model_uri,
                name=registered_name
            )

                mlflow.set_tag("registered_model_name", registered_name)
                mlflow.set_tag("registered_model_version", result.version)

        except Exception as e:
            print("Erro ao registrar modelo:", e)

        return TrainResult(
            leaderboard=leaderboard,
            best_key=str(best_key),
            best_run_id=str(run.info.run_id),
            best_model_uri=str(best_model_uri),
        )