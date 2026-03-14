from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Retail Forecast Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS / TEMA VISUAL
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.10), transparent 25%),
                radial-gradient(circle at top right, rgba(8,145,178,0.08), transparent 25%),
                linear-gradient(180deg, #07111f 0%, #0b1730 100%);
        }

        .block-container {
            max-width: 1540px;
            padding-top: 1rem;
            padding-bottom: 1.8rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1730 0%, #102042 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: #f5f7fb !important;
        }

        .hero-card {
            background:
                linear-gradient(135deg, rgba(37,99,235,0.22), rgba(8,145,178,0.18));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 24px;
            padding: 28px 30px;
            margin-bottom: 18px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        }

        .section-card {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.10);
        }

        .kpi-card {
            background: linear-gradient(180deg, rgba(15, 29, 55, 0.98) 0%, rgba(9, 20, 40, 0.98) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            min-height: 136px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.14);
        }

        .kpi-title {
            color: #b7c7df !important;
            font-size: 0.90rem;
            margin-bottom: 10px;
        }

        .kpi-value {
            color: #ffffff !important;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.08;
            margin-bottom: 8px;
        }

        .kpi-desc {
            color: #d6e0ef !important;
            font-size: 0.84rem;
            line-height: 1.35;
        }

        .mini-note {
            color: #b7c7df !important;
            font-size: 0.90rem;
            line-height: 1.45;
        }

        .chip {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            color: #dce7fb !important;
            font-size: 0.78rem;
            margin-right: 8px;
            margin-top: 4px;
        }

        .info-box {
            background: rgba(59,130,246,0.08);
            border: 1px solid rgba(96,165,250,0.22);
            border-left: 4px solid #60a5fa;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .warn-box {
            background: rgba(245,158,11,0.10);
            border: 1px solid rgba(245,158,11,0.25);
            border-left: 4px solid #f59e0b;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .good-box {
            background: rgba(16,185,129,0.10);
            border: 1px solid rgba(16,185,129,0.25);
            border-left: 4px solid #10b981;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 14px 14px 10px 14px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 10px 16px;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(59,130,246,0.14) !important;
            border: 1px solid rgba(96,165,250,0.28) !important;
        }

        .footer-note {
            color: #a9b8d1 !important;
            font-size: 0.82rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def fmt_money(valor: Optional[float]) -> str:
    """Formata valor monetário."""
    if valor is None or pd.isna(valor):
        return "N/A"
    return f"${valor:,.0f}"


def fmt_number(valor: Optional[float], casas: int = 2) -> str:
    """Formata número decimal."""
    if valor is None or pd.isna(valor):
        return "N/A"
    return f"{valor:,.{casas}f}"


def fmt_pct(valor: Optional[float], casas: int = 1) -> str:
    """Formata percentual."""
    if valor is None or pd.isna(valor):
        return "N/A"
    return f"{valor:.{casas}f}%"


def primeiro_arquivo_existente(candidatos: list[str]) -> Optional[str]:
    """Retorna o primeiro arquivo que existir."""
    for caminho in candidatos:
        if os.path.exists(caminho):
            return caminho
    return None


def card_kpi(titulo: str, valor: str, descricao: str) -> str:
    """Cria card visual de KPI."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{titulo}</div>
        <div class="kpi-value">{valor}</div>
        <div class="kpi-desc">{descricao}</div>
    </div>
    """


def caixa_secao(titulo: str, subtitulo: str) -> None:
    """Cria caixa visual de seção."""
    st.markdown(
        f"""
        <div class="section-card">
            <div style="font-size:1.15rem; font-weight:700;">{titulo}</div>
            <div class="mini-note">{subtitulo}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classificar_mape(mape: Optional[float]) -> tuple[str, str]:
    """Classifica o MAPE para leitura executiva."""
    if mape is None or pd.isna(mape):
        return "Sem leitura", "Não foi possível calcular o erro percentual médio."
    if mape <= 10:
        return "Alta confiabilidade", "O modelo está em faixa forte para apoiar decisão comercial."
    if mape <= 20:
        return "Boa confiabilidade", "A previsão ajuda o negócio, mas exige monitoramento."
    if mape <= 30:
        return "Atenção", "O erro já pode gerar impacto relevante em estoque e receita."
    return "Risco elevado", "O erro ainda está alto para decisões mais sensíveis."


def safe_mean(series: pd.Series) -> Optional[float]:
    """Retorna média segura."""
    if series is None or series.dropna().empty:
        return None
    return float(series.dropna().mean())


def safe_sum(series: pd.Series) -> Optional[float]:
    """Retorna soma segura."""
    if series is None or series.dropna().empty:
        return None
    return float(series.dropna().sum())


# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================

@st.cache_data(show_spinner=False)
def carregar_previsoes(caminho: str) -> pd.DataFrame:
    """
    Carrega o parquet de previsões e cria colunas auxiliares
    para análise de negócio e qualidade da previsão.
    """
    df = pd.read_parquet(caminho).copy()

    # Normalização mínima
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "prediction" in df.columns and "y_pred" not in df.columns:
        df["y_pred"] = df["prediction"]

    for col in ["Weekly_Sales", "y_pred"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Métricas de erro
    if {"Weekly_Sales", "y_pred"}.issubset(df.columns):
        df["erro"] = df["Weekly_Sales"] - df["y_pred"]
        df["erro_abs"] = df["erro"].abs()

        denominador = df["Weekly_Sales"].replace(0, np.nan).abs()
        df["erro_pct"] = (df["erro_abs"] / denominador) * 100

        # Quando o modelo previu acima do realizado
        df["superprevisao"] = np.where(
            df["y_pred"] > df["Weekly_Sales"],
            df["y_pred"] - df["Weekly_Sales"],
            0.0,
        )

        # Quando o modelo previu abaixo do realizado
        df["subprevisao"] = np.where(
            df["y_pred"] < df["Weekly_Sales"],
            df["Weekly_Sales"] - df["y_pred"],
            0.0,
        )

    # Nomes amigáveis
    if "Store" in df.columns:
        df["Store_Label"] = "Store " + df["Store"].astype(str)

    if "Dept" in df.columns:
        df["Dept_Label"] = "Department " + df["Dept"].astype(str)

    return df


@st.cache_data(show_spinner=False)
def carregar_feature_importance(caminho: Optional[str]) -> pd.DataFrame:
    """Carrega o CSV de importância das variáveis."""
    if caminho is None or not os.path.exists(caminho):
        return pd.DataFrame()

    df = pd.read_csv(caminho).copy()

    colunas_lower = {c.lower(): c for c in df.columns}
    col_feature = None
    col_importance = None

    for nome in ["feature", "variavel", "variable", "name"]:
        if nome in colunas_lower:
            col_feature = colunas_lower[nome]
            break

    for nome in ["importance", "gain", "score", "valor"]:
        if nome in colunas_lower:
            col_importance = colunas_lower[nome]
            break

    if col_feature and col_importance:
        df = df[[col_feature, col_importance]].copy()
        df.columns = ["feature", "importance"]
        df = df.sort_values("importance", ascending=False)
        return df

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def carregar_json_se_existir(caminho: Optional[str]) -> dict:
    """Carrega arquivo JSON se existir."""
    if caminho is None or not os.path.exists(caminho):
        return {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def carregar_meta_yaml_simples(caminho: Optional[str]) -> dict:
    """Carrega informações simples do meta.yaml sem parser externo."""
    if caminho is None or not os.path.exists(caminho):
        return {}

    info = {}
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            if ":" not in linha:
                continue
            chave, valor = linha.split(":", 1)
            chave = chave.strip()
            valor = valor.strip()
            if chave in {
                "name",
                "version",
                "run_id",
                "current_stage",
                "status",
                "source",
                "storage_location",
            }:
                info[chave] = valor
    return info


def calcular_impacto_negocio(
    df: pd.DataFrame,
    taxa_margem_perdida: float,
    taxa_custo_excesso: float,
    taxa_penalidade_ruptura: float,
) -> pd.DataFrame:
    """
    Traduz o erro do modelo para impacto financeiro estimado.
    """
    base = df.copy()

    if {"subprevisao", "superprevisao"}.issubset(base.columns):
        base["perda_venda_estimada"] = base["subprevisao"] * taxa_margem_perdida
        base["custo_excesso_estocado"] = base["superprevisao"] * taxa_custo_excesso
        base["penalidade_ruptura"] = base["subprevisao"] * taxa_penalidade_ruptura
        base["impacto_total_estimado"] = (
            base["perda_venda_estimada"]
            + base["custo_excesso_estocado"]
            + base["penalidade_ruptura"]
        )

    return base


def resumo_executivo(df: pd.DataFrame) -> dict:
    """Calcula KPIs principais do recorte."""
    resumo = {}

    resumo["linhas"] = len(df)
    resumo["lojas"] = int(df["Store"].nunique()) if "Store" in df.columns else 0
    resumo["departamentos"] = int(df["Dept"].nunique()) if "Dept" in df.columns else 0
    resumo["data_inicio"] = df["Date"].min() if "Date" in df.columns and not df.empty else None
    resumo["data_fim"] = df["Date"].max() if "Date" in df.columns and not df.empty else None

    resumo["receita_real"] = safe_sum(df["Weekly_Sales"]) if "Weekly_Sales" in df.columns else None
    resumo["receita_prevista"] = safe_sum(df["y_pred"]) if "y_pred" in df.columns else None
    resumo["erro_total"] = safe_sum(df["erro_abs"]) if "erro_abs" in df.columns else None
    resumo["mae"] = safe_mean(df["erro_abs"]) if "erro_abs" in df.columns else None
    resumo["mape"] = safe_mean(df["erro_pct"]) if "erro_pct" in df.columns else None
    resumo["impacto_total"] = safe_sum(df["impacto_total_estimado"]) if "impacto_total_estimado" in df.columns else None

    return resumo


def serie_temporal_agregada(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por data para gráficos temporais."""
    required = {"Date", "Weekly_Sales", "y_pred"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    colunas = ["Weekly_Sales", "y_pred"]
    if "impacto_total_estimado" in df.columns:
        colunas.append("impacto_total_estimado")
    if "erro_abs" in df.columns:
        colunas.append("erro_abs")

    ts = (
        df.groupby("Date", as_index=False)[colunas]
        .sum(numeric_only=True)
        .sort_values("Date")
    )

    ts["gap"] = ts["Weekly_Sales"] - ts["y_pred"]
    ts["gap_abs"] = ts["gap"].abs()
    ts["crescimento_semanal"] = ts["Weekly_Sales"].pct_change()

    return ts


def top_impacto_por_dimensao(df: pd.DataFrame, coluna: str, top_n: int = 10) -> pd.DataFrame:
    """Gera ranking por loja ou departamento."""
    required = {coluna, "Weekly_Sales", "y_pred", "erro_abs", "impacto_total_estimado"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    agg = (
        df.groupby(coluna, as_index=False)
        .agg(
            receita_real=("Weekly_Sales", "sum"),
            receita_prevista=("y_pred", "sum"),
            mae=("erro_abs", "mean"),
            impacto=("impacto_total_estimado", "sum"),
            linhas=(coluna, "count"),
        )
        .sort_values("impacto", ascending=False)
        .head(top_n)
    )
    return agg


def detectar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta anomalias simples usando z-score do erro absoluto.
    """
    base = df.copy()

    if "erro_abs" not in base.columns or base["erro_abs"].dropna().empty:
        return pd.DataFrame()

    media = base["erro_abs"].mean()
    desvio = base["erro_abs"].std()

    if pd.isna(desvio) or desvio == 0:
        base["z_erro"] = 0.0
    else:
        base["z_erro"] = (base["erro_abs"] - media) / desvio

    base["anomalia"] = base["z_erro"].abs() >= 2.5
    return base[base["anomalia"]].copy()


def gerar_insight_executivo(df: pd.DataFrame) -> str:
    """Gera um insight executivo automático simples."""
    if df.empty:
        return "Não há dados suficientes para gerar insight."

    partes = []

    if "erro_pct" in df.columns and not df["erro_pct"].dropna().empty:
        mape = df["erro_pct"].dropna().mean()
        partes.append(f"O erro percentual médio do recorte está em {mape:.2f}%.")

    if "impacto_total_estimado" in df.columns and not df["impacto_total_estimado"].dropna().empty:
        impacto = df["impacto_total_estimado"].sum()
        partes.append(f"O impacto financeiro estimado acumulado é de {fmt_money(impacto)}.")

    if {"Store_Label", "impacto_total_estimado"}.issubset(df.columns):
        risco_loja = (
            df.groupby("Store_Label", as_index=False)["impacto_total_estimado"]
            .sum()
            .sort_values("impacto_total_estimado", ascending=False)
        )
        if not risco_loja.empty:
            loja = risco_loja.iloc[0]["Store_Label"]
            partes.append(f"A maior concentração de risco está em {loja}.")

    partes.append(
        "A recomendação é priorizar ajustes nos segmentos com maior impacto antes de ampliar mudanças para toda a operação."
    )
    return " ".join(partes)


def categorizar_features(feature_names: list[str]) -> dict[str, list[str]]:
    """Organiza features em grupos de negócio/modelagem."""
    grupos = {
        "Histórico de vendas": [],
        "Estatísticas móveis": [],
        "Calendário": [],
        "Promoções e contexto": [],
        "Estrutura da loja": [],
        "Outras": [],
    }

    for f in feature_names:
        if f.startswith("lag_") or f == "diff_1":
            grupos["Histórico de vendas"].append(f)
        elif f.startswith("roll_"):
            grupos["Estatísticas móveis"].append(f)
        elif f in {"year", "month", "weekofyear", "dayofweek", "is_month_start", "is_month_end", "week_sin", "week_cos"}:
            grupos["Calendário"].append(f)
        elif f in {"Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment", "IsHoliday"}:
            grupos["Promoções e contexto"].append(f)
        elif f in {"Store", "Dept", "Size", "Type_A", "Type_B", "Type_C"}:
            grupos["Estrutura da loja"].append(f)
        else:
            grupos["Outras"].append(f)

    return grupos


# =========================================================
# CAMINHOS PADRÃO
# =========================================================

caminho_previsoes_padrao = primeiro_arquivo_existente(
    [
        "reports/batch_predictions.parquet",
        "../reports/batch_predictions.parquet",
        "/mnt/data/batch_predictions.parquet",
    ]
) or "reports/batch_predictions.parquet"

caminho_importancia_padrao = primeiro_arquivo_existente(
    [
        "feature_importance.csv",
        "reports/feature_importance.csv",
        "/mnt/data/feature_importance.csv",
    ]
) or "feature_importance.csv"

caminho_meta_yaml_padrao = primeiro_arquivo_existente(
    [
        "meta.yaml",
        "/mnt/data/meta.yaml",
    ]
)

caminho_best_meta_json_padrao = primeiro_arquivo_existente(
    [
        "best_model_meta.json",
        "/mnt/data/best_model_meta.json",
    ]
)

caminho_preprocess_json_padrao = primeiro_arquivo_existente(
    [
        "preprocess.json",
        "/mnt/data/preprocess.json",
    ]
)

caminho_features_json_padrao = primeiro_arquivo_existente(
    [
        "features.json",
        "/mnt/data/features.json",
    ]
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## Forecast Intelligence")
st.sidebar.caption("Painel executivo de previsão de vendas")

st.sidebar.markdown("---")
st.sidebar.markdown("### MLflow")
mlflow_uri = st.sidebar.text_input(
    "Tracking URI",
    value=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    help="file:./mlruns para local, http://... para servidor remoto.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Arquivos")
caminho_previsoes = st.sidebar.text_input("Arquivo de previsões", value=caminho_previsoes_padrao)

# Detecta modelos disponíveis em artifacts/feature_importance/
_artif_fi_dir = os.path.join("artifacts", "feature_importance")
_modelos_fi = sorted([
    d for d in os.listdir(_artif_fi_dir)
    if os.path.isdir(os.path.join(_artif_fi_dir, d))
]) if os.path.exists(_artif_fi_dir) else []

if _modelos_fi:
    _modelo_fi_sel = st.sidebar.selectbox("Modelo (feature importance)", _modelos_fi)
    caminho_importancia = os.path.join(_artif_fi_dir, _modelo_fi_sel, "feature_importance.csv")
else:
    caminho_importancia = st.sidebar.text_input("Arquivo de feature importance", value=caminho_importancia_padrao)

if caminho_best_meta_json_padrao:
    caminho_best_meta_json = st.sidebar.text_input("Arquivo best_model_meta.json", value=caminho_best_meta_json_padrao)
else:
    caminho_best_meta_json = st.sidebar.text_input("Arquivo best_model_meta.json", value="best_model_meta.json")

if caminho_preprocess_json_padrao:
    caminho_preprocess_json = st.sidebar.text_input("Arquivo preprocess.json", value=caminho_preprocess_json_padrao)
else:
    caminho_preprocess_json = st.sidebar.text_input("Arquivo preprocess.json", value="preprocess.json")

if caminho_features_json_padrao:
    caminho_features_json = st.sidebar.text_input("Arquivo features.json", value=caminho_features_json_padrao)
else:
    caminho_features_json = st.sidebar.text_input("Arquivo features.json", value="features.json")

if caminho_meta_yaml_padrao:
    caminho_meta_yaml = st.sidebar.text_input("Arquivo meta.yaml", value=caminho_meta_yaml_padrao)
else:
    caminho_meta_yaml = st.sidebar.text_input("Arquivo meta.yaml", value="meta.yaml")

st.sidebar.markdown("---")
st.sidebar.markdown("### Premissas de negócio")

taxa_margem_perdida = st.sidebar.slider(
    "Margem perdida por subprevisão",
    min_value=0.01,
    max_value=1.00,
    value=0.25,
    step=0.01,
)

taxa_custo_excesso = st.sidebar.slider(
    "Custo de excesso de estoque",
    min_value=0.01,
    max_value=1.00,
    value=0.10,
    step=0.01,
)

taxa_penalidade_ruptura = st.sidebar.slider(
    "Penalidade operacional por ruptura",
    min_value=0.00,
    max_value=1.00,
    value=0.08,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Essas premissas convertem o erro do modelo em impacto financeiro estimado."
)

# =========================================================
# CARREGAMENTO EFETIVO
# =========================================================

if not os.path.exists(caminho_previsoes):
    st.error(f"Arquivo de previsões não encontrado: {caminho_previsoes}")
    st.stop()

df = carregar_previsoes(caminho_previsoes)
df = calcular_impacto_negocio(
    df,
    taxa_margem_perdida=taxa_margem_perdida,
    taxa_custo_excesso=taxa_custo_excesso,
    taxa_penalidade_ruptura=taxa_penalidade_ruptura,
)

df_importancia = carregar_feature_importance(caminho_importancia)
best_meta = carregar_json_se_existir(caminho_best_meta_json)
preprocess_meta = carregar_json_se_existir(caminho_preprocess_json)
features_meta = carregar_json_se_existir(caminho_features_json)
meta_yaml = carregar_meta_yaml_simples(caminho_meta_yaml)

if df.empty:
    st.warning("O arquivo de previsões foi carregado, mas não há dados disponíveis.")
    st.stop()

# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:18px; flex-wrap:wrap;">
            <div>
                <div style="font-size:2rem; font-weight:800;">Retail Forecast Intelligence</div>
                <div class="mini-note" style="margin-top:8px;">
                    Plataforma analítica para traduzir previsões em decisões comerciais, risco operacional e impacto financeiro.
                </div>
                <div style="margin-top:12px;">
                    <span class="chip">Forecasting</span>
                    <span class="chip">Negócio</span>
                    <span class="chip">Interpretabilidade</span>
                    <span class="chip">MLOps</span>
                </div>
            </div>
            <div style="max-width:500px;">
                <div style="font-size:1rem; font-weight:600;">Objetivo</div>
                <div class="mini-note" style="margin-top:6px;">
                    Mostrar o que foi previsto, o que ocorreu, onde o modelo falha mais e quanto esse desvio pode custar ao negócio.
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FILTROS GLOBAIS
# =========================================================

caixa_secao(
    "Filtros globais",
    "Use os filtros para analisar lojas, departamentos e períodos específicos com nomes amigáveis.",
)

f1, f2, f3 = st.columns(3)
df_filtrado = df.copy()

if "Store_Label" in df_filtrado.columns:
    lojas = sorted(df_filtrado["Store_Label"].dropna().unique().tolist())
    lojas_escolhidas = f1.multiselect("Loja", lojas, default=[])
    if lojas_escolhidas:
        df_filtrado = df_filtrado[df_filtrado["Store_Label"].isin(lojas_escolhidas)]

if "Dept_Label" in df_filtrado.columns:
    depts = sorted(df_filtrado["Dept_Label"].dropna().unique().tolist())
    depts_escolhidos = f2.multiselect("Departamento", depts, default=[])
    if depts_escolhidos:
        df_filtrado = df_filtrado[df_filtrado["Dept_Label"].isin(depts_escolhidos)]

if "Date" in df_filtrado.columns and not df_filtrado["Date"].dropna().empty:
    data_min = df_filtrado["Date"].min().date()
    data_max = df_filtrado["Date"].max().date()

    periodo = f3.date_input(
        "Período",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max,
    )

    if isinstance(periodo, tuple) and len(periodo) == 2:
        inicio, fim = periodo
        df_filtrado = df_filtrado[
            (df_filtrado["Date"].dt.date >= inicio)
            & (df_filtrado["Date"].dt.date <= fim)
        ]

if df_filtrado.empty:
    st.warning("Os filtros aplicados não retornaram dados.")
    st.stop()

# =========================================================
# RESUMO E ESTRUTURAS DERIVADAS
# =========================================================

resumo = resumo_executivo(df_filtrado)
status_label, status_desc = classificar_mape(resumo["mape"])
ts = serie_temporal_agregada(df_filtrado)
anomalias = detectar_anomalias(df_filtrado)

feature_names = []
if isinstance(best_meta.get("features"), list):
    feature_names = best_meta.get("features", [])
elif isinstance(features_meta.get("features"), list):
    feature_names = features_meta.get("features", [])
elif isinstance(preprocess_meta.get("feature_columns"), list):
    feature_names = preprocess_meta.get("feature_columns", [])

grupos_features = categorizar_features(feature_names) if feature_names else {}

tabs = st.tabs(
    [
        "Resumo Executivo",
        "Performance Comercial",
        "Forecast Analysis",
        "Impacto no Negócio",
        "Store & Department Insights",
        "Model Explainability",
        "Model Leaderboard",
        "Model Intelligence",
        "Model Monitoring",
        "Dados",
    ]
)

# =========================================================
# TAB 1 - RESUMO EXECUTIVO
# =========================================================

with tabs[0]:
    caixa_secao(
        "Resumo executivo",
        "Visão rápida para liderança: tamanho da operação, confiabilidade da previsão, impacto financeiro e status do modelo.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        card_kpi(
            "Receita real",
            fmt_money(resumo["receita_real"]),
            "Total realizado no recorte filtrado.",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        card_kpi(
            "Receita prevista",
            fmt_money(resumo["receita_prevista"]),
            "Total projetado pelo modelo.",
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        card_kpi(
            "Erro absoluto total",
            fmt_money(resumo["erro_total"]),
            "Diferença agregada entre previsto e realizado.",
        ),
        unsafe_allow_html=True,
    )
    c4.markdown(
        card_kpi(
            "Impacto financeiro",
            fmt_money(resumo["impacto_total"]),
            "Estimativa baseada em ruptura e excesso de estoque.",
        ),
        unsafe_allow_html=True,
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(
        card_kpi(
            "MAE",
            fmt_number(resumo["mae"]),
            "Erro médio absoluto. Quanto menor, melhor.",
        ),
        unsafe_allow_html=True,
    )
    c6.markdown(
        card_kpi(
            "MAPE",
            fmt_pct(resumo["mape"]),
            "Erro percentual médio em linguagem de negócio.",
        ),
        unsafe_allow_html=True,
    )
    c7.markdown(
        card_kpi(
            "Cobertura",
            f"{resumo['lojas']} lojas / {resumo['departamentos']} departamentos",
            "Escopo do recorte analisado.",
        ),
        unsafe_allow_html=True,
    )
    c8.markdown(
        card_kpi(
            "Status executivo",
            status_label,
            status_desc,
        ),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.7, 1])

    with col1:
        st.markdown("### Receita real vs prevista")
        st.caption("Mostra se o modelo acompanha a trajetória da demanda ao longo do tempo.")

        if not ts.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ts["Date"],
                    y=ts["Weekly_Sales"],
                    mode="lines",
                    name="Receita real",
                    line=dict(width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ts["Date"],
                    y=ts["y_pred"],
                    mode="lines",
                    name="Receita prevista",
                    line=dict(width=3, dash="dash"),
                )
            )
            fig.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Data",
                yaxis_title="Vendas",
                legend_title="Série",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Insight executivo")
        st.markdown(
            f"""
            <div class="info-box">
                <b>Leitura principal</b><br><br>
                {gerar_insight_executivo(df_filtrado)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if resumo["data_inicio"] is not None and resumo["data_fim"] is not None:
            st.caption(
                f"Período analisado: {resumo['data_inicio'].date()} até {resumo['data_fim'].date()}"
            )

        if not anomalias.empty:
            st.markdown(
                f"""
                <div class="warn-box">
                    <b>Atenção</b><br>
                    Foram detectados <b>{len(anomalias)}</b> registros com erro atípico no recorte atual.
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# TAB 2 - PERFORMANCE COMERCIAL
# =========================================================

with tabs[1]:
    caixa_secao(
        "Performance comercial",
        "Leitura do comportamento de vendas, crescimento e concentração de receita.",
    )

    a, b = st.columns(2)

    with a:
        st.markdown("### Receita por loja")
        st.caption("Mostra concentração de vendas por loja.")

        if "Store_Label" in df.columns:
            receita_loja = (
                df.groupby("Store_Label", as_index=False)["Weekly_Sales"]
                .sum()
                .sort_values("Weekly_Sales", ascending=False)
            )

            fig_loja = px.bar(
                receita_loja.head(15),
                x="Store_Label",
                y="Weekly_Sales",
            )
            fig_loja.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Loja",
                yaxis_title="Receita",
            )
            st.plotly_chart(fig_loja, use_container_width=True)

    with b:
        st.markdown("### Receita por departamento")
        st.caption("Identifica categorias com maior peso comercial.")

        if "Dept_Label" in df.columns:
            receita_dept = (
                df.groupby("Dept_Label", as_index=False)["Weekly_Sales"]
                .sum()
                .sort_values("Weekly_Sales", ascending=False)
            )

            fig_dept = px.bar(
                receita_dept.head(15),
                x="Dept_Label",
                y="Weekly_Sales",
            )
            fig_dept.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Departamento",
                yaxis_title="Receita",
            )
            st.plotly_chart(fig_dept, use_container_width=True)

    c, d = st.columns(2)

    with c:
        st.markdown("### Distribuição de vendas")
        st.caption("Mostra dispersão dos valores de venda no recorte atual.")

        fig_hist_vendas = px.histogram(
            df_filtrado,
            x="Weekly_Sales",
            nbins=45,
        )
        fig_hist_vendas.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis_title="Weekly Sales",
            yaxis_title="Frequência",
        )
        st.plotly_chart(fig_hist_vendas, use_container_width=True)

    with d:
        st.markdown("### Crescimento semanal")
        st.caption("Mede aceleração ou desaceleração da receita ao longo do tempo.")

        if not ts.empty and "crescimento_semanal" in ts.columns:
            fig_growth = px.bar(
                ts,
                x="Date",
                y="crescimento_semanal",
            )
            fig_growth.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Data",
                yaxis_title="Crescimento semanal",
            )
            st.plotly_chart(fig_growth, use_container_width=True)

# =========================================================
# TAB 3 - FORECAST ANALYSIS
# =========================================================

with tabs[2]:
    caixa_secao(
        "Forecast analysis",
        "Análise da precisão, viés e estabilidade do erro do modelo.",
    )

    x1, x2 = st.columns(2)

    with x1:
        st.markdown("### Erro ao longo do tempo")
        st.caption("Ajuda a identificar períodos em que o modelo falha mais.")

        if not ts.empty and "erro_abs" in ts.columns:
            fig_erro_tempo = px.line(
                ts,
                x="Date",
                y="erro_abs",
                markers=True,
            )
            fig_erro_tempo.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Data",
                yaxis_title="Erro absoluto",
            )
            st.plotly_chart(fig_erro_tempo, use_container_width=True)

    with x2:
        st.markdown("### Distribuição do erro")
        st.caption("Mostra se o modelo tende a superprever ou subprever.")

        fig_hist_erro = px.histogram(
            df_filtrado,
            x="erro",
            nbins=45,
        )
        fig_hist_erro.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis_title="Erro",
            yaxis_title="Frequência",
        )
        st.plotly_chart(fig_hist_erro, use_container_width=True)

    viés = safe_mean(df_filtrado["erro"]) if "erro" in df_filtrado.columns else None
    if viés is not None and not pd.isna(viés):
        if viés > 0:
            msg_vies = "O modelo tende a subprever a demanda, elevando risco de ruptura."
        elif viés < 0:
            msg_vies = "O modelo tende a superprever a demanda, elevando risco de excesso de estoque."
        else:
            msg_vies = "O modelo está próximo de neutro em média."

        st.markdown(
            f"""
            <div class="info-box">
                <b>Diagnóstico de viés</b><br>
                {msg_vies}
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# TAB 4 - IMPACTO NO NEGÓCIO
# =========================================================

with tabs[3]:
    caixa_secao(
        "Impacto no negócio",
        "Tradução do erro do modelo para linguagem financeira e operacional.",
    )

    perda_venda = safe_sum(df_filtrado["perda_venda_estimada"]) if "perda_venda_estimada" in df_filtrado.columns else None
    custo_excesso = safe_sum(df_filtrado["custo_excesso_estocado"]) if "custo_excesso_estocado" in df_filtrado.columns else None
    penalidade = safe_sum(df_filtrado["penalidade_ruptura"]) if "penalidade_ruptura" in df_filtrado.columns else None
    impacto = safe_sum(df_filtrado["impacto_total_estimado"]) if "impacto_total_estimado" in df_filtrado.columns else None

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Impacto total", fmt_money(impacto))
    i2.metric("Perda por subprevisão", fmt_money(perda_venda))
    i3.metric("Custo por superprevisão", fmt_money(custo_excesso))
    i4.metric("Penalidade por ruptura", fmt_money(penalidade))

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("### Composição do impacto")
        st.caption("Divide o impacto entre venda perdida, excesso e penalidade operacional.")

        impacto_df = pd.DataFrame(
            {
                "categoria": [
                    "Perda por subprevisão",
                    "Custo por superprevisão",
                    "Penalidade por ruptura",
                ],
                "valor": [
                    perda_venda or 0.0,
                    custo_excesso or 0.0,
                    penalidade or 0.0,
                ],
            }
        )

        fig_pie = px.pie(
            impacto_df,
            names="categoria",
            values="valor",
            hole=0.55,
        )
        fig_pie.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with p2:
        st.markdown("### Impacto ao longo do tempo")
        st.caption("Mostra em quais semanas o erro gerou maior pressão financeira.")

        if not ts.empty and "impacto_total_estimado" in ts.columns:
            fig_impacto_tempo = px.bar(
                ts,
                x="Date",
                y="impacto_total_estimado",
            )
            fig_impacto_tempo.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Data",
                yaxis_title="Impacto estimado",
            )
            st.plotly_chart(fig_impacto_tempo, use_container_width=True)

# =========================================================
# TAB 5 - STORE & DEPARTMENT INSIGHTS
# =========================================================

with tabs[4]:
    caixa_secao(
        "Store & Department insights",
        "Identificação de bolsões de risco e oportunidades por loja e departamento.",
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("### Top lojas por impacto")
        st.caption("Prioriza onde agir primeiro na operação.")

        if "Store_Label" in df_filtrado.columns:
            top_lojas = top_impacto_por_dimensao(df_filtrado, "Store_Label", top_n=10)
            if not top_lojas.empty:
                fig_top_lojas = px.bar(
                    top_lojas.sort_values("impacto", ascending=True),
                    x="impacto",
                    y="Store_Label",
                    orientation="h",
                )
                fig_top_lojas.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    xaxis_title="Impacto estimado",
                    yaxis_title="Loja",
                )
                st.plotly_chart(fig_top_lojas, use_container_width=True)

    with s2:
        st.markdown("### Top departamentos por impacto")
        st.caption("Mostra onde o ganho de melhoria tende a ser maior.")

        if "Dept_Label" in df_filtrado.columns:
            top_depts = top_impacto_por_dimensao(df_filtrado, "Dept_Label", top_n=10)
            if not top_depts.empty:
                fig_top_depts = px.bar(
                    top_depts.sort_values("impacto", ascending=True),
                    x="impacto",
                    y="Dept_Label",
                    orientation="h",
                )
                fig_top_depts.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    xaxis_title="Impacto estimado",
                    yaxis_title="Departamento",
                )
                st.plotly_chart(fig_top_depts, use_container_width=True)

    st.markdown("### Heatmap loja × departamento")
    st.caption("Quanto mais intenso o bloco, maior o erro médio naquele cruzamento.")

    if {"Store_Label", "Dept_Label", "erro_abs"}.issubset(df.columns):
        heatmap = df.pivot_table(
            values="erro_abs",
            index="Store_Label",
            columns="Dept_Label",
            aggfunc="mean",
        )

        if not heatmap.empty:
            fig_heat = px.imshow(
                heatmap,
                aspect="auto",
            )
            fig_heat.update_layout(
                height=560,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# TAB 6 - MODEL EXPLAINABILITY
# =========================================================

with tabs[5]:
    caixa_secao(
        "Model explainability",
        "Explica quais variáveis mais influenciam a previsão.",
    )

    if not df_importancia.empty:
        e1, e2 = st.columns([1.1, 0.9])

        with e1:
            st.markdown("### Feature importance")
            st.caption("Principais direcionadores do comportamento do modelo.")

            top_imp = df_importancia.head(20).sort_values("importance", ascending=True)

            fig_imp = px.bar(
                top_imp,
                x="importance",
                y="feature",
                orientation="h",
            )
            fig_imp.update_layout(
                height=580,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Importância",
                yaxis_title="Variável",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with e2:
            st.markdown("### Leitura de negócio")
            st.markdown(
                """
                <div class="info-box">
                    <b>Como explicar para a área comercial</b><br><br>
                    As variáveis no topo do ranking são as que mais influenciam a previsão.
                    Em forecasting de varejo, o histórico recente de vendas e medidas de tendência
                    costumam pesar mais do que variáveis macroeconômicas.<br><br>
                    Isso indica que a demanda tende a ser explicada principalmente pelo comportamento
                    recente da própria série.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Principais mensagens")
            st.markdown(
                """
                - histórico recente normalmente domina a previsão  
                - variações bruscas tendem a aparecer em lags e diferenças  
                - promoções e fatores externos ajudam, mas raramente explicam tudo  
                - interpretabilidade ajuda a justificar por que o modelo sobe ou reduz a expectativa de vendas
                """
            )
    else:
        st.info(
            "Arquivo de feature importance não encontrado ou fora do padrão esperado. "
            "Use um CSV com colunas como 'feature' e 'importance'."
        )

# =========================================================
# TAB 7 - MODEL LEADERBOARD (MLflow)
# =========================================================

_CORES_MODELOS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#a855f7"]

with tabs[6]:
    caixa_secao(
        "Model Leaderboard",
        "Comparativo de performance dos modelos treinados — conectado diretamente ao MLflow.",
    )

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        _client = MlflowClient()
        _exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "forecast_estoque_walmart")
        _exp = _client.get_experiment_by_name(_exp_name)

        if _exp is None:
            st.info(
                f"Experimento '{_exp_name}' não encontrado no MLflow. "
                "Execute o pipeline de treinamento primeiro."
            )
        else:
            _parent_runs = _client.search_runs(
                experiment_ids=[_exp.experiment_id],
                filter_string="tags.`mlflow.runName` = 'train_compare'",
                order_by=["start_time DESC"],
                max_results=20,
            )

            if not _parent_runs:
                st.info("Nenhum run de treinamento encontrado. Execute o pipeline primeiro.")
            else:
                _run_labels = {}
                for _r in _parent_runs:
                    _dt = datetime.datetime.fromtimestamp(_r.info.start_time / 1000).strftime("%Y-%m-%d %H:%M")
                    _run_labels[f"{_dt}  —  run {_r.info.run_id[:8]}"] = _r.info.run_id

                _sel_label = st.selectbox("Selecione o run de treinamento", list(_run_labels.keys()))
                _sel_run_id = _run_labels[_sel_label]

                _child_runs = _client.search_runs(
                    experiment_ids=[_exp.experiment_id],
                    filter_string=f"tags.`mlflow.parentRunId` = '{_sel_run_id}'",
                    order_by=["metrics.rmse ASC"],
                )

                if _child_runs:
                    _lb_rows = []
                    for _run in _child_runs:
                        _lb_rows.append({
                            "Modelo": _run.data.tags.get("mlflow.runName", _run.info.run_id[:8]),
                            "RMSE": _run.data.metrics.get("rmse"),
                            "MAE": _run.data.metrics.get("mae"),
                            "SMAPE (%)": _run.data.metrics.get("smape"),
                            "Treino (s)": round(_run.data.metrics.get("train_seconds", 0), 1),
                        })

                    _lb_df = pd.DataFrame(_lb_rows).sort_values("RMSE").reset_index(drop=True)

                    st.markdown("### Comparativo de modelos")
                    st.dataframe(_lb_df, use_container_width=True)

                    _l1, _l2, _l3 = st.columns(3)

                    with _l1:
                        _fig_r = px.bar(
                            _lb_df.sort_values("RMSE"),
                            x="Modelo", y="RMSE",
                            color="Modelo",
                            color_discrete_sequence=_CORES_MODELOS,
                            title="RMSE por modelo",
                        )
                        _fig_r.update_layout(
                            height=320, showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.02)",
                        )
                        st.plotly_chart(_fig_r, use_container_width=True)

                    with _l2:
                        _fig_m = px.bar(
                            _lb_df.sort_values("MAE"),
                            x="Modelo", y="MAE",
                            color="Modelo",
                            color_discrete_sequence=_CORES_MODELOS,
                            title="MAE por modelo",
                        )
                        _fig_m.update_layout(
                            height=320, showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.02)",
                        )
                        st.plotly_chart(_fig_m, use_container_width=True)

                    with _l3:
                        _fig_s = px.bar(
                            _lb_df.sort_values("SMAPE (%)"),
                            x="Modelo", y="SMAPE (%)",
                            color="Modelo",
                            color_discrete_sequence=_CORES_MODELOS,
                            title="SMAPE por modelo",
                        )
                        _fig_s.update_layout(
                            height=320, showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.02)",
                        )
                        st.plotly_chart(_fig_s, use_container_width=True)

                    _best = _lb_df.iloc[0]
                    st.markdown(
                        f"""
                        <div class="good-box">
                            <b>Melhor modelo: {_best['Modelo']}</b><br>
                            RMSE = {fmt_number(_best.get('RMSE'))} &nbsp;|&nbsp;
                            MAE = {fmt_number(_best.get('MAE'))} &nbsp;|&nbsp;
                            SMAPE = {fmt_pct(_best.get('SMAPE (%)'))}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    _parent_run_data = _client.get_run(_sel_run_id)
                    st.markdown("### Metadados do run")
                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                    _mc1.metric("Modelos treinados", _parent_run_data.data.params.get("models", "N/A"))
                    _mc2.metric("Linhas de treino", _parent_run_data.data.params.get("train_rows", "N/A"))
                    _mc3.metric("Features", _parent_run_data.data.params.get("n_features", "N/A"))
                    _mc4.metric("Modo avaliação", _parent_run_data.data.params.get("eval_mode", "N/A"))
                else:
                    st.info("Runs filhos não encontrados para este experimento.")

    except Exception as _e:
        st.error(f"Erro ao conectar ao MLflow: {_e}")
        st.caption("Verifique o Tracking URI na barra lateral e se o pipeline de treinamento foi executado.")


# =========================================================
# TAB 8 - MODEL INTELLIGENCE
# =========================================================

with tabs[7]:
    caixa_secao(
        "Model intelligence",
        "Explica a estrutura do pipeline e como o modelo foi montado.",
    )

    col_a, col_b = st.columns(2)

    best_model_name = best_meta.get("best_model", "N/A")
    best_rmse = best_meta.get("best_rmse", None)

    with col_a:
        st.markdown("### Modelo selecionado")
        st.metric("Melhor modelo", str(best_model_name))
        st.metric("RMSE", fmt_number(best_rmse))

        st.markdown(
            """
            <div class="info-box">
                <b>Leitura</b><br>
                O RMSE mede o desvio médio da previsão em relação ao valor real.
                Quanto menor o valor, melhor a precisão do modelo.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("### Estratégia de features")
        total_features = len(feature_names)
        st.metric("Quantidade de features", total_features if total_features else "N/A")

        if preprocess_meta.get("target_col"):
            st.metric("Target", str(preprocess_meta.get("target_col")))
        elif best_meta.get("target_col"):
            st.metric("Target", str(best_meta.get("target_col")))
        else:
            st.metric("Target", "Weekly_Sales")

    if feature_names:
        st.markdown("### Grupos de variáveis")

        for grupo, feats in grupos_features.items():
            if feats:
                st.markdown(f"**{grupo}**")
                st.write(feats)

        st.markdown(
            """
            <div class="good-box">
                <b>Resumo técnico</b><br>
                O pipeline combina variáveis de calendário, histórico de vendas, estatísticas móveis,
                contexto promocional e estrutura da loja. Essa combinação é adequada para forecasting de varejo,
                porque captura sazonalidade, tendência, comportamento recente e contexto externo.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Não foi possível identificar a lista de features do pipeline.")

# =========================================================
# TAB 9 - MODEL MONITORING
# =========================================================

with tabs[8]:
    caixa_secao(
        "Model monitoring",
        "Camada de monitoramento do comportamento do erro e metadados do modelo.",
    )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("### Drift do erro")
        st.caption("Monitora se o erro muda de patamar ao longo do tempo.")

        if "Date" in df_filtrado.columns and "erro_abs" in df_filtrado.columns:
            drift_df = df_filtrado.copy()
            drift_df["mes_ref"] = drift_df["Date"].dt.to_period("M").astype(str)

            drift_agg = (
                drift_df.groupby("mes_ref", as_index=False)["erro_abs"]
                .mean()
                .sort_values("mes_ref")
            )

            fig_drift = px.line(
                drift_agg,
                x="mes_ref",
                y="erro_abs",
                markers=True,
            )
            fig_drift.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Mês",
                yaxis_title="Erro absoluto médio",
            )
            st.plotly_chart(fig_drift, use_container_width=True)

    with m2:
        st.markdown("### Registros anômalos")
        st.caption("Detecção simples de pontos com erro atípico.")

        if not anomalias.empty:
            cols_show = [
                c for c in [
                    "Store_Label",
                    "Dept_Label",
                    "Date",
                    "Weekly_Sales",
                    "y_pred",
                    "erro_abs",
                    "z_erro",
                ]
                if c in anomalias.columns
            ]
            st.dataframe(
                anomalias[cols_show].sort_values("erro_abs", ascending=False).head(20),
                use_container_width=True,
            )
        else:
            st.info("Nenhuma anomalia relevante foi detectada no recorte atual.")

    st.markdown("### Metadados do modelo")
    st.caption("Informações úteis para auditoria e versionamento.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Nome do modelo", meta_yaml.get("name", "N/A"))
    c2.metric("Versão", meta_yaml.get("version", "N/A"))
    c3.metric("Status", meta_yaml.get("status", "N/A"))

    st.markdown(
        f"""
        <div class="section-card">
            <div><b>Run ID:</b> {meta_yaml.get("run_id", "N/A")}</div>
            <div><b>Stage:</b> {meta_yaml.get("current_stage", "N/A")}</div>
            <div><b>Source:</b> {meta_yaml.get("source", "N/A")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# TAB 10 - DADOS
# =========================================================

with tabs[9]:
    caixa_secao(
        "Dados",
        "Camada operacional para inspeção do recorte analisado.",
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Linhas", f"{len(df_filtrado):,}")
    d2.metric("Lojas", resumo["lojas"])
    d3.metric("Departamentos", resumo["departamentos"])
    d4.metric("MAPE", fmt_pct(resumo["mape"]))

    st.markdown("### Amostra dos dados")
    cols = [
        c for c in [
            "Store_Label",
            "Dept_Label",
            "Date",
            "Weekly_Sales",
            "y_pred",
            "erro",
            "erro_abs",
            "erro_pct",
            "impacto_total_estimado",
        ]
        if c in df_filtrado.columns
    ]

    st.dataframe(
        df_filtrado[cols].sort_values("Date"),
        use_container_width=True,
    )

    st.markdown("### Estatísticas descritivas")
    numericas = df_filtrado.select_dtypes(include="number")
    if not numericas.empty:
        st.dataframe(numericas.describe().T, use_container_width=True)

# =========================================================
# RODAPÉ
# =========================================================

st.markdown("---")
st.markdown(
    """
    <div class="footer-note">
        Retail Forecast Intelligence • Dashboard orientado a negócio, risco operacional, interpretabilidade e monitoramento do modelo.
    </div>
    """,
    unsafe_allow_html=True,
)