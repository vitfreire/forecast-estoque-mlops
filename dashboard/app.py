from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# =========================================================
# BASE DIR — funciona localmente e no Streamlit Cloud
# =========================================================

BASE_DIR = Path(__file__).parent.parent  # raiz do repositório


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

        .alert-red {
            background: rgba(239,68,68,0.12);
            border: 1px solid rgba(239,68,68,0.30);
            border-left: 4px solid #ef4444;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .alert-yellow {
            background: rgba(245,158,11,0.10);
            border: 1px solid rgba(245,158,11,0.25);
            border-left: 4px solid #f59e0b;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .alert-green {
            background: rgba(16,185,129,0.10);
            border: 1px solid rgba(16,185,129,0.25);
            border-left: 4px solid #10b981;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 10px;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def fmt_money(valor: Optional[float]) -> str:
    if valor is None or pd.isna(valor):
        return "N/A"
    return f"${valor:,.0f}"


def fmt_number(valor: Optional[float], casas: int = 2) -> str:
    if valor is None or pd.isna(valor):
        return "N/A"
    return f"{valor:,.{casas}f}"


def fmt_pct(valor: Optional[float], casas: int = 1) -> str:
    if valor is None or pd.isna(valor):
        return "N/A"
    return f"{valor:.{casas}f}%"


def primeiro_arquivo_existente(candidatos: list[str]) -> Optional[str]:
    for caminho in candidatos:
        if os.path.exists(caminho):
            return caminho
    return None


def card_kpi(titulo: str, valor: str, descricao: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{titulo}</div>
        <div class="kpi-value">{valor}</div>
        <div class="kpi-desc">{descricao}</div>
    </div>
    """


def caixa_secao(titulo: str, subtitulo: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div style="font-size:1.15rem; font-weight:700;">{titulo}</div>
            <div class="mini-note">{subtitulo}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classificar_smape(smape: Optional[float]) -> tuple[str, str, str]:
    """Retorna (label, classe_css, descricao) baseado no SMAPE."""
    if smape is None or pd.isna(smape):
        return "Sem leitura", "info-box", "Não foi possível calcular o erro percentual."
    if smape <= 8:
        return "Alta confiabilidade", "alert-green", f"SMAPE de {smape:.1f}% — excelente para apoiar decisões de estoque."
    if smape <= 15:
        return "Boa confiabilidade", "alert-green", f"SMAPE de {smape:.1f}% — previsão útil, monitorar segmentos críticos."
    if smape <= 25:
        return "Atenção", "alert-yellow", f"SMAPE de {smape:.1f}% — erro pode impactar estoque e receita."
    return "Risco elevado", "alert-red", f"SMAPE de {smape:.1f}% — revisar modelo antes de usar em produção."


def safe_mean(series: pd.Series) -> Optional[float]:
    if series is None or series.dropna().empty:
        return None
    return float(series.dropna().mean())


def safe_sum(series: pd.Series) -> Optional[float]:
    if series is None or series.dropna().empty:
        return None
    return float(series.dropna().sum())


# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================

@st.cache_data(show_spinner=False)
def carregar_previsoes(caminho: str) -> pd.DataFrame:
    df = pd.read_parquet(caminho).copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "prediction" in df.columns and "y_pred" not in df.columns:
        df["y_pred"] = df["prediction"]

    for col in ["Weekly_Sales", "y_pred"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"Weekly_Sales", "y_pred"}.issubset(df.columns):
        df["erro"] = df["Weekly_Sales"] - df["y_pred"]
        df["erro_abs"] = df["erro"].abs()
        # SMAPE simétrico: 2*|e|/(|real|+|pred|) — mais estável que |e|/|real|
        denominador = (df["Weekly_Sales"].abs() + df["y_pred"].abs()).replace(0, np.nan)
        df["erro_pct"] = (2 * df["erro_abs"] / denominador) * 100
        df["superprevisao"] = np.where(df["y_pred"] > df["Weekly_Sales"], df["y_pred"] - df["Weekly_Sales"], 0.0)
        df["subprevisao"] = np.where(df["y_pred"] < df["Weekly_Sales"], df["Weekly_Sales"] - df["y_pred"], 0.0)

    if "Store" in df.columns:
        df["Store_Label"] = "Loja " + df["Store"].astype(str)
    if "Dept" in df.columns:
        df["Dept_Label"] = "Depto " + df["Dept"].astype(str)

    return df


@st.cache_data(show_spinner=False)
def carregar_feature_importance(caminho: Optional[str]) -> pd.DataFrame:
    if caminho is None or not os.path.exists(caminho):
        return pd.DataFrame()
    df = pd.read_csv(caminho).copy()
    colunas_lower = {c.lower(): c for c in df.columns}
    col_feature, col_importance = None, None
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
        return df.sort_values("importance", ascending=False)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def carregar_json(caminho: Optional[str]) -> dict:
    if caminho is None or not os.path.exists(caminho):
        return {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def calcular_impacto(df: pd.DataFrame, taxa_margem: float, taxa_excesso: float, taxa_ruptura: float) -> pd.DataFrame:
    base = df.copy()
    if {"subprevisao", "superprevisao"}.issubset(base.columns):
        base["perda_venda"] = base["subprevisao"] * taxa_margem
        base["custo_excesso"] = base["superprevisao"] * taxa_excesso
        base["penalidade_ruptura"] = base["subprevisao"] * taxa_ruptura
        base["impacto_total"] = base["perda_venda"] + base["custo_excesso"] + base["penalidade_ruptura"]
    return base


def serie_temporal(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "Weekly_Sales", "y_pred"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    colunas = ["Weekly_Sales", "y_pred"]
    for c in ["impacto_total", "erro_abs"]:
        if c in df.columns:
            colunas.append(c)
    ts = df.groupby("Date", as_index=False)[colunas].sum(numeric_only=True).sort_values("Date")
    ts["gap"] = ts["Weekly_Sales"] - ts["y_pred"]
    return ts


def detectar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    if "erro_abs" not in df.columns or df["erro_abs"].dropna().empty:
        return pd.DataFrame()
    media = df["erro_abs"].mean()
    desvio = df["erro_abs"].std()
    if pd.isna(desvio) or desvio == 0:
        return pd.DataFrame()
    base = df.copy()
    base["z_erro"] = (base["erro_abs"] - media) / desvio
    base["anomalia"] = base["z_erro"].abs() >= 2.5
    return base[base["anomalia"]].copy()


# =========================================================
# CAMINHOS — BASE_DIR garante funcionar no Streamlit Cloud
# =========================================================

def _resolver(candidatos: list[str]) -> Optional[str]:
    """Tenta caminhos relativos ao repo root e ao CWD."""
    for c in candidatos:
        full = BASE_DIR / c
        if full.exists():
            return str(full)
        if os.path.exists(c):
            return c
    return None


caminho_previsoes_padrao = _resolver([
    "artifacts/reports/valid_predictions.parquet",
    "reports/batch_predictions.parquet",
]) or str(BASE_DIR / "artifacts/reports/valid_predictions.parquet")

caminho_best_meta_padrao = _resolver([
    "artifacts/models/best_model_meta.json",
    "best_model_meta.json",
])

caminho_preprocess_padrao = _resolver([
    "artifacts/preprocess.json",
    "preprocess.json",
])

caminho_leaderboard_padrao = _resolver([
    "artifacts/reports/leaderboard.csv",
])

_charts_dir = BASE_DIR / "artifacts" / "charts"

def _chart(nome: str) -> Optional[str]:
    """Retorna o caminho de um PNG pré-gerado, ou None se não existir."""
    p = _charts_dir / nome
    return str(p) if p.exists() else None

_artif_fi_dir = BASE_DIR / "artifacts" / "feature_importance"
_modelos_fi = sorted([
    d for d in os.listdir(_artif_fi_dir)
    if (_artif_fi_dir / d).is_dir()
]) if _artif_fi_dir.exists() else []


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## Retail Forecast Intelligence")
st.sidebar.caption("Dashboard executivo de previsão de vendas")
st.sidebar.markdown("---")

st.sidebar.markdown("### Arquivo de dados")
caminho_previsoes = st.sidebar.text_input("Previsões", value=caminho_previsoes_padrao)

if _modelos_fi:
    _modelo_fi_sel = st.sidebar.selectbox("Modelo (feature importance)", _modelos_fi)
    caminho_importancia = str(_artif_fi_dir / _modelo_fi_sel / "feature_importance.csv")
else:
    caminho_importancia = _resolver(["feature_importance.csv"]) or "feature_importance.csv"

st.sidebar.markdown("---")
st.sidebar.markdown("### Premissas financeiras")
st.sidebar.caption("Definem como o erro do modelo é traduzido em impacto financeiro estimado.")

taxa_margem = st.sidebar.slider("Margem perdida por ruptura", 0.01, 1.00, 0.25, 0.01)
taxa_excesso = st.sidebar.slider("Custo de excesso de estoque", 0.01, 1.00, 0.10, 0.01)
taxa_ruptura = st.sidebar.slider("Penalidade operacional por ruptura", 0.00, 1.00, 0.08, 0.01)


# =========================================================
# CARREGAMENTO EFETIVO
# =========================================================

if not os.path.exists(caminho_previsoes):
    st.error(f"Arquivo de previsões não encontrado: `{caminho_previsoes}`")
    st.info("Verifique se o pipeline de inferência foi executado: `python -m flows.batch_inference_flow`")
    st.stop()

with st.spinner("Carregando dados..."):
    df = carregar_previsoes(caminho_previsoes)
    df = calcular_impacto(df, taxa_margem, taxa_excesso, taxa_ruptura)

df_importancia = carregar_feature_importance(caminho_importancia)
best_meta = carregar_json(caminho_best_meta_padrao)
preprocess_meta = carregar_json(caminho_preprocess_padrao)

if df.empty:
    st.warning("Arquivo carregado, mas sem dados disponíveis.")
    st.stop()


# =========================================================
# HERO
# =========================================================

best_model_nome = best_meta.get("best_model", "N/A").upper() if best_meta else "N/A"
best_rmse_hero = best_meta.get("best_rmse", None) if best_meta else None

st.markdown(
    f"""
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:18px; flex-wrap:wrap;">
            <div>
                <div style="font-size:2rem; font-weight:800;">Retail Forecast Intelligence</div>
                <div class="mini-note" style="margin-top:8px;">
                    Plataforma de previsão de demanda para varejo — traduz previsões em decisões de estoque e impacto financeiro.
                </div>
                <div style="margin-top:12px;">
                    <span class="chip">Forecasting</span>
                    <span class="chip">Gestão de Estoque</span>
                    <span class="chip">Impacto Financeiro</span>
                    <span class="chip">MLOps</span>
                </div>
            </div>
            <div style="max-width:420px;">
                <div style="font-size:1rem; font-weight:600;">Modelo ativo</div>
                <div class="mini-note" style="margin-top:6px;">
                    <b>{best_model_nome}</b> · RMSE {fmt_number(best_rmse_hero, 0) if best_rmse_hero else "N/A"}
                    &nbsp;·&nbsp; Dataset Walmart · {len(df):,} registros de validação
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

caixa_secao("Filtros", "Selecione lojas, departamentos e período para detalhar a análise.")

f1, f2, f3 = st.columns(3)
df_filtrado = df.copy()

if "Store_Label" in df_filtrado.columns:
    lojas = sorted(df_filtrado["Store_Label"].dropna().unique().tolist())
    lojas_sel = f1.multiselect("Loja", lojas, default=[])
    if lojas_sel:
        df_filtrado = df_filtrado[df_filtrado["Store_Label"].isin(lojas_sel)]

if "Dept_Label" in df_filtrado.columns:
    depts = sorted(df_filtrado["Dept_Label"].dropna().unique().tolist())
    depts_sel = f2.multiselect("Departamento", depts, default=[])
    if depts_sel:
        df_filtrado = df_filtrado[df_filtrado["Dept_Label"].isin(depts_sel)]

if "Date" in df_filtrado.columns and not df_filtrado["Date"].dropna().empty:
    data_min = df_filtrado["Date"].min().date()
    data_max = df_filtrado["Date"].max().date()
    periodo = f3.date_input("Período", value=(data_min, data_max), min_value=data_min, max_value=data_max)
    if isinstance(periodo, tuple) and len(periodo) == 2:
        inicio, fim = periodo
        df_filtrado = df_filtrado[
            (df_filtrado["Date"].dt.date >= inicio) & (df_filtrado["Date"].dt.date <= fim)
        ]

if df_filtrado.empty:
    st.warning("Os filtros aplicados não retornaram dados.")
    st.stop()

# Estruturas derivadas
ts = serie_temporal(df_filtrado)
anomalias = detectar_anomalias(df_filtrado)

mae = safe_mean(df_filtrado["erro_abs"]) if "erro_abs" in df_filtrado.columns else None
smape = safe_mean(df_filtrado["erro_pct"]) if "erro_pct" in df_filtrado.columns else None
receita_real = safe_sum(df_filtrado["Weekly_Sales"]) if "Weekly_Sales" in df_filtrado.columns else None
receita_prev = safe_sum(df_filtrado["y_pred"]) if "y_pred" in df_filtrado.columns else None
impacto_total = safe_sum(df_filtrado["impacto_total"]) if "impacto_total" in df_filtrado.columns else None
vies = safe_mean(df_filtrado["erro"]) if "erro" in df_filtrado.columns else None

status_label, status_classe, status_desc = classificar_smape(smape)
n_lojas = int(df_filtrado["Store"].nunique()) if "Store" in df_filtrado.columns else 0
n_depts = int(df_filtrado["Dept"].nunique()) if "Dept" in df_filtrado.columns else 0


# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "Visão Executiva",
    "Risco Operacional",
    "Forecast & Erros",
    "Impacto Financeiro",
    "Modelo",
])


# =========================================================
# TAB 1 — VISÃO EXECUTIVA
# =========================================================

with tabs[0]:
    caixa_secao(
        "Visão executiva",
        "KPIs principais para liderança: cobertura da operação, precisão da previsão e impacto financeiro estimado.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card_kpi("Receita real", fmt_money(receita_real), f"{n_lojas} lojas · {n_depts} departamentos"), unsafe_allow_html=True)
    c2.markdown(card_kpi("Receita prevista", fmt_money(receita_prev), "Total projetado pelo modelo no período."), unsafe_allow_html=True)
    c3.markdown(card_kpi("SMAPE", fmt_pct(smape), "Erro percentual médio — benchmark de mercado é 10–30%."), unsafe_allow_html=True)
    c4.markdown(card_kpi("Impacto estimado", fmt_money(impacto_total), "Custo de ruptura + excesso estimado pelo modelo."), unsafe_allow_html=True)

    st.markdown("")

    # Status e viés
    col_status, col_vies = st.columns(2)
    with col_status:
        st.markdown(
            f"""<div class="{status_classe}"><b>{status_label}</b><br>{status_desc}</div>""",
            unsafe_allow_html=True,
        )
    with col_vies:
        if vies is not None and not pd.isna(vies):
            if vies > 50:
                msg_vies = f"<b>⚠️ Tendência a subprever</b> (viés médio: +{fmt_money(vies)})<br>O modelo tende a estimar menos do que a demanda real — risco de ruptura de estoque."
                classe_vies = "alert-yellow"
            elif vies < -50:
                msg_vies = f"<b>⚠️ Tendência a superprever</b> (viés médio: {fmt_money(vies)})<br>O modelo tende a estimar mais do que a demanda real — risco de excesso de estoque."
                classe_vies = "alert-yellow"
            else:
                msg_vies = f"<b>✓ Modelo equilibrado</b> (viés médio: {fmt_money(vies)})<br>Sem tendência sistemática de erro — bom para decisões de reposição."
                classe_vies = "alert-green"
            st.markdown(f"""<div class="{classe_vies}">{msg_vies}</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Gráfico principal: histórico de receita (imagem pré-gerada com 3 anos de dados)
    col_chart, col_insight = st.columns([2, 1])

    with col_chart:
        st.markdown("### Receita semanal histórica (2010–2012)")
        st.caption("Série completa de 143 semanas — 45 lojas, padrão sazonal com pico em Black Friday e Natal.")
        _img_hist = _chart("receita_historica.png")
        if _img_hist:
            st.image(_img_hist, use_column_width=True)
        elif not ts.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts["Date"], y=ts["Weekly_Sales"], mode="lines", name="Real", line=dict(width=3, color="#3b82f6")))
            fig.add_trace(go.Scatter(x=ts["Date"], y=ts["y_pred"], mode="lines", name="Previsto", line=dict(width=2, dash="dash", color="#10b981")))
            fig.update_layout(
                height=400, margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_insight:
        st.markdown("### Resumo do período")
        if receita_real and receita_prev:
            delta_pct = ((receita_prev - receita_real) / receita_real) * 100
            st.metric("Receita real", fmt_money(receita_real))
            st.metric("Receita prevista", fmt_money(receita_prev), delta=f"{delta_pct:+.1f}%")
        st.metric("MAE médio", fmt_money(mae))

        if not anomalias.empty:
            st.markdown(
                f"""<div class="alert-yellow"><b>⚠️ {len(anomalias)} registros anômalos</b><br>
                Desvios atípicos detectados no recorte atual. Verifique a aba Risco Operacional.</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="alert-green"><b>✓ Sem anomalias detectadas</b><br>Nenhum desvio atípico no recorte atual.</div>""",
                unsafe_allow_html=True,
            )


# =========================================================
# TAB 2 — RISCO OPERACIONAL
# =========================================================

with tabs[1]:
    caixa_secao(
        "Risco operacional",
        "Identifica onde o modelo erra mais — prioriza lojas e departamentos que precisam de atenção.",
    )

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("### Top lojas por impacto financeiro")
        st.caption("Quanto maior a barra, maior o custo estimado de erros de previsão nessa loja.")
        if "Store_Label" in df_filtrado.columns and "impacto_total" in df_filtrado.columns:
            top_lojas = (
                df_filtrado.groupby("Store_Label", as_index=False)
                .agg(impacto=("impacto_total", "sum"), mae=("erro_abs", "mean"), receita=("Weekly_Sales", "sum"))
                .sort_values("impacto", ascending=False)
                .head(15)
            )
            if not top_lojas.empty:
                fig_lojas = px.bar(
                    top_lojas.sort_values("impacto", ascending=True),
                    x="impacto", y="Store_Label", orientation="h",
                    color="mae",
                    color_continuous_scale="Reds",
                    labels={"impacto": "Impacto estimado (USD)", "Store_Label": "Loja", "mae": "MAE médio"},
                )
                fig_lojas.update_layout(
                    height=460, margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_lojas, use_container_width=True)

    with r2:
        st.markdown("### Top departamentos por impacto financeiro")
        st.caption("Departamentos com maior distorção entre previsto e real acumulam mais custo operacional.")
        if "Dept_Label" in df_filtrado.columns and "impacto_total" in df_filtrado.columns:
            top_depts = (
                df_filtrado.groupby("Dept_Label", as_index=False)
                .agg(impacto=("impacto_total", "sum"), mae=("erro_abs", "mean"), receita=("Weekly_Sales", "sum"))
                .sort_values("impacto", ascending=False)
                .head(15)
            )
            if not top_depts.empty:
                fig_depts = px.bar(
                    top_depts.sort_values("impacto", ascending=True),
                    x="impacto", y="Dept_Label", orientation="h",
                    color="mae",
                    color_continuous_scale="Oranges",
                    labels={"impacto": "Impacto estimado (USD)", "Dept_Label": "Departamento", "mae": "MAE médio"},
                )
                fig_depts.update_layout(
                    height=460, margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_depts, use_container_width=True)

    # Heatmap — usa imagem pré-gerada (mais confiável no Streamlit Cloud)
    st.markdown("### Mapa de calor — erro médio por loja × departamento")
    st.caption("Blocos mais escuros indicam combinações loja/departamento com maior erro de previsão. Top 20 lojas × 20 departamentos por MAE.")

    _img_hm = _chart("heatmap_erro.png")
    if _img_hm:
        st.image(_img_hm, use_column_width=True)
    elif {"Store_Label", "Dept_Label", "erro_abs"}.issubset(df_filtrado.columns):
        heatmap_df = df_filtrado.copy()
        # Limita a top-20 lojas e top-20 departamentos por erro médio (evita heatmap ilegível)
        top_stores = (
            heatmap_df.groupby("Store_Label")["erro_abs"].mean()
            .sort_values(ascending=False).head(20).index.tolist()
        )
        top_depts_hm = (
            heatmap_df.groupby("Dept_Label")["erro_abs"].mean()
            .sort_values(ascending=False).head(20).index.tolist()
        )
        heatmap_df = heatmap_df[
            heatmap_df["Store_Label"].isin(top_stores) & heatmap_df["Dept_Label"].isin(top_depts_hm)
        ]
        # Ordena labels numericamente
        def _sort_label(labels):
            return sorted(labels, key=lambda s: int(s.split()[-1]) if s.split()[-1].isdigit() else 0)
        top_stores = _sort_label(top_stores)
        top_depts_hm = _sort_label(top_depts_hm)
        heatmap = heatmap_df.pivot_table(
            values="erro_abs",
            index="Store_Label",
            columns="Dept_Label",
            aggfunc="mean",
        ).reindex(index=top_stores, columns=top_depts_hm)
        if not heatmap.empty:
            fig_heat = px.imshow(
                heatmap,
                x=heatmap.columns.tolist(),
                y=heatmap.index.tolist(),
                color_continuous_scale="Blues",
                aspect="auto",
                labels={"color": "MAE médio (USD)", "x": "Departamento", "y": "Loja"},
                text_auto=False,
            )
            fig_heat.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=11)),
                coloraxis_colorbar=dict(title="MAE (USD)"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Mostrando top 20 lojas × top 20 departamentos por erro médio absoluto.")
        else:
            st.info("Dados insuficientes para gerar o heatmap com os filtros atuais.")

    # Registros anômalos
    if not anomalias.empty:
        with st.expander(f"Ver {len(anomalias)} registros anômalos detectados"):
            cols_show = [c for c in ["Store_Label", "Dept_Label", "Date", "Weekly_Sales", "y_pred", "erro_abs"] if c in anomalias.columns]
            st.dataframe(
                anomalias[cols_show].sort_values("erro_abs", ascending=False).head(30),
                use_container_width=True,
            )


# =========================================================
# TAB 3 — FORECAST & ERROS
# =========================================================

with tabs[2]:
    caixa_secao(
        "Forecast & Erros",
        "Analisa a qualidade da previsão ao longo do tempo e a distribuição dos erros.",
    )

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("### Sazonalidade semanal (dataset completo)")
        st.caption("Picos na semana 47-48 (Black Friday/Thanksgiving) e 51-52 (Natal) — padrão dos 3 anos de histórico.")
        _img_saz = _chart("sazonalidade_semanal.png")
        if _img_saz:
            st.image(_img_saz, use_column_width=True)

    with g2:
        st.markdown("### Distribuição do erro de previsão")
        st.caption("Erro centrado em zero indica modelo equilibrado. Assimetria indica viés sistemático.")
        _img_err = _chart("distribuicao_erro.png")
        if _img_err:
            st.image(_img_err, use_column_width=True)
        elif "erro" in df_filtrado.columns:
            fig_hist_erro = px.histogram(df_filtrado, x="erro", nbins=50, color_discrete_sequence=["#3b82f6"])
            fig_hist_erro.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
            fig_hist_erro.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title="Erro (Real − Previsto)", yaxis_title="Frequência")
            st.plotly_chart(fig_hist_erro, use_container_width=True)

    g3, g4 = st.columns(2)

    with g3:
        st.markdown("### Vendas por tipo de loja")
        st.caption("Lojas tipo A (grandes) vendem em média 3× mais que tipo C — justifica o modelo por segmento.")
        _img_tipo = _chart("vendas_por_tipo.png")
        if _img_tipo:
            st.image(_img_tipo, use_column_width=True)

    with g4:
        st.markdown("### Real vs Previsto por registro")
        st.caption("Pontos próximos da diagonal indicam boa previsão. Cor indica magnitude do erro.")
        _img_rvp = _chart("real_vs_previsto.png")
        if _img_rvp:
            st.image(_img_rvp, use_column_width=True)
        elif {"Weekly_Sales", "y_pred"}.issubset(df_filtrado.columns):
            sample = df_filtrado.sample(min(2000, len(df_filtrado)), random_state=42)
            fig_scatter = px.scatter(sample, x="Weekly_Sales", y="y_pred",
                color="erro_abs" if "erro_abs" in sample.columns else None,
                color_continuous_scale="RdYlGn_r", opacity=0.5,
                labels={"Weekly_Sales": "Real (USD)", "y_pred": "Previsto (USD)", "erro_abs": "Erro abs"})
            max_val = max(sample["Weekly_Sales"].max(), sample["y_pred"].max())
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                  line=dict(color="rgba(255,255,255,0.25)", dash="dash"))
            fig_scatter.update_layout(height=340, margin=dict(l=10,r=10,t=20,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                coloraxis_showscale=False)
            st.plotly_chart(fig_scatter, use_container_width=True)


# =========================================================
# TAB 4 — IMPACTO FINANCEIRO
# =========================================================

with tabs[3]:
    caixa_secao(
        "Impacto financeiro",
        "Traduz o erro do modelo em custo estimado de ruptura e excesso de estoque.",
    )

    perda_venda = safe_sum(df_filtrado["perda_venda"]) if "perda_venda" in df_filtrado.columns else None
    custo_excesso = safe_sum(df_filtrado["custo_excesso"]) if "custo_excesso" in df_filtrado.columns else None
    penalidade = safe_sum(df_filtrado["penalidade_ruptura"]) if "penalidade_ruptura" in df_filtrado.columns else None

    i1, i2, i3, i4 = st.columns(4)
    i1.markdown(card_kpi("Impacto total estimado", fmt_money(impacto_total), "Custo combinado de ruptura e excesso."), unsafe_allow_html=True)
    i2.markdown(card_kpi("Perda por subprevisão", fmt_money(perda_venda), "Receita perdida quando o modelo previu menos do que a demanda."), unsafe_allow_html=True)
    i3.markdown(card_kpi("Custo de excesso", fmt_money(custo_excesso), "Custo de estoque parado quando o modelo superestimou a demanda."), unsafe_allow_html=True)
    i4.markdown(card_kpi("Penalidade por ruptura", fmt_money(penalidade), "Custo operacional de rupturas (logística, emergências)."), unsafe_allow_html=True)

    st.markdown("")

    fi1, fi2 = st.columns(2)

    with fi1:
        st.markdown("### Composição do impacto")
        st.caption("Ruptura (subprevisão) tende a ser mais cara que excesso — ajuste as premissas na barra lateral.")
        if perda_venda is not None:
            impacto_comp = pd.DataFrame({
                "Tipo": ["Perda por subprevisão", "Custo de excesso", "Penalidade por ruptura"],
                "Valor": [perda_venda or 0, custo_excesso or 0, penalidade or 0],
            })
            fig_pie = px.pie(
                impacto_comp, names="Tipo", values="Valor",
                hole=0.55,
                color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6"],
            )
            fig_pie.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with fi2:
        st.markdown("### Top 15 lojas por impacto financeiro estimado")
        st.caption("Prioridade de ação: lojas onde reduzir erro tem maior retorno financeiro.")
        _img_imp = _chart("impacto_por_loja.png")
        if _img_imp:
            st.image(_img_imp, use_column_width=True)
        elif not ts.empty and "impacto_total" in ts.columns:
            fig_impact_time = px.bar(ts, x="Date", y="impacto_total", color_discrete_sequence=["#ef4444"])
            fig_impact_time.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)")
            st.plotly_chart(fig_impact_time, use_container_width=True)

    # Tabela de maiores impactos
    st.markdown("### Maiores impactos por loja × departamento")
    st.caption("Prioridade de ação: onde reduzir erro tem maior retorno financeiro.")
    if {"Store_Label", "Dept_Label", "impacto_total", "erro_abs", "Weekly_Sales"}.issubset(df_filtrado.columns):
        tabela_impacto = (
            df_filtrado.groupby(["Store_Label", "Dept_Label"], as_index=False)
            .agg(
                Receita_Real=("Weekly_Sales", "sum"),
                MAE=("erro_abs", "mean"),
                Impacto_Estimado=("impacto_total", "sum"),
            )
            .sort_values("Impacto_Estimado", ascending=False)
            .head(20)
        )
        tabela_impacto["Receita_Real"] = tabela_impacto["Receita_Real"].map("${:,.0f}".format)
        tabela_impacto["MAE"] = tabela_impacto["MAE"].map("${:,.0f}".format)
        tabela_impacto["Impacto_Estimado"] = tabela_impacto["Impacto_Estimado"].map("${:,.0f}".format)
        tabela_impacto.columns = ["Loja", "Departamento", "Receita Real", "MAE", "Impacto Estimado"]
        st.dataframe(tabela_impacto, use_container_width=True)


# =========================================================
# TAB 5 — MODELO
# =========================================================

with tabs[4]:
    caixa_secao(
        "Modelo",
        "Performance técnica, importância de variáveis e monitoramento do modelo em produção.",
    )

    m_tabs = st.tabs(["Leaderboard", "Feature Importance", "Monitoramento", "Pipeline"])

    # --- Leaderboard ---
    with m_tabs[0]:
        st.markdown("### Comparativo de modelos")
        st.caption("Modelo vencedor por menor RMSE médio — treinado com Rolling Time Series CV (3 folds).")

        _CORES = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#a855f7"]

        _lb_df = pd.DataFrame()

        # 1) CSV primeiro — funciona no Streamlit Cloud sem MLflow server
        if caminho_leaderboard_padrao and os.path.exists(caminho_leaderboard_padrao):
            try:
                _lb_df = pd.read_csv(caminho_leaderboard_padrao)
                _col_map = {"model": "Modelo", "rmse": "RMSE", "mae": "MAE", "smape": "SMAPE (%)"}
                _lb_df = _lb_df.rename(columns={k: v for k, v in _col_map.items() if k in _lb_df.columns})
                _display = [c for c in ["Modelo", "RMSE", "MAE", "SMAPE (%)"] if c in _lb_df.columns]
                _lb_df = _lb_df[_display].sort_values("RMSE").reset_index(drop=True)
            except Exception:
                _lb_df = pd.DataFrame()

        # 2) Fallback MLflow — só funciona com mlruns/ local
        if _lb_df.empty and _MLFLOW_AVAILABLE:
            try:
                _mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
                if _mlflow_uri and not _mlflow_uri.startswith("file:"):
                    # só tenta se for servidor remoto (não file store local)
                    mlflow.set_tracking_uri(_mlflow_uri)
                    _client = MlflowClient()
                    _exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "forecast_estoque_walmart")
                    _exp = _client.get_experiment_by_name(_exp_name)
                    if _exp:
                        _parent_runs = _client.search_runs(
                            experiment_ids=[_exp.experiment_id],
                            filter_string="tags.`mlflow.runName` = 'train_compare'",
                            order_by=["start_time DESC"],
                            max_results=1,
                        )
                        if _parent_runs:
                            _child_runs = _client.search_runs(
                                experiment_ids=[_exp.experiment_id],
                                filter_string=f"tags.`mlflow.parentRunId` = '{_parent_runs[0].info.run_id}'",
                            )
                            _rows = []
                            for _r in _child_runs:
                                _rmse  = _r.data.metrics.get("rmse")  or _r.data.metrics.get("RMSE")
                                _mae   = _r.data.metrics.get("mae")   or _r.data.metrics.get("MAE")
                                _smape = _r.data.metrics.get("smape") or _r.data.metrics.get("SMAPE")
                                _rows.append({
                                    "Modelo":     _r.data.tags.get("mlflow.runName", _r.info.run_id[:8]),
                                    "RMSE":       float(_rmse)  if _rmse  else None,
                                    "MAE":        float(_mae)   if _mae   else None,
                                    "SMAPE (%)":  float(_smape) if _smape else None,
                                })
                            if _rows:
                                _lb_df = pd.DataFrame(_rows).sort_values("RMSE").reset_index(drop=True)
            except Exception:
                _lb_df = pd.DataFrame()

        if not _lb_df.empty:
            # Cards de métricas por modelo
            _model_cols = st.columns(len(_lb_df))
            _medals = ["🥇", "🥈", "🥉"]
            for _idx, (_mc, (_, _row)) in enumerate(zip(_model_cols, _lb_df.iterrows())):
                _medal = _medals[_idx] if _idx < len(_medals) else ""
                _card_class = "alert-green" if _idx == 0 else "section-card"
                _mc.markdown(
                    f"""<div class="{_card_class}" style="text-align:center; padding:16px;">
                    <div style="font-size:1.4rem;">{_medal}</div>
                    <div style="font-size:1rem; font-weight:700; margin:4px 0;">{str(_row.get('Modelo','N/A')).upper()}</div>
                    <div style="font-size:0.82rem; color:#b7c7df;">RMSE: {fmt_number(_row.get('RMSE'))}</div>
                    <div style="font-size:0.82rem; color:#b7c7df;">MAE: {fmt_number(_row.get('MAE'))}</div>
                    <div style="font-size:0.82rem; color:#b7c7df;">SMAPE: {fmt_pct(_row.get('SMAPE (%)'))}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Tabela completa
            st.dataframe(_lb_df, use_container_width=True)

            # Gráfico comparativo (imagem pré-gerada ou Plotly)
            _img_lb = _chart("leaderboard.png")
            if _img_lb:
                st.image(_img_lb, use_column_width=True)
            else:
                _l1, _l2, _l3 = st.columns(3)
                for _col_widget, _metric, _title in [(_l1, "RMSE", "RMSE (USD)"), (_l2, "MAE", "MAE (USD)"), (_l3, "SMAPE (%)", "SMAPE (%)")]:
                    if _metric in _lb_df.columns and _lb_df[_metric].notna().any():
                        with _col_widget:
                            _fig = px.bar(_lb_df.dropna(subset=[_metric]).sort_values(_metric),
                                x="Modelo", y=_metric, color="Modelo",
                                color_discrete_sequence=_CORES, title=_title)
                            _fig.update_layout(height=300, showlegend=False,
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                                yaxis=dict(rangemode="tozero"))
                            st.plotly_chart(_fig, use_container_width=True)

            st.markdown(
                """<div class="info-box"><b>Como interpretar</b><br>
                Métricas são <b>médias dos 3 folds</b> do Rolling Time Series CV — mais honestas do que pegar o melhor fold isolado.
                SMAPE abaixo de 10% é considerado excelente para forecasting de varejo (benchmark: 10–30%).</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Leaderboard não encontrado em `artifacts/reports/leaderboard.csv`. Execute o pipeline de treinamento.")

    # --- Feature Importance ---
    with m_tabs[1]:
        st.markdown("### Variáveis mais importantes para a previsão")
        st.caption("Mostra quais features o modelo mais usa para gerar as previsões.")

        # Tenta PNG pré-gerado do modelo selecionado; fallback para Plotly dinâmico
        _fi_png = None
        if _modelo_fi_sel:
            _fi_png_path = _artif_fi_dir / _modelo_fi_sel / "feature_importance.png"
            if _fi_png_path.exists():
                _fi_png = str(_fi_png_path)

        if _fi_png:
            st.image(_fi_png, use_column_width=True)
        elif not df_importancia.empty:
            top_feats = df_importancia.head(20)
            fig_fi = px.bar(
                top_feats.sort_values("importance", ascending=True),
                x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Blues",
                labels={"importance": "Importância", "feature": "Variável"},
            )
            fig_fi.update_layout(height=560, margin=dict(l=10,r=10,t=20,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown(
                """<div class="info-box"><b>Leitura para negócio</b><br>
                Variáveis de histórico recente (lags) e estatísticas móveis dominam a previsão — o modelo aprende
                padrões de venda recentes. Variáveis de calendário capturam sazonalidade. Features de contexto
                (temperatura, CPI, markdowns) adicionam sensibilidade a eventos externos.</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Feature importance não disponível para o modelo selecionado.")

    # --- Monitoramento ---
    with m_tabs[2]:
        st.markdown("### Drift do erro ao longo do tempo")
        st.caption("Monitora se o erro aumenta ao longo dos meses — sinal de que o modelo precisa de retreinamento.")

        if "Date" in df_filtrado.columns and "erro_abs" in df_filtrado.columns:
            drift_df = df_filtrado.copy()
            # Usa semana quando há poucos meses distintos, mês quando há mais de 2
            n_meses = drift_df["Date"].dt.to_period("M").nunique()
            if n_meses <= 2:
                drift_df["periodo_ref"] = drift_df["Date"].dt.to_period("W").dt.to_timestamp()
                periodo_label = "Semana"
            else:
                drift_df["periodo_ref"] = drift_df["Date"].dt.to_period("M").dt.to_timestamp()
                periodo_label = "Mês"
            drift_agg = (
                drift_df.groupby("periodo_ref", as_index=False)["erro_abs"]
                .mean()
                .sort_values("periodo_ref")
            )
            fig_drift = px.line(drift_agg, x="periodo_ref", y="erro_abs", markers=True, color_discrete_sequence=["#3b82f6"])
            tick_fmt = "%d/%m/%Y" if n_meses <= 2 else "%b %Y"
            fig_drift.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis_title=periodo_label,
                yaxis_title="Erro absoluto médio (USD)",
                xaxis=dict(tickformat=tick_fmt, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", rangemode="tozero"),
            )
            st.plotly_chart(fig_drift, use_container_width=True)

        if not anomalias.empty:
            st.markdown(f"**{len(anomalias)} registros anômalos** detectados (z-score ≥ 2.5)")
            cols_show = [c for c in ["Store_Label", "Dept_Label", "Date", "Weekly_Sales", "y_pred", "erro_abs"] if c in anomalias.columns]
            st.dataframe(anomalias[cols_show].sort_values("erro_abs", ascending=False).head(15), use_container_width=True)

    # --- Pipeline ---
    with m_tabs[3]:
        st.markdown("### Configuração do pipeline")
        st.caption("Parâmetros e features utilizados no treinamento do modelo.")

        pa, pb = st.columns(2)
        with pa:
            st.markdown("**Modelo selecionado**")
            st.metric("Melhor modelo", str(best_meta.get("best_model", "N/A")).upper())
            st.metric("RMSE de validação", fmt_number(best_meta.get("best_rmse")))
            st.metric("Fração de validação", fmt_pct(best_meta.get("valid_frac", 0) * 100, 0) if best_meta.get("valid_frac") else "N/A")

        with pb:
            st.markdown("**Features do modelo**")
            feature_names = (
                best_meta.get("features")
                or preprocess_meta.get("feature_columns")
                or []
            )
            st.metric("Total de features", len(feature_names) if feature_names else "N/A")
            target = preprocess_meta.get("target_col") or "Weekly_Sales"
            st.metric("Target", target)

        if feature_names:
            st.markdown("**Lista de features**")
            # Agrupa por categoria
            grupos = {
                "Histórico (lags)": [f for f in feature_names if f.startswith("lag_") or f == "diff_1"],
                "Estatísticas móveis": [f for f in feature_names if f.startswith("roll_")],
                "Calendário": [f for f in feature_names if f in {"year", "month", "weekofyear", "dayofweek", "is_month_start", "is_month_end", "week_sin", "week_cos"}],
                "Contexto externo": [f for f in feature_names if f in {"Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment", "IsHoliday"}],
                "Estrutura da loja": [f for f in feature_names if f in {"Store", "Dept", "Size", "Type_A", "Type_B", "Type_C"}],
            }
            for grupo, feats in grupos.items():
                if feats:
                    with st.expander(f"{grupo} ({len(feats)})"):
                        st.write(", ".join(feats))


# =========================================================
# RODAPÉ
# =========================================================

st.markdown("---")
st.markdown(
    """<div style="color:#a9b8d1; font-size:0.82rem; text-align:center;">
    Retail Forecast Intelligence · Pipeline MLOps com LightGBM, XGBoost e Random Forest ·
    Treinado com Rolling Time Series CV · Dados: Walmart Store Sales Forecasting (Kaggle)
    </div>""",
    unsafe_allow_html=True,
)
