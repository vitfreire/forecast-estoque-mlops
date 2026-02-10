import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import settings
from src.cost import cost_error
from src.metrics import mae, rmse, mape

load_dotenv()

st.set_page_config(page_title="Forecast + Estoque (Walmart)", layout="wide")

st.title("Forecast de Demanda + Custo do Erro (R$)")
st.caption(
    "Comparação por loja/departamento no conjunto de validação (últimos dias).")

pred_path = os.path.join(settings.REPORTS_DIR, "valid_predictions.parquet")
if not os.path.exists(pred_path):
    st.error("Arquivo de previsões não encontrado. Rode o flow do Prefect antes.")
    st.stop()

df = pd.read_parquet(pred_path)
df["Date"] = pd.to_datetime(df["Date"])

st.sidebar.header("Filtros")
store = st.sidebar.selectbox("Store", sorted(df["Store"].unique()))
dept = st.sidebar.selectbox("Dept", sorted(
    df[df["Store"] == store]["Dept"].unique()))

cost_under = st.sidebar.number_input(
    "Custo subprevisão (R$ / unidade)", value=float(settings.COST_UNDER), step=0.5)
cost_over = st.sidebar.number_input(
    "Custo superprevisão (R$ / unidade)", value=float(settings.COST_OVER), step=0.5)

view = df[(df["Store"] == store) & (df["Dept"] == dept)].sort_values("Date")

y_true = view["y_true"].values
y_pred = view["y_pred"].values

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae(y_true, y_pred):.2f}")
col2.metric("RMSE", f"{rmse(y_true, y_pred):.2f}")
col3.metric("MAPE", f"{mape(y_true, y_pred):.3f}")
col4.metric("Custo total (R$)",
            f"{cost_error(y_true, y_pred, cost_under, cost_over):,.2f}")

st.subheader("Série temporal (validação)")
chart_df = view[["Date", "y_true", "y_pred"]].rename(
    columns={"y_true": "Real", "y_pred": "Previsto"}).set_index("Date")
st.line_chart(chart_df)

st.subheader("Tabela")
st.dataframe(view, use_container_width=True)
