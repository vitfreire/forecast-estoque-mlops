# Retail Forecast Intelligence — Previsão de Demanda com MLOps

[![CI](https://github.com/vitfreire/forecast-estoque-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/vitfreire/forecast-estoque-mlops/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://forecas-tml.streamlit.app)

Sistema **end-to-end** de previsão de vendas semanais para varejo, construído com foco em **MLOps real**: pipeline orquestrado com Prefect, rastreamento completo de experimentos via MLflow, comparação automática de múltiplos modelos com rolling time series CV, monitoramento de drift em produção e dashboard executivo interativo no Streamlit Cloud.

> **Dataset:** Walmart Store Sales Forecasting (Kaggle) — 418 mil registros, 45 lojas, 79 departamentos, 143 semanas (2010–2012).

---

## Dashboard ao vivo

**[→ forecas-tml.streamlit.app](https://forecas-tml.streamlit.app)**

O dashboard é voltado para dois perfis:
- **Diretoria** — KPIs de receita, SMAPE, impacto financeiro estimado e status do modelo
- **Gestores de loja** — mapa de calor de erros por loja/departamento, anomalias e alertas de risco

---

## Resultados do modelo (Rolling CV — 3 folds, dataset completo)

| Modelo | RMSE (USD) | MAE (USD) | SMAPE |
|---|---|---|---|
| **LightGBM** | ~813 | ~188 | ~5,8% |
| XGBoost | ~801 | ~168 | ~4,0% |
| Random Forest | ~212* | ~26* | ~1,0%* |

> \* O Random Forest apresenta métricas suspeitas — investigação de possível vazamento de dados em andamento. O modelo selecionado para produção é definido pela **média dos 3 folds**, não pelo melhor fold isolado.
>
> **Benchmark de mercado para forecasting de varejo:** SMAPE 10–30%. SMAPE < 10% indica modelo de alta confiabilidade.

---

## O que este projeto faz

O problema central é prever, com antecedência, quanto cada departamento de cada loja vai vender na próxima semana.

- **Errar para baixo** → ruptura de estoque → venda perdida e cliente insatisfeito
- **Errar para cima** → excesso de estoque → capital imobilizado e custo de armazenagem

O custo desses dois tipos de erro é assimétrico, e o projeto trata isso explicitamente com uma função de custo configurável.

### Pipeline completo

1. **EDA e ABT** (`notebooks/`) — análise exploratória documentada e tabela analítica base com justificativa de cada feature escolhida

2. **Engenharia de features** — calendário (semana do ano, componentes cíclicos seno/cosseno), histórico de vendas (lags 1–4 semanas), estatísticas móveis (médias e desvios em janelas 4, 8 e 12 semanas), contexto macroeconômico (CPI, desemprego, temperatura, combustível) e promoções (markdowns, feriados)

3. **Treinamento multi-modelo com Rolling CV** — LightGBM, XGBoost e Random Forest treinados com 3-fold rolling time series cross-validation, preservando a ordem cronológica dos dados. O **melhor modelo é selecionado pela média do RMSE nos 3 folds** (não pelo melhor fold isolado)

4. **Rastreamento com MLflow** — cada execução registra métricas (RMSE, MAE, SMAPE), hiperparâmetros, datasets amostrados, feature importances e o modelo serializado. O melhor modelo é promovido automaticamente ao MLflow Model Registry

5. **Inferência em batch** — carrega o modelo do Registry (Production > Staging > versão mais recente), gera previsões para o horizonte configurado e salva para consumo pelo dashboard

6. **Monitoramento de drift** — PSI (Population Stability Index) comparando distribuição das features da batch atual vs referência do treino. PSI > 0.2 sinaliza retreinamento necessário

7. **Dashboard executivo** — 5 abas focadas em negócio: Visão Executiva, Risco Operacional, Forecast & Erros, Impacto Financeiro, e Modelo Intelligence. Métricas técnicas traduzidas em linguagem de negócio

---

## Stack técnica

| Camada | Tecnologia |
|---|---|
| Modelos | LightGBM, XGBoost, Scikit-learn (Random Forest) |
| Tracking | MLflow (experimentos, métricas, artifacts, Model Registry) |
| Orquestração | Prefect |
| Dashboard | Streamlit + Plotly |
| Features | Pandas, NumPy |
| Config | Pydantic Settings |
| Testes | pytest (47 testes unitários) |
| CI/CD | GitHub Actions |
| Containerização | Docker + Docker Compose |
| Deploy | Streamlit Cloud (branch `multi-model`) |
| Ambiente | Python 3.11 |

---

## Estrutura do projeto

```
.
├── notebooks/
│   ├── 01_eda.ipynb            # Análise exploratória — sazonalidade, outliers, autocorrelação
│   └── 02_abt.ipynb            # ABT — dicionário de dados, distribuições, leaderboard
├── src/                        # Biblioteca central
│   ├── config.py               # Settings (Pydantic) com modo dev/prod
│   ├── split.py                # Splits temporais e rolling CV
│   ├── metrics.py              # MAE, RMSE, SMAPE
│   ├── cost.py                 # Função de custo assimétrica (ruptura > excesso)
│   ├── predict.py              # Pipeline de inferência com alinhamento de features
│   ├── features/
│   │   └── features.py         # Lags, rolling stats, calendário, componentes cíclicos
│   └── models/
│       ├── trainer.py          # Orquestração multi-modelo + MLflow + valid_predictions
│       ├── lgbm.py             # LightGBM
│       ├── xgb.py              # XGBoost
│       └── rf.py               # Random Forest
├── flows/                      # Pipelines Prefect
│   ├── training_flow.py        # Treino (holdout ou rolling CV) — seleção por média de folds
│   ├── batch_inference_flow.py # Inferência + drift monitoring
│   └── master_flow.py          # Treino → inferência em sequência
├── dashboard/
│   └── app.py                  # Interface Streamlit (5 abas executivas)
├── tests/                      # 47 testes unitários
├── artifacts/                  # Leaderboard, feature importances, valid_predictions
├── .github/workflows/ci.yml    # GitHub Actions (lint + testes automáticos)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Como rodar localmente

### Pré-requisitos

- Python 3.10+
- Dataset Walmart em `data/raw/` (arquivos `train.csv`, `features.csv`, `stores.csv`)
  - Fonte: [Kaggle — Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)

```bash
kaggle competitions download -c walmart-recruiting-store-sales-forecasting
unzip walmart-recruiting-store-sales-forecasting.zip -d data/raw/
```

### Instalação

```bash
git clone https://github.com/vitfreire/forecast-estoque-mlops.git
cd forecast-estoque-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Executar o pipeline

```bash
# Modo dev (rápido — 5 lojas × 5 depts)
APP_ENV=dev EVAL_MODE=rolling python -m flows.training_flow

# Modo produção (dataset completo — ~2h com rolling CV)
APP_ENV=prod EVAL_MODE=rolling python -m flows.training_flow

# Inferência em batch (após treino)
python -m flows.batch_inference_flow

# Dashboard local
streamlit run dashboard/app.py
```

### MLflow UI

```bash
mlflow ui --port 5000
# Acesse: http://localhost:5000
```

### Docker

```bash
# Sobe MLflow server + dashboard
docker compose up mlflow dashboard

# Pipeline via Docker
APP_ENV=dev docker compose run --rm training
```

---

## Variáveis de ambiente

```env
MLFLOW_TRACKING_URI=file:./mlruns       # local sem servidor
MLFLOW_EXPERIMENT_NAME=forecast_estoque_walmart
MLFLOW_REGISTERED_MODEL_NAME=walmart_forecast_lgbm_cost

HORIZON_DAYS=28                          # horizonte de previsão em dias
N_FOLDS=3                                # folds do rolling CV
RUN_MODELS=lightgbm,xgboost,random_forest

EVAL_MODE=rolling                        # holdout ou rolling
APP_ENV=prod                             # dev (reduzido) ou prod (completo)

COST_UNDER=6.0                           # penalidade por ruptura de estoque
COST_OVER=1.5                            # penalidade por excesso
```

---

## Testes

```bash
pytest tests/ -v
```

| Módulo | Testes | O que verifica |
|---|---|---|
| `src/metrics.py` | 9 | MAE, RMSE, SMAPE — valores conhecidos e propriedades matemáticas |
| `src/cost.py` | 5 | Custo assimétrico — ruptura penalizada 4× mais que excesso |
| `src/features/features.py` | 13 | Lags, rolling stats, calendário — ausência de leakage de features |
| `src/split.py` | 10 | Temporal split e rolling CV — sem vazamento, shapes corretos |
| `src/models/preprocessing.py` | 10 | Encoding, alinhamento de colunas treino/validação |

---

## Como funciona o MLflow

```
run: train_compare (run pai)
├── params: modelos, n_features, linhas, hash do dataset
├── métricas: best_rmse, avg_rmse, std_rmse (e MAE/SMAPE equivalentes)
├── artifacts: leaderboard.csv (média dos folds), features.json, valid_predictions.parquet
└── runs filhos (um por modelo)
    ├── lightgbm/   → métricas, hiperparâmetros, model/, feature_importance/
    ├── xgboost/
    └── random_forest/
```

O melhor modelo (menor **RMSE médio** nos folds) é registrado automaticamente no MLflow Model Registry.

---

## Monitoramento de drift (PSI)

O `batch_inference_flow` compara a distribuição das features da batch atual com a referência do treino:

| PSI | Interpretação |
|---|---|
| < 0.1 | Estável — sem ação necessária |
| 0.1 – 0.2 | Mudança leve — monitorar |
| ≥ 0.2 | Drift relevante — avaliar retreinamento |
