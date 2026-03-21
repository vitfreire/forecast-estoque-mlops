# Retail Forecast Intelligence — Previsão de Demanda com MLOps

![CI](https://github.com/vitfreire/forecast-estoque-mlops/actions/workflows/ci.yml/badge.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://forecas-tml.streamlit.app)

Sistema end-to-end de previsão de vendas semanais para varejo, construído com foco em MLOps real: treinamento multi-modelo com comparação automática, rastreamento completo de experimentos via MLflow, orquestração de pipeline com Prefect, otimização de estoque baseada em custo assimétrico, monitoramento de drift em produção e dashboard analítico interativo.

---

## Interface

> Dashboard interativo com 10 abas — da visão executiva ao monitoramento de modelos.

### Resumo Executivo
![Resumo Executivo](docs/screenshots/01_resumo_executivo.png)

### Performance Comercial
![Performance Comercial](docs/screenshots/02_performance_comercial.png)

### Forecast Analysis
![Forecast Analysis](docs/screenshots/03_forecast_analysis.png)

### Store & Department Insights — Heatmap de Erro
![Heatmap Loja × Departamento](docs/screenshots/04_heatmap.png)

### Model Leaderboard
![Leaderboard](docs/screenshots/05_leaderboard.png)

### Model Monitoring — Drift do Erro
![Drift do Erro](docs/screenshots/06_monitoring.png)

---

## O que este projeto faz

O problema central é prever, com antecedência, quanto cada departamento de cada loja vai vender na próxima semana. Errar para baixo significa ruptura de estoque — vendas perdidas, experiência do cliente prejudicada. Errar para cima significa excesso de estoque parado, custo de armazenagem e capital imobilizado. O custo desses dois tipos de erro é diferente, e o projeto trata isso explicitamente.

O pipeline completo faz:

1. **Engenharia de features** — constrói variáveis de calendário (semana do ano, dia da semana, componentes cíclicos seno/cosseno para sazonalidade), histórico de vendas (lags 1–4 semanas), estatísticas móveis (médias e desvios em janelas de 4, 8 e 12 semanas), contexto macroeconômico (temperatura, preço do combustível, CPI, desemprego) e contexto promocional (markdowns, feriados).

2. **Treinamento multi-modelo** — treina em paralelo LightGBM, XGBoost e Random Forest no mesmo conjunto de dados, registra cada modelo em runs aninhados no MLflow com suas métricas (RMSE, MAE, SMAPE), hiperparâmetros e feature importances. O melhor modelo por RMSE é promovido automaticamente ao MLflow Model Registry.

3. **Avaliação rigorosa** — suporta dois modos: holdout temporal (corte fixo) e rolling time series cross-validation com janela expansiva, preservando a ordem cronológica dos dados para evitar vazamento de informação do futuro.

4. **Otimização de custo assimétrico** — a função de custo penaliza subprevisão 4× mais do que superprevisão (configurável via variáveis de ambiente), refletindo a assimetria real entre ruptura de estoque e excesso de mercadoria.

5. **Inferência em batch** — carrega o melhor modelo do Registry (Production > Staging > versão mais recente), gera previsões para o horizonte configurado e salva para consumo pelo dashboard e outros sistemas.

6. **Monitoramento de drift** — calcula PSI (Population Stability Index) comparando a distribuição de features da batch atual com a distribuição de referência do treino. Features com PSI > 0.2 são sinalizadas como drift relevante.

7. **Dashboard analítico** — interface Streamlit com 10 abas cobrindo: resumo executivo, performance comercial, análise de forecast, impacto financeiro (ruptura + excesso), insights por loja e departamento, explicabilidade do modelo (feature importance), leaderboard de modelos via MLflow, inteligência do pipeline e monitoramento de anomalias.

---

## Stack

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
| Ambiente | Python 3.11, venv |

---

## Pré-requisitos

- Python 3.10 ou superior
- Dataset Walmart disponível em `data/raw/` (arquivos `train.csv`, `features.csv`, `stores.csv`)
  - Fonte: [Kaggle — Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)

---

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/vitfreire/forecast-estoque-mlops.git
cd forecast-estoque-mlops

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env conforme necessário
```

---

## Variáveis de ambiente

Copie `.env.example` para `.env` e ajuste os valores:

```env
# MLflow — onde os experimentos são rastreados
# Local (sem servidor):  file:./mlruns
# Servidor local:        http://127.0.0.1:5000
# DagsHub:               https://dagshub.com/<usuario>/<repo>.mlflow
MLFLOW_TRACKING_URI=file:./mlruns

# Nome do experimento no MLflow
MLFLOW_EXPERIMENT_NAME=forecast_estoque_walmart

# Nome do modelo no MLflow Model Registry
MLFLOW_REGISTERED_MODEL_NAME=walmart_forecast_lgbm_cost

# Horizonte de previsão em dias
HORIZON_DAYS=28

# Modelos a treinar (separados por vírgula)
# Opções: lightgbm, xgboost, random_forest
RUN_MODELS=lightgbm,xgboost,random_forest

# Modo de avaliação: holdout ou rolling
EVAL_MODE=holdout

# Custo assimétrico de erro (ajusta a função objetivo)
COST_UNDER=6.0   # penalidade por subprevisão (ruptura)
COST_OVER=1.5    # penalidade por superprevisão (excesso)
```

---

## Como rodar

### 1. Pipeline de treinamento

```bash
# Modo desenvolvimento (dataset reduzido, 5 lojas, 5 departamentos)
APP_ENV=dev python -m flows.training_flow

# Modo produção (dataset completo)
python -m flows.training_flow

# Treinar modelos específicos
RUN_MODELS=lightgbm,xgboost APP_ENV=dev python -m flows.training_flow

# Usar rolling cross-validation
EVAL_MODE=rolling APP_ENV=dev python -m flows.training_flow
```

O treinamento:
- Lê os CSVs de `data/raw/`
- Constrói o dataset com todas as features
- Treina cada modelo configurado em `RUN_MODELS`
- Registra experimentos no MLflow (métricas, params, artifacts, feature importances)
- Promove o melhor modelo ao Model Registry
- Salva o dataset processado em `data/processed/dataset.parquet`

### 2. Visualizar experimentos no MLflow

```bash
# Inicia a interface do MLflow na porta 5000
mlflow ui --port 5000
# Acesse: http://localhost:5000
```

### 3. Inferência em batch

```bash
python -m flows.batch_inference_flow
```

Gera previsões usando o melhor modelo do Registry e salva em `reports/batch_predictions.parquet`.

### 4. Pipeline completo (treino + inferência)

```bash
python -m flows.master_flow
```

### 5. Dashboard

```bash
streamlit run dashboard/app.py
```

O dashboard carrega automaticamente as previsões de `reports/batch_predictions.parquet` e os artifacts de `artifacts/`. Conecta ao MLflow para exibir o leaderboard de modelos.

---

## Testes

O projeto possui 47 testes unitários cobrindo as camadas críticas do pipeline:

```bash
# Instalar dependências de teste (se necessário)
pip install pytest

# Rodar todos os testes
pytest tests/ -v
```

| Módulo testado | Testes | O que verifica |
|---|---|---|
| `src/metrics.py` | 9 | MAE, RMSE, SMAPE — valores conhecidos e propriedades matemáticas |
| `src/cost.py` | 5 | Função de custo assimétrico — subprevisão é mais cara que superprevisão |
| `src/features/features.py` | 13 | Features de calendário, lags, rolling stats — ausência de leakage |
| `src/split.py` | 10 | Temporal split e rolling CV — sem vazamento, shapes corretos |
| `src/models/preprocessing.py` | 10 | Encoding, alinhamento de colunas entre treino e validação |

---

## Docker

### Rodar com Docker Compose (recomendado)

```bash
# Sobe MLflow server + dashboard
docker compose up mlflow dashboard

# Dashboard:  http://localhost:8501
# MLflow UI:  http://localhost:5000
```

### Rodar o pipeline de treinamento via Docker

```bash
# Modo dev (rápido, 5 lojas)
APP_ENV=dev docker compose run --rm training

# Modo produção
docker compose run --rm training
```

### Build manual

```bash
docker build -t forecast-intelligence .
docker run -p 8501:8501 forecast-intelligence
```

---

## Estrutura do projeto

```
.
├── src/                        # Biblioteca central
│   ├── config.py               # Settings (Pydantic) com modo dev/prod
│   ├── io.py                   # Leitura e escrita de dados
│   ├── split.py                # Splits temporais e rolling CV
│   ├── metrics.py              # MAE, RMSE, SMAPE
│   ├── cost.py                 # Função de custo assimétrica
│   ├── predict.py              # Pipeline de inferência com alinhamento de features
│   ├── mlflow_utils.py         # Setup do MLflow
│   ├── baseline.py             # Baseline sazonal (naive)
│   ├── model_selection.py      # Protocolos e metadados de modelos
│   ├── features/
│   │   ├── features.py         # Engenharia de features (lags, rolling, calendário)
│   │   └── prepare.py          # Orquestração da preparação
│   ├── models/
│   │   ├── base.py             # Protocolo BaseModel
│   │   ├── preprocessing.py    # TabularPreprocessor (encoding, alinhamento)
│   │   ├── registry.py         # Registro dinâmico de modelos
│   │   ├── trainer.py          # Orquestração de treino multi-modelo + MLflow
│   │   ├── lgbm.py             # LightGBM
│   │   ├── xgb.py              # XGBoost
│   │   └── rf.py               # Random Forest
│   ├── data/
│   │   └── time_series_cv.py   # Rolling folds com expanding window
│   └── monitoring/
│       └── drift.py            # Detecção de drift via PSI
├── flows/                      # Pipelines Prefect
│   ├── training_flow.py        # Treino (holdout ou rolling CV)
│   ├── batch_inference_flow.py # Inferência em batch + drift
│   └── master_flow.py          # Orquestra treino → inferência
├── dashboard/
│   ├── app.py                  # Interface Streamlit (10 abas)
│   └── mlflow_reader.py        # Helpers de leitura do MLflow
├── data/
│   ├── raw/                    # CSVs originais do Kaggle (não versionados)
│   └── processed/              # Dataset processado (não versionado)
├── tests/                      # Testes unitários (pytest)
│   ├── test_metrics.py
│   ├── test_cost.py
│   ├── test_features.py
│   ├── test_split.py
│   └── test_preprocessing.py
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions — lint + testes automáticos
├── artifacts/                  # Artifacts do pipeline (feature importances, leaderboard)
├── mlruns/                     # Banco de dados local do MLflow (não versionado)
├── Dockerfile
├── docker-compose.yml
├── .env.example                # Template de variáveis de ambiente
├── pyproject.toml
└── requirements.txt
```

---

## Modelos

| Modelo | Implementação | Características |
|---|---|---|
| LightGBM | `src/models/lgbm.py` | Gradient boosting em árvores, early stopping, log via `mlflow.lightgbm` |
| XGBoost | `src/models/xgb.py` | Gradient boosting com `tree_method=hist`, log via `mlflow.xgboost` |
| Random Forest | `src/models/rf.py` | Ensemble de árvores, sem boosting, log via `mlflow.sklearn` |

Todos os modelos implementam o protocolo `BaseModel` com os métodos `fit()`, `predict()` e `log_to_mlflow()`, o que permite adicionar novos modelos sem modificar o orquestrador de treino.

---

## Como funciona o MLflow

Cada execução de `training_flow` cria:

```
run: train_compare (run pai)
├── params: modelos treinados, n_features, linhas, hash do dataset
├── métricas: best_rmse, avg_rmse, std_rmse (e equivalentes para MAE/SMAPE)
├── artifacts: leaderboard.csv, features.json, preprocess.json, amostras de dados
└── runs filhos (um por modelo)
    ├── lightgbm/
    │   ├── métricas: rmse, mae, smape, train_seconds
    │   ├── hiperparâmetros do modelo
    │   ├── artifact: model/ (registrado no Model Registry)
    │   └── artifact: feature_importance/ (CSV + PNG)
    ├── xgboost/
    └── random_forest/
```

O melhor modelo (menor RMSE) é registrado automaticamente no MLflow Model Registry com o nome configurado em `MLFLOW_REGISTERED_MODEL_NAME`.

---

## Monitoramento de drift

O `batch_inference_flow` compara a distribuição das features da batch atual com a distribuição de referência registrada no treino, usando PSI:

- **PSI < 0.1** — estável
- **0.1 ≤ PSI < 0.2** — mudança leve, monitorar
- **PSI ≥ 0.2** — drift relevante, avaliar retreinamento

---

## Dataset

O projeto usa o dataset público **Walmart Store Sales Forecasting** do Kaggle:

- `train.csv` — vendas semanais por loja e departamento (2010–2012)
- `features.csv` — temperatura, combustível, markdowns, CPI, desemprego, feriados
- `stores.csv` — tipo e tamanho de cada loja

Os arquivos de dados brutos não são versionados no Git. Use a Kaggle API ou download manual:

```bash
kaggle competitions download -c walmart-recruiting-store-sales-forecasting
unzip walmart-recruiting-store-sales-forecasting.zip -d data/raw/
```
