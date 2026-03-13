import os
from pathlib import Path


class Settings:

     # diretórios de dados (relativos ao projeto)
    DATA_RAW_DIR: str = "data/raw"
    DATA_PROCESSED_DIR: str = "data/processed"

    ARTIFACTS_DIR: str = "artifacts"
    REPORTS_DIR: str = "reports"

    HORIZON_DAYS: int = int(os.getenv("HORIZON_DAYS", 28))
    APP_ENV: str = os.getenv("APP_ENV", "dev")

    # parâmetros diferentes para dev e prod
    if APP_ENV == "dev":
        SAMPLE_ROWS = 100_000
        LGBM_ROUNDS = 300
        EARLY_STOPPING = 30
    else:
        SAMPLE_ROWS = None
        LGBM_ROUNDS = 3000
        EARLY_STOPPING = 100


settings = Settings()