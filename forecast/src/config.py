import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "forecast_estoque_walmart")
    HORIZON_DAYS: int = int(os.getenv("HORIZON_DAYS", "28"))

    COST_UNDER: float = float(os.getenv("COST_UNDER", "6.0"))
    COST_OVER: float = float(os.getenv("COST_OVER", "1.5"))

    DATA_RAW_DIR: str = os.getenv("DATA_RAW_DIR", "data/raw")
    DATA_PROCESSED_DIR: str = os.getenv("DATA_PROCESSED_DIR", "data/processed")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")


settings = Settings()
