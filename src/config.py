from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/config.py -> src/ -> repository root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    project_root: Path = Field(default=_PROJECT_ROOT)
    data_root: Path = Field(default=_PROJECT_ROOT / "data")
    logs_root: Path = Field(default=_PROJECT_ROOT / "logs")
    models_root: Path = Field(default=_PROJECT_ROOT / "models")

    grid_eso_api_base: str = Field(default="https://api.carbonintensity.org.uk")
    met_office_api_base: str = Field(
        default="https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point"
    )
    met_office_api_key: str = Field(default="")

    wandb_api_key: str = Field(default="")
    wandb_project: str = Field(default="gridpulse-tft")

    llm_provider: str = Field(default="deepseek")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="deepseek-chat")

    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)

    seed: int = Field(default=42)
    rate_limit_per_min: int = Field(default=60)
    llm_timeout_seconds: float = Field(default=5.0)
    drift_ks_threshold: float = Field(default=0.15)

    @property
    def raw_grid_root(self) -> Path:
        return self.data_root / "raw" / "grid"

    @property
    def raw_weather_root(self) -> Path:
        return self.data_root / "raw" / "weather"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    def validate_required(self) -> list[str]:
        missing = []
        required = [
            ("MET_OFFICE_API_KEY", self.met_office_api_key),
            ("WANDB_API_KEY", self.wandb_api_key),
            ("LLM_API_KEY", self.llm_api_key),
        ]
        for name, value in required:
            if not value:
                missing.append(name)
        return missing


settings = Settings()
