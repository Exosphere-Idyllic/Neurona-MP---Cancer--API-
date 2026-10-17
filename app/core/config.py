from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "API Cáncer de mama (MP)"
    model_path: Path = Path("artifacts/model.json")

settings = Settings()
