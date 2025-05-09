from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ICD_DB_PATH: str = "data/icd10/icd10_am.sqlite"
    OPENAI_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")