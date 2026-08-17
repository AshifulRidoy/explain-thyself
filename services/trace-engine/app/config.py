"""Engine settings — ETS_* environment variables (see root .env.example)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ETS_", env_file=".env", extra="ignore")

    model: str = "gpt2-small"
    # auto | mps | cpu
    device: str = "auto"
    # off skips the ~2s CPU numerics self-check at model load
    self_check: str = "on"
    cors_origins: str = "http://localhost:3000"
    database_url: str = "postgresql://ets:ets@localhost:5432/explain_the_self"
    # hard safety cap on total context (prompt + generation); no KV reuse in MVP
    max_context: int = 256
    # SSE heartbeat cadence
    heartbeat_s: float = 15.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
