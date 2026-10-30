import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator


class Settings(BaseModel):
    base_url: str = ""
    api_key: SecretStr = SecretStr("")
    model: str = ""
    model_timeout: float = Field(default=45, gt=0, le=300)
    retrieval_mode: str = "hybrid"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_revision: str = "7999e1d3359715c523056ef9478215996d62a620"
    embedding_local_only: bool = False
    vector_min_score: float = Field(default=0.45, ge=-1, le=1)
    max_model_steps: int = Field(default=8, ge=1, le=100)
    max_tool_calls: int = Field(default=12, ge=1, le=200)
    tool_timeout: float = Field(default=15, gt=0, le=300)
    data_path: Path = Path("data/documents.json")
    cache_dir: Path = Path(".cache")
    vector_backend: str = "qdrant"
    vector_db_path: Path = Path(".cache/qdrant")

    @field_validator("vector_backend")
    @classmethod
    def valid_backend(cls, value):
        if value not in {"qdrant", "numpy"}:
            raise ValueError("VECTOR_BACKEND 必须是 qdrant 或 numpy")
        return value

    @field_validator("retrieval_mode")
    @classmethod
    def valid_mode(cls, value):
        if value not in {"hybrid", "bm25"}:
            raise ValueError("RETRIEVAL_MODE 必须是 hybrid 或 bm25")
        return value

    def require_live_model(self):
        url = urlparse(self.base_url)
        if (
            not self.model
            or not self.api_key.get_secret_value()
            or url.scheme not in {"http", "https"}
            or not url.netloc
        ):
            raise ValueError("请在本地 .env 配置 LLM_BASE_URL、LLM_API_KEY 和 LLM_MODEL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("LLM_BASE_URL 不允许包含凭据、查询参数或 fragment")
        if self.base_url.rstrip("/").endswith("/chat/completions"):
            raise ValueError("LLM_BASE_URL 应为 API 根地址，不包含 /chat/completions")

    @classmethod
    def from_env(cls):
        load_dotenv(override=False)
        mapping = {
            "base_url": "LLM_BASE_URL",
            "api_key": "LLM_API_KEY",
            "model": "LLM_MODEL",
            "model_timeout": "LLM_TIMEOUT_SECONDS",
            "retrieval_mode": "RETRIEVAL_MODE",
            "embedding_model": "EMBEDDING_MODEL",
            "embedding_revision": "EMBEDDING_REVISION",
            "embedding_local_only": "EMBEDDING_LOCAL_ONLY",
            "vector_min_score": "VECTOR_MIN_SCORE",
            "max_model_steps": "MAX_MODEL_STEPS",
            "max_tool_calls": "MAX_TOOL_CALLS",
            "tool_timeout": "TOOL_TIMEOUT_SECONDS",
            "data_path": "DATA_PATH",
            "cache_dir": "CACHE_DIR",
            "vector_backend": "VECTOR_BACKEND",
            "vector_db_path": "VECTOR_DB_PATH",
        }
        return cls(**{key: os.environ[env] for key, env in mapping.items() if os.environ.get(env)})
