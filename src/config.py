from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Redis connection
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None

    # Embedding model (public HF model, no auth required)
    EMBED_MODEL: str = "Alibaba-NLP/gte-modernbert-base"
    EMBED_DIM: int = 768
    EMBED_MAX_LENGTH: int = 2048
    EMBED_BATCH_SIZE: int = 4

    # Chunking parameters
    CODE_CHUNK_LINES: int = 80
    CODE_OVERLAP_LINES: int = 20
    DOC_CHUNK_SIZE: int = 512  # approximate token count

    # Search defaults
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 20

    # HNSW index parameters for RediSearch
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_RUNTIME: int = 10

    # Ingestion filtering: skip files matching these patterns
    # Comma-separated patterns (e.g., "*.test.py,*_test.py,conftest.py")
    SKIP_FILES: str = "*.test.py,*_test.py,test_*.py,conftest.py,*.min.js,*.min.css,*.map"

    # API server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Security: ingestion endpoints validate all paths are within BASE_PATH
    BASE_PATH: str = "/data"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    @property
    def skip_files_patterns(self) -> list[str]:
        """Parse SKIP_FILES into a list of glob patterns."""
        return [p.strip() for p in self.SKIP_FILES.split(",") if p.strip()]


settings = Settings()
