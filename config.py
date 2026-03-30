from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    anthropic_api_key: str # Set this in your .env file as ANTHROPIC_API_KEYs

    chroma_persist_dir: str = "data/chroma"
    sqlite_path: str = "data/stocks.db"
    pdf_folder: str = "data/pdfs"
    embed_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        for path in (self.chroma_persist_dir, self.pdf_folder):
            Path(path).mkdir(parents=True, exist_ok=True)
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
