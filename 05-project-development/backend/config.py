import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    應用程式設定
    """
    # OpenAI API Key
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # LLM 模型
    LLM_MODEL_NAME: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.0

    # Embedding 模型
    EMBEDDING_MODEL: str = "text-embedding-ada-002"

    # 文檔路徑
    PAPERS_PATH: str = "/app/papers"

    # 文本分割
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Qdrant
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION_NAME: str = "rag_collection"

    # API 服務
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# 創建一個全局可用的設定實例
settings = Settings()
