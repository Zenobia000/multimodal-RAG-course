#!/usr/bin/env python3
"""
Embedding Application 配置文件
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

class EmbeddingAppSettings(BaseSettings):
    """Embedding 應用程式設定"""

    # OpenAI 設定
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = "text-embedding-ada-002"

    # Qdrant 設定
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "olmocr_documents"

    # 文本處理設定
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MIN_CHUNK_SIZE: int = 50

    # 路徑設定
    CURRENT_DIR: Path = Path(__file__).parent
    OLMOCR_OUTPUT_PATH: str = str(CURRENT_DIR.parent / "03.1-papers_knowledge_base" / "outputs" / "aggregated_chunks")

    # 處理設定
    BATCH_SIZE: int = 50
    MAX_RETRIES: int = 3

    # 日誌設定
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(CURRENT_DIR / "logs" / "embedding_app.log")

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# 創建全局設定實例
settings = EmbeddingAppSettings()

# 支援的文檔類型
SUPPORTED_DOCUMENT_TYPES = {
    'olmocr_jsonl': {
        'extensions': ['.jsonl'],
        'description': 'OLMoCR 處理結果文件',
        'processor': 'olmocr_processor'
    },
    'markdown': {
        'extensions': ['.md'],
        'description': 'Markdown 文檔',
        'processor': 'markdown_processor'
    },
    'text': {
        'extensions': ['.txt'],
        'description': '純文本文檔',
        'processor': 'text_processor'
    }
}

# 向量資料庫配置
VECTOR_DB_CONFIG = {
    'qdrant': {
        'vector_size': 1536,  # OpenAI ada-002 embedding size
        'distance_metric': 'cosine',
        'collection_config': {
            'optimizers_config': {
                'default_segment_number': 2,
                'max_segment_size': 20000,
                'memmap_threshold': 20000,
                'indexing_threshold': 20000,
                'flush_interval_sec': 5,
                'max_optimization_threads': 1
            }
        }
    }
}

# 文本處理策略
TEXT_PROCESSING_STRATEGIES = {
    'markdown': {
        'primary_splitter': 'MarkdownTextSplitter',
        'fallback_splitter': 'RecursiveCharacterTextSplitter',
        'chunk_size': 1000,
        'chunk_overlap': 200
    },
    'academic_paper': {
        'primary_splitter': 'RecursiveCharacterTextSplitter',
        'separators': ["\n\n", "\n", ". ", " ", ""],
        'chunk_size': 1500,
        'chunk_overlap': 300
    },
    'general_text': {
        'primary_splitter': 'RecursiveCharacterTextSplitter',
        'separators': ["\n\n", "\n", ". ", " ", ""],
        'chunk_size': 1000,
        'chunk_overlap': 200
    }
}