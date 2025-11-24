#!/usr/bin/env python3
"""
Embedding 資料庫寫入邏輯 Pipeline
整合 OLMoCR 處理結果並寫入向量資料庫的完整流程
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingConfig(BaseSettings):
    """Embedding Pipeline 配置"""
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

    # OLMoCR 路徑設定
    OLMOCR_OUTPUT_PATH: str = "/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03-advanced-tools/olmocr/output/workspace"

    # 處理設定
    BATCH_SIZE: int = 50
    MAX_RETRIES: int = 3

class DocumentProcessor:
    """文檔處理器"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_olmocr_results(self) -> List[Dict[str, Any]]:
        """載入 OLMoCR 處理結果"""
        results = []
        results_dir = Path(self.config.OLMOCR_OUTPUT_PATH) / "results"

        if not results_dir.exists():
            logger.warning(f"OLMoCR 結果目錄不存在: {results_dir}")
            return results

        jsonl_files = list(results_dir.glob("output_*.jsonl"))
        logger.info(f"找到 {len(jsonl_files)} 個 OLMoCR 結果文件")

        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                data['source_file'] = jsonl_file.name
                                data['line_number'] = line_num
                                results.append(data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON 解析失敗 {jsonl_file}:{line_num} - {e}")

                logger.info(f"從 {jsonl_file.name} 載入了 {line_num} 筆記錄")
            except Exception as e:
                logger.error(f"讀取文件失敗 {jsonl_file}: {e}")

        logger.info(f"總共載入 {len(results)} 筆 OLMoCR 記錄")
        return results

    def create_document_id(self, content: str, source: str, chunk_index: int = 0) -> str:
        """創建唯一的文檔 ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"{source_hash}_{content_hash}_chunk_{chunk_index}"

    def process_documents(self, olmocr_results: List[Dict[str, Any]]) -> List[Document]:
        """將 OLMoCR 結果轉換為 LangChain 文檔"""
        documents = []

        for result in olmocr_results:
            try:
                text_content = result.get('text', '').strip()
                if not text_content or len(text_content) < self.config.MIN_CHUNK_SIZE:
                    continue

                # 基本元數據
                base_metadata = {
                    'source_file': result.get('source_file', 'unknown'),
                    'document_id': result.get('id', 'unknown'),
                    'line_number': result.get('line_number', 0),
                    'processed_by': 'olmocr',
                    'processed_date': datetime.now().isoformat(),
                    'content_length': len(text_content),
                    'processing_pipeline': 'embedding_pipeline_v1'
                }

                # 嘗試使用 Markdown 分割器
                try:
                    splits = self.text_splitter.split_text(text_content)
                    logger.debug(f"Markdown 分割: {len(splits)} 個塊")
                except Exception as e:
                    logger.warning(f"Markdown 分割失敗，使用備用分割器: {e}")
                    splits = self.fallback_splitter.split_text(text_content)

                # 處理每個文本塊
                for i, chunk in enumerate(splits):
                    chunk = chunk.strip()
                    if len(chunk) < self.config.MIN_CHUNK_SIZE:
                        continue

                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': self.create_document_id(chunk, base_metadata['source_file'], i),
                        'chunk_index': i,
                        'chunk_size': len(chunk),
                        'total_chunks': len(splits)
                    })

                    doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    documents.append(doc)

            except Exception as e:
                logger.error(f"處理文檔失敗: {e}")
                continue

        logger.info(f"生成了 {len(documents)} 個文檔塊")
        return documents

class VectorDBManager:
    """向量資料庫管理器"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)

    def ensure_collection_exists(self) -> bool:
        """確保 Qdrant 集合存在"""
        try:
            collection_exists = self.qdrant_client.collection_exists(self.config.QDRANT_COLLECTION_NAME)

            if collection_exists:
                logger.info(f"集合 {self.config.QDRANT_COLLECTION_NAME} 已存在")
                collection_info = self.qdrant_client.get_collection(self.config.QDRANT_COLLECTION_NAME)
                logger.info(f"現有集合向量數量: {collection_info.points_count}")
            else:
                logger.info(f"創建新集合: {self.config.QDRANT_COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=self.config.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI text-embedding-ada-002 向量維度
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfig(
                        default_segment_number=2,
                        max_segment_size=20000,
                        memmap_threshold=20000,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=1
                    )
                )
                logger.info("✅ 集合創建成功")

            return True

        except Exception as e:
            logger.error(f"集合操作失敗: {e}")
            return False

    def check_document_exists(self, document_id: str) -> bool:
        """檢查文檔是否已存在"""
        try:
            search_result = self.qdrant_client.scroll(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="chunk_id",
                            match=rest.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1
            )
            return len(search_result[0]) > 0
        except Exception as e:
            logger.warning(f"檢查文檔存在性失敗: {e}")
            return False

    def add_documents_batch(self, documents: List[Document]) -> int:
        """批量添加文檔到向量資料庫"""
        try:
            # 過濾已存在的文檔
            new_documents = []
            for doc in documents:
                doc_id = doc.metadata.get('chunk_id', '')
                if not self.check_document_exists(doc_id):
                    new_documents.append(doc)
                else:
                    logger.debug(f"文檔已存在，跳過: {doc_id}")

            if not new_documents:
                logger.info("沒有新文檔需要添加")
                return 0

            logger.info(f"準備添加 {len(new_documents)} 個新文檔")

            # 使用 LangChain Qdrant 包裝器
            vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                embeddings=self.embeddings,
            )

            # 批量處理
            added_count = 0
            batch_size = self.config.BATCH_SIZE

            for i in range(0, len(new_documents), batch_size):
                batch = new_documents[i:i + batch_size]
                retry_count = 0

                while retry_count < self.config.MAX_RETRIES:
                    try:
                        vector_store.add_documents(batch)
                        added_count += len(batch)
                        logger.info(f"成功添加批次 {i//batch_size + 1}: {len(batch)} 個文檔")
                        break
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"批次添加失敗 (嘗試 {retry_count}/{self.config.MAX_RETRIES}): {e}")
                        if retry_count >= self.config.MAX_RETRIES:
                            logger.error(f"批次 {i//batch_size + 1} 最終添加失敗")
                            continue

            # 驗證結果
            collection_info = self.qdrant_client.get_collection(self.config.QDRANT_COLLECTION_NAME)
            logger.info(f"📊 集合總向量數量: {collection_info.points_count}")

            return added_count

        except Exception as e:
            logger.error(f"批量添加文檔失敗: {e}")
            return 0

class EmbeddingPipeline:
    """完整的 Embedding Pipeline"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = EmbeddingConfig()
        self.processor = DocumentProcessor(self.config)
        self.vector_db = VectorDBManager(self.config)

        # 驗證配置
        if not self.config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未設定")

    def run_pipeline(self) -> Dict[str, Any]:
        """執行完整的 embedding pipeline"""
        results = {
            'success': False,
            'processed_documents': 0,
            'added_documents': 0,
            'errors': [],
            'start_time': datetime.now().isoformat()
        }

        try:
            logger.info("🚀 開始 Embedding Pipeline 執行...")

            # 1. 載入 OLMoCR 結果
            logger.info("📁 載入 OLMoCR 結果...")
            olmocr_results = self.processor.load_olmocr_results()

            if not olmocr_results:
                raise ValueError("沒有找到 OLMoCR 結果文件")

            # 2. 處理文檔
            logger.info("📝 處理文檔...")
            documents = self.processor.process_documents(olmocr_results)
            results['processed_documents'] = len(documents)

            if not documents:
                raise ValueError("沒有生成任何文檔")

            # 3. 確保向量資料庫集合存在
            logger.info("🗄️ 準備向量資料庫...")
            if not self.vector_db.ensure_collection_exists():
                raise ValueError("無法創建或訪問向量資料庫集合")

            # 4. 添加文檔到向量資料庫
            logger.info("💾 添加文檔到向量資料庫...")
            added_count = self.vector_db.add_documents_batch(documents)
            results['added_documents'] = added_count

            results['success'] = True
            results['end_time'] = datetime.now().isoformat()

            logger.info("🎉 Embedding Pipeline 執行完成！")
            logger.info(f"📊 處理統計: 處理文檔 {results['processed_documents']} 個，添加 {results['added_documents']} 個")

        except Exception as e:
            logger.error(f"❌ Pipeline 執行失敗: {e}")
            results['errors'].append(str(e))
            results['end_time'] = datetime.now().isoformat()

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """獲取集合統計信息"""
        try:
            collection_info = self.vector_db.qdrant_client.get_collection(self.config.QDRANT_COLLECTION_NAME)
            return {
                'collection_name': self.config.QDRANT_COLLECTION_NAME,
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"獲取集合統計失敗: {e}")
            return {}

def main():
    """主函數"""
    print("🧪 Embedding Pipeline - OLMoCR 向量資料庫寫入工具")
    print("=" * 70)

    try:
        # 創建並執行 pipeline
        pipeline = EmbeddingPipeline()

        # 顯示配置信息
        logger.info("🔧 Pipeline 配置:")
        logger.info(f"   - Qdrant URL: {pipeline.config.QDRANT_URL}")
        logger.info(f"   - 集合名稱: {pipeline.config.QDRANT_COLLECTION_NAME}")
        logger.info(f"   - Embedding 模型: {pipeline.config.EMBEDDING_MODEL}")
        logger.info(f"   - 文本塊大小: {pipeline.config.CHUNK_SIZE}")
        logger.info(f"   - 批次大小: {pipeline.config.BATCH_SIZE}")

        # 執行 pipeline
        results = pipeline.run_pipeline()

        # 顯示結果
        print("\n" + "=" * 50)
        if results['success']:
            print("🎉 Pipeline 執行成功！")
            print(f"📊 處理文檔: {results['processed_documents']} 個")
            print(f"💾 新增文檔: {results['added_documents']} 個")

            # 顯示集合統計
            stats = pipeline.get_collection_stats()
            if stats:
                print(f"🗄️ 資料庫統計:")
                print(f"   - 總向量數: {stats.get('total_points', 'N/A')}")
                print(f"   - 向量維度: {stats.get('vector_size', 'N/A')}")
        else:
            print("❌ Pipeline 執行失敗")
            for error in results['errors']:
                print(f"   錯誤: {error}")

        print(f"⏱️ 執行時間: {results.get('start_time', '')} - {results.get('end_time', '')}")

    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
        print(f"\n❌ 程式執行失敗: {e}")

if __name__ == "__main__":
    main()