#!/usr/bin/env python3
"""
OLMoCR 整合腳本
將 OLMoCR 處理過的 PDF 文件結果載入到 Qdrant 向量資料庫中
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models

from config import Settings

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OLMoCRIntegration:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        # Use localhost instead of docker hostname when running outside container
        qdrant_url = settings.QDRANT_URL.replace("qdrant:6333", "localhost:6333")
        self.qdrant_client = QdrantClient(url=qdrant_url)

        # OLMoCR 輸出路徑
        self.olmocr_output_path = Path("/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03-advanced-tools/olmocr/output/workspace")

    def load_olmocr_results(self) -> List[Dict[str, Any]]:
        """載入所有 OLMoCR 處理結果"""
        results = []
        results_dir = self.olmocr_output_path / "results"

        if not results_dir.exists():
            logger.warning(f"OLMoCR 結果目錄不存在: {results_dir}")
            return results

        # 查找所有 .jsonl 文件
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
                                # 添加文件信息
                                data['source_file'] = jsonl_file.name
                                data['line_number'] = line_num
                                results.append(data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"解析 JSON 失敗 {jsonl_file}:{line_num} - {e}")
                logger.info(f"從 {jsonl_file.name} 載入了 {line_num} 筆記錄")
            except Exception as e:
                logger.error(f"讀取文件失敗 {jsonl_file}: {e}")

        return results

    def process_olmocr_documents(self, results: List[Dict[str, Any]]) -> List[Document]:
        """將 OLMoCR 結果轉換為 LangChain 文檔"""
        documents = []

        for result in results:
            try:
                # 提取文本內容
                text_content = result.get('text', '')
                if not text_content or len(text_content.strip()) < 10:
                    continue

                # 創建元數據
                metadata = {
                    'source': result.get('source_file', 'unknown'),
                    'document_id': result.get('id', 'unknown'),
                    'line_number': result.get('line_number', 0),
                    'processed_by': 'olmocr',
                    'processed_date': datetime.now().isoformat(),
                    'content_length': len(text_content),
                }

                # 分割長文本
                splits = self.text_splitter.split_text(text_content)

                for i, chunk in enumerate(splits):
                    if len(chunk.strip()) < 10:  # 跳過太短的塊
                        continue

                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': f"{metadata['document_id']}_chunk_{i}",
                        'chunk_index': i,
                        'chunk_size': len(chunk)
                    })

                    doc = Document(
                        page_content=chunk.strip(),
                        metadata=chunk_metadata
                    )
                    documents.append(doc)

            except Exception as e:
                logger.error(f"處理文檔失敗: {e}")
                continue

        logger.info(f"生成了 {len(documents)} 個文檔塊")
        return documents

    def create_qdrant_collection(self):
        """創建或更新 Qdrant 集合"""
        try:
            # 檢查集合是否已存在
            collection_exists = self.qdrant_client.collection_exists(self.settings.QDRANT_COLLECTION_NAME)

            if collection_exists:
                logger.info(f"集合 {self.settings.QDRANT_COLLECTION_NAME} 已存在")
                # 獲取集合信息
                collection_info = self.qdrant_client.get_collection(self.settings.QDRANT_COLLECTION_NAME)
                logger.info(f"現有集合向量數量: {collection_info.points_count}")
            else:
                logger.info(f"創建新集合: {self.settings.QDRANT_COLLECTION_NAME}")
                # 創建新集合，使用與 OpenAI embeddings 相容的向量大小
                self.qdrant_client.create_collection(
                    collection_name=self.settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI text-embedding-ada-002 的向量維度
                        distance=models.Distance.COSINE
                    )
                )
                logger.info("✅ 集合創建成功")

        except Exception as e:
            logger.error(f"創建集合失敗: {e}")
            raise

    def add_documents_to_qdrant(self, documents: List[Document]):
        """將文檔添加到 Qdrant"""
        try:
            logger.info(f"開始將 {len(documents)} 個文檔添加到 Qdrant...")

            # 使用 LangChain Qdrant 包裝器
            vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.settings.QDRANT_COLLECTION_NAME,
                embeddings=self.embeddings,
            )

            # 批量添加文檔
            batch_size = 50  # 批次大小
            added_count = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    # 添加文檔批次
                    vector_store.add_documents(batch)
                    added_count += len(batch)
                    logger.info(f"已添加 {added_count}/{len(documents)} 個文檔...")
                except Exception as e:
                    logger.error(f"添加批次失敗 (索引 {i}-{i+len(batch)}): {e}")
                    continue

            logger.info(f"✅ 成功添加 {added_count} 個文檔到 Qdrant")

            # 驗證結果
            collection_info = self.qdrant_client.get_collection(self.settings.QDRANT_COLLECTION_NAME)
            logger.info(f"📊 集合總向量數量: {collection_info.points_count}")

        except Exception as e:
            logger.error(f"添加文檔到 Qdrant 失敗: {e}")
            raise

    def integrate_olmocr_data(self):
        """執行完整的 OLMoCR 資料整合"""
        try:
            logger.info("🚀 開始 OLMoCR 資料整合...")

            # 1. 載入 OLMoCR 結果
            logger.info("📁 載入 OLMoCR 結果...")
            results = self.load_olmocr_results()

            if not results:
                logger.warning("❌ 沒有找到 OLMoCR 結果文件")
                return False

            logger.info(f"✅ 載入了 {len(results)} 筆 OLMoCR 記錄")

            # 2. 處理文檔
            logger.info("📝 處理文檔...")
            documents = self.process_olmocr_documents(results)

            if not documents:
                logger.warning("❌ 沒有生成任何文檔")
                return False

            # 3. 確保 Qdrant 集合存在
            logger.info("🗄️ 準備 Qdrant 集合...")
            self.create_qdrant_collection()

            # 4. 添加到向量資料庫
            logger.info("💾 添加到向量資料庫...")
            self.add_documents_to_qdrant(documents)

            logger.info("🎉 OLMoCR 資料整合完成！")
            return True

        except Exception as e:
            logger.error(f"❌ 整合失敗: {e}")
            return False

def main():
    """主函數"""
    print("🧪 OLMoCR 向量資料庫整合工具")
    print("=" * 60)

    try:
        # 初始化設定
        settings = Settings()

        # 檢查必要的環境變數
        if not settings.OPENAI_API_KEY:
            logger.error("❌ 請設定 OPENAI_API_KEY 環境變數")
            return

        logger.info(f"🔧 配置資訊:")
        logger.info(f"   - Qdrant URL: {settings.QDRANT_URL}")
        logger.info(f"   - 集合名稱: {settings.QDRANT_COLLECTION_NAME}")
        logger.info(f"   - Embedding 模型: {settings.EMBEDDING_MODEL}")
        logger.info(f"   - 文本塊大小: {settings.CHUNK_SIZE}")

        # 執行整合
        integration = OLMoCRIntegration(settings)
        success = integration.integrate_olmocr_data()

        if success:
            print("\n🎉 整合成功完成！")
            print("📊 現在可以使用 RAG 系統查詢 OLMoCR 處理過的文檔內容")
        else:
            print("\n❌ 整合失敗")

    except Exception as e:
        logger.error(f"執行失敗: {e}")
        print(f"\n❌ 程式執行失敗: {e}")

if __name__ == "__main__":
    main()