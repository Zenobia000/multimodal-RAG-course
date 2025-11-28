import os
import sys
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 從同級目錄導入
from config import Settings

# 導入已有的 embedding 管道功能
embedding_app_path = Path(__file__).parent.parent.parent / "04-embedding-application"
if str(embedding_app_path) not in sys.path:
    sys.path.append(str(embedding_app_path))

try:
    from embedding_pipeline import EmbeddingPipeline
except ImportError as e:
    logging.warning(f"無法導入 embedding_pipeline: {e}")
    EmbeddingPipeline = None

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def elements_to_markdown(elements: List) -> str:
    """將 unstructured 元素轉換為 Markdown 格式"""
    markdown_lines = []
    for element in elements:
        elem_type = element.category
        text = element.text.strip()
        
        if not text:
            continue
        
        if elem_type == "Title":
            markdown_lines.append(f"## {text}\n")
        elif elem_type == "ListItem":
            markdown_lines.append(f"- {text}\n")
        else:
            markdown_lines.append(f"{text}\n")
    
    return "\n".join(markdown_lines)

class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_store = None
        self.qa_chain = None
        self.embedding_pipeline = None

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model_name=self.settings.LLM_MODEL_NAME,
            temperature=self.settings.LLM_TEMPERATURE
        )

        # 初始化 Qdrant 客戶端
        self.qdrant_client = QdrantClient(url=self.settings.QDRANT_URL)

        # 初始化 embedding pipeline (如果可用)
        if EmbeddingPipeline:
            try:
                # 從 04-embedding-application 載入配置
                embedding_config_path = Path(__file__).parent.parent.parent / "04-embedding-application" / "config.py"
                if embedding_config_path.exists():
                    self.embedding_pipeline = EmbeddingPipeline()
                    logger.info("✅ 成功載入 embedding pipeline")
                else:
                    logger.warning("⚠️ embedding pipeline 配置文件未找到")
            except Exception as e:
                logger.warning(f"⚠️ 初始化 embedding pipeline 失敗: {e}")
        else:
            logger.warning("⚠️ EmbeddingPipeline 類不可用，將使用基礎功能")

    async def init_rag(self) -> bool:
        """
        初始化RAG系統：載入文檔、建立向量索引和QA鏈。
        如果索引已存在，則直接載入。
        優先使用 04-embedding-application 的功能。
        """
        try:
            collection_exists = self.qdrant_client.collection_exists(self.settings.QDRANT_COLLECTION_NAME)

            if collection_exists:
                logger.info(f"從 Qdrant 載入現有集合: {self.settings.QDRANT_COLLECTION_NAME}")

                # 如果有 embedding pipeline，使用它的 embeddings
                if self.embedding_pipeline and hasattr(self.embedding_pipeline, 'embeddings'):
                    embeddings = self.embedding_pipeline.embeddings
                    logger.info("✅ 使用 embedding pipeline 的 embeddings")
                else:
                    # 回退到基本的 OpenAI embeddings
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(model=self.settings.EMBEDDING_MODEL)
                    logger.info("⚠️ 使用基本的 OpenAI embeddings")

                self.vector_store = Qdrant(
                    client=self.qdrant_client,
                    collection_name=self.settings.QDRANT_COLLECTION_NAME,
                    embeddings=embeddings,
                )
                logger.info("✅ 索引成功載入")
            else:
                logger.info("找不到現有索引，建議使用 04-embedding-application 建立索引")
                return False

            # 創建QA鏈
            template = """你是一個問答助手。根據以下檢索到的文檔內容來回答問題。
如果檢索到的文檔中沒有相關信息，請明確說明。

Context: {context}

Question: {question}

Answer:"""

            prompt = ChatPromptTemplate.from_template(template)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self.qa_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            logger.info("✅ RAG服務初始化成功")
            return True

        except Exception as e:
            logger.error(f"RAG服務初始化失敗: {e}", exc_info=True)
            return False

    def use_embedding_pipeline_search(self, question: str, top_k: int = 3) -> Dict:
        """
        使用 embedding pipeline 的搜索功能 (如果可用)
        """
        if not self.embedding_pipeline:
            raise RuntimeError("Embedding pipeline 不可用")

        try:
            # 使用 embedding pipeline 的搜索功能
            # 假設 embedding pipeline 有類似的搜索方法
            if hasattr(self.embedding_pipeline, 'search_documents'):
                results = self.embedding_pipeline.search_documents(question, top_k=top_k)
                return results
            else:
                logger.warning("Embedding pipeline 沒有 search_documents 方法，回退到基本搜索")
                return None
        except Exception as e:
            logger.error(f"使用 embedding pipeline 搜索失敗: {e}")
            return None

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        執行RAG查詢。
        優先嘗試使用 embedding pipeline 功能，回退到基本 RAG 功能。
        """
        # 嘗試使用 embedding pipeline
        if self.embedding_pipeline:
            pipeline_result = self.use_embedding_pipeline_search(question, top_k)
            if pipeline_result:
                logger.info("✅ 使用 embedding pipeline 搜索結果")
                return pipeline_result

        # 回退到基本 RAG 功能
        if not self.qa_chain:
            raise RuntimeError("RAG系統未初始化，無法執行查詢。")

        try:
            logger.info("⚠️ 使用基本 RAG 搜索")
            # 更新檢索的 top_k 參數
            self.retriever.search_kwargs = {"k": top_k}

            # 獲取答案和源文檔
            answer = self.qa_chain.invoke(question)
            source_docs = self.retriever.invoke(question)

            # 格式化來源文檔
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:500] + "...",
                    "metadata": doc.metadata
                })

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"執行查詢時出錯: {e}", exc_info=True)
            raise RuntimeError(f"查詢失敗: {str(e)}")
