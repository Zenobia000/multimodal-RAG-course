import os
import logging
from typing import Dict, List
from datetime import datetime

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 從同級目錄導入
from config import Settings

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
        self.qa_chain = None
        self.retriever = None

        # 初始化核心組件
        self.embeddings = OpenAIEmbeddings(model=self.settings.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(
            model_name=self.settings.LLM_MODEL_NAME,
            temperature=self.settings.LLM_TEMPERATURE
        )
        self.qdrant_client = QdrantClient(url=self.settings.QDRANT_URL)

        logger.info("✅ RAG 服務核心組件初始化完成")

    async def init_rag(self) -> bool:
        """
        初始化RAG系統：檢查 Qdrant 中是否有數據並建立 QA 鏈。
        """
        try:
            # 檢查集合是否存在
            collection_exists = self.qdrant_client.collection_exists(self.settings.QDRANT_COLLECTION_NAME)

            if not collection_exists:
                logger.warning(f"❌ Qdrant 集合 '{self.settings.QDRANT_COLLECTION_NAME}' 不存在")
                logger.info("💡 請先在 04-embedding-application 中建立向量索引")
                return False

            # 檢查集合中是否有數據
            collection_info = self.qdrant_client.get_collection(self.settings.QDRANT_COLLECTION_NAME)
            if collection_info.points_count == 0:
                logger.warning(f"❌ Qdrant 集合 '{self.settings.QDRANT_COLLECTION_NAME}' 中沒有數據")
                logger.info("💡 請先在 04-embedding-application 中載入文檔數據")
                return False

            logger.info(f"✅ Qdrant 集合 '{self.settings.QDRANT_COLLECTION_NAME}' 可用，包含 {collection_info.points_count} 個向量")

            # 創建QA鏈
            template = """你是一個問答助手。根據以下檢索到的文檔內容來回答問題。
如果檢索到的文檔中沒有相關信息，請明確說明。

Context: {context}

Question: {question}

Answer:"""

            self.prompt = ChatPromptTemplate.from_template(template)

            logger.info("✅ RAG服務初始化成功")
            return True

        except Exception as e:
            logger.error(f"RAG服務初始化失敗: {e}", exc_info=True)
            return False

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        執行RAG查詢：將查詢向量化，在 Qdrant 中搜索，用 LLM 生成回答。
        """
        if not self.prompt:
            raise RuntimeError("RAG系統未初始化，無法執行查詢。")

        try:
            # 1. 將查詢轉換為向量
            query_vector = self.embeddings.embed_query(question)
            logger.info(f"✅ 查詢向量化完成，維度: {len(query_vector)}")

            # 2. 在 Qdrant 中搜索相似向量
            search_results = self.qdrant_client.query_points(
                collection_name=self.settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                with_payload=True,
                limit=top_k
            ).points
            logger.info(f"✅ 找到 {len(search_results)} 個相關文檔")

            # 3. 準備上下文
            context_texts = []
            sources = []

            for hit in search_results:
                content = hit.payload.get('page_content', '')
                metadata = hit.payload.get('metadata', {})

                context_texts.append(content)
                sources.append({
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "metadata": metadata,
                    "score": hit.score
                })

            context = "\n\n".join(context_texts)

            # 4. 用 LLM 生成回答
            messages = self.prompt.format_messages(context=context, question=question)
            answer = self.llm.invoke(messages).content

            logger.info("✅ RAG 查詢完成")
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"執行查詢時出錯: {e}", exc_info=True)
            raise RuntimeError(f"查詢失敗: {str(e)}")
