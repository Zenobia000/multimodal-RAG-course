import os
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from langchain_core.documents import Document
from unstructured.partition.auto import partition
from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
        self.vector_store = None
        self.qa_chain = None

        # 初始化組件
        self.embeddings = OpenAIEmbeddings(model=self.settings.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(
            model_name=self.settings.LLM_MODEL_NAME,
            temperature=self.settings.LLM_TEMPERATURE
        )
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
        
        self.qdrant_client = QdrantClient(url=self.settings.QDRANT_URL)

    async def init_rag(self) -> bool:
        """
        初始化RAG系統：載入文檔、建立向量索引和QA鏈。
        如果索引已存在，則直接載入。
        """
        try:
            collection_exists = self.qdrant_client.collection_exists(self.settings.QDRANT_COLLECTION_NAME)
            
            if collection_exists:
                logger.info(f"從 Qdrant 載入現有集合: {self.settings.QDRANT_COLLECTION_NAME}")
                self.vector_store = Qdrant(
                    client=self.qdrant_client,
                    collection_name=self.settings.QDRANT_COLLECTION_NAME,
                    embeddings=self.embeddings,
                )
                logger.info("✅ 索引成功載入")
            else:
                logger.info("找不到現有索引，開始建立新的索引...")
                await self._build_index()

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

    async def _build_index(self):
        """
        私有方法：從PDF文檔建立新的向量索引。
        """
        papers_dir = Path(self.settings.PAPERS_PATH)
        if not papers_dir.exists() or not papers_dir.is_dir():
            logger.error(f"指定的文檔路徑不存在: {papers_dir}")
            raise FileNotFoundError(f"指定的文檔路徑不存在: {papers_dir}")

        pdf_files = list(papers_dir.rglob('*.pdf'))
        if not pdf_files:
            logger.warning(f"在 '{papers_dir}' 中沒有找到PDF文件")
            return

        documents_to_index = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"正在處理: {pdf_file.name}")
                elements = partition(filename=str(pdf_file), strategy="hi_res")
                markdown_content = elements_to_markdown(elements)
                
                # 將整個 Markdown 內容分割
                chunks = self.text_splitter.split_text(markdown_content)
                
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": str(pdf_file.relative_to(papers_dir.parent)),
                            "file_name": pdf_file.name,
                            "category": pdf_file.parent.name,
                        }
                    )
                    documents_to_index.append(doc)
                
                logger.info(f"成功處理: {pdf_file.name}，產生 {len(chunks)} 個文本塊")
                
            except Exception as e:
                logger.error(f"處理 '{pdf_file}' 失敗: {e}")
                continue

        if not documents_to_index:
            logger.warning("沒有成功處理任何文檔，索引未建立")
            return

        logger.info(f"共分割出 {len(documents_to_index)} 個文本塊，準備寫入 Qdrant")

        # 創建並持久化向量存儲
        self.vector_store = await Qdrant.afrom_documents(
            documents_to_index,
            embedding=self.embeddings,
            url=self.settings.QDRANT_URL,
            collection_name=self.settings.QDRANT_COLLECTION_NAME,
            force_recreate=True,  # 確保創建新集合
        )
        logger.info(f"✅ 向量索引建立完成並保存至 Qdrant 集合 '{self.settings.QDRANT_COLLECTION_NAME}'")

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        執行RAG查詢。
        """
        if not self.qa_chain:
            raise RuntimeError("RAG系統未初始化，無法執行查詢。" )

        try:
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
