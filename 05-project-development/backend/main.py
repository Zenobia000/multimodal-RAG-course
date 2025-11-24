import logging
from datetime import datetime
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# 依賴注入和服務模組
from config import settings
from schemas import (
    QueryRequest, QueryResponse, HealthCheckResponse,
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatCompletionChoice,
    ChatCompletionChunk, ChatCompletionChunkChoice
)
from rag_service import RAGService

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 全局變數 ---
lifespan_context = {}

# --- FastAPI 生命週期事件 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI應用生命週期管理
    """
    logger.info("🚀 啟動RAG API服務...")
    
    if not settings.OPENAI_API_KEY:
        logger.error("❌ 嚴重錯誤: OPENAI_API_KEY環境變數未設定。")
    
    rag_service = RAGService(settings=settings)
    # Run in background
    import asyncio
    asyncio.create_task(rag_service.init_rag())
    
    lifespan_context["rag_service"] = rag_service
    
    logger.info("📖 API文檔可在 /docs 查閱")
    
    yield
    
    logger.info(" gracefully shutting down...")
    lifespan_context.clear()

# --- 依賴注入 ---
def get_rag_service() -> RAGService:
    """
    依賴注入函數，獲取RAG服務實例。
    """
    service = lifespan_context.get("rag_service")
    if not service:
        raise HTTPException(status_code=503, detail="RAG服務尚未初始化。 সন")
    return service

# --- FastAPI應用實例 ---
app = FastAPI(
    title="OpenAI相容的RAG API服務",
    description="一個為Open-WebUI提供支援的、相容OpenAI API的RAG服務",
    version="3.0.0",
    lifespan=lifespan
)

# --- 中間件 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API 端點 ---

# --- OpenAI相容端點 ---
@app.get("/v1/models")
async def list_models():
    """
    模擬OpenAI的 `list models` API。
    返回一個包含此RAG服務模型的列表。
    """
    return {
        "object": "list",
        "data": [
            {
                "id": settings.LLM_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "system"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    模擬OpenAI的 `chat completions` API。
    """
    if not rag_service.qa_chain:
        raise HTTPException(status_code=503, detail="RAG服務正在初始化，請稍後再試。 সন")
        
    last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="請求中沒有用戶消息。 সন")

    if request.stream:
        async def stream_generator():
            # 模擬流式響應
            result = rag_service.query(last_user_message)
            answer = result["answer"]
            
            # 第一個塊發送角色
            yield f"data: {ChatCompletionChunk(choices=[ChatCompletionChunkChoice(index=0, delta=ChatMessage(role='assistant', content=''))]).model_dump_json()}\n\n"
            
            # 逐字發送內容
            for char in answer:
                chunk = ChatCompletionChunk(
                    choices=[ChatCompletionChunkChoice(index=0, delta=ChatMessage(role='assistant', content=char))]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            # 結束標誌
            yield f"data: [DONE]\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        result = rag_service.query(last_user_message)
        
        response_message = ChatMessage(role="assistant", content=result["answer"])
        choice = ChatCompletionChoice(index=0, message=response_message)
        
        return ChatCompletionResponse(model=request.model, choices=[choice])

# --- 原有端點 ---
@app.get("/health", response_model=HealthCheckResponse, summary="健康檢查")
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    is_rag_ready = rag_service.qa_chain is not None
    service_status = "healthy" if is_rag_ready else "degraded"
    
    return HealthCheckResponse(
        service="模組化RAG API",
        status=service_status,
        version=app.version,
        rag_ready=is_rag_ready,
        timestamp=datetime.now().isoformat()
    )

@app.post("/query", response_model=QueryResponse, summary="RAG問答（舊版）")
async def query_documents(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    if not rag_service.qa_chain:
        raise HTTPException(status_code=503, detail="RAG服務正在初始化，請稍後再試。 সন")
    try:
        result = rag_service.query(request.question, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"查詢處理失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="伺服器發生內部錯誤。 সন")
