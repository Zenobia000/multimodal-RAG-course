# Part 4. 多模態RAG項目開發實戰

## 課程內容概述

本期公開課包含以下模組內容：

- **實操項目一**：從零到一快速搭建多模態RAG系統
- **實操項目二**：企業級多模態RAG系統開發實戰

---

## 階段一：多模態RAG項目需求描述、技術棧規劃與接口設計

### 1. 項目背景與需求

本項目是一個教學型多模態 RAG（Retrieval-Augmented Generation，檢索增強生成）系統，目標是幫助學習者理解 RAG 系統的完整開發流程。

我們希望實現以下功能：

#### 1.1 PDF 文檔處理
- 用戶上傳 PDF 文件
- 後端完成 OCR、版面解析、Markdown 轉換
- 提供文檔解析狀態查詢與預覽

#### 1.2 索引構建
- 將解析得到的 Markdown 文檔進行切分（chunking）
- 使用 Embedding 模型（OpenAI Embeddings）將片段向量化
- 保存至 FAISS 向量資料庫，用於後續檢索

#### 1.3 對話問答（RAG）
- 用戶輸入問題，系統先在向量資料庫中檢索相關片段
- 將檢索結果與用戶問題一起交給 LLM 生成答案
- 答案中附帶引用（citations），方便追溯來源
- 支持流式輸出（SSE），提升交互體驗
- 支持歷史對話記憶（基於 InMemorySaver），並提供清空功能

#### 1.4 健壯性需求
- 系統在沒有上傳文檔時也能正常回答（使用模型自帶知識）
- 所有接口有清晰的 API 約定，方便前端對接

### 2. 技術棧規劃

#### 2.1 後端技術
- **FastAPI**：高效能 Python Web 框架，自動生成 Swagger UI
- **Uvicorn**：ASGI 伺服器，支持非同步處理
- **LangChain / LangGraph**：RAG 框架，管理對話狀態與檢索邏輯
- **Unstructured / fitz (PyMuPDF)**：PDF 解析、OCR、圖片處理
- **FAISS**：Facebook 開源的向量資料庫，用於相似檢索

#### 2.2 前端技術
- **Figma**：快速完成 UI 原型設計
- **React / Next.js**：實現流式 SSE 接口調用、前端展示

#### 2.3 AI 模型與 API
- **OpenAI Embeddings**：用於向量化（text-embedding-3-small / large）
- **對話模型（LLM）**：支持通用對話（如 DeepSeek-Chat / OpenAI GPT-4）

#### 2.4 環境與依賴
- **Python** >= 3.9
- **主要依賴**：fastapi、uvicorn、python-multipart、langchain、faiss-cpu、unstructured、pymupdf、paddleocr

### 3. 接口規劃

我們採用 RESTful API 風格，分為 4 大模塊：

#### 3.1 健康檢查（Health）
- `/health`：確認服務正常運行

#### 3.2 PDF 處理（PDF Service）
- `/pdf/upload`：上傳 PDF 文件
- `/pdf/parse`：觸發解析任務
- `/pdf/status`：查詢解析進度
- `/pdf/page`：獲取 PDF 頁圖（原始/解析）
- `/pdf/chunk`：根據 citationId 獲取片段

#### 3.3 索引構建（Index Service）
- `/index/build`：構建向量索引
- `/index/search`：檢索相似片段

#### 3.4 對話（Chat Service）
- `/chat`：RAG 聊天（SSE 流式輸出，包含 citations）
- `/chat/clear`：清空會話歷史

### 4. OpenAPI 文件編寫

基於以上規劃，我們使用 OpenAPI 3.1.0 來定義後端接口。

**OpenAPI 的作用**：
- 是前後端溝通的「契約」
- 能被 FastAPI 自動識別並生成交互式文檔
- 可生成前端 SDK / TS 類型，避免手寫錯誤

下面是一份完整的 openapi.yaml（已簡化描述，完整版本可參考專案倉庫）：

```yaml
openapi: 3.1.0
info:
  title: RAG Demo API
  version: 1.0.0
servers:
  - url: http://localhost:8001/api/v1

paths:
  /health:
    get:
      summary: 健康檢查
      responses:
        "200":
          description: OK

  /pdf/upload:
    post:
      summary: 上傳 PDF
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file]
              properties:
                file: { type: string, format: binary }
      responses:
        "200":
          description: 上傳成功

  /pdf/parse:
    post:
      summary: 觸發解析
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                fileId: { type: string }
      responses:
        "202": { description: 已接受 }

  /pdf/status:
    get:
      summary: 查詢解析狀態
      parameters:
        - in: query
          name: fileId
          required: true
          schema: { type: string }
      responses:
        "200": { description: 狀態返回 }

  /index/build:
    post:
      summary: 構建索引
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                fileId: { type: string }
      responses:
        "200": { description: 構建完成 }

  /chat:
    post:
      summary: 聊天接口（SSE）
      description: 返回流式回答 + 引用
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                message: { type: string }
                fileId: { type: string }
      responses:
        "200": { description: text/event-stream }

  /chat/clear:
    post:
      summary: 清空會話
      responses:
        "200": { description: 會話已清空 }
```

### 小結

在開發的第一部分，我們完成了：

1. **明確了系統需求**：PDF → 索引 → RAG 問答
2. **規劃了技術棧**：FastAPI + LangChain + FAISS + 前端 React
3. **梳理了接口模塊**：PDF、索引、聊天
4. **編寫了 openapi.yaml**，形成前後端開發的「契約」

下一步我們將基於 OpenAPI 文檔，逐步實現後端服務（FastAPI + Mock + 真正邏輯），並在 Swagger UI 中進行測試。

---

## 階段二：後端功能思路規劃與Mock功能實現

### 1. 專案結構與職責邊界

**目標**：前後端「用接口說話」；後端內部「用服務分層說話」。

```
backend/
├─ app.py                        # 入口與路由（FastAPI）
├─ services/
│  ├─ pdf_service.py             # 上傳/解析/頁圖/可視化
│  ├─ index_service.py           # 切分/向量化/索引/檢索
│  └─ rag_service.py             # RAG 檢索+生成流（SSE）與會話歷史
├─ data/                         # 解析與索引產物（按 fileId 分目錄）
│  └─ f_xxx/
│     ├─ original.pdf
│     ├─ output.md
│     ├─ pages/{original|parsed}/page-0001.png
│     └─ index_faiss/{index.faiss,index.pkl}
├─ .env
└─ requirements.txt
```

**為什麼這麼拆？**
- `app.py` 專注「路由+協議」（HTTP/SSE），不摻業務細節
- `services/*` 可單獨測試、復用、替換（比如換別的向量庫/LLM）
- `data/<fileId>/...` 是教學友好的本地可觀測產物（便於課堂講解與排錯）

### 2. 從Mock到真實現：增量替換的策略

**先Mock**：接口通、前端能調、邊界清晰

**再替換**：把Mock的「假資料/睡眠」替換成真實調用與產物

這能最大化減少「前後端互相等待」的時間，也讓你每步都有可驗證成果。

#### 2.1 PDF 管線（/pdf/*）

**Mock版思路**
- `/pdf/upload` 返回一個臨時 fileId（UUID），把文件暫存
- `/pdf/parse` 啟動一個後台任務（不要阻塞 HTTP 回應），假裝解析，寫個「進度文件」
- `/pdf/status` 根據進度文件返回 parsing/ready
- `/pdf/page` 返回佔位圖/原圖

**逐步替換為真實現**
- 解析任務裡接入 Unstructured + PaddleOCR + PyMuPDF：
  - 導出 output.md、每頁 original.png、parsed.png、抽取圖片 images/
  - 把頁號、bbox、類別等結構寫入中間文件（便於 /pdf/chunk/可視化疊框）
- 進度更新：按階段寫入（20/50/80/100），讓 /status 有「在動」的感覺
- 頁圖加載：/pdf/page 用 FileResponse 或 StreamingResponse 回圖

**關鍵示意代碼**：

```python
# app.py
@app.post("/api/v1/pdf/parse")
def start_parse(req: ParseBody, bg: BackgroundTasks):
    bg.add_task(run_full_parse_pipeline, req.fileId)
    return {"jobId": f"j_{shortid()}"}

# services/pdf_service.py
def run_full_parse_pipeline(file_id: str):
    write_status(file_id, "parsing", 10)
    # 1) OCR/結構解析 -> elements
    # 2) 導出 output.md、images/
    # 3) 渲染 original/parsed/page-*.png
    write_status(file_id, "ready", 100)
```

#### 2.2 索引管線（/index/*）

**Mock版思路**
- `/index/build` 直接 time.sleep(1) 返回 "chunks": 42
- `/index/search` 返回固定三條佔位片段

**真實現替換**
- 構建階段讀取 `data/<fileId>/output.md`，做「按標題切分（Header1/2）→ 清洗 → 向量化（OpenAI Embeddings）→ FAISS」
- 檢索階段加載本地 FAISS，similarity_search_with_score(query, k)

**關鍵示意代碼**：

```python
# services/index_service.py
def build_faiss_index(file_id: str):
    md = load_markdown(file_id)
    docs = header_split(md)               # MarkdownHeaderTextSplitter
    vs = FAISS.from_documents(docs, embedding=load_embeddings())
    vs.save_local(index_dir(file_id))
    return {"ok": True, "chunks": len(docs)}

def search_faiss(file_id: str, query: str, k=5):
    vs = FAISS.load_local(index_dir(file_id), load_embeddings(), allow_dangerous_deserialization=True)
    return vs.similarity_search_with_score(query, k=k)
```

**為什麼先做「標題切分」？**
- 講解性強（學生容易理解「結構化」切分）
- 對課程 PPT/講義類文檔命中率高
- 之後再引入「遞歸字符切分」作對照，討論召回/粒度權衡

#### 2.3 RAG 聊天（/chat SSE）

**Mock版思路**
- 原來 /chat 返回「偽流」：sleep + 逐句 token 事件
- 先不發 citation，只發 token/done

**真實現替換**
- 收到 message + pdfFileId + sessionId：
  - 如果有 pdfFileId 且索引存在：先 retrieve → 生成 citations
  - 先發若干 event: citation（前端角標立刻出現）
  - 組裝歷史 + 上下文 → 調 LLM 流式生成 → event: token 連發
  - 結尾 event: done 帶 {"used_retrieval": true|false}
- 無文檔也能聊：當索引缺失或 pdfFileId 為空，直接走「通識回答」

**關鍵示意代碼**：

```python
# app.py
@app.post("/api/v1/chat")
def chat_sse(req: ChatReq):
    async def gen():
        citations, context = [], ""
        if req.pdfFileId:
            citations, context_text = await retrieve(req.message, req.pdfFileId)
            for c in citations:
                yield "event: citation\n"
                yield f"data: {json.dumps(c)}\n\n"

        async for ev in answer_stream(
            question=req.message,
            citations=citations,
            context_text=context_text,
            branch="with_context" if context else "no_context",
            session_id=req.sessionId or "default"
        ):
            # token / done
            ...
    return StreamingResponse(gen(), media_type="text/event-stream")
```

#### 2.4 多輪對話（/chat/clear）

**思路**
- 教學場景只需進程內記憶體保存歷史（sessionId → [messages]）
- `/chat/clear` 清空該 sessionId
- 重啟後自然丟失（即是你要的效果）

**關鍵示意代碼**：

```python
# services/rag_service.py
_sessions = defaultdict(list)

def get_history(sid):
    return _sessions.get(sid, [])

def append_history(sid, role, content):
    _sessions[sid].append({"role":role,"content":content})

def clear_history(sid):
    _sessions.pop(sid, None)
```

**為什麼不立刻引入資料庫？**
- 教學項目先證明閉環
- 學生先理解「多輪上下文對生成的影響」
- 需要持久化時，再引入 Redis / SQLite 講「狀態存儲」

### 3. 錯誤處理與返回規範

**目標**：前端可預測、便於排查。

統一錯誤體：
```json
{ "error": "CODE", "message": "人類可讀描述" }
```

**常見錯誤碼建議**：
- `FILE_NOT_FOUND`（fileId 不存在）
- `NEED_PARSE_FIRST`（沒 output.md 就建索引）
- `INDEX_NOT_FOUND`（索引沒建）
- `PAGE_NOT_FOUND`（頁碼越界）
- `OCR_FAILED` / `PARSE_FAILED` / `INDEX_BUILD_ERROR`

示例：
```python
return JSONResponse({"error":"INDEX_NOT_FOUND","message":"請先構建索引"}, status_code=400)
```

### 4. 性能、可觀測性與穩定性

- **進度可見**：解析階段分階段更新 /status
- **日誌可讀**：關鍵里程碑打印（上傳成功、開始 OCR、導出 MD、渲染頁圖、完成）
- **冪等**：/index/build 再次調用可復用已有索引（返回 {"ok":true,"reused":true}）
- **資源控制**：解析時限制並發，避免 OCR 佔滿 CPU
- **超時與回退**：LLM 流式異常時，回退為整段生成 + 手動切片
- **Windows 兼容**：路徑用 pathlib；端口衝突換 8001；中文文件名注意編碼

---

## 階段三：多模態RAG系統後端功能開發與測試

### 後端目錄與角色回顧

```
backend/
├─ app.py                    # FastAPI 入口 & 路由（HTTP/SSE 協議層）
├─ services/
│  ├─ pdf_service.py         # PDF 上傳/解析/OCR/可視化導出
│  ├─ index_service.py       # Markdown切分、向量化、FAISS索引/檢索
│  └─ rag_service.py         # RAG檢索+生成、SSE流式輸出、多輪會話
├─ data/
│  └─ <fileId>/
│     ├─ original.pdf
│     ├─ output.md
│     ├─ images/                 # Markdown內引用的圖片
│     ├─ pages/original/page-*.png
│     ├─ pages/parsed/page-*.png # 疊框可視化
│     └─ index_faiss/{index.faiss,index.pkl}
├─ .env
└─ requirements.txt
```

### 1. app.py 路由與協議層功能開發

定位：只負責「HTTP/SSE 協議 + 入參校驗 + 調用 services」，不直接寫業務細節。這樣方便單測與替換實現。

#### 核心路由

- `GET /api/v1/health`：可用性探針
- `POST /api/v1/pdf/upload`：保存文件，返回 fileId/name/pages
- `POST /api/v1/pdf/parse`：啟動後台任務（BackgroundTasks），馬上返回 202 風格的結果，解析過程在後台跑
- `GET /api/v1/pdf/status?fileId=`：輪詢解析狀態（idle/parsing/ready/error + 進度）
- `GET /api/v1/pdf/page?fileId=&page=&type=`：回傳 PNG（原始頁/疊框頁）
- `GET /api/v1/pdf/chunk?citationId=`：按角標ID回查片段
- `POST /api/v1/index/build`：對 data/<fileId>/output.md 切分→向量化→FAISS 存盤；返回 chunks
- `POST /api/v1/index/search`：Top-K 相似檢索，返回 text/score/metadata
- `POST /api/v1/chat`（SSE）：按 RAG 流生成事件：citation→token→done（錯誤時 error）
- `POST /api/v1/chat/clear`：清空指定 sessionId 的記憶體會話

#### 設計要點

- **後台任務**：/pdf/parse 解析耗時，必須放 BackgroundTasks，避免阻塞請求
- **錯誤體統一**：用 JSONResponse({"error":"CODE","message":"..."} , status_code=xxx)
- **SSE**：StreamingResponse(gen(), media_type="text/event-stream")，事件格式嚴格：

```
event: token
data: {"text":"..."}

event: done
data: {"used_retrieval": true}
```

- **跨域**：本地聯調需要 CORSMiddleware（允許前端端口）

#### 完整代碼示例

```python
from fastapi import FastAPI, UploadFile, File, Query, Body, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio, time, os, random, string
from typing import Optional, Dict, Any, List
import json

from services.pdf_service import (
    save_upload, run_full_parse_pipeline,
    original_pdf_path, dir_original_pages, dir_parsed_pages, markdown_output
)
from services.index_service import build_faiss_index, search_faiss
from services.rag_service import retrieve, answer_stream, clear_history

app = FastAPI(
    title="多模態RAG系統API",
    version="1.0.0",
    description="多模態RAG系統開發實戰後端API"
)

# 允許前端本地聯調
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 課堂演示方便，生產請收緊
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"

# 內存態存儲（教學Mock）
current_pdf: Dict[str, Any] = {
    "fileId": None,
    "name": None,
    "pages": 0,
    "status": "idle",      # idle | parsing | ready | error
    "progress": 0
}
citations: Dict[str, Dict[str, Any]] = {}   # citationId -> { fileId, page, snippet, bbox, previewUrl }

# 工具函數
def rid(prefix: str) -> str:
    return f"{prefix}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"

def now_ts() -> int:
    return int(time.time())

def err(code: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message}, "requestId": rid("req"), "ts": now_ts()}

# Pydantic 模型（契約）
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    pdfFileId: Optional[str] = None

class BuildIndexRequest(BaseModel):
    fileId: str

class SearchRequest(BaseModel):
    fileId: str
    query: str
    k: Optional[int] = 5

class ClearChatRequest(BaseModel):
    sessionId: Optional[str] = None

# Health 檢查
@app.get(f"{API_PREFIX}/health", tags=["Health"])
async def health():
    return {"ok": True, "version": "1.0.0"}

# Chat（SSE，POST 返回 event-stream）
@app.post(f"{API_PREFIX}/chat", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """SSE 事件：token | citation | done | error"""
    async def gen():
        try:
            question = (req.message or "").strip()
            session_id = (req.sessionId or "default").strip()
            file_id = (req.pdfFileId or "").strip()

            citations, context_text = [], ""
            branch = "no_context"
            if file_id:
                try:
                    citations, context_text = await retrieve(question, file_id)
                    branch = "with_context" if context_text else "no_context"
                except FileNotFoundError:
                    branch = "no_context"

            # 先推送引用（若有）
            if branch == "with_context" and citations:
                for c in citations:
                    yield "event: citation\n"
                    yield f"data: {json.dumps(c)}\n\n"

            # 再推送 token 流
            async for evt in answer_stream(
                question=question,
                citations=citations,
                context_text=context_text,
                branch=branch,
                session_id=session_id
            ):
                if evt["type"] == "token":
                    yield "event: token\n"
                    text = evt["data"].replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
                    yield f'data: {{"text":"{text}"}}\n\n'
                elif evt["type"] == "citation":
                    yield "event: citation\n"
                    yield f"data: {json.dumps(evt['data'])}\n\n"
                elif evt["type"] == "done":
                    used = "true" if evt["data"].get("used_retrieval") else "false"
                    yield "event: done\n"
                    yield f"data: {{\"used_retrieval\": {used}}}\n\n"

        except Exception as e:
            yield "event: error\n"
            esc = str(e).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
            yield f'data: {{"message":"{esc}"}}\n\n'

    headers = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# Chat: 清除對話
@app.post(f"{API_PREFIX}/chat/clear", tags=["Chat"])
async def chat_clear(req: ClearChatRequest):
    sid = (req.sessionId or "default").strip()
    clear_history(sid)
    return {"ok": True, "sessionId": sid, "cleared": True}

# PDF: 上傳
@app.post(f"{API_PREFIX}/pdf/upload", tags=["PDF"])
async def pdf_upload(file: UploadFile = File(...), replace: Optional[bool] = True):
    if not file:
        return JSONResponse(err("NO_FILE", "缺少文件"), status_code=400)

    # 生成新的 fileId
    fid = rid("f")
    saved = save_upload(fid, await file.read(), file.filename)
    current_pdf.update({**saved, "status": "idle", "progress": 0})
    citations.clear()
    return saved

# PDF: 觸發解析
@app.post(f"{API_PREFIX}/pdf/parse", tags=["PDF"])
async def pdf_parse(payload: Dict[str, Any] = Body(...), bg: BackgroundTasks = None):
    file_id = payload.get("fileId")
    if not current_pdf["fileId"] or current_pdf["fileId"] != file_id:
        return JSONResponse(err("FILE_NOT_FOUND", "未找到該文件"), status_code=400)

    current_pdf["status"] = "parsing"
    current_pdf["progress"] = 5

    def _job():
        try:
            current_pdf["progress"] = 20
            run_full_parse_pipeline(file_id)
            current_pdf["progress"] = 100
            current_pdf["status"] = "ready"
        except Exception as e:
            current_pdf["status"] = "error"
            current_pdf["progress"] = 0
            print("Parse error:", e)

    if bg is not None:
        bg.add_task(_job)
    else:
        _job()

    return {"jobId": rid("j")}

# PDF: 狀態
@app.get(f"{API_PREFIX}/pdf/status", tags=["PDF"])
async def pdf_status(fileId: str = Query(...)):
    if not current_pdf["fileId"] or current_pdf["fileId"] != fileId:
        return {"status": "idle", "progress": 0}

    resp = {"status": current_pdf["status"], "progress": current_pdf["progress"]}
    if current_pdf["status"] == "error":
        resp["errorMsg"] = "解析失敗"
    return resp

# PDF: 頁面圖
@app.get(f"{API_PREFIX}/pdf/page", tags=["PDF"])
async def pdf_page(
    fileId: str = Query(...),
    page: int = Query(..., ge=1),
    type: str = Query(..., regex="^(original|parsed)$")
):
    if not current_pdf["fileId"] or current_pdf["fileId"] != fileId:
        return JSONResponse(status_code=404, content=None)

    if current_pdf["status"] != "ready" and type == "parsed":
        return JSONResponse(status_code=204, content=None)

    base = dir_original_pages(fileId) if type == "original" else dir_parsed_pages(fileId)
    img = base / f"page-{page:04d}.png"
    if not img.exists():
        return JSONResponse(err("PAGE_NOT_FOUND", "頁面不存在或未渲染"), status_code=404)
    return FileResponse(str(img), media_type="image/png")

# PDF: 引用片段
@app.get(f"{API_PREFIX}/pdf/chunk", tags=["PDF"])
async def pdf_chunk(citationId: str = Query(...)):
    ref = citations.get(citationId)
    if not ref:
        return JSONResponse(err("NOT_FOUND", "無該引用"), status_code=404)
    return ref

# Index: 構建
@app.post(f"{API_PREFIX}/index/build", tags=["Index"])
async def index_build(req: BuildIndexRequest):
    if not current_pdf["fileId"] or current_pdf["fileId"] != req.fileId:
        raise HTTPException(status_code=400, detail="FILE_NOT_FOUND_OR_NOT_CURRENT")
    if current_pdf["status"] != "ready":
        raise HTTPException(status_code=409, detail="NEED_PARSE_FIRST")

    out = build_faiss_index(req.fileId)
    if not out.get("ok"):
        return JSONResponse(err(out.get("error", "INDEX_BUILD_ERROR"), "索引構建失敗"), status_code=500)
    return {"ok": True, "chunks": out["chunks"]}

# Index: 搜索
@app.post(f"{API_PREFIX}/index/search", tags=["Index"])
async def index_search(req: SearchRequest):
    out = search_faiss(req.fileId, req.query, req.k or 5)
    if not out.get("ok"):
        code = out.get("error", "INDEX_NOT_FOUND")
        return JSONResponse(err(code, "請先構建索引"), status_code=400)
    return out
```

### 2. services/pdf_service.py PDF解析功能開發

定位：把 PDF 變成 RAG 友好的產物：output.md + 頁圖 + 可視化疊框 + 圖片資源。

#### 關鍵步驟

1. **保存上傳文件**
   - 命名為 `data/<fileId>/original.pdf`（fileId 由 upload 生成）
   - 讀取頁數用於返回給前端（PyMuPDF fitz.open）

2. **異步解析（後台任務）**
   - 進度管理：write_status(fileId, "parsing", 10/30/60/100)
   - Unstructured.partition_pdf(...)：
     - strategy="hi_res"、ocr_engine="paddleocr"、ocr_languages="chi_sim+eng"
     - 得到 elements（含文本、表格、圖片、標題等類別）
   - 導出 Markdown：
     - 標題轉 # / ##，表格優先 text_as_html→html2text
     - 圖片導出到 images/ 並在 MD 裡用相對路徑引用
   - 頁圖渲染：
     - pages/original/page-0001.png：用 fitz 渲染頁面
     - pages/parsed/page-0001.png：把 elements 的 bbox 按類別上色疊加（matplotlib / PIL）

#### 完整代碼示例

```python
# services/pdf_service.py
from __future__ import annotations
import os, io, math, json
from pathlib import Path
from typing import Dict, Any, List
import fitz
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # 伺服器無頭
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from langchain_unstructured import UnstructuredLoader
from unstructured.partition.pdf import partition_pdf
from html2text import html2text

from dotenv import load_dotenv
load_dotenv(override=True)

# 統一的根目錄：每個 fileId 一個子目錄
DATA_ROOT = Path("data")

def workdir(file_id: str) -> Path:
    d = DATA_ROOT / file_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def dir_original_pages(file_id: str) -> Path:
    p = workdir(file_id) / "pages" / "original"
    p.mkdir(parents=True, exist_ok=True)
    return p

def dir_parsed_pages(file_id: str) -> Path:
    p = workdir(file_id) / "pages" / "parsed"
    p.mkdir(parents=True, exist_ok=True)
    return p

def original_pdf_path(file_id: str) -> Path:
    return workdir(file_id) / "original.pdf"

def markdown_output(file_id: str) -> Path:
    return workdir(file_id) / "output.md"

def images_dir(file_id: str) -> Path:
    p = workdir(file_id) / "images"
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_upload(file_id: str, upload_bytes: bytes, filename: str) -> Dict[str, Any]:
    """保存上傳的 PDF，並返回頁數"""
    pdf_path = original_pdf_path(file_id)
    pdf_path.write_bytes(upload_bytes)
    with fitz.open(pdf_path) as doc:
        pages = doc.page_count
    return {"fileId": file_id, "name": filename, "pages": pages}

def render_original_pages(file_id: str, dpi: int = 144):
    """把原始 PDF 渲染為 PNG，存到 pages/original/"""
    pdf_path = original_pdf_path(file_id)
    out_dir = dir_original_pages(file_id)
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            (out_dir / f"page-{idx:04d}.png").write_bytes(pix.tobytes("png"))

def _plot_boxes_to_ax(ax, pix, segments):
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }
    categories = set()
    for seg in segments:
        points = seg["coordinates"]["points"]
        lw = seg["coordinates"]["layout_width"]
        lh = seg["coordinates"]["layout_height"]
        scaled = [(x * pix.width / lw, y * pix.height / lh) for x, y in points]
        color = category_to_color.get(seg.get("category"), "deepskyblue")
        categories.add(seg.get("category", "Text"))
        poly = patches.Polygon(scaled, linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(poly)

    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for cat, color in category_to_color.items():
        if cat in categories:
            legend_handles.append(patches.Patch(color=color, label=cat))
    ax.legend(handles=legend_handles, loc="upper right")

def render_parsed_pages_with_boxes(file_id: str, docs_local: List[Dict[str, Any]], dpi: int = 144):
    """根據 UnstructuredLoader 的 metadata（含坐標）在原圖上疊框，輸出到 pages/parsed/"""
    pdf_path = original_pdf_path(file_id)
    out_dir = dir_parsed_pages(file_id)
    with fitz.open(pdf_path) as doc:
        # 預聚合：按 page_number 分組 segments
        segments_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for d in docs_local:
            meta = d.metadata if hasattr(d, "metadata") else d["metadata"]
            pno = meta.get("page_number")
            if pno is None:
                continue
            segments_by_page.setdefault(pno, []).append(meta)

        for page_number in range(1, doc.page_count + 1):
            page = doc.load_page(page_number - 1)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(pil)
            ax.axis("off")
            _plot_boxes_to_ax(ax, pix, segments_by_page.get(page_number, []))
            fig.tight_layout()
            fig.savefig(out_dir / f"page-{page_number:04d}.png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)

def unstructured_segments(file_id: str) -> List[Any]:
    """用 UnstructuredLoader 產生高分辨率布局段"""
    pdf_path = str(original_pdf_path(file_id))
    loader = UnstructuredLoader(
        file_path=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        ocr_languages="chi_sim+eng",
        ocr_engine="paddleocr",
    )
    out = []
    for d in loader.lazy_load():
        out.append(d)
    return out

def pdf_to_markdown(file_id: str):
    pdf_path = str(original_pdf_path(file_id))
    out_md = markdown_output(file_id)
    img_dir = images_dir(file_id)

    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        ocr_languages="chi_sim+eng",
        ocr_engine="paddleocr"
    )

    # 提取圖片
    image_map = {}
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            image_map[page_num] = []
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                img_path = img_dir / f"page{page_num}_img{img_index}.png"
                if pix.n < 5:
                    pix.save(str(img_path))
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(str(img_path))
                image_map[page_num].append(img_path.name)

    md_lines: List[str] = []
    inserted_images = set()
    for el in elements:
        cat = getattr(el, "category", None)
        text = (getattr(el, "text", "") or "").strip()
        meta = getattr(el, "metadata", None)
        page_num = getattr(meta, "page_number", None) if meta else None

        if not text and cat != "Image":
            continue

        if cat == "Title" and text.startswith("- "):
            md_lines.append(text + "\n")
        elif cat == "Title":
            md_lines.append(f"# {text}\n")
        elif cat in ["Header", "Subheader"]:
            md_lines.append(f"## {text}\n")
        elif cat == "Table":
            html = getattr(meta, "text_as_html", None) if meta else None
            if html:
                md_lines.append(html2text(html) + "\n")
            else:
                md_lines.append((text or "") + "\n")
        elif cat == "Image" and page_num:
            for name in image_map.get(page_num, []):
                if (page_num, name) not in inserted_images:
                    md_lines.append(f"![Image](./images/{name})\n")
                    inserted_images.add((page_num, name))
        else:
            md_lines.append(text + "\n")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return {"markdown": out_md.name, "images_dir": "images"}

def run_full_parse_pipeline(file_id: str) -> Dict[str, Any]:
    """完整流程：原始頁圖渲染 → Unstructured 布局段 → 疊框圖 → 輸出 Markdown"""
    render_original_pages(file_id)
    docs = unstructured_segments(file_id)
    render_parsed_pages_with_boxes(file_id, docs)
    md_info = pdf_to_markdown(file_id)
    return {"md": md_info["markdown"]}
```

### 3. services/index_service.py 切分/向量化/FAISS 索引

定位：把 output.md 變成「可檢索的 chunks + 向量索引」。

#### 核心流程

1. **加載 .env & Embeddings**
   - OpenAIEmbeddings(model="text-embedding-3-small|large", base_url/key from .env)

2. **Markdown 切分**
   - 用 MarkdownHeaderTextSplitter 按 # / ## 切段，保留 metadata
   - 二次切分（可選）：對超長段用字符長度遞歸切分

3. **構建 & 保存索引**
   - FAISS.from_documents(docs, embeddings) → save_local("data/<fileId>/index_faiss")
   - 冪等：若目錄已存在，返回 reused:true

4. **檢索**
   - load_local(..., allow_dangerous_deserialization=True)
   - similarity_search_with_score(query, k) → 返回 (Document, score)

#### 完整代碼示例

```python
# services/index_service.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv(override=True)

DATA_ROOT = Path("data")

def workdir(file_id: str) -> Path:
    p = DATA_ROOT / file_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def markdown_path(file_id: str) -> Path:
    return workdir(file_id) / "output.md"

def index_dir(file_id: str) -> Path:
    p = workdir(file_id) / "index_faiss"
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_EMBEDDING_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_EMBEDDING_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(model="text-embedding-3-small", **kwargs)

def split_markdown(md_text: str) -> List[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(md_text)

    cleaned: List[Document] = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        # 限制太長的段落
        if len(txt) > 8000:
            txt = txt[:8000]
        cleaned.append(Document(page_content=txt, metadata=d.metadata))
    return cleaned

def build_faiss_index(file_id: str) -> Dict[str, Any]:
    md_file = markdown_path(file_id)
    if not md_file.exists():
        return {"ok": False, "error": "MARKDOWN_NOT_FOUND"}
    md_text = md_file.read_text(encoding="utf-8")

    docs = split_markdown(md_text)
    if not docs:
        return {"ok": False, "error": "EMPTY_MD"}

    embeddings = load_embeddings()
    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(str(index_dir(file_id)))
    return {"ok": True, "chunks": len(docs)}

def search_faiss(file_id: str, query: str, k: int = 5) -> Dict[str, Any]:
    idx = index_dir(file_id)
    if not (idx / "index.faiss").exists():
        return {"ok": False, "error": "INDEX_NOT_FOUND"}

    embeddings = load_embeddings()
    vs = FAISS.load_local(str(idx), embeddings, allow_dangerous_deserialization=True)
    hits = vs.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in hits:
        results.append({
            "text": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata,
        })
    return {"ok": True, "results": results}
```

### 4. services/rag_service.py RAG 檢索+生成（SSE）與會話歷史功能開發

定位：提供一次問答的完整閉環：檢索 → 判斷是否相關 → 先發 citation → 帶歷史與上下文生成流式回答 → done。

#### 關鍵組成

1. **可配置項**
   - LLM：init_chat_model("deepseek-chat")（或其他提供商）
   - Embedding：與 index_service 保持一致
   - 閾值：SCORE_TAU_TOP1/MEAN3（FAISS L2：越小越相似）
   - K 值：top-k 片段數量（默認 3~5）

2. **檢索與相關性判定**
   - 先做向量檢索，得到 (doc, score) 列表
   - 構造 citations：{citation_id, fileId, rank, page, snippet, score, previewUrl}
   - 雙信號判定：
     - 規則：分數閾值（top-1/前3均值）
     - LLM 複核：GRADE_PROMPT 讓模型判斷「檢索上下文是否能回答該問」
   - 只要一個為真→用檢索上下文；否則走「無上下文回答」

3. **SSE 輸出順序**
   - 如果走 with_context：先 event: citation n 條
   - 再通過 astream 流式生成 event: token
   - 結尾 event: done，包含 {"used_retrieval": true|false}
   - 若 astream 不可用 → 回退整段生成 + 手工切片

4. **多輪會話（內存態）**
   - _sessions: Dict[sessionId, List[{"role","content"}]]
   - 每次 /chat：把歷史 + 本輪用戶拼進消息列表
   - 生成後把本輪問答追加進歷史
   - /chat/clear 清空 sessionId；重啟即丟失

#### 完整代碼示例

```python
# services/rag_service.py
from __future__ import annotations
import os, asyncio, textwrap
from typing import List, Dict, Any, Tuple, AsyncGenerator
from typing_extensions import TypedDict
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 配置
MODEL_NAME = "deepseek-chat"
MODEL_PROVIDER = "deepseek"
TEMPERATURE = 0

EMBED_MODEL = "text-embedding-3-large"
K = 3
# FAISS L2：越小越相似
SCORE_TAU_TOP1 = 0.45
SCORE_TAU_MEAN3 = 0.60

SYSTEM_INSTRUCTION = (
    "You are a technical training assistant. "
    "Prefer using retrieved course materials to answer. If no relevant context is found, "
    "answer from your general knowledge and explicitly mention that no matching course content was found."
)

GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved context to the user's question.\n"
    "Context snippets:\n{context}\n\nQuestion: {question}\n"
    "Return exactly 'yes' if the context is helpful to answer the question, otherwise 'no'."
)

ANSWER_WITH_CONTEXT = (
    "Answer the user's question using the provided context.\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n\n"
    "Write in Markdown. Be concise but complete. If code is relevant, use fenced code blocks.\n"
    "Do not fabricate information not present or entailed by the context."
)

ANSWER_NO_CONTEXT = (
    "No relevant course context was found for the question.\n"
    "Answer from your general knowledge. Be clear and accurate.\n"
    "Question:\n{question}"
)

# 模型/向量函數
def _get_llm():
    return init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=TEMPERATURE)

def _get_grader():
    return init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=0)

def _get_embeddings():
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_EMBEDDING_BASE_URL"),
        model=EMBED_MODEL,
    )

def _vs_dir(file_id: str) -> str:
    return os.path.join("data", file_id, "index_faiss")

def _load_vs(file_id: str) -> FAISS:
    vs_path = _vs_dir(file_id)
    idx_file = os.path.join(vs_path, "index.faiss")
    if not os.path.exists(idx_file):
        raise FileNotFoundError(f"FAISS index not found at {vs_path}; build index first.")
    return FAISS.load_local(vs_path, _get_embeddings(), allow_dangerous_deserialization=True)

def _score_ok(scores: List[float]) -> bool:
    if not scores:
        return False
    top1 = scores[0]
    mean3 = sum(scores[:3]) / min(3, len(scores))
    return (top1 <= SCORE_TAU_TOP1) or (mean3 <= SCORE_TAU_MEAN3)

# 會話管理
_sessions = defaultdict(list)

def get_history(sid):
    return _sessions.get(sid, [])

def append_history(sid, role, content):
    _sessions[sid].append({"role": role, "content": content})

def clear_history(sid):
    _sessions.pop(sid, None)

# 主流程：檢索 + 判定 + 生成
async def retrieve(question: str, file_id: str) -> tuple[list[dict], str]:
    """
    返回 (citations, context_text)
    citations: [{citation_id, fileId, rank, page, snippet, score, previewUrl}]
    context_text: 供 LLM 使用的拼接上下文
    """
    vs = _load_vs(file_id)
    hits = vs.similarity_search_with_score(question, k=K)
    citations = []
    ctx_snippets = []
    scores = []

    for i, (doc, score) in enumerate(hits, start=1):
        snippet_short = (doc.page_content or "").strip()
        if len(snippet_short) > 500:
            snippet_short = snippet_short[:500] + "..."
        page = doc.metadata.get("page") or doc.metadata.get("page_number")

        citations.append({
            "citation_id": f"{file_id}-c{i}",
            "fileId": file_id,
            "rank": i,
            "page": page,
            "snippet": (doc.page_content or "")[:4000],
            "score": float(score),
            "previewUrl": f"/api/v1/pdf/page?fileId={file_id}&page={(page or 1)}&type=original",
        })
        ctx_snippets.append(f"[{i}] {snippet_short}")
        scores.append(float(score))

    context_text = "\n\n".join(ctx_snippets) if ctx_snippets else "(no hits)"

    # 規則 + LLM 複核
    ok_by_score = _score_ok(scores)
    if not ok_by_score:
        grader = _get_grader()
        grade_prompt = GRADE_PROMPT.format(context=context_text, question=question)
        decision = await grader.ainvoke([{"role": "user", "content": grade_prompt}])
        ok_by_llm = "yes" in (decision.content or "").lower()
    else:
        ok_by_llm = True

    branch = "with_context" if ok_by_llm else "no_context"
    return citations, context_text if branch == "with_context" else ""

async def answer_stream(
    question: str,
    citations: list[dict],
    context_text: str,
    branch: str,
    session_id: str = "default"
) -> AsyncGenerator[dict, None]:
    """
    以增量事件的形式產出：
      {"type":"citation", "data": citation_dict}
      {"type":"token", "data": "text chunk"}
      {"type":"done", "data": {"used_retrieval": bool}}
    """
    # 先把 citations 全部發給前端
    if branch == "with_context" and citations:
        for c in citations:
            yield {"type": "citation", "data": c}

    # 構建消息歷史
    history = get_history(session_id)
    llm = _get_llm()

    if branch == "with_context" and context_text:
        prompt = ANSWER_WITH_CONTEXT.format(question=question, context=context_text)
    else:
        prompt = ANSWER_NO_CONTEXT.format(question=question)

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # 優先嘗試流式
    assistant_content = ""
    try:
        async for chunk in llm.astream(messages):
            delta = getattr(chunk, "content", None)
            if delta:
                assistant_content += delta
                yield {"type": "token", "data": delta}
    except Exception:
        # 回退：一次性生成
        resp = await llm.ainvoke(messages)
        text = resp.content or ""
        assistant_content = text
        for i in range(0, len(text), 20):
            yield {"type": "token", "data": text[i:i+20]}
            await asyncio.sleep(0.01)

    # 更新會話歷史
    append_history(session_id, "user", question)
    append_history(session_id, "assistant", assistant_content)

    yield {"type": "done", "data": {"used_retrieval": branch == "with_context"}}
```

---

## 階段四：後端功能驗證與前後端功能聯調

### 後端啟動與測試

首先啟動後端服務並進行基本功能測試：

```bash
cd backend
uvicorn app:app --reload --port 8001
```

然後可以在網址：http://127.0.0.1:8001/docs 中進行功能測試，使用 FastAPI 自動生成的 Swagger UI 界面。

### 完整項目運行流程

#### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

#### 2. 創建環境變量

需要手動創建 `.env` 文件，並輸入相應的 API 密鑰：

```env
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://ai.devtool.tech/proxy/v1
```

#### 3. 開啟後端

```bash
cd backend
uvicorn app:app --reload --port 8001
```

#### 4. 開啟前端

在另一個終端中運行：

```bash
# 1. 安裝依賴
npm install

# 2. 啟動開發服務器
npm run dev
```

#### 5. 進行對話測試

完整的多模態 RAG 系統已經可以進行端到端的測試，包括：

- PDF 文檔上傳與解析
- 向量索引構建
- 基於檢索增強的智能問答
- 流式輸出與引用追溯
- 多輪對話記憶

### 系統特點總結

1. **模組化設計**：前後端分離，服務層解耦，便於維護和擴展
2. **異步處理**：PDF 解析和索引構建採用後台任務，不阻塞用戶界面
3. **流式交互**：支持 SSE 流式輸出，提升用戶體驗
4. **智能檢索**：結合向量相似度和 LLM 判斷，提高檢索準確性
5. **可視化支持**：提供 PDF 頁面渲染和結構化解析可視化
6. **容錯設計**：完善的錯誤處理和狀態管理機制

這個多模態 RAG 系統為企業級應用提供了完整的技術方案和實現參考，可以根據具體業務需求進行定制和擴展。

---

## 課程總結

本課程完整展示了多模態 RAG 系統的開發全流程：

1. **需求分析與架構設計**：明確系統功能、技術選型和接口規劃
2. **後端開發策略**：從 Mock 到真實現的增量開發方式
3. **核心功能實現**：PDF 處理、向量索引、智能問答的完整實現
4. **系統集成與測試**：前後端聯調和端到端功能驗證

通過本項目的學習和實踐，可以深入理解 RAG 系統的核心原理和工程實現，為開發更複雜的多模態 AI 應用奠定堅實基礎。