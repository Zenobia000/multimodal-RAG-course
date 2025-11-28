# 專案四：生產級RAG系統開發實戰 (重構版)

本專案旨在演示如何將一個單體式的RAG原型，重構成為一個結構清晰、易於維護和擴展的模組化微服務應用。這次重構的核心是**關注點分離 (Separation of Concerns)**，這是現代軟體工程中的一個關鍵原則，特別適合教學和團隊協作。

## ⚡ 重要更新

**此專案現在直接引用 `04-embedding-application/` 中的 embedding 功能，避免重複實現：**

- 移除了不必要的 `olmocr_integration.py` 文件
- 移除了 `qdrant_storage` 目錄 (建議使用外部 Qdrant 服務)
- `rag_service.py` 現在智能引用 `04-embedding-application/` 中的 embedding pipeline
- 優先使用已有的 embedding 功能，回退到基本功能

**建議工作流程：**
1. 先在 `04-embedding-application/` 中建立和測試 embedding 索引
2. 再啟動此專案的 API 服務來提供生產級接口

## 專案架構

我們將原有的單一`04_api_server.py`文件拆分為以下結構：

```
04-project-development/
├── backend/
│   ├── Dockerfile
│   ├── main.py         # FastAPI 應用主入口、API端點
│   ├── rag_service.py  # RAG核心業務邏輯
│   ├── config.py       # Pydantic 應用配置
│   └── schemas.py      # API數據模型 (請求/響應)
│
├── docker-compose.yml  # 微服務編排文件
└── README.md           # 就是你正在閱讀的這份文件
```

### 架構優勢

1.  **關注點分離**:
    *   `backend`: 專注於核心的RAG功能、數據處理和API提供。
    *   `open-webui`: 作為一個獨立、功能豐富的前端服務。
    *   `qdrant`: 作為獨立的向量數據庫服務。
    *   `docker-compose.yml`: 專注於服務的部署、網絡和數據卷管理。

2.  **可擴展性**:
    *   前後端可以獨立擴展。例如，您可以運行多個`backend`實例來處理高併發請求。
    *   可以輕易地替換或添加新的服務。

3.  **可維護性**:
    *   每個模組職責單一，代碼更易於理解、調試和修改。

4.  **貼近生產環境**:
    *   使用`docker-compose`進行容器化編排是現代雲原生應用的標準實踐。
    *   通過環境變數進行配置 (`.env`)，而不是硬編碼，增強了安全性及不同環境部署的靈活性。

## 如何運行

在開始之前，請確保您已安裝 Docker 和 Docker Compose。

### 步驟 1: 環境配置

1.  在專案根目錄 (`multimodel-RAG/`) 下創建一個 `.env` 文件。
2.  在 `.env` 文件中，填寫您的 `OPENAI_API_KEY`。這是RAG系統運作的必要條件。

    ```
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

### 步驟 2: 使用 Docker Compose 啟動服務

1.  打開終端，進入當前目錄 (`04-project-development/`)。
2.  運行以下命令來構建並啟動所有服務：

    ```bash
    docker compose up --build -d
    ```
    *   `--build`: 強制重新構建鏡像，確保應用最新的代碼改動。
    *   `-d`: 在後台運行服務。

3.  等待所有服務啟動。您可以通過以下命令查看後端服務的日誌：

    ```bash
    docker compose logs -f backend
    ```
    第一次啟動時，後端會需要一些時間來處理文檔並建立索引。當您看到類似 `RAG服務已成功初始化` 的日誌時，表示後端已準備就緒。

### 步驟 3: 訪問與設定應用

在以下步驟中，請將 `<your_server_ip>` 替換為您 Docker 主機的實際 IP 地址 (例如：`127.0.0.1` 或 `0.0.0.0`)。

1.  **訪問 Open-WebUI**:
    打開瀏覽器，訪問 `http://<your_server_ip>:8081`。首次進入需要註冊一個管理員帳號。

2.  **設定模型**:
    *   登入 `Open-WebUI` 後，點擊右上角的頭像，選擇 "Settings"。
    *   在 "Connections" -> "OpenAI" 中，你應該會看到已經預設連接好的後端服務 (`http://backend:8000/v1`)。請注意，`http://backend:8000/v1` 是 `Open-WebUI` 容器內部訪問 `backend` 服務的地址。
    *   回到主頁面，在模型選擇下拉列表中，選擇 `gpt-3.5-turbo` (或其他你在 `config.py` 中設定的模型)。

3.  **開始對話**:
    現在您可以開始與您的RAG系統進行對話了！

*   **Qdrant Web UI**:
    打開瀏覽器，訪問 `http://<your_server_ip>:6333/dashboard`
    您可以在這裡查看向量數據庫的集合和數據點。

*   **後端API文檔 (Swagger UI)**:
    打開瀏覽器，訪問 `http://<your_server_ip>:8000/docs`

### 如何停止

當您想停止所有服務時，運行：

```bash
docker compose down
```
這會停止並移除所有相關的容器和網絡。如果您想同時刪除數據卷（例如Qdrant的索引），可以添加 `-v` 參數： `docker compose down -v`。
