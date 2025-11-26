# 多模態 RAG 課程 (Multimodal RAG Course)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)

一個全面的多模態檢索增強生成 (Multimodal Retrieval-Augmented Generation) 教學課程，涵蓋從基礎概念到高級應用的完整知識體系。

## 🎯 課程概覽

本課程採用循序漸進的教學方式，從理論基礎到實戰應用，幫助學習者全面掌握多模態 RAG 技術：

### 📚 課程結構

| 模組 | 內容 | 狀態 |
|------|------|------|
| **00-course-overview** | 課程介紹與學習路徑 | ✅ |
| **01-fundamentals** | 基礎概念與理論知識 | ✅ |
| **02-hands-on-basics** | 基礎實作練習 | ✅ |
| **03-advanced-tools** | 進階工具與技術 | ✅ |
| **04-embedding-application** | 嵌入向量應用實戰 | ✅ |
| **05-project-development** | 完整專案開發 | ✅ |
| **06-assessment** | 評估與測驗 | ✅ |
| **07-supplementary** | 補充材料 | ✅ |
| **08-instructor-materials** | 教學資源 | ✅ |

### 📖 完整課程文檔

詳細的課程理論與實作指南位於 `docs/` 目錄：

- **[Part1: 多模態RAG技術體系介紹](./docs/Part1-多模態RAG技術體系介紹.md)**
  - 第一原理分析
  - 核心理論基礎
  - 技術架構概覽

- **[Part2: 從零到一快速搭建多模態RAG引擎](./docs/Part2-從零到一快速搭建多模態RAG引擎.md)**
  - 環境配置
  - 基礎系統搭建
  - 快速原型開發

- **[Part3: 多模態RAG系統進階](./docs/Part3-多模態RAG系統進階-olmOCR與MinerU工具使用.md)**
  - OLMoCR 工具使用
  - MinerU 進階技術
  - 系統優化策略

- **[Part4: 多模態RAG項目開發實戰](./docs/Part4-多模態RAG項目開發實戰.md)**
  - 完整專案架構
  - 生產環境部署
  - 性能優化實踐

- **[Part5: 基於VLM構建多模態RAG問答系統實戰](./docs/Part5-基於VLM構建多模態RAG問答系統實戰.md)**
  - 視覺語言模型整合
  - 問答系統構建
  - 進階應用場景

## 🚀 快速開始

### 環境要求

- Python 3.8+
- OpenAI API 密鑰
- Qdrant 向量數據庫
- 8GB+ RAM (推薦16GB)

### 安裝步驟

1. **克隆倉庫**
```bash
git clone https://github.com/Zenobia000/multimodal-RAG-course.git
cd multimodal-RAG-course
```

2. **建立虛擬環境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

3. **安裝依賴**
```bash
cd 04-embedding-application
pip install -r requirements.txt
```

4. **配置環境變數**
```bash
cp .env.example .env
# 編輯 .env 文件，添加你的 API 密鑰
```

5. **啟動向量數據庫**
```bash
# 使用 Docker 啟動 Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

6. **運行示例**
```bash
python embedding_pipeline.py
```

## 🔧 核心功能

### 🎯 向量檢索策略

- **語義搜索**: 基於向量相似度的檢索
- **混合搜索**: 結合關鍵字和語義檢索
- **查詢擴展**: LLM 輔助的查詢優化
- **發現搜索**: 正負樣本引導的探索性檢索
- **推薦系統**: 基於文檔相似度的推薦
- **分組檢索**: 按源文件分組的結果聚合
- **LLM 重排**: 智能結果重排序
- **自適應檢索**: 動態策略選擇

### 📊 學術論文知識庫

包含30+篇重要AI/ML論文的完整處理流程：

- **語言模型**: Word2Vec, BERT, GPT 系列, InstructGPT, TULU-3
- **多模態模型**: CLIP, ViT, GAN, 擴散模型
- **基礎設施**: ZeRO, Scaling Laws, LAION-5B, MegaScale
- **方法論**: Transformer, LoRA, Chain of Thought, ReAct

### 🛠 技術棧

- **向量數據庫**: Qdrant
- **嵌入模型**: OpenAI text-embedding-ada-002
- **文檔處理**: OLMoCR, MinerU
- **語言模型**: GPT-3.5/4
- **框架**: LangChain, FastAPI
- **測試**: pytest, 性能基準測試

## 📁 專案結構

```
multimodal-RAG-course/
├── 00-course-overview/           # 課程概覽
├── 01-fundamentals/              # 基礎理論
├── 02-hands-on-basics/           # 基礎實作
│   └── sample-data/              # 範例數據
├── 03-advanced-tools/            # 進階工具
│   └── olmocr/                   # OCR 工具
├── 03.1-papers_knowledge_base/   # 學術論文知識庫
│   ├── logs/                     # 處理日誌
│   ├── outputs/                  # 處理結果
│   └── scripts/                  # 處理腳本
├── 04-embedding-application/     # 向量應用實戰
│   ├── config.py                 # 配置管理
│   ├── embedding_pipeline.py     # 主要管道
│   ├── test_search_strategies.py # 測試套件
│   └── vector_search_strategies_tutorial.ipynb
├── 05-project-development/       # 專案開發
│   ├── backend/                  # 後端服務
│   ├── frontend/                 # 前端介面
│   └── qdrant_storage/           # 向量數據庫存儲
├── 06-assessment/                # 評估測驗
├── 07-supplementary/             # 補充材料
├── 08-instructor-materials/      # 教學資源
└── docs/                         # 完整課程文檔
    ├── Part1-多模態RAG技術體系介紹.md
    ├── Part2-從零到一快速搭建多模態RAG引擎.md
    ├── Part3-多模態RAG系統進階-olmOCR與MinerU工具使用.md
    ├── Part4-多模態RAG項目開發實戰.md
    └── Part5-基於VLM構建多模態RAG問答系統實戰.md
```

## 🎓 學習路徑

### 初學者路徑
1. 閱讀 [Part1: 技術體系介紹](./docs/Part1-多模態RAG技術體系介紹.md)
2. 完成 `01-fundamentals` 基礎練習
3. 跟隨 [Part2: 快速搭建](./docs/Part2-從零到一快速搭建多模態RAG引擎.md)
4. 實作 `02-hands-on-basics` 範例

### 中級路徑
1. 學習 `03-advanced-tools` 進階工具
2. 閱讀 [Part3: 系統進階](./docs/Part3-多模態RAG系統進階-olmOCR與MinerU工具使用.md)
3. 完成 `04-embedding-application` 實戰項目

### 高級路徑
1. 研讀 [Part4: 開發實戰](./docs/Part4-多模態RAG項目開發實戰.md)
2. 實作 `05-project-development` 完整專案
3. 學習 [Part5: VLM問答系統](./docs/Part5-基於VLM構建多模態RAG問答系統實戰.md)

## 🧪 測試

```bash
# 運行完整測試套件
cd 04-embedding-application
pytest test_search_strategies.py -v

# 運行性能基準測試
python test_search_strategies.py
```

## 📈 性能指標

- **檢索精度**: mAP@10 > 0.85
- **響應時間**: < 500ms (P95)
- **並發處理**: 100+ QPS
- **向量維度**: 1536 (text-embedding-ada-002)

## 🤝 貢獻指南

歡迎提交問題和改進建議！

1. Fork 本倉庫
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 開源協議

本專案採用 MIT 協議 - 詳見 [LICENSE](LICENSE) 文件

## 🙏 致謝

- [OpenAI](https://openai.com/) - 提供強大的嵌入模型和語言模型
- [Qdrant](https://qdrant.tech/) - 高效的向量數據庫解決方案
- [LangChain](https://langchain.com/) - 優秀的LLM開發框架
- 所有貢獻者和學習者的支持

## 📞 聯繫方式

如有問題或建議，請通過以下方式聯繫：

- GitHub Issues: [提交問題](https://github.com/Zenobia000/multimodal-RAG-course/issues)
- Email: [聯繫作者]

---

⭐ 如果這個專案對您有幫助，請給我們一個星星！

**Happy Learning! 🚀**