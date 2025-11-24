# 論文知識庫建立指南

## 📋 概述

本指南說明如何使用 OLMoCR 工具批量處理學術論文 PDF，建立結構化的知識庫。

### 🎯 目標
- 處理 31 個 PDF 學術論文（4個分類）
- 使用 OLMoCR 進行 OCR 和內容提取
- 輸出統一管理的結構化知識庫
- 生成易於查找和檢索的檔案命名系統

## 📁 目錄結構

本知識庫採用優化的統一輸出架構設計：

### 🎯 統一輸出架構

#### 統一輸出目錄 (集中管理)
- `outputs/`: 所有處理結果的統一收集點
  - `processed_papers/`: 按分類組織的論文處理結果
    - `MP/`: 模型範式相關文件
    - `IF/`: 基礎設施相關文件
    - `LM/`: 語言模型相關文件
    - `MM/`: 多模態相關文件
  - `aggregated_chunks/`: 所有論文的JSONL文件（統一命名）
  - `embeddings/`: 向量嵌入文件
  - `indices/`: 全局索引文件
  - `logs/`: 統一處理日誌
  - `temp/`: 臨時處理工作區

### 全局目錄
- `scripts/`: 處理腳本

### 📊 源文件分析
```
源目錄: /home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/papers
├── 01_model_paradigm/     (10 files) - 模型範式
├── 02_infrastructure/     (5 files)  - 基礎設施
├── 03_language_models/    (8 files)  - 語言模型
└── 04_multimodal/        (8 files)  - 多模態

總計: 31 個 PDF 文件
```

## 🚀 快速開始

### 方法 1: 使用主啟動腳本（推薦）
```bash
./start_processing.sh
```

這個腳本會：
- ✅ 檢查目錄結構完整性
- ✅ 驗證環境和依賴
- ✅ 提供三種處理模式選擇
- ✅ 按照 SOP 標準執行

### 方法 2: 手動執行（進階用戶）

#### 前置準備
```bash
# 1. 檢查環境
source /home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03-advanced-tools/.venv/bin/activate

# 2. 設置 GPU 環境（如需要）
export CUDA_VISIBLE_DEVICES=1
nvidia-smi  # 檢查 GPU 狀態
```

#### 批量處理（單線程推薦）
```bash
# 處理單個分類（例如：基礎設施）- 單線程處理
python3 scripts/batch_process_papers.py \
    --input_dir "/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/papers/02_infrastructure" \
    --output_dir "." \
    --category "IF" \
    --max_workers 1

# 註: 處理結果會輸出到 outputs/processed_papers/ 和 outputs/aggregated_chunks/

# 生成全局索引
python3 scripts/generate_global_index.py

# 驗證結果
python3 scripts/verify_processing_results.py
```

## 📊 詳細處理流程

### 論文分類對應表

| 原始分類 | 新分類代碼 | 中文名稱 | 英文名稱 | 文件數 |
|----------|------------|----------|----------|-------|
| 01_model_paradigm | MP | 模型範式 | Model_Paradigm | 10 |
| 02_infrastructure | IF | 基礎設施 | Infrastructure | 5 |
| 03_language_models | LM | 語言模型 | Language_Models | 8 |
| 04_multimodal | MM | 多模態 | Multimodal | 8 |

### 處理模式
1. **🧪 測試模式**: 先處理 5 個文件驗證流程
2. **🏃 分類模式**: 手動控制每個分類的處理
3. **🚀 自動化模式**: 全自動處理所有文件（建議單線程）

### 處理順序建議
1. **先處理小文件**: 從 02_infrastructure (5 files) 開始
2. **測試穩定性**: 完成一個分類後驗證結果
3. **逐步擴展**: 然後處理其他分類
4. **監控資源**: 注意 GPU 記憶體和磁碟空間

## 🔧 腳本說明

### 核心腳本（位於 scripts/ 目錄）
- `create_directory_structure.py`: 創建目錄結構
- `batch_process_papers.py`: 批量處理論文（已優化支援單線程和防衝突）
- `generate_global_index.py`: 生成全局索引
- `verify_processing_results.py`: 驗證處理結果
- `quick_start.sh`: 快速開始腳本

### 主控腳本
- `start_processing.sh`: 主啟動腳本，提供完整的交互式處理流程

## 📋 檔案命名規則

處理後的文件按照以下規則命名：
```
{分類代碼}_{年份}_{論文名稱}_{時間戳}
```

示例：
- `MP_2017_Transformer_20251124`
- `MM_2021_CLIP_20251124`
- `LM_2018_BERT_20251124`

**統一輸出位置**:
- **分類處理結果**: `outputs/processed_papers/{分類}/{論文名稱}/` - 完整的處理結果
- **統一JSONL文件**: `outputs/aggregated_chunks/` - 所有JSONL文件統一管理

## 🔍 品質保證

### 自動檢查項目
- ✅ 目錄結構完整性
- ✅ 源文件覆蓋率
- ✅ 處理結果完整性
- ✅ 內容品質驗證
- ✅ 檔名規範一致性

### 生成的索引文件 (位於 outputs/indices/)
- `master_index.json`: 完整的論文元數據
- `category_index.json`: 分類索引
- `search_index.json`: 搜索索引
- `statistics.json`: 統計信息

### 統一輸出結果
- `outputs/aggregated_chunks/`: 所有論文的JSONL文件（統一管理）
- `outputs/processed_papers/`: 按分類組織的詳細處理結果
- `outputs/logs/`: 統一處理日誌
- `outputs/embeddings/`: 向量嵌入文件（未來使用）

### 品質檢查點
1. **文件完整性**: 每個 PDF 都有對應的輸出
2. **命名規範**: 檔名符合預定義規則
3. **內容品質**: 抽檢 OCR 結果準確性
4. **結構完整**: 目錄結構符合設計

## ⚠️ 注意事項和故障排除

### 錯誤處理
- **GPU OOM**: 使用單線程 `max_workers 1`
- **處理中斷**: 腳本支援斷點續傳
- **格式錯誤**: 檢查日誌文件排查問題

### GPU 記憶體管理
```bash
# 清理 GPU 記憶體（處理大文件前執行）
python3 cleanup_gpu.py

# 監控 GPU 使用
watch -n 1 nvidia-smi
```

### 檢查命令
```bash
# 檢查統一處理完成度
find outputs/aggregated_chunks -name "*.jsonl" | wc -l

# 檢查分類處理完成度
find outputs/processed_papers -name "*.jsonl" | wc -l

# 檢查錯誤日誌
grep -r "ERROR" outputs/logs/

# 驗證文件大小
du -sh outputs/*
```

## 📊 預期輸出

### 處理統計
- **總文件數**: 31 個 PDF
- **預估處理時間**: 2-4 小時（依硬體而定，單線程）
- **輸出文件類型**:
  - JSONL（結構化數據）
  - Markdown（可讀格式）
  - 元數據文件
  - 處理日誌

### 文件大小估算
- **每個 PDF**: 約 50-200 MB 輸出
- **總輸出大小**: 預估 3-6 GB

## 📝 使用說明

1. **首次使用**: 執行 `./start_processing.sh` 並選擇測試模式
2. **定期處理**: 選擇分類模式逐步處理（建議單線程）
3. **批量處理**: 選擇自動化模式（需要幾小時）

## 🎯 完成標準

### 成功標準
1. ✅ 31 個 PDF 全部處理完成
2. ✅ 按統一結構組織輸出
3. ✅ 檔名規範化
4. ✅ 生成檢索索引
5. ✅ 無重大處理錯誤

### 完成後檢查清單
- [ ] 所有 PDF 文件都已處理
- [ ] 檔名符合命名規範
- [ ] 統一輸出目錄結構完整
- [ ] 無處理錯誤
- [ ] 生成全局索引
- [ ] 元數據完整

### 交付物
1. **統一的知識庫輸出目錄**
2. **全局索引文件**
3. **處理報告和統計**
4. **錯誤日誌和解決方案**

## 📖 相關文檔

技術細節和操作程序請參閱：
- `outputs/logs/`: 處理日誌和錯誤記錄
- `outputs/indices/`: 全局索引文件
- 腳本內置說明文檔

## 🔄 版本更新

### v2.0 (2025-11-24 17:15)
- ✅ 重新設計為統一輸出架構
- ✅ 新增 processed_papers/ 和 aggregated_chunks/ 目錄
- ✅ 優化批處理腳本支援單線程處理
- ✅ 新增臨時工作區避免並發衝突
- ✅ 簡化目錄結構，移除重複路徑

### v1.3 (2025-11-24 16:35)
- ✅ 修正日誌文件輸出路徑統一到 outputs/logs/
- ✅ 移除重複的日誌和索引目錄結構
- ✅ 更新腳本以動態創建正確的日誌路徑
- ✅ 簡化全局目錄結構，僅保留 scripts/

### v1.2 (2025-11-24 16:20)
- ✅ 實現雙重架構設計：分類目錄 + 統一輸出
- ✅ 分類目錄保持詳細的處理結果（包含workspace）
- ✅ 統一輸出提供便於管理的JSONL集合
- ✅ 修改批處理腳本支援雙重輸出
- ✅ 更新README反映新架構

### v1.1 (2025-11-24 15:50)
- ✅ 重新規劃輸出目錄結構
- ✅ 統一所有處理結果到 outputs/ 目錄
- ✅ 解決輸出文件分散問題
- ✅ 更新腳本以支持新的目錄結構
- ✅ 整合README和SOP文檔

### v1.0 (2025-11-24 15:11)
- ✅ 初始版本建立
- ✅ 基本目錄結構和處理流程

---
建立時間: 2025-11-24 15:11:49
最後更新: 2025-11-24 17:15:00