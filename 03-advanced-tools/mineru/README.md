# MinerU PDF 解析工具：用戶指南

## 1. 工具簡介

MinerU 是一個強大的 PDF 解析工具，專為高精度處理複雜文檔而設計。它特別擅長處理包含大量圖表、表格和複雜排版的學術論文或技術報告，能最大程度地還原原始文檔的佈局。

本目錄下的 `demo.py` 腳本提供了一個自動化的流程，讓您能輕鬆地對示例 PDF 進行處理和評估。

## 2. 快速上手 (Quick Start)

請遵循以下步驟來安裝和運行 MinerU。

### 步驟 1: 進入工作目錄並啟動虛擬環境

本項目共用一個位於 `03-advanced-tools/` 的虛擬環境。
```bash
# 進入 OCR 工具比較目錄
cd 03-advanced-tools

# 啟動虛擬環境
source .venv/bin/activate
```

### 步驟 2: 安裝 MinerU

`demo.py` 腳本在運行時會自動檢查 `mineru` 是否安裝。如果尚未安裝，請使用以下指令進行安裝（建議使用 `uv`）：

```bash
# （推薦）使用 uv 安裝
uv pip install -U "mineru[core]"

# 或者使用 pip
pip install -U "mineru[core]"
```

### 步驟 3: 執行處理腳本

進入 `mineru` 目錄並運行 `demo.py`。

```bash
# 進入本工具目錄
cd mineru

# 運行演示腳本
python demo.py
```

腳本將自動開始處理 `test_pdfs` 目錄下的所有 PDF 文件。

## 3. 腳本執行說明

`demo.py` 腳本會自動完成以下任務：
1.  **檢查環境**：確認 `mineru` 指令是否可用。
2.  **掃描文件**：查找 `../test_pdfs/` 目錄下的所有 PDF 文件。
3.  **批量處理**：逐一調用 `mineru` 工具處理每個 PDF。
4.  **保存結果**：
    *   每個 PDF 的詳細解析結果（包含 Markdown 文件和圖片）會保存在 `output/<PDF文件名>/` 目錄下。
    *   一份包含所有文件處理時間和成功狀態的匯總報告將保存為 `mineru_results.json`。

## 4. 查看處理結果

處理完成後，最重要的產出是 Markdown 文件，它保留了原始 PDF 的佈局結構。

-   **Markdown 內容**：位於 `output/<PDF文件名>/<PDF文件名>/auto/` 目錄下，例如：
    ```
    output/2015_ResNet/2015_ResNet/auto/2015_ResNet.md
    ```
-   **提取的圖片**：保存在上述目錄的 `images/` 子目錄中。
-   **運行報告**：根目錄下的 `mineru_results.json` 文件記錄了每份文件的處理效能。

## 5. 主要特性與經驗總結 (Lessons Learnt)

根據在 NVIDIA RTX 4070 環境下的測試，總結出 MinerU 的關鍵特性：

-   **優點 (Strengths)**
    -   **格式保持極佳**：對於學術論文、多欄位報告等複雜排版，其格式還原度非常高，優於許多其他工具。
    -   **端到端處理**：能一次性提取文本、表格和圖片，並以 Markdown 格式重組。

-   **權衡與考量 (Trade-offs)**
    -   **資源消耗高**：處理期間需要 **GPU** 支援，記憶體佔用約 4GB。
    -   **處理速度慢**：平均處理一個 20 頁左右的 PDF 約需 30-40 秒，不適合追求極速的場景。

-   **適用場景 (Best For)**
    -   當 **內容保真度** 和 **佈局還原** 是首要目標時，MinerU 是最佳選擇。
    -   適合用於構建高質量的知識庫或 RAG 應用，因為它提供了結構清晰的 Markdown 輸出。

## 6. 進階使用：直接調用 CLI

如果您想處理 `test_pdfs` 之外的單個文件，可以直接使用 `mineru` 的命令行界面 (CLI)。

**基本語法：**
```bash
mineru -p <輸入文件路徑> -o <輸出目錄>
```

**示例：**
```bash
mineru -p my_document.pdf -o ./output_folder
```

## 7. 疑難排解 (Troubleshooting)

### Q: 提示 "mineru: command not found"
**A:** 這表示 `mineru` 未成功安裝或不在系統的 PATH 中。請確保您已啟動正確的虛擬環境 (`.venv`)，並執行[步驟 2](#步驟-2-安裝-mineru) 中的安裝指令。

### Q: 處理時間過長或超時
**A:** `demo.py` 中設置的默認超時為 600 秒（10 分鐘）。對於非常大或複雜的 PDF，這可能不夠。MinerU 本身速度較慢是正常現象，請耐心等待。

### Q: GPU 記憶體不足
**A:** MinerU 對 GPU 資源有一定要求。如果遇到此問題，建議關閉其他正在使用 GPU 的應用程序，或嘗試處理較小的文件。