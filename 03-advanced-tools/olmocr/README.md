# olmOCR PDF 處理工具：用戶指南

## 1. 環境總覽與主線版本 (Environment Overview & Main Version)

本專案以單一 `uv` 虛擬環境為基準，實際運行的主要版本如下 (2025-11):

- **Python**: 3.x (以 `python --version` 為準)
- **olmocr**: 0.4.6
- **vLLM**: 0.11.0
- **PyTorch**: 2.8.0
- **CUDA**: cu12x (由 `torch` / `nvidia-*-cu12` 等依賴套件提供)
- **FlashInfer / flash-attn**: **未安裝** (此為可選的加速項)

**結論**: 本指南聚焦於 `olmocr[gpu]` 搭配本地 `vLLM 0.11.0` 的使用方式。所有關於 SGLang 後端、NVIDIA Triton 容器 (vLLM 0.5.5) 或舊版 PyTorch 的討論均被視為歷史或平行場景，並已移至附錄。

---

## 2. 快速上手 (Quick Start)

請遵循以下步驟，在您本地的 GPU 環境中安裝和運行 olmOCR。

### 步驟 1: 環境設置 (Environment Setup)

本專案共享一個位於 `03-advanced-tools/` 的虛擬環境。

```bash
# 進入 OCR 工具比較目錄
cd 03-advanced-tools

# 啟動虛擬環境
source .venv/bin/activate
```

### 步驟 2: 安裝 olmOCR (GPU 版本)

使用 `uv` 安裝 `olmocr[gpu]`，它會自動處理 `vLLM`、`PyTorch` 及 CUDA 相關依賴。

```bash
# 推薦使用 uv 安裝
uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
```
> **註**：官方推薦在乾淨的虛擬環境中執行此命令，以避免依賴衝突。安裝過程可能需要 10-20 分鐘，請耐心等待。

### 步驟 3: 驗證安裝

```bash
python -c "import olmocr; print(f'olmOCR: {olmocr.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
```
預期應看到 `olmocr 0.4.6`, `vllm 0.11.0`, `torch 2.8.0` (或兼容版本)，且 `CUDA 可用: True`。

### 步驟 4: 模型下載 (可選)

olmOCR 運行時會自動下載模型。為避免處理中斷，可手動預下載。

```bash
# 手動下載模型
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/olmOCR-2-7B-1025-FP8')"

# 若網絡不佳，可使用鏡像
export HF_ENDPOINT=https://hf-mirror.com
# (然後再次運行上面的指令)
```

### 步驟 5: 執行處理

進入 `olmocr` 目錄，運行 `demo.py` 腳本。

```bash
# 進入本工具目錄
cd olmocr

# 處理指定的 PDF 檔案
python demo.py /path/to/your/document.pdf

# 或使用預設測試檔案
python demo.py
```

---

## 3. 核心概念與經驗總結 (Core Concepts & Lessons Learnt)

-   **環境隔離是關鍵**：務必在虛擬環境中操作，確保 `olmocr`, `vllm`, `torch` 版本匹配。
-   **vLLM 是當前標準後端**：自 `v0.1.75` 起，olmOCR 已使用 `vLLM` 作為標準後端，取代了早期的 SGLang。本專案也基於 `vLLM` 運行。
-   **`max_model_len` 的重要性**：處理時需確保 `--max_model_len` 參數足夠大，以容納模型的上下文窗口，但又不能超過 GPU 記憶體限制。優先通過 `vLLM` 啟動參數或 `olmocr.pipeline` 的 CLI 參數調整，而非直接修改套件源碼。
-   **FlashInfer 是可選加速項**：官方文件推薦的 `FlashInfer` (非 `flash-attn`) 是一個可選的推理加速庫。本環境目前未安裝，證明其並非運行的必要條件。

---

## 4. `demo.py` 腳本說明

`demo.py` 腳本封裝了 `olmocr.pipeline` 的調用，功能如下：
-   處理單個 PDF 文件（來自命令列參數或預設路徑）。
-   通過 `CUDA_VISIBLE_DEVICES` 指定 GPU。
-   配置 `--gpu_memory_utilization 0.7` 和 `--max_model_len 8192` 等關鍵參數。
-   將所有輸出保存至 `olmocr/output/workspace/`。

---

## 5. 詳細安裝與疑難排解 (Detailed Installation & Troubleshooting)

### 系統套件

在 Ubuntu/Debian 系統上，可能需要安裝字體和 PDF 工具：
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

### 疑難排解 (FAQ)
-   **GPU 記憶體不足**: 嘗試在 `demo.py` 或 `olmocr.pipeline` 命令中降低 `--gpu-memory-utilization` (例如 `0.7`) 和 `--max_model_len` (例如 `8192`)。
-   **CUDA 版本問題**: `olmocr[gpu]` 的安裝與 PyTorch 的 CUDA 版本緊密相關。`--extra-index-url .../cu128` 通常對應 CUDA 12.x 系列驅動。若安裝失敗，請檢查 `nvcc --version` 和 `nvidia-smi` 的輸出，並參考 PyTorch 官網尋找匹配的 Wheel 文件。
-   **FlashInfer 安裝問題**: `FlashInfer` 是可選的。如果安裝失敗或導致 PyTorch 版本降級，可以安全地跳過此步驟。olmOCR 仍可正常運行。

---

## 6. 附錄 (Appendices)

### 附錄 A: 版本脈絡 (Version Context)

-   **v0.1.58 (2025-02-25)**: 初代公開版本，後端為 SGLang (歷史版本)。
-   **v0.1.75 (2025-06-17)**: **後端改為 vLLM**，Docker 也更新至 CUDA 12.8。這是從 SGLang 到 vLLM 的關鍵轉折點。
-   **v0.4.x (2025-10–)**: 引入新模型 (FP8 / RL 強化)，提升效能與準確度。本環境使用的 `0.4.6` 屬於此系列。

### 附錄 B: 歷史問題紀錄 (Historical Issues)

> **注意**: 以下內容僅為歷史調試記錄，對當前 `v0.4.6 + vLLM 0.11.0` 環境**不再適用**。

-   **SGLang 後端問題 (v0.1.x 早期)**: 曾遇到記憶體配置困難 (SIGKILL)、伺服器連線不穩等問題，這些都隨著切換到 vLLM 而解決。
-   **硬改 `pipeline.py` (舊版 Workaround)**: 早期為了解決 `max_tokens` 錯誤，曾直接修改 `site-packages` 中的 `olmocr/pipeline.py` 文件。這不是標準做法，目前應通過 CLI 參數 (`--max_model_len`) 解決。

### 附錄 C: NVIDIA Triton 容器運行場景 (Alternative: NVIDIA Triton Container)

官方文件提到的 `vLLM 0.5.5`、`Python 3.10.12`、`CUDA 12.6.2` 等配置，源自 NVIDIA 的 `tritonserver:24.10-vllm-python-py3` NGC 容器。這是一個與本地 `uv/.venv` 不同的獨立運行環境，用於伺服器部署。在本專案中，我們主要關注本地環境的配置。