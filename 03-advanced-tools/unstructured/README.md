# Unstructured PDF 處理工具

## 簡介

Unstructured 是一個易於使用的 PDF 解析工具，適合處理大多數常見的 PDF 文檔。它部署簡單，處理速度快，是 80% 場景下的最佳選擇。

## 特點

- ✅ **部署簡單**：`pip install` 即可使用，無需額外配置
- ✅ **處理速度快**：平均處理時間 2-6 秒/文件
- ✅ **資源消耗低**：僅需約 512MB 記憶體，無需 GPU
- ✅ **通用性強**：適合處理大多數常見 PDF 文檔
- ⚠️ **掃描文檔準確率較低**：對掃描版 PDF 的準確率約 60%
- ⚠️ **格式保持一般**：複雜格式的保持率約 50%

## ⚠️ 版本兼容性重點

**必須使用 NumPy < 2.0**

Unstructured 及其依賴（特別是 `onnxruntime`）目前**不支持 NumPy 2.x**，必須使用 NumPy < 2.0。

**常見錯誤：**
- `AttributeError: _ARRAY_API not found`
- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
- `onnxruntime` 導入失敗

**快速修復：**
```bash
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

---

## 安裝要求

### 1. 安裝依賴

#### 方法一：使用 requirements.txt（推薦）

```bash
# 使用 requirements.txt 安裝（會自動固定 NumPy 版本）
pip install -r requirements.txt
```

#### 方法二：手動安裝並固定 NumPy 版本

```bash
# 1. 先降級 NumPy（如果已安裝 NumPy 2.x）
pip install "numpy<2.0"

# 2. 安裝 unstructured
pip install unstructured

# 3. 安裝 PDF 處理依賴
pip install "unstructured[pdf]"
```

#### 方法三：如果已安裝 NumPy 2.x，先降級再安裝

```bash
# 檢查當前 NumPy 版本
python -c "import numpy; print(numpy.__version__)"

# 如果版本 >= 2.0，先降級
pip install "numpy<2.0" --force-reinstall

# 然後安裝 unstructured
pip install unstructured "unstructured[pdf]"
```

### 2. 系統要求

- Python 3.8+
- 無需 GPU
- 至少 512MB 可用記憶體

## 使用方法

### 基本使用

**推薦：使用公用 OCR 環境**

```bash
# 進入項目目錄
cd 03-advanced-tools

# 方法 1：使用 activate 腳本（推薦）
source ocr-env/bin/activate
cd unstructured
python demo.py

# 方法 2：直接使用 ocr-env 的 Python（如果 activate 有問題）
./ocr-env/bin/python3 unstructured/demo.py
```

**注意：**
- 如果 `activate` 腳本無法正確激活環境（可能因為舊環境變量），使用方法 2 直接調用
- 確保 `ocr-env` 中已安裝 `numpy<2.0` 和 `unstructured`，否則會出現 NumPy 兼容性錯誤

**或直接運行（使用系統環境，需要 NumPy < 2.0）**

```bash
cd unstructured
python demo.py
```

### 功能說明

1. **自動掃描測試文件**：程序會自動掃描父目錄 `../test_pdfs/` 下的所有 PDF 文件
2. **批量處理**：自動處理所有找到的 PDF 文件
3. **結果分析**：處理完成後會顯示統計信息
4. **結果保存**：所有處理結果會保存到 `output/` 目錄

### 輸出說明

處理完成後，所有輸出文件會保存在 `output/` 目錄：

- **處理結果 JSON**：`output/unstructured_results.json` 包含每個文件的處理詳情
  - 文件信息（名稱、大小、處理時間）
  - 成功/失敗狀態
  - 輸出文件路徑
  - 錯誤信息（如有）

- **提取的 Markdown 文件**：`output/{PDF文件名}.md`
  - 每個成功處理的 PDF 會生成對應的 `.md` Markdown 文件
  - 包含從 PDF 中提取並格式化為 Markdown 的文本內容
  - 自動識別標題、段落、列表等結構元素並轉換為對應的 Markdown 格式

### 進階使用

```python
from unstructured.partition.auto import partition

# 處理單個文件
elements = partition(filename="document.pdf")

# 獲取文本
text = "\n".join([str(element) for element in elements])
```

## 性能指標

根據測試數據：

**純文字 PDF (30 個樣本)**：
- **平均處理時間**：2.3 秒/文件
- **準確率**：95%+
- **記憶體消耗**：約 512MB

**掃描 PDF (25 個樣本)**：
- **平均處理時間**：5.8 秒/文件
- **準確率**：60%（不推薦用於掃描文檔）
- **記憶體消耗**：約 512MB

**複雜格式 PDF (20 個樣本)**：
- **平均處理時間**：4.1 秒/文件
- **格式保持率**：50%
- **記憶體消耗**：約 512MB

## 適用場景

✅ **推薦使用**：
- 純文本 PDF 文檔
- 簡單格式的技術文檔
- 批量處理大量文檔
- 對處理速度要求高的場景
- 資源受限的環境

❌ **不推薦使用**：
- 掃描版 PDF（準確率低）
- 複雜格式文檔（格式保持差）
- 需要高格式保持率的場景

## 錯誤排查與常見問題

### ❌ 錯誤 1: `AttributeError: _ARRAY_API not found`

**錯誤信息：**
```
AttributeError: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

**原因：**
- `onnxruntime`（unstructured 的依賴）是使用 NumPy 1.x 編譯的
- NumPy 2.x 移除了 `_ARRAY_API`，導致不兼容

**解決方案：**
```bash
# 檢查當前版本
python -c "import numpy; print('NumPy 版本:', numpy.__version__)"

# 降級到 NumPy 1.x
pip install "numpy<2.0" --force-reinstall

# 重新安裝依賴
pip install -r requirements.txt
```

---

### ❌ 錯誤 2: `unstructured套件未安裝`

**錯誤信息：**
```
ImportError: No module named 'unstructured'
```

**解決方案：**
```bash
# 使用 requirements.txt（推薦，會自動處理版本兼容）
pip install -r requirements.txt

# 或手動安裝
pip install "numpy<2.0" unstructured "unstructured[pdf]"
```

---

### ❌ 錯誤 3: `onnxruntime` 導入失敗

**錯誤信息：**
```
ImportError: cannot import name 'ExecutionMode' from 'onnxruntime.capi._pybind_state'
AttributeError: _ARRAY_API not found
```

**原因：** NumPy 版本不兼容

**解決方案：**
```bash
# 先降級 NumPy
pip install "numpy<2.0" --force-reinstall

# 重新安裝 onnxruntime 和 unstructured
pip install -r requirements.txt
```

---

### ❌ 錯誤 4: 環境激活問題

**問題：** 即使激活了 `ocr-env`，仍然使用系統 Python 或舊環境

**診斷：**
```bash
cd 03-advanced-tools

# 檢查環境變量
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Python 路徑: $(which python3)"

# 使用診斷腳本
./debug_ocr_env.sh

# 檢查 ocr-env 中的包
./ocr-env/bin/python3 -c "import numpy; print('NumPy:', numpy.__version__)"
./ocr-env/bin/python3 -c "import unstructured; print('✅ Unstructured OK')"
```

**解決方案：**

```bash
# 方法 1：清理舊環境變量
deactivate
unset VIRTUAL_ENV
source ocr-env/bin/activate

# 方法 2：直接使用 ocr-env 的 Python（推薦）
cd 03-advanced-tools
./ocr-env/bin/python3 unstructured/demo.py
```

**修復環境中的包：**

```bash
# 使用 uv pip 安裝（ocr-env 使用 uv 管理）
cd 03-advanced-tools
uv pip install "numpy<2.0" --python ocr-env/bin/python3
uv pip install unstructured "unstructured[pdf]" --python ocr-env/bin/python3
```

---

### ❌ 錯誤 5: 版本檢查與診斷

**快速診斷命令：**
```bash
# 檢查 NumPy 版本
python -c "import numpy; print('NumPy:', numpy.__version__)"

# 檢查 unstructured 是否安裝
python -c "import unstructured; print('unstructured: OK')"

# 檢查 onnxruntime 是否正常
python -c "import onnxruntime; print('onnxruntime: OK')"
```

**如果出現錯誤，執行修復：**
```bash
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

---

### ✅ 其他常見問題

**Q: 處理掃描 PDF 準確率低**

A: Unstructured 不適合處理掃描版 PDF。建議使用 OLMoCR 或 MinerU 處理掃描文檔。

**Q: 處理複雜格式時格式丟失**

A: Unstructured 的格式保持能力有限。對於複雜格式文檔，建議使用 MinerU。

**Q: 如何提高處理速度？**

A: Unstructured 已經是最快的工具之一。如果需要進一步優化，可以：
1. 處理較小的文件
2. 使用多進程批量處理
3. 考慮使用其他工具處理特定類型的文檔

## 文件結構

```
unstructured/
├── demo.py                    # 主程序
├── README.md                 # 本文件
├── requirements.txt          # 依賴列表（包含 NumPy 版本限制）
└── output/                   # 輸出目錄（運行後生成）
    ├── unstructured_results.json  # 處理結果統計
    └── *.md                  # 提取的 Markdown 文件（每個PDF對應一個md文件）
```

## 相關資源

- [Unstructured 官方文檔](https://unstructured.io/)
- [Unstructured GitHub](https://github.com/Unstructured-IO/unstructured)
- [API 文檔](https://unstructured-io.github.io/unstructured/)

