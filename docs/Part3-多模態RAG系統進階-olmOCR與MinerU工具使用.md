# Part 3. 多模態RAG系統進階：olmOCR與MinerU工具使用

## 課程說明

本體驗課內容節選自《2025大模型Agent智能體開發實戰》(秋招衝刺班)完整版付費課程。體驗課時間有限，若想深度學習大模型技術，歡迎大家報名由我主講的《2025大模型Agent智能體開發實戰》(秋招衝刺班)。

**學習資料下載**：公開課全套學習資料，已上傳至網盤（https://pan.baidu.com/s/1vV-8ew5tAfZOfwG3c4WAFw 提取碼: i345）需要更系統深入學習大模型技術可掃碼添加助教諮詢。

## 課程內容概覽

### 本期公開課四大模組內容

- **演示專案一**：從零到一快速搭建多模態RAG系統
- **演示專案二**：企業級多模態RAG系統開發實戰

---

# 一、最強開源OCR模型：olmOCR部署與調用流程

## 1.1 PDF轉MD功能重要性說明

在多模態 RAG 系統中，「PDF → Markdown（MD）」是整條鏈路最關鍵的入口：PDF 更偏向「版面/座標」，而檢索需要的是可切塊、可對齊語義與結構的文字。

### 核心轉換價值

將 PDF 線性化成 MD 後，標題/段落/列表/表格/公式等要素被清晰地暴露，具備以下優勢：

1. **便於細粒度切分**：可以使用 partition_markdown + chunk_by_title 做細粒度切分
2. **支援多模態索引**：與圖片、表格截圖等「資產軌」對齊進行多模態索引（文字向量、關鍵詞 BM25、圖像向量）
3. **提升檢索精度**：從而提升召回與答案可解釋性

### 主流技術方案對比

圍繞「PDF→MD」，目前社群有兩條代表性路徑：

#### olmOCR 方案特點
- **技術來源**：由 AI2 開源
- **核心技術**：基於視覺-語言模型進行高品質線性化
- **技術優勢**：強調自然閱讀順序與對公式、表格、手寫體等複雜版式的魯棒支援
- **部署支援**：提供面向大規模的推理/部署方案（相容 vLLM/SGLang 等）
- **適用場景**：非常適合作為「文字軌」起點，配合後續的結構化與檢索流程使用

#### MinerU 方案特點
- **技術定位**：主打一站式 PDF→Markdown/JSON 的開源工具鏈
- **應用場景**：在科研文獻等場景中表現活躍
- **整合優勢**：便於與下游的資料加工、結構抽取與標註流程銜接
- **注意事項**：需關注其開源許可

### 方案價值總結

兩者都能把「難啃的 PDF」轉成「檢索友好」的語料，為多模態 RAG 的高精度檢索與可追溯引用打下堅實基礎。
## 1.2 olmOCR專案介紹

olmOCR 是由 AI2（Allen Institute for AI）開源的 PDF 線性化工具包，主要功能是將 PDF/PNG/JPEG 等基於圖像的文件轉成乾淨的 Markdown/純文字格式。

### 核心特性

olmOCR 具備以下核心特性：

1. **高品質轉換**：保留自然閱讀順序
2. **複雜場景支援**：對公式、表格、手寫體、多欄版式等複雜場景進行專項最佳化
3. **自動清理**：能夠自動去除頁眉/頁尾
4. **大規模處理**：面向大規模批次處理提供高效推理與叢集/雲端處理能力

### 專案特點

官方 README 概述的核心要點包括：
- **功能特性**：完整的PDF解析能力
- **版本更新**：v0.3.x 修復自動旋轉與空白頁幻覺，v0.2.x 預設 FP8 更快等
- **安裝與用法**：完整的部署指南
- **技術整合**：外接 vLLM、Docker、S3/多機並行等
- **使用文檔**：完整命令說明

### 技術本質

換言之，**olmOCR本質上是一個經過特定功能微調的多模態大模型**，具備以下優勢：
- 能夠實現明顯優於其他普通OCR模型的光學字元辨識效果
- 借助官方發布的各種腳本，能夠非常便捷地實現PDF到markdown的一鍵轉化

### 相關資源

- **專案地址**：https://github.com/allenai/olmocr
## 1.3 模型架構與訓練

**模型規格**：
- **模型規模**：7B 等級的 VLM 權重
- **基礎模型**：微調自 Qwen2.5-VL-7B-Instruct
- **訓練數據**：olmOCR-mix-0225 資料集（約 25 萬頁，保持自然閱讀順序）
- **量化支援**：提供 FP8 量化版本便於推理

**模型資源**：
- **專案模型**：https://huggingface.co/allenai/olmOCR-7B-0825-FP8

## 1.4 性能對比分析

### 技術定位

olmOCR 更像是「面向 PDF→Markdown 的 VLM 型 OCR 系統」，具備以下特點：

**優勢領域**：
- **自然閱讀順序**：能夠保持文檔的邏輯閱讀順序
- **複雜版式處理**：對多欄/表格/公式/頁眉頁腳等複雜版式處理能力強
- **一鍵產出**：能夠一鍵產出乾淨的 Markdown 格式
- **便利性**：比傳統 OCR 流水線（如 PaddleOCR）或單一辨識模型更省事且更穩定

**適用場景對比**：
- **olmOCR優勢**：結構化文檔處理、多模態內容解析
- **傳統OCR優勢**：純字元級辨識的極致精度、低算力部署

### 在線測試與效果演示

**測試平台**：https://olmocr.allenai.org/
**測試文件**：《GSPO原論文》

通過實際測試，olmOCR 在複雜學術論文的解析中展現出優異的性能，能夠準確識別並保持文檔的原有結構。

## 1.5 olmOCR部署與調用流程

### 1.5.1 硬體與系統需求

目前 olmOCR 只支援本地部署，硬體條件如下（注意：後續介紹的MinerU可以通過API进行部署）：

#### 硬體要求
- **GPU要求**：NVIDIA GPU，建議顯存 ≥ 15 GB
- **測試硬體**：官方測試過 RTX 4090、L40S、A100、H100
- **存儲空間**：磁碟需約 30 GB

#### 系統要求
- **作業系統**：Linux
### 1.5.2 系統依賴安裝

在安裝 olmOCR 之前，需要先安裝系統依賴（用於 PDF 渲染/字體）：

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts \
  fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

**依賴說明**：
- 以上為 README 推薦依賴
- 主要用於將 PDF 頁渲染為圖像、補齊字體
- 確保系統能夠正確處理各種字體和PDF格式
### 1.5.3 建立虛擬環境

接下來建立Python虛擬環境：

```bash
conda create -n olmocr python=3.11 -y
conda activate olmocr
```
### 1.5.4 安裝olmOCR

#### 安裝選項

```bash
# 可選，CPU 僅用於執行評測腳本（不能做 7B 模型推理）
# pip install "olmocr[bench]"

# 可選，設置代理環境
# set http_proxy=http://127.0.0.1:10080
# set https_proxy=http://127.0.0.1:10080

# GPU 推理（推薦）
pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

# 可選：FlashInfer 加速（CUDA 12.8 + torch2.7 對應版本）
# pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
```

#### 重要說明

**計算要求**：
- CPU 只能執行 bench 相關（評分/統計）
- 真正的 OCR/VLM 推理必須用 GPU

**安裝說明**：
- 這條安裝命令的核心是安裝帶 GPU 支援的 olmOCR 依賴
- 確保 pip 能從 PyTorch 官方 CUDA 12.8 倉庫拉取到正確的 CUDA 版 torch
- 從而讓後續的 VLM 推理真正執行在 GPU 上

**vLLM整合**：
- 本條安裝命令包含自動安裝推理工具vLLM
- 如果當前環境已經安裝了vLLM，則可以直接使用 `pip install "olmocr[gpu]"`
- 然後使用vLLM服務來調用腳本

#### 安裝驗證

安裝完成後可以查看實際安裝結果：

```bash
# 查看olmOCR安裝信息
pip show olmocr

# 查看vLLM安裝信息（自動附帶安裝）
pip show vllm
```
### 1.5.5 下載olmOCR模型權重

#### 安裝ModelScope

首先需要安裝魔搭社區的下載工具：

```bash
pip install modelscope
```

#### 模型下載

**模型來源**：https://www.modelscope.cn/models/allenai/olmOCR-7B-0825-FP8/

使用如下命令開始下載：

```bash
# mkdir ./olmOCR-7B-0825-FP8
modelscope download --model allenai/olmOCR-7B-0825-FP8 --local_dir ./olmOCR-7B-0825-FP8
```

#### 下載結果

下載完成後，會在本地目錄中看到完整的模型文件結構，包括：
- 模型權重文件
- 配置文件
- tokenizer 相關文件

#### 替代下載方式

此外，模型權重也可以從網盤中直接進行下載（適用於網絡環境受限的情況）。

### 1.5.6 olmOCR模型調用流程

接下來可以嘗試調用olmOCR模型。需要注意的是，olmOCR模型本質上是Qwen2.5-VL模型經過微調後的模型，我們仍然可以採用大模型基本調用流程來調用olmOCR模型。

#### 模型調用原理

由於微調改變了模型的輸入、輸出格式，我們需要先了解olmOCR模型微調資料集，來確認微調模型可以接受的輸入和輸出格式。

**微調資料集**：https://huggingface.co/datasets/allenai/olmOCR-mix-0225
#### 資料集格式說明

每條資料集包含以下格式：
- **輸入**：PDF中的一頁圖像
- **輸出**：結構化的文字解析結果
#### 輸出格式示例

模型的輸出是結構化文字解析，具體格式如下：
{"primary_language":"en","is_rotation_valid":true,"rotation_correction":0,"is_table":false,"is_diagram":true,"natural_text":"HIGHLIGHTS/SITUATION UPDATE (03/02/2022)\n\nCUMULATIVE\n\n- Tested 926,848\n- Confirmed 156,187\n- Active 6,024\n- Recovered 146,174\n- Vaccinated\n - 1st doses 424,912\n - 2nd doses 246,268\n - 3rd doses 17,617\n- Deaths 3,974\n\nTOTAL TODAY\n\n- Tested 1,598\n- Confirmed 85\n- Active 6,024\n- Recovered 60\n- Vaccinated\n - 1st doses 1,385\n - 2nd doses 246\n - 3rd doses 617\n- Deaths 1\n\n- A total of 156,187 cases have been recorded to-date, representing 6% of the total population (2,550,226).\n- More female cases 82,860 (53%) have been recorded.\n- Of the total confirmed cases, 5,285 (3%) are Health Workers, with no new confirmation today.\n - 4,474 (85%) State; 803 (15%) Private, 8 (0.2%) Non-Governmental Organizations.\n - 5,261 (99%) recoveries and 25 (0.5%) deaths.\n- The recovery rate now stands at 94%.\n- Khomas and Erongo regions reported the highest number of cases with 50,844 (33%) and 22,507 (14%) respectively.\n- Of the total fatalities, 3,650 (92%) are COVID-19 deaths while 324 (8%) are COVID-19 related deaths.\n- The case fatality rate now stands at 2.5%.\n\nTable 1: Distribution of confirmed COVID-19 cases by region, 03 February 2022\n\n| Region | Total cases daily | New reported re-infections | Total No. of cases | Active cases | Recoveries | Cumulative Deaths | Cumulative deaths with co-morbidities | Non-COVID deaths | Health Workers |\n|--------------|-------------------|----------------------------|--------------------|--------------|------------|-------------------|---------------------------------------|-----------------|---------------|\n| Erongo | 8 | 0 | 22,507 | 3,649 | 18,427 | 426 | 353 | 5 | 491 |\n| Hardap | 0 | 0 | 8,372 | 9 | 8,099 | 264 | 166 | 0 | 160 |\n| ||Khomas | 10 | 0 | 50,844 | 1,378 | 48,567 | 899 | 703 | 1 | 1,812 |\n| Kunene | 2 | 0 | 4,972 | 7 | 4,816 | 149 | 107 | 0 | 150 |\n| Ohangwena | 5 | 0 | 5,964 | 88 | 5,710 | 194 | 118 | 2 | 220 |\n| Omaheke | 40 | 0 | 4,961 | 81 | 4,590 | 289 | 204 | 1 | 142 |\n| Omusati | 7 | 0 | 7,524 | 66 | 7,125 | 333 | 221 | 0 | 265 |\n| Oshana | 2 | 0 | 10,579 | 55 | 10,132 | 391 | 249 | 0 | 607 |\n| Oshikoto | 0 | 0 | 7,852 | 0 | 7,632 | 220 | 150 | 2 | 365 |\n| Otjozondjupa | 5 | 0 | 12,109 | 88 | 11,736 | 284 | 184 | 1 | 339 |\n| Zambezi | 1 | 0 | 3,522 | 147 | 3,243 | 132 | 94 | 0 | 125 |\n\nTotal: 85 cases, 156,187 total cases, 6,024 active cases, 146,174 recoveries, 3,974 deaths, 2,810 cumulative deaths with co-morbidities, 15 non-COVID deaths, 5,285 health workers."} 
#### 輸出字段說明

翻譯為中文的字段含義如下：
{
  "primary_language": "zh",
  "is_rotation_valid": true,
  "rotation_correction": 0,
  "is_table": false,
  "is_diagram": true,
  "natural_text": "重要內容/情況更新 (2022年2月3日)\n\n累計數據\n\n- 檢測 926,848例\n- 確診 156,187例\n- 現有病例 6,024例\n- 康復 146,174例\n- 接種疫苗\n - 第1劑 424,912例\n - 第2劑 246,268例\n - 第3劑 17,617例\n- 死亡 3,974例\n\n今日總計\n\n- 檢測 1,598例\n- 確診 85例\n- 現有病例 6,024例\n- 康復 60例\n- 接種疫苗\n - 第1劑 1,385例\n - 第2劑 246例\n - 第3劑 617例\n- 死亡 1例\n\n- 迄今為止，共記錄156,187例病例，佔總人口（2,550,226）的6%。\n- 記錄了更多的女性病例，共82,860例（53%）。\n- 在所有確診病例中，有5,285例（3%）是醫護人員，今天沒有新增確診病例。\n - 4,474例（85%）為國家公立醫院醫護人員；803例（15%）為私人醫院醫護人員；8例（0.2%）為非政府組織醫護人員。\n - 5,261例（99%）康復，25例（0.5%）死亡。\n- 康復率目前為94%。\n- 赫馬斯（Khomas）和埃龍戈（Erongo）地區報告的病例數最多，分別為50,844例（33%）和22,507例（14%）。\n- 在所有死亡病例中，3,650例（92%）是因新冠病毒死亡，而324例（8%）是與新冠病毒相關的死亡。\n- 病死率目前為2.5%。\n\n表1：按地區劃分的COVID-19確診病例分佈，2022年2月3日\n\n| 地區 | 每日總病例數 | 新報告的重複感染病例 | 病例總數 | 現有病例 | 康復病例 | 累計死亡人數 | 累計伴有基礎疾病的死亡人數 | 非新冠病毒死亡人數 | 醫護人員病例數 |\n|--------------|-------------------|----------------------------|--------------------|--------------|------------|-------------------|---------------------------------------|-----------------|---------------|\n| 埃龍戈（Erongo） | 8 | 0 | 22,507 | 3,649 | 18,427 | 426 | 353 | 5 | 491 |\n| 哈達普（Hardap） | 0 | 0 | 8,372 | 9 | 8,099 | 264 | 166 | 0 | 160 |\n| 赫馬斯（Khomas） | 10 | 0 | 50,844 | 1,378 | 48,567 | 899 | 703 | 1 | 1,812 |\n| 庫內內（Kunene） | 2 | 0 | 4,972 | 7 | 4,816 | 149 | 107 | 0 | 150 |\n| 奧漢圭納（Ohangwena） | 5 | 0 | 5,964 | 88 | 5,710 | 194 | 118 | 2 | 220 |\n| 奧馬赫克（Omaheke） | 40 | 0 | 4,961 | 81 | 4,590 | 289 | 204 | 1 | 142 |\n| 奧穆薩蒂（Omusati） | 7 | 0 | 7,524 | 66 | 7,125 | 333 | 221 | 0 | 265 |\n| 奧沙納（Oshana） | 2 | 0 | 10,579 | 55 | 10,132 | 391 | 249 | 0 | 607 |\n| 奧希科托（Oshikoto） | 0 | 0 | 7,852 | 0 | 7,632 | 220 | 150 | 2 | 365 |\n| 奥特乔宗杜帕（Otjozondjupa） | 5 | 0 | 12,109 | 88 | 11,736 | 284 | 184 | 1 | 339 |\n| 贊比西（Zambezi） | 1 | 0 | 3,522 | 147 | 3,243 | 132 | 94 | 0 | 125 |\n\n總計：85例病例，累計156,187例病例，6,024例現有病例，146,174例康復病例，3,974例死亡病例，2,810例累計伴有基礎疾病的死亡病例，15例非新冠病毒死亡病例，5,285名醫護人員病例。"}
因此，後續我們也需要據此對模型進行提問。

#### vLLM 服務啟動

首先需要啟動vLLM模型服務：

```bash
vllm serve ./olmOCR-7B-0825-FP8 \
  --served-model-name olmocr \
  --max-model-len 16384
```

服務成功啟動後，模型將可以通過API接口進行調用。

#### 環境配置

在命令行中將當前虛擬環境添加到Jupyter kernel中：

```bash
conda install jupyterlab
conda install ipykernel
python -m ipykernel install --user --name olmocr --display-name "Python (olmocr)"
```

#### 測試文檔準備

下載官方提供的測試文檔：https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

這是一個3頁的學術論文樣本，包含各種複雜的版面結構，適合測試模型的解析能力。

#### 模型調用測試

打開Jupyter，輸入以下程式碼進行調用測試：

```python
# Jupyter最小可複現實驗：PDF -> (pdf2image) -> vLLM(olmocr) -> Markdown
# 需要安裝!pip install pdf2image pillow requests tqdm
import os, base64, requests, textwrap
from pdf2image import convert_from_path
from PIL import Image

VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"  # 改成你的host
MODEL_NAME    = "olmocr"   # 必須與 vLLM 的 --served-model-name 一致
PDF_PATH      = "olmocr_3pg_sample.pdf"
OUT_MD        = "out.md"
MAX_PAGES     = 5          # 只測前N頁，長文檔避免一次性太大

# 1) PDF -> images（可按需調 dpi 或對最長邊做resize以控顯存/上下文）
pages = convert_from_path(PDF_PATH, dpi=200)   # 200~300 dpi 常用
images = []
for i, img in enumerate(pages[:MAX_PAGES], start=1):
    # 可選：限制最長邊（例：最長邊不超過 1600px，減少上下文佔用）
    max_side = max(img.size)
    if max_side > 1600:
        scale = 1600 / max_side
        img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
    buf_path = f"__page_{i}.png"
    img.save(buf_path, "PNG")
    images.append(buf_path)

def to_data_uri(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# 2) 構造每頁的聊天消息並調用 vLLM（OpenAI兼容協議）
def ocr_page(img_path: str) -> str:
    content = [
        {
            "type": "text",
            "text": (
                "Convert this page into clean Markdown in natural reading order. "
                "Remove headers/footers. Keep tables as Markdown tables. "
                "Represent math as LaTeX ($...$ or $$...$$). "
                "Do not invent missing content."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": to_data_uri(img_path),  # 注意這裡是 dict 裡放 url
                "detail": "auto"               # 可選: "low" | "high" | "auto"
            },
        },
    ]

    payload = {
        "model": "olmocr",     # 要與 vLLM --served-model-name 一致
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 4096,
    }

    r = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# 3) 逐頁解析並合併
md_pages = []
for p in images:
    try:
        md_pages.append(ocr_page(p))
    except Exception as e:
        md_pages.append(f"\n\n<!-- ERROR on {p}: {e} -->\n\n")

full_md = "\n\n\\pagebreak\n\n".join(md_pages)
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(full_md)

print(f"Done. Saved Markdown to: {OUT_MD}")
```

#### 測試結果分析

運行過程中後台會顯示模型推理進度，最終創建的 `out.md` 文檔包含了完整的PDF解析內容。

**重要說明**：由於olmOCR微調過程專門針對OCR任務，並未帶入VLM圖片語義解析的訓練資料集，因此olmOCR本身並不具備傳統VLM的視覺理解功能，而是一個專門針對文檔結構化解析優化的OCR模型。

## 1.6 借助olmOCR腳本高效轉化PDF文檔

除了可以使用最底層的OpenAI風格API來調用模型完成解析外，olmOCR還提供了更加便捷的腳本，可以直接將PDF轉化為MD。官方 olmocr.pipeline 提供了自動旋轉檢測、頁眉頁腳清理、重試策略、採樣溫度選擇、閱讀順序增強等一攬子工程最佳化，質量通常更好。

### 基本用法

#### vLLM服務啟動狀態下使用

```bash
# vLLM啟動時：
python -m olmocr.pipeline ./workspace \
  --server http://localhost:8000 \
  --markdown \
  --pdfs ./olmocr_3pg_sample.pdf
```

#### 本地模型推理

```bash
# vLLM未啟動時
python -m olmocr.pipeline ./workspace --markdown --pdfs olmocr_3pg_sample.pdf
```

### 輸出結果說明

輸出會寫到 `./workspace/markdown/` 目錄中，包含以下內容：
- **results**：模型的直接輸出結果（JSON格式）
- **markdown**：提取的純文字Markdown結果

### 圖片文檔解析

除了PDF文檔外，我們還可以將圖片直接輸入到olmOCR模型中進行解析：

```bash
# 圖片解析示例
python -m olmocr.pipeline ./workspace_image \
  --server http://localhost:8000 \
  --markdown \
  --pdfs ./olmocr_sample.png
```

### 解析效果展示

解析完成後會創建相應的工作目錄，包含：
- 模型推理的原始JSON輸出
- 格式化後的Markdown文檔
- 解析統計信息
## 1.7 olmOCR.pipeline 啟動參數列表

以下為olmOCR pipeline完整的啟動參數說明：

| 類別 | 參數 | 含義 / 作用 | 典型取值 / 示例 | 備註 / 建議 |
|------|------|-------------|----------------|-------------|
| 位置參數 | workspace | 工作區路徑（保存中間產物與結果）。支援本地目錄或 S3 路徑。 | ./ws，s3://bucket/prefix/ | 多機協同時建議用 S3。 |
| 輸入/模型 | --pdfs [PDFS ...] | 向工作區添加要處理的 PDF 列表；可傳通配符或"路徑清單文件"。 | ./a.pdf ./b.pdf，s3://bucket/x/*.pdf，或 list.txt | list.txt 一行一個 PDF 路徑。 |
| 輸入/模型 | --model MODEL | 模型位置或名稱。預設 allenai/olmOCR-7B-0725-FP8。可本地目錄、S3、或 HF 倉庫名。 | /models/olmocr-7b，allenai/olmOCR-7B-0825-FP8 | 首次用倉庫名會自動下載到緩存。 |
| S3 訪問 | --workspace_profile | 訪問 workspace（S3） 的配置檔（profile）。 | default | 僅當 workspace 在 S3 時需要。 |
| S3 訪問 | --pdf_profile | 訪問 原始 PDF（S3） 的配置檔。 | pdf-profile | 僅當 PDF 在 S3 時需要。 |
| 任務切分/容錯 | --pages_per_group | 每個"工作項分組"包含的頁數（控制批大小/顯存峰值）。 | 4、8 | 顯存緊張時調小，更穩。 |
| 任務切分/容錯 | --max_page_retries | 單頁渲染/推理的最大重試次數。 | 2、3 | 異常頁可自動重試。 |
| 任務切分/容錯 | --max_page_error_rate | 文檔允許失敗頁比例；超出則判定該文檔失敗。預設 1/250。 | 0.004（≈1/250） | 髒數據多時適當放寬。 |
| 並行/統計 | --workers | 本機並發 worker 數量。 | 1、2、4 | 結合 CPU/IO 能力調整。 |
| 並行/統計 | --stats | 僅輸出工作區統計信息，不執行任務。 | (開關) | 巡檢/觀測用。 |
| 質量過濾 | --apply_filter | 開啟基礎過濾：英文、非表單、非 SEO 垃圾。 | (開關) | 提升語料質量（非必需）。 |
| 輸出/渲染 | --markdown | 產出 Markdown 文件（保留輸入目錄結構）。 | (開關) | 結果在 workspace/markdown/。 |
| 輸出/渲染 | --target_longest_image_dim | PDF 渲染為圖片時的"最長邊像素"。 | 1400、1600、1800 | 調大可改善結構判別（標題/表格），但更耗顯存。 |
| 輸出/渲染 | --target_anchor_text_len | 錨點文字最大長度（字符）。新模型已不使用。 | 0 或省略 | 通常忽略。 |
| 輸出/渲染 | --guided_decoding | 啟用引導式解碼（YAML 類輸出時）。 | (開關) | OCR→MD 場景下一般不用。 |
| 推理（vLLM 本地） | --gpu-memory-utilization | vLLM 可用顯存比例（0~1）。 | 0.85、0.6 | 防 OOM；與其他任務共存時下調。 |
| 推理（vLLM 本地） | --max_model_len | 最大上下文長度（tokens）。 | 16384 | 受模型/引擎限制，過大可能報錯。 |
| 推理（vLLM 本地） | --tensor-parallel-size | 張量並行份數（多 GPU 切分同一模型）。 | 1、2 | 多卡推理設為 >1。 |
| 推理（vLLM 本地） | --data-parallel-size | 數據並行副本數（同模型多份並行）。 | 1、2 | 提高吞吐用，需更多 GPU。 |
| 推理（服務端） | --server | 連接外部 vLLM OpenAI 兼容服務地址。 | http://host:8000 | 指定後不再使用本地 vLLM。 |
| 推理（服務端） | --port | 本地服務監聽端口（需要本地起服務時）。 | 8000 等 | 一般無需改；避讓端口衝突時用。 |
| 集群（Beaker） | --beaker | 啟用 Beaker 集群模式。 | (開關) | 非 Beaker 用戶可忽略。 |
| 集群（Beaker） | --beaker_workspace | Beaker 工作空間名。 | ai2/xyz | 與組織環境對應。 |
| 集群（Beaker） | --beaker_cluster | 目標集群名。 | ai2/general-gpu | 選擇可用 GPU 集群。 |
| 集群（Beaker） | --beaker_gpus | 每個作業申請的 GPU 數。 | 1、2、4 | 結合模型/吞吐需求。 |
| 集群（Beaker） | --beaker_priority | 作業優先級。 | normal、preemptible | 隊列/成本策略相關。 |
## 1.8 借助olmOCR實現元素感知OCR

### 環境準備

首先安裝所需的依賴包：

```bash
pip install "unstructured[all-docs]"   # 支援 PDF / Word / PPT / HTML 等文檔解析
pip install paddlenlp paddleocr        # OCR 引擎
pip install PyMuPDF pillow matplotlib  # PDF 和圖片處理
pip install html2text                  # 用於 HTML 表格轉 Markdown
```

### PDF元素提取與結構化

結合上一節的PDF轉化MD流程，我們可以實現更精細的元素感知解析：

```python
import os
import fitz
from unstructured.partition.pdf import partition_pdf

pdf_path = "0.LangChain技術生態介紹.pdf"
output_dir = "pdf_images"
os.makedirs(output_dir, exist_ok=True)

# Step 1: 提取文本/結構化內容
elements = partition_pdf(
    filename=pdf_path,
    infer_table_structure=True,   # 開啟表格結構檢測
    strategy="hi_res",            # 高分辨率 OCR，適合複雜表格
    ocr_languages="chi_sim+eng",  # 中英文混合識別
    ocr_engine="paddleocr"        # 指定 PaddleOCR 引擎
)

# Step 2: 提取圖片並保存
doc = fitz.open(pdf_path)
image_map = {}  # 映射 page_num -> list of image paths

for page_num, page in enumerate(doc, start=1):
    image_map[page_num] = []
    for img_index, img in enumerate(page.get_images(full=True), start=1):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        img_path = os.path.join(output_dir, f"page{page_num}_img{img_index}.png")
        if pix.n < 5:  # RGB / Gray
            pix.save(img_path)
        else:  # CMYK 轉 RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(img_path)
        image_map[page_num].append(img_path)

# Step 3: 轉換為 Markdown
md_lines = []
inserted_images = set()  # 用來記錄已經插入過的圖片，避免重複

for el in elements:
    cat = el.category
    text = el.text
    page_num = el.metadata.page_number

    if cat == "Title" and text.strip().startswith("- "):
        md_lines.append(text + "\n")
    elif cat == "Title":
        md_lines.append(f"# {text}\n")
    elif cat in ["Header", "Subheader"]:
        md_lines.append(f"## {text}\n")
    elif cat == "Table":
        if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
            from html2text import html2text
            md_lines.append(html2text(el.metadata.text_as_html) + "\n")
        else:
            md_lines.append(el.text + "\n")
    elif cat == "Image":
        # 避免重複插入：只插入當前圖片對應的文件
        for img_path in image_map.get(page_num, []):
            if img_path not in inserted_images:
                md_lines.append(f"![Image](./{img_path})\n")
                inserted_images.add(img_path)
    else:
        md_lines.append(text + "\n")

# Step 4: 寫入 Markdown 文件
output_md = "output.md"
with open(output_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"✅ 轉換完成，已生成 {output_md} 和 {output_dir}/ 圖片文件夾")
```

### 結合olmOCR進行圖片智能解析

在提取圖片後，我們可以結合olmOCR進行更精準的圖片內容解析：

```python
import os, re, io, base64, requests, json
from PIL import Image

DEFAULT_PROMPT = (
    "You are an OCR & document understanding assistant.\n"
    "Analyze this image region and produce:\n"
    "1) ALT: a very short alt text (<=12 words).\n"
    "2) CAPTION: a 1-2 sentence concise caption.\n"
    "3) TEXT: any readable text found in the image.\n"
    "Format your response as JSON."
)

def analyze_image_with_olmocr(image_path, server_url="http://localhost:8000"):
    """使用olmOCR分析圖片內容"""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    content = [
        {"type": "text", "text": DEFAULT_PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        }
    ]

    payload = {
        "model": "olmocr",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 512
    }

    try:
        response = requests.post(f"{server_url}/v1/chat/completions",
                               json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        return result
    except Exception as e:
        return f"Error analyzing image: {e}"

# 使用示例
image_folder = "pdf_images"
for image_file in os.listdir(image_folder):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, image_file)
        analysis = analyze_image_with_olmocr(image_path)
        print(f"Image: {image_file}")
        print(f"Analysis: {analysis}")
        print("-" * 50)
```

---

# 二、MinerU工具部署與使用

## 2.1 MinerU專案介紹

MinerU 是由阿里巴巴達摩院與OpenDataLab聯合開源的PDF文檔解析工具，專注於提供高品質的PDF到Markdown轉換能力。

### 核心特性

MinerU具備以下核心特性：

1. **高精度解析**：專門優化的PDF版面分析和內容提取算法
2. **科研友好**：特別針對學術論文、科研文獻進行優化
3. **表格與公式支持**：精確識別和解析複雜表格、數學公式
4. **多語言支持**：支持中英文等多種語言的文檔解析
5. **API與本地部署**：既支持本地部署也提供API服務

### 技術架構

MinerU 採用多模組架構設計：
- **版面分析模組**：負責識別文檔的版面結構
- **OCR引擎**：集成先進的OCR技術
- **結構重建模組**：將解析結果重建為結構化格式
- **後處理模組**：優化輸出格式和內容質量

### 相關資源

- **專案地址**：https://github.com/opendatalab/MinerU
- **在線演示**：https://opendatalab.com/OpenSourceTools/Extractor/PDF
- **技術文檔**：完整的部署和使用指南

## 2.2 MinerU安裝與配置

### 環境要求

#### 硬體要求
- **CPU**：支持CPU推理，建議4核以上
- **內存**：建議8GB以上
- **存儲**：至少5GB可用空間
- **GPU**：可選，支持CUDA加速

#### 系統要求
- **作業系統**：Linux、macOS、Windows
- **Python**：3.8+
- **依賴**：詳見requirements.txt

### 安裝步驟

#### 1. 環境準備

```bash
# 創建虛擬環境
conda create -n mineru python=3.9 -y
conda activate mineru

# 安裝基礎依賴
pip install -r requirements.txt
```

#### 2. 安裝MinerU

```bash
# 從PyPI安裝（推薦）
pip install magic-pdf

# 或從源碼安裝
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
pip install -e .
```

#### 3. 配置模型

MinerU支持多種模型配置：

```bash
# 下載預訓練模型
python -m magic_pdf.tools.download_models

# 配置模型路徑
export MODEL_PATH="/path/to/models"
```

## 2.3 MinerU使用方法

### 基本用法

#### 命令行使用

```bash
# 基本PDF轉換
magic-pdf pdf-command --pdf input.pdf --output-dir ./output

# 指定輸出格式
magic-pdf pdf-command --pdf input.pdf --output-dir ./output --output-format markdown

# 批量處理
magic-pdf pdf-command --pdf-dir ./pdfs --output-dir ./output
```

#### Python API使用

```python
from magic_pdf.pdf_parse_main import pdf_parse_main
import json

def convert_pdf_with_mineru(pdf_path, output_dir):
    """使用MinerU轉換PDF"""
    try:
        # 配置參數
        parse_mode = "auto"  # 可選：auto, ocr, txt
        output_format = "markdown"

        # 執行轉換
        result = pdf_parse_main(
            pdf_path=pdf_path,
            output_dir=output_dir,
            parse_mode=parse_mode,
            output_format=output_format
        )

        print(f"轉換完成：{result}")
        return result

    except Exception as e:
        print(f"轉換失敗：{e}")
        return None

# 使用示例
pdf_file = "example.pdf"
output_directory = "./mineru_output"
convert_pdf_with_mineru(pdf_file, output_directory)
```

### 進階配置

#### 自定義解析參數

```python
# 詳細配置示例
config = {
    "parse_mode": "ocr",  # 解析模式：auto, ocr, txt
    "ocr_engine": "paddleocr",  # OCR引擎選擇
    "layout_model": "yolov5",   # 版面分析模型
    "formula_enable": True,     # 啟用公式識別
    "table_enable": True,       # 啟用表格識別
    "figure_enable": True,      # 啟用圖片提取
    "lang": ["zh", "en"],       # 支持語言
    "output_format": "markdown" # 輸出格式
}
```

## 2.4 MinerU與olmOCR對比分析

### 技術特點對比

| 特性 | olmOCR | MinerU |
|------|--------|---------|
| 技術基礎 | VLM微調模型 | 傳統OCR+深度學習 |
| 部署方式 | 本地GPU推理 | 本地/API/雲服務 |
| 處理速度 | 中等（需GPU） | 較快（CPU可運行） |
| 解析精度 | 高（結構保持好） | 高（特別是表格公式） |
| 資源消耗 | 高（需15GB+顯存） | 中等（8GB內存） |
| 商用許可 | 開放 | AGPL-3.0（受限） |

### 應用場景建議

#### 選用olmOCR的情況
- 對文檔結構保持要求極高
- 有充足的GPU資源
- 需要處理複雜多欄版面
- 對閱讀順序要求嚴格

#### 選用MinerU的情況
- 處理科研論文、學術文獻
- 需要精確的表格和公式識別
- 資源受限環境（CPU推理）
- 需要批量處理大量文檔

## 2.5 工具組合使用策略

### 混合處理流程

在實際項目中，可以結合兩種工具的優勢：

```python
def hybrid_pdf_processing(pdf_path):
    """混合使用olmOCR和MinerU進行PDF處理"""

    # Step 1: 使用MinerU進行初步解析（快速、高效）
    mineru_result = convert_pdf_with_mineru(pdf_path, "./mineru_temp")

    # Step 2: 對複雜頁面使用olmOCR精細化處理
    complex_pages = identify_complex_pages(mineru_result)

    for page in complex_pages:
        olmocr_result = process_with_olmocr(page)
        # 合併結果
        merge_results(mineru_result, olmocr_result, page)

    return mineru_result

def identify_complex_pages(parse_result):
    """識別需要精細化處理的複雜頁面"""
    complex_pages = []

    # 根據版面複雜度、表格數量等指標判斷
    for page_info in parse_result:
        if (page_info.get('table_count', 0) > 3 or
            page_info.get('column_count', 1) > 2 or
            page_info.get('figure_count', 0) > 5):
            complex_pages.append(page_info['page_num'])

    return complex_pages
```

---

# 三、課程總結與最佳實踐

## 3.1 技術方案總結

本課程深入介紹了兩個重要的多模態文檔解析工具：

### olmOCR核心要點
1. **技術本質**：基於Qwen2.5-VL微調的7B級VLM模型
2. **核心優勢**：保持自然閱讀順序，支持複雜版面結構
3. **部署要求**：需要15GB+顯存的GPU環境
4. **適用場景**：高質量文檔線性化，複雜學術論文解析

### MinerU核心要點
1. **技術定位**：一站式PDF解析工具鏈
2. **核心優勢**：科研友好，表格公式識別精度高
3. **部署靈活性**：支持CPU/GPU，API/本地部署
4. **適用場景**：批量文檔處理，學術文獻解析

## 3.2 選型建議

### 技術選型決策樹

```
是否有充足GPU資源？
├─ 是 → 對結構保持要求是否極高？
│   ├─ 是 → 選擇 olmOCR
│   └─ 否 → 需要處理大量科研文獻？
│       ├─ 是 → 選擇 MinerU
│       └─ 否 → 根據具體需求選擇
└─ 否 → 選擇 MinerU
```

### 實際項目建議

1. **原型階段**：使用MinerU快速驗證可行性
2. **精度優化**：針對關鍵文檔使用olmOCR精細化處理
3. **生產部署**：根據業務需求和資源狀況選擇最適合的方案
4. **混合策略**：結合兩者優勢，實現最優的處理效果

## 3.3 多模態RAG系統建構指導

### 完整工作流程

1. **文檔預處理**
   - 使用olmOCR或MinerU進行PDF解析
   - 提取文字、表格、圖片等多模態內容
   - 生成結構化的Markdown文檔

2. **內容分塊與向量化**
   - 按照語義邊界進行內容切分
   - 使用多模態Embedding模型進行向量化
   - 建立多模態索引（文字+圖片+表格）

3. **檢索與生成**
   - 多路檢索：文字檢索+圖片檢索+表格檢索
   - 結果融合與重排序
   - 結合檢索結果進行增強生成

### 工程化考慮

1. **性能優化**
   - 合理選擇解析工具
   - 實施分佈式處理
   - 建立緩存機制

2. **質量控制**
   - 建立解析質量評估指標
   - 實施人工校驗流程
   - 持續優化處理pipeline

3. **可擴展性**
   - 支持多種文檔格式
   - 模組化設計，便於升級
   - 預留API接口，支持第三方整合

---

## 課程結語

多模態RAG系統的建構需要在技術選型、工程實踐、業務需求之間找到平衡。olmOCR和MinerU作為當前最具代表性的開源方案，各有其技術特色和適用場景。

通過本課程的學習，您已經掌握了：
- olmOCR的部署、調用和優化技巧
- MinerU的安裝、配置和使用方法
- 兩種工具的對比分析和選型策略
- 多模態RAG系統的工程化實踐指南

在實際項目中，建議根據具體業務需求、資源限制和質量要求，選擇最合適的技術方案，並持續優化和迭代，以實現最佳的用戶體驗和業務效果。

---

體驗課內容節選自《2025大模型Agent智能體開發實戰》(秋招衝刺班)完整版付費課程。體驗課時間有限，若想深度學習大模型技術，歡迎大家報名由我主講的《2025大模型Agent智能體開發實戰》(秋招衝刺班)。

此外，若是對大模型底層原理感興趣，也歡迎報名由我和菜菜老師共同主講的《2025大模型原理與實戰課程》(秋招衝刺班)。

大模型秋招衝刺班開班特惠進行中，直播間享五折特價+全套SVIP新班特定福利，合購還有更多優惠！詳細資訊可掃碼添加助教，回覆「大模型」，即可領取課程大綱&查看課程詳情。

《2025大模型Agent智能體開發實戰》(秋招衝刺班)為【100+小時】體系大課，總共20大模組精講精析，零基礎直達大模型企業級應用！