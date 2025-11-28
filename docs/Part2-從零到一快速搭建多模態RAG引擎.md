# Part 2. 從零到一快速搭建多模態RAG引擎

## 課程說明

本體驗課內容節選自《2025大模型Agent智慧體開發實戰》(秋招衝刺班)完整版付費課程。體驗課時間有限，若想深度學習大模型技術，歡迎大家報名由我主講的《2025大模型Agent智慧體開發實戰》(秋招衝刺班)。

**學習資料下載**：公開課全套學習資料，已上傳至網盤（https://pan.baidu.com/s/1vV-8ew5tAfZOfwG3c4WAFw 提取碼: i345）需要更系統深入學習大模型技術可掃碼添加助教諮詢。

## 課程內容概覽

### 本期公開課四大模組內容

- **演示項目一**：從零到一快速搭建多模態RAG系統
- **演示項目二**：企業級多模態RAG系統開發實戰

---

# 一、結構解析重建法多模態檢索流程

## 1.1 從零到一快速搭建多模態RAG系統基本思路

接下來讓我們實作結構解析重建法來從零搭建多模態檢索流程。在前面對多模態PDF檢索的難點與主流開源項目的梳理之後，我們已經建立起一個清晰的認知框架：

- 單純依賴文字檢索無法應對PDF文件中複雜的多模態內容
- 僅僅依賴OCR也難以保留完整的結構資訊
- 真正可落地的解決方案需要結合文件解析與結構化重建

### 結構解析重建法核心概念

所謂「結構解析重建」，本質上是對原始PDF文件進行分層解析，將其中的：
- 標題、段落
- 表格、圖片
- 公式等元素逐一擷取

並依據其在文件中的位置和語義關係重新組織，轉化為更適合下游檢索系統（如RAG）的結構化表示形式。

### 技術方案概述

在本節中，我們將以 **Unstructured + PaddleOCR** 為核心工具鏈，演示如何：
1. 從PDF文件中自動解析多模態內容
2. 將其重建為Markdown格式文件
3. 保留段落的層次結構
4. 擷取並本地保存圖片、表格等元素

最終得到一份既可讀又可檢索的中間產物，為後續的向量化與知識檢索打下堅實基礎。

### 最終效果演示

本節將實現以下核心功能：

1. **多模態PDF文件元素識別**
   - 文字、標題識別
   - 圖片、表格多模態識別

2. **圖片文字識別與表格內容識別**
   - OCR文字識別精度高
   - 表格結構解離清晰

3. **多模態PDF逆向轉化為Markdown**
   - 保留原有結構層次
   - 支援標題、段落、表格等元素

4. **搭建Agentic RAG系統檢索多模態PDF文件**
   - 智慧化檢索流程
   - 上下文感知回答

5. **多模態PDF檢索與結果生成**
   - 結合文字與表格資訊
   - 生成結構化的答案

## 1.2 Unstructured框架介紹

Unstructured是目前業界最具影響力的**文件解析與預處理框架**之一，由Unstructured-IO團隊開源並持續維護。與MarkItDown偏向輕量化的Markdown轉換不同，Unstructured更強調**多模態文件的細粒度分解（partitioning）與結構化輸出**。

### 核心設計理念

該項目的核心設計理念是：**無論文件來源是PDF、Word、PPT、HTML、E-mail，甚至是圖像和掃描件，都能夠被解析為一個個結構化的Element（元素）物件**。

每個Element都帶有完整的：
- **類別資訊**：Title、Paragraph、Table、Image、List、Formula等
- **元數據**（metadata）：頁碼、座標、置信度等

這種解析方式不僅能：
- 保留文本的層次邏輯
- 為下游任務提供定位資訊
- 支援表格重建、圖像OCR、公式識別等複雜的多模態處理

### PDF解析策略

在 PDF 場景中，Unstructured 提供了多種解析策略：

- **fast 模式**：僅利用 PDF 內置文本層，快速提取文字，適合機器生成的 PDF
- **hi_res 模式**：結合 OCR（支持 Tesseract、PaddleOCR 等）與版面分析，精確分割文本塊、表格與圖片，適合掃描件與版面複雜的 PDF
- **chunking**：可以將解析結果進一步切分為適合向量檢索的語義片段

### 框架優勢

與其他工具相比，Unstructured 的一大優勢在於 **高度可擴展**：它既能作為獨立的 Python 庫使用，也能以 API 服務形式 部署，甚至與 LangChain、Haystack、LlamaIndex 等主流 RAG 框架無縫集成。這種靈活性，使其在 企業知識管理、合規性文檔解析、科研論文分析 等場景中廣泛應用。

因此，Unstructured 已經成為 多模態 PDF 文檔檢索 技術棧中的重要基石。它不僅能提供高精度的結構化解析結果，還能與後續的向量數據庫、檢索模型和大模型推理環節形成天然的銜接，是目前最接近「工業級標準」的開源解決方案之一。

### 相關資源

- **項目地址**：https://github.com/Unstructured-IO/unstructured

### 實驗文檔素材

這裡我們先嘗試使用一個Demo PDF文檔進行檢索嘗試，在跑通流程之後，我們再將方法應用於更加複雜的文檔檢索。

### 更多參考學習資料

- [工業級智能體開發實踐，LangChain從零入門與智能體開發實戰](https://www.bilibili.com/video/BV1pYKgzAE5C/)
- [超越LangChain！LangGraph快速入門與智能體開發實戰](https://www.bilibili.com/video/BV1Kx3CzyE6Q/)

---

# 二、多模態PDF文件解析流程

## 2.1 基礎環境準備

在正式上手PDF → Markdown的結構化解析之前，我們需要先準備好實驗環境。由於本文的實驗在Windows系統上進行，下面的步驟也以Windows為例。

**整體思路**：配置Python環境 → 安裝基礎依賴 → 安裝OCR引擎（PaddleOCR） → 安裝PDF處理與輔助庫。

### 2.1.1 建立 Python 環境

建議使用Python 3.9+（推薦 3.10 或 3.11），以保證相容性，同時推薦使用conda或venv來建立虛擬環境：

```bash
# 使用 conda 建立虛擬環境
conda create -n pdf_rag python=3.10 -y
conda activate pdf_rag

# 或使用 venv
python -m venv pdf_rag
pdf_rag\Scripts\activate
```

### 2.1.2 安裝基礎庫

本次項目需要依賴的核心套件包括：

- **PyMuPDF (fitz)**：負責讀取PDF文件、提取頁面和圖片
- **matplotlib / pillow**：用於視覺化和圖像處理
- **unstructured**：微軟 / LangChain 推薦的PDF文件解析庫，支援結構化分塊
- **paddleocr**：OCR引擎，用於文本區域的識別

安裝指令如下：

```bash
pip install "unstructured[all-docs]"   # 支援 PDF / Word / PPT / HTML 等文件解析
pip install paddlenlp paddleocr        # OCR 引擎
pip install PyMuPDF pillow matplotlib  # PDF 和圖片處理
pip install html2text                  # 用於 HTML 表格轉 Markdown
```

> **⚠️ 注意事項**：
> - `unstructured[all-docs]` 會自動安裝PDF解析相關的依賴（如pdfminer, PyMuPDF）
> - `paddleocr` 在第一次運行時會自動下載模型（中英文模型），如果網路不暢，可以提前下載後手動指定路徑
> - 在Windows上安裝`paddleocr`時，可能需要先裝Visual C++運行庫，否則會遇到paddlepaddle的動態鏈接庫錯誤
> - 可以添加`--index-url https://mirrors.huaweicloud.com/repository/pypi/simple`華為鏡像源來加速下載

## 2.2 載入 PDF 並進行元素提取

有了依賴庫之後，我們就可以使用 UnstructuredLoader 來解析 PDF 文檔了，對於給定的文檔，我們可以按照如下方式進行解析：

```python
from langchain_unstructured import UnstructuredLoader

file_path = "0.LangChain技術生態介紹.pdf"

loader_local = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",              # 高解析度模式，支援複雜文檔
    infer_table_structure=True,     # 自動解析表格結構
    ocr_languages="chi_sim+eng",    # 支援中英文 OCR
    ocr_engine="paddleocr"          # 指定 PaddleOCR 作為 OCR 引擎
)

docs_local = []
for doc in loader_local.lazy_load():
    docs_local.append(doc)

docs_local
```

此時docs_local就包含了每個解析的元素。其中每個 doc 都包含 page_content（文本內容）以及 metadata（頁碼、座標、類型等）。這就意味著我們的 PDF 文檔已經被拆解為一個個 可檢索的基本單元，接下來便可以進一步做結構化處理。

### 代碼詳細解釋

這段代碼的核心目標是用 Unstructured + PaddleOCR 從 PDF 中提取結構化內容，並輸出為文檔物件列表。

#### 1. 導入UnstructuredLoader

```python
from langchain_unstructured import UnstructuredLoader
```

導入 UnstructuredLoader，這是 LangChain 封裝的一個介面，可以直接用來載入 PDF 等非結構化文檔。

#### 2. 配置載入器參數

```python
loader_local = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",              # 高解析度模式，支援複雜文檔
    infer_table_structure=True,     # 自動解析表格結構
    ocr_languages="chi_sim+eng",    # 支援中英文 OCR
    ocr_engine="paddleocr"          # 指定 PaddleOCR 作為 OCR 引擎
)
```

**參數說明：**
- `file_path`：指定 PDF 路徑
- `strategy="hi_res"`：高解析度 OCR 模式，適合複雜表格和排版
- `infer_table_structure=True`：啟用表格解析，把表格恢復為結構化數據
- `ocr_languages="chi_sim+eng"`：設置 OCR 支持簡體中文 + 英文
- `ocr_engine="paddleocr"`：指定 OCR 引擎為 PaddleOCR（相比 Tesseract 更強）

#### 3. 載入文檔

```python
docs_local = []
for doc in loader_local.lazy_load():
    docs_local.append(doc)
```

- `lazy_load()` 會逐頁載入 PDF 並調用 OCR/解析
- `doc` 是 LangChain 的 Document 物件，裡面包含：
  - `doc.page_content` → 文本內容
  - `doc.metadata` → 額外資訊（頁碼、座標、分類、OCR 置信度等）

## 2.3 元素提取效果視覺化

接下來為了驗證實際元素提取效果，我們這裡進一步把 PDF 頁面渲染成圖片，並在上面繪製出分塊框（標題、表格、圖片、文本等），實現視覺化。

```python
import fitz
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def plot_pdf_with_boxes(pdf_page, segments):
    """在PDF頁面上繪製元素分塊框"""
    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    categories = set()
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }

    for segment in segments:
        points = segment["coordinates"]["points"]
        layout_width = segment["coordinates"]["layout_width"]
        layout_height = segment["coordinates"]["layout_height"]

        # 座標縮放到實際像素
        scaled_points = [
            (x * pix.width / layout_width, y * pix.height / layout_height)
            for x, y in points
        ]

        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(
            scaled_points, linewidth=1, edgecolor=box_color, facecolor="none"
        )
        ax.add_patch(rect)

    # 建立圖例
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category in ["Title", "Image", "Table"]:
        if category in categories:
            legend_handles.append(
                patches.Patch(color=category_to_color[category], label=category)
            )
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.show()


def render_page(doc_list: list, page_number: int, print_text=True) -> None:
    """渲染指定頁面的元素提取結果"""
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    page_docs = [
        doc for doc in doc_list if doc.metadata.get("page_number") == page_number
    ]
    segments = [doc.metadata for doc in page_docs]
    plot_pdf_with_boxes(pdf_page, segments)
    if print_text:
        for doc in page_docs:
            print(f"{doc.page_content}\n")
```

此時我們就能看到每一個PDF頁面裡面提取的元素了：

```python
render_page(docs_local, 1)  # 渲染第一頁
render_page(docs_local, 3)  # 渲染第三頁
```

表格識別效果會顯示出表格的精確分割和結構化數據提取。

### 代碼詳細解釋

#### plot_pdf_with_boxes 函式解釋

**1. PDF 頁面渲染**
```python
pix = pdf_page.get_pixmap()
pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
```
- 使用 PyMuPDF (fitz) 把一頁 PDF 渲染為像素圖（pixmap）
- 轉換為 PIL Image，方便後續視覺化

**2. 建立視覺化的背景**
```python
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(pil_image)
```
- 用 matplotlib 顯示 PDF 頁的圖像作為背景

**3. 定義顏色映射**
```python
category_to_color = {
    "Title": "orchid",
    "Image": "forestgreen",
    "Table": "tomato",
}
```
- 定義不同類別（標題、圖片、表格）對應的高亮顏色
- 其他未定義類別（例如普通文本）預設用 deepskyblue

**4. 座標處理和框架繪製**
```python
for segment in segments:
    points = segment["coordinates"]["points"]
    layout_width = segment["coordinates"]["layout_width"]
    layout_height = segment["coordinates"]["layout_height"]
```
- 遍歷所有 segments（這些是前面 Unstructured/OCR 輸出的結果，每個 segment 對應一個識別塊）
- 拿到每個塊的座標（通常是 PDF 頁面的相對座標，單位化為 0~1 或文檔尺寸）

**5. 座標縮放**
```python
scaled_points = [
    (x * pix.width / layout_width, y * pix.height / layout_height)
    for x, y in points
]
```
- 座標縮放：把邏輯座標映射到實際像素座標
- 這樣矩形框才能精確覆蓋到圖像上的正確位置

**6. 繪製多邊形框**
```python
rect = patches.Polygon(
    scaled_points, linewidth=1, edgecolor=box_color, facecolor="none"
)
ax.add_patch(rect)
```
- 使用 matplotlib.patches.Polygon 繪製多邊形框（通常是矩形）
- 設置 edgecolor 表示不同類型的顏色

**7. 建立圖例**
```python
legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
```
- 手動繪製圖例，幫助區分不同類型的標註框

#### render_page 函式解釋

**1. 加載指定頁面**
```python
pdf_page = fitz.open(file_path).load_page(page_number - 1)
```
- 打開 PDF，定位到第 page_number 頁

**2. 過濾頁面文檔**
```python
page_docs = [
    doc for doc in doc_list if doc.metadata.get("page_number") == page_number
]
segments = [doc.metadata for doc in page_docs]
```
- 從識別結果中過濾出屬於該頁的所有 doc（segment）
- 提取它們的元數據（metadata，裡面有 coordinates、category 等資訊）

**3. 視覺化渲染**
```python
plot_pdf_with_boxes(pdf_page, segments)
```
- 調用上面的方法，把這一頁繪製出來，並加上分塊框

**4. 文本輸出**
```python
if print_text:
    for doc in page_docs:
        print(f"{doc.page_content}\n")
```
- 如果需要，打印出 OCR/解析得到的實際文本

## 2.4 PDF逆向轉化為Markdown文檔

而更進一步的，我們就能將其轉化為markdown文檔：

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
    strategy="hi_res",            # 高解析度 OCR，適合複雜表格
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

運行結束後即可看到創建的md文檔，將PDF成功轉換為結構化的Markdown格式，保留了原有的層次結構和多模態元素。

### 完整代碼解釋

#### 1. 基礎準備
```python
import os
import fitz
from unstructured.partition.pdf import partition_pdf

pdf_path = "0.LangChain技術生態介紹.pdf"
output_dir = "pdf_images"
os.makedirs(output_dir, exist_ok=True)
```
- `fitz` → PyMuPDF 庫，用來讀取 PDF、提取圖片
- `partition_pdf` → Unstructured 提供的 PDF 解析介面，可以自動調用 OCR
- 設置 PDF 文件路徑和輸出目錄 `pdf_images`，保存提取出的圖片

#### 2. Step 1：提取文本與結構化內容
```python
elements = partition_pdf(
    filename=pdf_path,
    infer_table_structure=True,   # 開啟表格結構檢測
    strategy="hi_res",            # 高解析度 OCR，適合複雜表格
    ocr_languages="chi_sim+eng",  # 中英文混合識別
    ocr_engine="paddleocr"        # 指定 PaddleOCR 引擎
)
```
- 核心作用：調用 Unstructured 的分區解析，把 PDF 切分為 標題、正文、表格、圖片等元素
- `infer_table_structure=True` → 表格會被解析成結構化數據
- `strategy="hi_res"` → 高解析度 OCR，能更好地識別複雜排版（比如學術論文）
- `ocr_languages="chi_sim+eng"` → 同時支持中文簡體和英文
- `ocr_engine="paddleocr"` → 使用 PaddleOCR 引擎（比默認 Tesseract 更強大）

返回的 `elements` 是一個 Element 列表，每個元素有：
- `.category` → 類型（Title、Paragraph、Table、Image …）
- `.text` → 文本內容
- `.metadata` → 頁碼、座標、表格的 HTML 等資訊

#### 3. Step 2：提取圖片並保存
```python
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
```
- 使用 fitz 遍歷 PDF 的每一頁，提取圖片
- 每張圖片保存為 `page{頁碼}_img{索引}.png`
- `pix.n < 5` → 說明是 RGB 或灰度圖，可以直接保存
- 否則是 CMYK 色彩空間，需要轉為 RGB 再保存
- 最後存入 `image_map`，便於後續和 Markdown 內容對應

#### 4. Step 3：組裝 Markdown 文檔
```python
md_lines = []
inserted_images = set()  # 避免重複插入相同圖片

for el in elements:
    cat = el.category
    text = el.text
    page_num = el.metadata.page_number
```
- 遍歷前面解析出的所有 elements，根據不同類型拼接 Markdown

##### (1) 標題
```python
if cat == "Title" and text.strip().startswith("- "):
    md_lines.append(text + "\n")
elif cat == "Title":
    md_lines.append(f"# {text}\n")
elif cat in ["Header", "Subheader"]:
    md_lines.append(f"## {text}\n")
```
- `Title` → 轉換為 Markdown 一級標題 `#`
- `Header / Subheader` → 轉換為二級標題 `##`
- 特殊情況：如果標題開頭是 `-`，說明其實是列表項而不是標題，直接保持原樣

##### (2) 表格
```python
elif cat == "Table":
    if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
        from html2text import html2text
        md_lines.append(html2text(el.metadata.text_as_html) + "\n")
    else:
        md_lines.append(el.text + "\n")
```
- 如果表格有 HTML 格式（`text_as_html`），用 `html2text` 轉換為 Markdown 表格
- 否則直接寫入文本

##### (3) 圖片
```python
elif cat == "Image":
    for img_path in image_map.get(page_num, []):
        if img_path not in inserted_images:
            md_lines.append(f"![Image](./{img_path})\n")
            inserted_images.add(img_path)
```
- 對 Image 元素，插入對應的圖片路徑
- 使用 `inserted_images` 避免重複添加相同的圖片

##### (4) 普通文本
```python
else:
    md_lines.append(text + "\n")
```
- 其餘情況（正文段落等）直接作為普通文本寫入

#### 5. Step 4：寫入 Markdown 文件
```python
output_md = "output.md"
with open(output_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"✅ 轉換完成，已生成 {output_md} 和 {output_dir}/ 圖片文件夾")
```
- 把拼接好的 Markdown 行寫入 `output.md`
- 所有圖片保存在 `pdf_images/` 文件夾中
- 最終得到一個結構化良好的 Markdown 文件 + 圖片資源目錄，可直接用於 RAG

---

# 三、搭建基於多模態MarkDown文檔的Agentic RAG檢索引擎

在跑通了多模態文檔轉化之後，接下來我們基於轉化後的多模態MarkDown文檔來創建一個Agentic RAG引擎。

## 3.1 基礎環境搭建

### 創建項目主目錄

建立項目結構，包含主要的代碼文件和配置文件。

### 安裝基礎依賴

創建 `requirements.txt` 文件，包含以下依賴：

```text
pydantic
python-dotenv
langgraph
langchain-core
langchain-deepseek
langchain-tavily
langsmith
langchain-openai
langchain-text-splitters
langchain-community
faiss-cpu
langgraph_supervisor
graphrag
```

輸入如下命令完成安裝：

```bash
pip install -r requirements.txt
```

### 配置環境變量

創建 `.env` 文件，並輸入如下API-KEY：

```bash
DEEPSEEK_API_KEY=sk-c1a253**
OPENAI_API_KEY=sk-proj-gE**
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_b44**
LANGSMITH_PROJECT=langraph_studio_chatbot
```

### 安裝前端框架Agent Chat UI

```bash
# 如需要設置代理
# git config --global http.proxy http://127.0.0.1:10080
# git config --global https.proxy http://127.0.0.1:10080

git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
```

然後安裝前端依賴：

```bash
pnpm install
```

安裝LangGraph項目部署工具：

```bash
pip install -U "langgraph-cli[inmem]"
```

## 3.2 知識庫檢索數據集準備

接下來我們繼續準備檢索用的數據集，這裡我們採用此前系列公開課《MCP技術實戰》課件作為檢索材料，課件總共約6萬字。

接下來我們需要先將其逆向轉化為md文檔：

```python
import os
import fitz
from unstructured.partition.pdf import partition_pdf

pdf_path = "MCP實戰課件【合集】.pdf"
output_dir = "pdf_images"
os.makedirs(output_dir, exist_ok=True)

# Step 1: 提取文本/結構化內容
elements = partition_pdf(
    filename=pdf_path,
    infer_table_structure=True,   # 開啟表格結構檢測
    strategy="hi_res",            # 高解析度 OCR，適合複雜表格
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
                # 移除圖片參考，用文字說明替代
                md_lines.append("（圖片：相關技術架構圖或示例圖）\n")
                inserted_images.add(img_path)
    else:
        md_lines.append(text + "\n")

# Step 4: 寫入 Markdown 文件
output_md = "output.md"
with open(output_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"✅ 轉換完成，已生成 {output_md} 和 {output_dir}/ 圖片文件夾")
```

生成md文檔後，注意md版課件統一進行了圖床上傳，並且按照正則規則進行了數據清洗，整體文檔結構更加規範。

然後，為了搭建RAG系統，我們還需要對原始文檔進行處理，來創建詞向量數據庫：

```python
import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS

OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_BASE_URL = "https://ai.devtool.tech/proxy/v1"

embed = OpenAIEmbeddings(
    api_key=OPENAI_EMBEDDING_API_KEY,
    base_url=OPENAI_EMBEDDING_BASE_URL,
    model="text-embedding-3-small"
)

file_path = "MCP實戰課件【合集】.md"

with open(file_path, "r", encoding="utf-8") as f:
    md_content = f.read()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_content)

vector_store = FAISS.from_documents(md_header_splits, embedding=embed)
vector_store.save_local("mcp_course_materials_db")
```

生成的詞向量數據庫將用於後續的檢索任務。

## 3.3 多模態RAG系統開發

接下來編寫多模態RAG系統代碼：

```python
from __future__ import annotations

import os
import asyncio
from typing import Literal
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM & Embeddings
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-chat"
model = init_chat_model(model=MODEL_NAME, model_provider="deepseek", temperature=0)
grader_model = init_chat_model(model=MODEL_NAME, model_provider="deepseek", temperature=0)

embed = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://ai.devtool.tech/proxy/v1",
    model="text-embedding-3-small",
)

# ---------------------------------------------------------------------------
# Vector store & Retriever tool
# ---------------------------------------------------------------------------
VS_PATH = "mcp_course_materials_db"

vector_store = FAISS.load_local(
    folder_path=VS_PATH,
    embeddings=embed,
    allow_dangerous_deserialization=True,
)
retriever_tool = create_retriever_tool(
    vector_store.as_retriever(search_kwargs={"k": 3}),
    name="retrieve_mcp_course",
    description="Search and return relevant sections from the mcp course materials.",
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "您是一個MCP技術培訓助手。'MCP' 指的是 **Model Context Protocol**，"
    "一個開放的框架，用於讓大模型調用外部工具。請勿與Microsoft Certified Professional混淆。\n"
    "只回答與MCP實戰課程內容相關的問題，包括工具調用、串流、LangGraph、API設計等。"
    "如果用戶問題與課程無關，請回覆：'我不能回答與 MCP 技術實戰公開課無關的問題。' "
    "需要額外上下文時，您可以調用提供的工具 `retriever_tool`。"
)

GRADE_PROMPT = (
    "您是一個評估檢索文檔與用戶問題相關性的評分員。\n"
    "檢索文檔：\n{context}\n\n用戶問題：{question}\n"
    "如果相關請返回 'yes'，否則返回 'no'。"
)

REWRITE_PROMPT = (
    "您要重寫用戶問題，使其更貼近MCP技術實戰課程。\n"
    "注意：在此語境下，**MCP代表Model Context Protocol**，一個讓大模型使用外部工具和結構化API的開放框架。\n"
    "請勿將MCP解釋為Microsoft Certified Professional。\n"
    "您的工作是精煉或澄清用戶的問題，使其更好地符合Model Context Protocol課程的關鍵概念，如工具調用、工具註冊、串流API、LangGraph工作流等。\n\n"
    "原問題：\n{question}\n改進問題："
)

ANSWER_PROMPT = (
    "您是一個回答MCP技術實戰課程相關問題的助手。"
    "請使用提供的上下文盡可能完整和準確地回答問題。"
    "如果相關，請包含源材料中出現的示例、程式碼區塊。"
    "使用標準Markdown格式輸出。\n\n"

    "指導原則：\n"
    "- 優先使用三重反引號（```）引用程式碼片段以保持格式。\n"
    "- 如果答案未知或上下文中不存在，請說：'我不知道。'\n"
    "- 保持回應結構化，如需要請使用適當的Markdown區段。\n\n"

    "問題：{question}\n"
    "上下文：{context}"
)

# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------
async def generate_query_or_respond(state: MessagesState):
    """LLM決定直接回答或調用檢索工具。"""
    response = await model.bind_tools([retriever_tool]).ainvoke(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


class GradeDoc(BaseModel):
    binary_score: str = Field(description="相關性分數 'yes' 或 'no'。")


async def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """評估檢索文檔的相關性。"""
    question = state["messages"][0].content  # 原用戶問題
    ctx = state["messages"][-1].content      # 檢索器輸出
    prompt = GRADE_PROMPT.format(question=question, context=ctx)
    result = await grader_model.with_structured_output(GradeDoc).ainvoke([
        {"role": "user", "content": prompt}
    ])
    return "generate_answer" if result.binary_score.lower().startswith("y") else "rewrite_question"


async def rewrite_question(state: MessagesState):
    """重寫用戶問題使其更相關。"""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": resp.content}]}


async def generate_answer(state: MessagesState):
    """生成最終答案。"""
    question = state["messages"][0].content
    ctx = state["messages"][-1].content
    prompt = ANSWER_PROMPT.format(question=question, context=ctx)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [resp]}

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
workflow = StateGraph(MessagesState)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_edge("generate_query_or_respond", "retrieve")
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

rag_agent = workflow.compile(name="rag_agent")
```

### 代碼解釋

#### 1. 環境與依賴加載
```python
from __future__ import annotations
import os
import asyncio
from typing import Literal
from dotenv import load_dotenv
load_dotenv(override=True)
```
- `__future__.annotations`: 使 Python 3.7+ 支持延遲注解解析（避免循環引用等問題）
- `load_dotenv(override=True)`: 加載 `.env` 環境變量文件中的內容（如 API key），並允許覆蓋已有變量

#### 2. 初始化 LLM 模型與嵌入模型
```python
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

MODEL_NAME = "deepseek-chat"
model = init_chat_model(model=MODEL_NAME, model_provider="deepseek", temperature=0)
grader_model = init_chat_model(model=MODEL_NAME, model_provider="deepseek", temperature=0)
```
- `init_chat_model(...)`: 初始化 deepseek-chat 模型（來自 DeepSeek 的對話式大模型）
- `OpenAIEmbeddings(...)`: 使用 OpenAI 的 text-embedding-3-small 嵌入模型，將文本轉為向量用於向量檢索
- `model`：主模型，用於用戶對話處理
- `grader_model`：用於判斷文檔相關性的小助手模型

#### 3. 向量數據庫加載與檢索工具構建
```python
VS_PATH = "mcp_course_materials_db"
vector_store = FAISS.load_local(...)
retriever_tool = create_retriever_tool(...)
```
- 從本地加載一個名為 `mcp_course_materials_db` 的 FAISS 向量庫
- 使用 `create_retriever_tool` 構造了一個可供 LangGraph 工具調用的 Retriever 工具（用於查找與問題相關的文本塊）
- `k=3` 表示每次檢索返回3條上下文

#### 4. 提示詞設計（Prompt Engineering）

這些提示詞是對智能體行為的指令設計。

**系統提示詞**
```python
SYSTEM_INSTRUCTION = (...)
```
- 限定助手只能回答"Model Context Protocol (MCP)"技術實戰相關問題
- 如果不是相關問題，就回復：我不能回答與 MCP 技術實戰公開課無關的問題

**評估 Prompt**
```python
GRADE_PROMPT = (...)
```
- 指導 `grader_model` 判斷檢索結果是否與用戶提問相關，輸出 yes 或 no

**重寫 Prompt**
```python
REWRITE_PROMPT = (...)
```
- 如果用戶問題偏離主題，讓模型改寫問題，使其更貼近"工具調用 / LangGraph / MCP"等關鍵概念

**回答 Prompt**
```python
ANSWER_PROMPT = (...)
```
- 給模型一個問題和上下文，引導它用 Markdown、程式碼區塊等方式生成結構化答案

#### 5. LangGraph 節點邏輯定義

LangGraph 是有狀態多節點的圖結構，這裡定義了智能體對話的各個節點功能。

**generate_query_or_respond**
```python
async def generate_query_or_respond(state: MessagesState):
    ...
```
- 調用 LLM，根據當前訊息決定是否要調用 `retriever_tool`
- 本質上是一個具備工具調用能力的交互節點（如果上下文不足，模型會自動決定調用檢索器）

**grade_documents**
```python
async def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    ...
```
- 使用 `grader_model` 判斷：檢索到的文檔是否與提問有關
- 如果是 → `generate_answer`
- 如果不是 → `rewrite_question`

**rewrite_question**
```python
async def rewrite_question(state: MessagesState):
    ...
```
- 調用 LLM 改寫用戶問題，使其更符合 MCP 課程範疇

**generate_answer**
```python
async def generate_answer(state: MessagesState):
    ...
```
- 用 LLM + 上下文生成最終答覆
- 支援程式碼區塊與 Markdown 格式

#### 6. 構建 LangGraph 工作流
```python
workflow = StateGraph(MessagesState)
```

我們將所有邏輯節點通過圖結構組合成一個完整智能體流程：

```
START → generate_query_or_respond → retrieve → grade_documents
                                                      ↓
                                              [yes] generate_answer → END
                                                      ↓
                                              [no] rewrite_question → generate_query_or_respond
```

解釋如下：

| 步驟 | 節點名稱 | 功能 |
|------|----------|------|
| 1 | START | 起點 |
| 2 | generate_query_or_respond | 模型判斷是否調用工具 |
| 3 | retrieve | 調用檢索工具返回上下文 |
| 4 | grade_documents | 判斷檢索結果是否相關 |
| 5a | generate_answer | 生成回答 → 終點 |
| 5b | rewrite_question | 改寫問題 → 回到第2步 |

#### 7. 編譯智能體並生成入口
```python
rag_agent = workflow.compile(name="rag_agent")
```

最終將整個 LangGraph 圖編譯為可調用的 `rag_agent`，你可以在主函式中這樣調用：

```python
await rag_agent.ainvoke({"messages": [{"role": "user", "content": "MCP 是什麼？"}]})
```

實際測試效果可以針對MCP技術課程內容進行精準檢索和回答。

---

## 課程總結

本節課程展示了從零到一快速搭建多模態RAG引擎的完整流程：

1. **結構解析重建法**：使用 Unstructured + PaddleOCR 進行多模態PDF文檔解析
2. **元素提取與視覺化**：精確識別標題、表格、圖片等不同類型的內容元素
3. **Markdown轉換**：將PDF逆向轉化為結構化的Markdown文檔，保留層次關係
4. **Agentic RAG系統**：基於LangGraph構建智能檢索問答系統

通過本課程學習，您可以：
- 掌握多模態文檔處理的核心技術
- 理解結構化解析與重建的實現原理
- 學會搭建企業級的智能文檔檢索系統
- 具備處理複雜PDF文檔的實戰能力

---

體驗課內容節選自《2025大模型Agent智能體開發實戰》(秋招衝刺班)完整版付費課程。體驗課時間有限，若想深度學習大模型技術，歡迎大家報名由我主講的《2025大模型Agent智能體開發實戰》(秋招衝刺班)。

此外，若是對大模型底層原理感興趣，也歡迎報名由我和菜菜老師共同主講的《2025大模型原理與實戰課程》(秋招衝刺班)。

大模型秋招衝刺班開班特惠進行中，直播間享五折特價+全套SVIP新班特定福利，合購還有更多優惠！詳細資訊可掃碼添加助教，回覆「大模型」，即可領取課程大綱&查看課程詳情。

《2025大模型Agent智能體開發實戰》(秋招衝刺班)為【100+小時】體系大課，總共20大模組精講精析，零基礎直達大模型企業級應用！