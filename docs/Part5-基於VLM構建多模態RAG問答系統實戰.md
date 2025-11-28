# 基於VLM構建多模態RAG問答系統實戰

## 課程說明

- 體驗課內容節選自《2025大模型Agent智能體開發實戰》(秋季班)完整版付費課程
- 體驗課時間有限，若想深度學習大模型技術，歡迎大家報名由我主講的《2025大模型Agent智能體開發實戰》(秋季班)

此外，公開課訓練專案程式碼、資料，已上傳至課件網盤，聯繫助教回覆“vlm”，即可領取（目前諮詢人數較多，助教老師在加急一一回覆啦，小夥伴們發送後耐心等待一下哦🌹～）。

---

## 本期公開課案例功能介紹

- **核心功能一**：支援線上上傳並自動解析多模態PDF及CAD、工程圖紙和複雜架構原型圖；
- **核心功能二**：通過自然語言問答，直接檢索圖片原型及文檔原件，並支援溯源和線上預覽，實現“以文搜圖”、“以圖搜圖”；
- **核心功能三**：支援即時上傳多模態PDF及CAD、工程圖紙和複雜架構原型圖，並直接對文件內容進行提問，實現“以文搜文”；

本期公開課，我們要探討的是目前大模型技術領域非常前沿且具有實際應用落地的技術領域——多模態RAG系統。

在開始之前，大家可以先思考下在實際的開發需求是否存在類似這樣的問題：當你拿到一份包含複雜圖表、公式和文字說明的學術論文時，如果想快速找到某個實驗結果對應的圖表說明，你會怎麼做？傳統方法可能需要逐頁翻閱、人工對照，這無疑是個耗時的過程。而今天我們要學習的多模態RAG技術，讓AI不僅能"讀懂"文字，還能"看懂"圖像、理解表格、識別公式，甚至處理音視頻內容。設想一下，你只需上傳文檔，然後用自然語言提問："第5頁的實驗結果圖表說明了什麼？"，AI就能準確定位圖表並給出專業解答。這就是多模態RAG帶來的變革。

因此在本期公開課中，我將帶領大家從RAG的基礎概念出發，深入理解多模態擴展的技術原理，掌握主流的實現方案，並了解目前的落地應用。相信通過本節課程，大家能夠建立起對這一前沿技術的系統性認知。

# 一、多模態RAG的產品落地形式剖析

## 從真實需求出發，看見“多模態”的價值

過去我們用文字RAG解決“從海量文檔中找準依據”的問題，如今業務資料已高度多媒體化：投研報告的圖表、醫學影像與病歷、客服錄音、培訓視頻、設計圖紙與原始碼等。如果仍然只索引文字，就像只看“一個維度”的世界，關鍵資訊（版式、圖形、音訊語義、視頻時序）會被丟失。因此在目前的真實業務裡，我們很少只面對純文字：圖片裡的表格與圖表、視頻中的鏡頭與字幕、會議裡的語音、工程裡的程式碼與文檔，這些資訊共同構成了知識的全貌。傳統僅靠文字的RAG（Retrieval-Augmented Generation）在此時往往力不從心：它無法“看圖”、不會“聽音”、也難以“看視頻”。

下表中展示了多模態RAG的主流形態與典型場景：

| 類型 | 典型輸入 | 核心輸出 | 適用場景 |
| :--- | :--- | :--- | :--- |
| **視覺問答型** | 圖片、文檔頁面截圖 | 基於圖像的問答與解釋 | 智能客服、報表/圖表理解 |
| **多模態搜索型** | 文字/圖像/音訊/視頻混合 | 跨模態檢索結果與證據 | 企業知識庫、媒體庫檢索 |
| **視頻理解型** | 視頻幀+字幕/音訊 | 片段定位與內容問答 | 課程/直播/培訓視頻問答 |
| **語音/音訊處理型** | 會議錄音、通話音訊 | 轉寫+基於內容的問答 | 會議紀要檢索、客服質檢 |
| **程式碼+文檔檢索型** | 原始碼+技術文檔 | 程式碼片段定位與說明 | 工程知識庫、研發助手 |

這類應用的直觀感受是：你不再需要關心“文件格式”，而是用自然語言直接提問，系統跨模態檢索到相關證據，再組織出可核驗的回答。這正是多模態RAG的“可用之處”。

## 如何判斷你的需求是不是多模態RAG？

多模態RAG的通用鏈路可以概括為：資料採集 → 解析與預處理（如OCR、切幀、轉寫） → 跨模態向量化與對齊（如CLIP/Whisper） → 統一向量索引與存儲 → 檢索重排與證據拼接 → 大模型生成與來源標註 → 反饋閉環與評估迭代。實踐中我們會因場景（如醫療、法律、金融）而替換具體組件，但鏈路形態不變。

| 需求場景（你遇到的情況） | 推薦形態 | 關鍵做法/理由 | 是否多模態RAG |
| :--- | :--- | :--- | :--- |
| 報表/論文/說明書解析，需要“看圖說話” | 視覺問答型/文檔按圖像檢索 | 保留版式、公式、表格結構，不丟關鍵資訊 | 是 |
| 希望“以文搜圖/以圖搜圖/跨模態檢索”，資料庫含文字+圖像/音視頻 | 多模態搜索型 | 不同模態向量化進同一空間，統一索引，無需區分格式 | 是 |
| 大量課程視頻/培訓錄影，想問“某段視頻發生了什麼” | 視頻理解型 | 切幀、抽字幕/音訊，構建時間感知索引再問答 | 是 |
| 以會議錄音/客服通話為主，希望先轉文字再問答且可回溯原音 | 語音/音訊處理型 | Whisper轉寫 + 檢索 + 生成，效率與可追蹤兼顧 | 是 |
| 知識主要在程式碼+文檔裡，想同時定位API實現與設計說明 | 程式碼與文檔混合檢索 | 原始碼與文檔統一索引，一次問清程式碼與說明 | 是 |

# 二、多模態RAG的基礎：傳統RAG

檢索增強生成（Retrieval-Augmented Generation, RAG） 的核心思想非常直觀：在大模型生成回答之前，先從外部知識庫中檢索相關資訊供模型參考。這就像我們在回答問題前先查閱資料一樣。

傳統的RAG系統存在一個明顯的短板——它們主要面向純文字場景。想像這樣的場景：
- 一份技術文檔中包含架構示意圖，但OCR只能提取"圖1：系統架構"這樣的標註文字，圖中的模組關係、資料流向等關鍵資訊完全丟失
- 一張財務報表的複雜表格結構，轉成純文字後行列關係混亂，很難準確回答"第二季度銷售額"這類問題
- 學術論文中的數學公式，即使識別成文字也難以理解其含義

這些場景暴露出一個根本問題：現實世界的知識不僅以文字形式存在，圖像、圖表、公式、音視頻都承載著豐富的資訊。比如用戶先上傳一些圖片、音訊，然後提出一個關於這些內容的問題，傳統的RAG系統是根本無法處理的。

正是基於這樣的需求，多模態RAG應運而生。它將RAG的思想拓展到圖像、音訊、視頻等多種資料形式。我們可以把多模態RAG理解為讓AI具備了"眼睛"和"耳朵"的檢索增強系統。這涵蓋了多種應用場景，例如：給定一張圖片或截圖，讓系統回答其中內容的問題；或者提供包含文字、圖表、公式的 PDF 文檔，讓系統整合文中圖文資訊進行問答。

當前，大模型（如 GPT-4o、Claude 3.5 Sonnet、Qwen-VL）已經展現出基礎的多模態能力，能夠將圖像與文字混合作為輸入並生成回答。例如 GPT-4 的視覺版允許用戶上傳圖像並就其提問，實現視覺問答。谷歌的 Bard 也增加了圖像輸入的支援，能夠分析用戶插入的圖片內容進行對話。同樣地，在文檔分析領域，一些應用可以處理 PDF 中的表格、圖片甚至公式，實現對複雜報告的問答理解。可以說，多模態 RAG 正在將 AI 應用從純文字擴大到“所見即所得”的更廣闊資訊空間。

多模態RAG的目標非常明確：在查詢和知識庫兩端都支援多模態——既能理解多模態形式的問題，又能檢索和利用文字以外的豐富資訊來源。用一句話概括，就是實現"Ask in Any Modality"——用任意模態提問並得到專業答案。

# 三、多模態RAG的核心概念

接下來我們深入理解多模態RAG的具體含義。簡單來說，多模態RAG是將檢索增強技術應用於多模態資料的系統。它可以處理不同模態的查詢和知識，包括圖像、語音、文字、視頻等。根據目前的技術發展及落地實踐經驗，多模態RAG主要涵蓋兩個維度：

- **維度一：富媒體文檔問答**
  這是最常見的應用形態。給定包含文字、圖片、表格、公式的PDF文檔，系統需要：
  1. **解析階段**：將文檔中的視覺和文字資訊分別提取
  2. **索引階段**：建立跨模態的知識檢索源
  3. **查詢階段**：用戶用自然語言提問
  4. **生成階段**：系統結合文字與圖片內容作答

  這要求系統能夠理解圖中的結構、表格資料，並將其與文字一起作為知識檢索源。而完整的流程是：系統先解析出其中的視覺和文字資訊，然後用戶可以通過自然語言提問，讓系統結合文字與圖片內容作答。這需要系統理解圖中的結構或表格資料，並將其與文字一起作為知識檢索源。從技術的角度上來看，我們需要做到這樣：

- **維度二：非文字輸入問答**
  這個維度更進一步——查詢本身就是圖像或音訊等模態。比如：
  - 直接上傳一張產品故障圖片，讓AI診斷問題
  - 提供一段音訊，讓系統分析其中討論的主題
  - 給出視頻片段，詢問其中的關鍵資訊

  這種情況下，系統需要先將非文字輸入轉化為可處理的資訊（通過圖像理解模型或語音識別），然後從多模態索引的知識庫中檢索相關資訊。從技術的角度我們需要做到這樣：

但實際上，多模態RAG並不只是簡單地把圖片和文字混在一起處理？在落地應用中遠比這複雜。它涉及檢索管線和生成模型兩方面的多模態處理：

| 技術環節 | 核心挑戰 | 解決方案 |
| :--- | :--- | :--- |
| **資料預處理** | 如何從複雜文檔中提取結構化資訊 | OCR、表格識別、公式解析 |
| **向量表示** | 如何將不同模態映射到統一語義空間 | 多模態嵌入模型（CLIP等） |
| **檢索匹配** | 如何實現跨模態的相似度計算 | 統一向量空間或多管線檢索 |
| **答案生成** | 如何融合多模態資訊生成回答 | 視覺語言模型（VLM） |

這張表格展示了多模態RAG的核心技術鏈條。並且值得注意的是，多模態 RAG 涉及檢索管線和生成模型兩方面的多模態處理。檢索階段需要能理解圖像、音訊等內容的索引和搜索演算法；生成階段則可能需要具備視覺、語言混合理解能力的模型（例如視覺語言模型，Vision-Language Model）來綜合檢索結果回答問題。因此，多模態 RAG 通常被視為對傳統文字RAG能力的擴展和增強，使系統能夠Ask in Any Modality（用任意模態提問並得到答案），每個環節都需要專門的技術模組支撐，這也是為什麼多模態RAG被視為對傳統文字RAG能力的重大擴展和增強。

至此，相信大家已經非常清晰地明白了多模態RAG的核心理念。接下來我們需要重點來看實際工程中如何構建這樣的系統。

# 四、多模態技術實現路線深度剖析

根據目前的技術發展及實踐經驗，當前主要有三種技術路線，每種都有其適用場景和權衡考量。接下來我將逐一介紹每種路線的核心思想、典型代表、優勢與局限。

## 4.1 技術路線一：統一向量空間檢索

這種方法的理念非常優雅：使用多模態嵌入模型將不同模態的資訊投影到同一向量空間。就像把不同語言翻譯成"世界語"一樣，無論是圖像還是文字，都用同一種向量表示方式。

典型代表就是CLIP模型，其中OpenAI的CLIP模型是這一路線的標杆。它可以將圖像和文字編碼到同一語義向量空間中，實現：
- 文字查詢可以直接檢索到相關圖像
- 圖像查詢可以匹配到描述文字
- 跨模態的相似度計算變得簡單直接

如果大家還不太理解CLIP的話，可以簡單將其想像成一個“翻譯器”，把圖像翻譯成一種向量語言（表示語義的向量），也把文字翻譯成相同 “語言”的向量。然後在這個向量“語言”裡做比較：圖像向量 vs 文字向量 — 看它們在向量空間裡的距離 /相似性。所以CLIP能對圖像和文字都進行編碼，如下所示：

目前有很多諸如 OpenAI / Qwen（或其生態）提供線上／API 模型或服務，可以支援如上圖所示的檢索＋融合 / 多模態輸入的能力。比如：OpenAI 的官方Cookbook 提供了一個示例，先對圖像做 CLIP 類 embedding（或類似方法）來做檢索，然後把檢索出的上下文 + 圖像內容組合起來給 GPT 模型做多模態 reasoning。這就是 embedding 圖像 (用 CLIP) 來做相似性檢索，然後把結果與 prompt 一起輸入模型回答的非常經典的技術實現過程。
https://cookbook.openai.com/examples/custom_image_embedding_search

其中最小可運行的核心程式碼如下所示：
```python
import torch
import clip  # 可以用 openai 的 clip 包，或者用 open_clip 等替代
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加載模型 + 圖像預處理器
model, preprocess = clip.load("ViT-B/32", device=device)

# 準備圖片列表
image_paths = ["dog.jpg", "cat.jpg", "car.jpg"]
images = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in image_paths]
images = torch.cat(images, dim=0).to(device)  # shape (N, 3, H, W)

# 準備文字查詢（可以多個描述）
texts = ["a photo of a dog", "a photo of a cat", "a photo of a vehicle"]
tokens = clip.tokenize(texts).to(device)

# 計算圖像和文字的嵌入
with torch.no_grad():
    image_embeddings = model.encode_image(images)  # 形狀 (N, D)
    text_embeddings = model.encode_text(tokens)    # 形狀 (M, D)

# 通常會做歸一化，這樣內積就等價於餘弦相似度
image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

# 計算相似度矩陣：每個文字與每張圖像的相似度
similarity = text_embeddings @ image_embeddings.T  # (M, N)

# 輸出最匹配的圖片 index
for i, txt in enumerate(texts):
    scores = similarity[i]  # 對應每張圖片的分數
    best_idx = scores.argmax().item()
    print(f"文字 ‘{txt}’ 最匹配圖片：{image_paths[best_idx]} (分數 {scores[best_idx].item():.4f})")
```
上述程式碼示例可以做“文字 → 圖像檢索”。同理，如果你有圖片查詢，也可以把圖片的 embedding 與所有文字 embedding 做相似度比對，就可以做“圖像 → 文字檢索”。其核心流程為：
- 用 clip.load(...) 加載預訓練模型
- 用 model.encode_image() 和 model.encode_text() 分別對圖像和文字編碼
- 對兩個嵌入向量做歸一化
- 計算內積（或餘弦相似度）來評估文字與圖像之間的匹配性

同時，這個基本流程就是用 CLIP 做跨模態檢索的核心。目前超90%的落地專案都是在這個基礎上再加上索引（例如用 FAISS、Milvus、Pinecone 等），以及批處理、緩存、加速等最佳化。

該技術路線實現過程還是比較複雜的，公開課時間有限，這裡就不再展開介紹。如若想深入學習的小夥伴，可以報名我們的正式課程。大模型秋季班封班特惠進行時，直播間享五折特價+全套SVIP新班特定福利，合購還有更多優惠哦~詳細資訊掃碼添加助教，回覆“大模型”，即可領取課程大綱&查看課程詳情👇

這種技術路線的優勢是架構簡潔，檢索階段不區分模態，只需查詢一個向量資料庫，但是局限越狠明顯，訓練統一模型非常困難，當前多模態嵌入模型往往只針對兩兩模態（如圖文）效果較好，對於更多模態或複雜格式（自然圖像 vs. 掃描文檔 vs. 圖表）泛化能力不足。

因此，在實際應用中，這種方案需要大量訓練資料和精調，而且對於包含特殊結構的資訊（如公式、合成圖表）效果可能不理想。

這裡要注意的是：在多模態 RAG（Multimodal RAG）裡，有一條重要的區別／設計抉擇，就是“用多模態模型直接解析 + 問答”與“用檢索 + 生成（RAG）”這兩條路徑的關係和優劣：

| 路徑 | 核心流程 /機制 | 輸入／輸出特徵 | 優點 | 缺點 /挑戰 |
| :--- | :--- | :--- | :--- | :--- |
| **直接多模態解析 /問答** | 給定圖像 + 文字 prompt → 多模態模型（例如 Qwen-Omni、VL 模型）內部理解 + 推理 → 直接輸出答案 | 輸入可能是圖像 + 文字，輸出是文字（或語音） | 簡潔，不需要檢索模組、向量資料庫、索引、召回等流程；適合即時互動 | 模型容量 /知識覆蓋受限；容易 “忘記”長尾知識或外部知識；當問題涉及知識庫內容或歷史文檔，模型可能石沉大海（hallucination 風險高） |
| **檢索增強（RAG）路線** | 先把知識庫裡的圖文 / 多模態內容編碼成向量、做索引 /檢索；給定用戶 query（可能包含圖 + 文字資訊），檢索最相關資料；把這些檢索結果 + query 一起交給多模態 / 混合模型生成答案 | 有檢索模組 + 向量資料庫 + 編碼器 + 生成模型 | 能顯著擴展知識覆蓋範圍、增強外部知識支援、減少模型的“記憶負擔”、提升答案可驗證性 | 檢索質量、向量表示對齊、模態差異對齊、查詢-檢索-融合策略設計複雜；若檢索結果無關或噪聲，會誤導生成模型 |

- **直接多模態模型解析**
```python
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

client = OpenAI(
    # 若沒有配置環境變數，請用阿里雲百煉API Key將下行替換為：api_key="sk-xxx",
    # 新加坡和北京地域的API Key不同。獲取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 以下是北京地域base_url，如果使用新加坡地域的模型，需要將base_url替換為：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-omni-flash", # 模型為Qwen3-Omni-Flash時，請在非思考模式下運行
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                    },
                },
                {"type": "text", "text": "圖中描繪的是什麼景象？"},
            ],
        },
    ],
    # 設置輸出資料的模態，當前支援兩種：["text","audio"]、["text"]
    modalities=["text", "audio"],
    audio={"voice": "Cherry", "format": "wav"},
    # stream 必須設置為 True，否則會報錯
    stream=True,
    stream_options={
        "include_usage": True
    }
)

for chunk in completion:
    if chunk.choices:
        print(chunk.choices[0].delta)
    else:
        print(chunk.usage)
```
## 4.2 技術路線二：多路並行檢索

既然統一空間很難，那就針對每種模態各自建立獨立的檢索管線和索引.針對每種模態各自建立獨立的檢索管線和索引。例如文字用文字向量索引，圖像用圖像向量索引，音訊用音訊索引。當收到查詢時，讓它並行地查詢多個檢索器，各自取 Top K 結果，然後匯總所有模態的結果提供給生成模型。

這樣做的好處是保持了每種模態檢索的專業性，不需要一個模型通吃所有模態。然而缺點也很明顯：第一，返回的候選片段數量會成倍增加，最終可能需要在生成階段處理海量跨模態資訊；第二，生成模型本身必須能同時理解多模態輸入，否則無法把不同來源的資訊融合起來。因此，多路並行方案實際是將問題從檢索階段轉移到了生成階段，並帶來了更高的計算開銷。在工程中，這種方案一般用於小規模實驗或配合強多模態模型時採用，但並非主流。

## 4.3 技術方案三：轉化為統一模態（文字）處理

這是目前應用最廣泛也最務實的方案，即將所有非文字資訊在預處理階段轉成文字表示。“統一以文字為基礎”也被稱作模態歸一化（grounding），例如對圖像運行OCR提取文字說明，對表格轉成CSV/文字，對音訊跑語音識別得到文字，對視頻提取字幕或說明性文字。通過這一過程，把多模態內容全部變成可索引的文字塊，再用常規文字向量檢索技術構建索引。查詢時同樣將問題轉成文字向量檢索相關片段，然後提供給語言模型生成答案。這種方法的優點在於架構簡單、複用成熟的文字RAG技術，避免了訓練複雜多模態模型。

比如很多文檔問答產品直接對PDF進行文字抽取和OCR，把圖文混排的內容轉成純文字索引，讓大模型基於提取的文字回答問題。

對於含有大量文字的圖像（如掃描文檔、截圖）和結構化資料（如表格，提取成文字表述）而言，這種方案相對有效。但缺點是可能損失模態專有的資訊和細節。例如OCR無法捕獲圖片中的視覺圖形含義，表格純文字可能丟失單元格對應關係，公式轉成文字往往不可讀。儘管如此，在多模態大模型尚未普及前，這是工業界落地最穩妥快捷的路線，也常與大模型結合使用（如讓大模型先讀OCR文字，再回答）。

# 5. 【實戰】多模態RAG系統架構設計與實現

在了解了多模態RAG的核心概念和實現路線後，接下來我們將通過一個實戰專案，為大家展示如何從零開始構建一個完整的多模態RAG系統。

接下來，我們就深入探索如何構建一個基於多模態RAG的CAD圖紙智能問答鏈路。相信大家在工業製造、建築設計等領域中，都會遇到大量的技術圖紙需要管理和查詢。傳統的方式是打開圖紙逐個查看，效率低下且容易遺漏關鍵資訊。

接下來我們就從零開始，逐步實現一個能夠"讀懂"CAD圖紙、自動提取關鍵資訊、並智能回答用戶問題的系統，其核心實現思路如下：
第一步：接入VLM模型
↓
第二步：解析本地CAD圖片
↓
第三步：提取結構化元資料
↓
第四步：存入向量資料庫
↓
第五步：智能問答（直接問答 + 圖像檢索）

- **處理圖片示例**

這套系統可以直接應用於：
- **房地產銷售**：快速回答客戶關於戶型的問題（"有幾個臥室？"、"主臥面積多大？"）
- **室內設計**：分析戶型優缺點，提供設計建議
- **智能選房**：根據用戶需求（如"3室2廳，面積100平以上"）自動篩選戶型
- **戶型對比**：智能對比多個戶型的優劣

同時，只要針對性地修改提示詞，即可快速遷移到其他的圖像分析場景。

## 5.1 環境準備與依賴安裝

首先，我們需要安裝必要的Python包。這個系統的核心依賴包括：
- **openai**：用於調用多模態大模型API
- **chromadb**：向量資料庫，用於存儲和檢索
- **langchain**：RAG框架的核心組件
- **Pillow**：圖像處理庫

```python
# 安裝依賴（如果需要）
# !pip install openai chromadb langchain langchain-community langchain-openai Pillow python-dotenv -q

# 導入必要的庫
import os
import json
import base64
import io
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
from dataclasses import dataclass
import sys
# 添加專案路徑（使用絕對路徑）
project_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
backend_path = project_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# OpenAI SDK
from openai import OpenAI

# LangChain 組件
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qwen_embeddings import QwenEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
```
上述程式碼中，導入了構建多模態RAG系統所需的所有核心庫：
- **openai**：提供了與 gpt-4o 等視覺語言模型互動的介面
- **chromadb 相關**：通過 langchain_community.vectorstores.Chroma 實現向量存儲
- **HuggingFaceEmbeddings**：用於將文字轉換為向量表示
- **PIL.Image**：處理圖像文件的加載和轉換

至此，我們的環境已經準備完畢，接下來就開始真正的系統構建。

## 5.2 接入視覺語言模型（VLM）

多模態RAG的核心能力來自於視覺語言模型（Vision-Language Model, VLM）。這類模型能夠同時理解圖像和文字，對於CAD圖紙這種技術圖像來說，VLM可以識別其中的結構、尺寸標註、技術參數等關鍵資訊。

本課程中，我們使用 gpt-4o 作為VLM模型。首先，我們需要配置API密鑰和模型接入點。需要配置三個關鍵參數：
- **API_KEY**：你的OpenAI API密鑰（或相容的API服務密鑰）
- **BASE_URL**：API服務的基礎URL
- **MODEL_NAME**：使用的模型名稱（這裡是 gpt-4o）

提示：如果你使用的是OpenAI官方API，BASE_URL 設置為 https://api.openai.com/v1 即可。
```python
from dotenv import load_dotenv

# ========== VLM 模型配置 ========== 
MODEL_NAME = "gpt-4o"

load_dotenv(override=True)

# 初始化 OpenAI 客戶端
vlm_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
```
這段程式碼創建了一個 OpenAI 客戶端實例，這是我們與視覺語言模型互動的橋樑。通過這個客戶端，我們後續可以發送圖像和問題給模型，並接收模型的分析結果。

接下來，我們需要構建一個CAD圖紙分析器，它能夠將圖像轉換為模型可以理解的格式，並調用VLM進行智能分析。

## 5.3 構建CAD圖紙分析器

多模態RAG的核心能力來自於視覺語言模型（VLM）。對於室內平面圖來說，VLM需要能夠識別：
- 房間佈局和功能區劃分
- 尺寸標註和面積計算
- 門窗位置和開啟方向
- 家具佈置和空間利用
- 動線設計和連通關係

在實際應用中，針對不同類型的圖紙（CAD、平面圖、架構圖等），我們需要設計不同的提示詞範本，以獲得最佳的分析效果。
```python
@dataclass
class AnalysisResult:
    """圖像分析結果資料類"""
    answer: str  # VLM的回答
    extracted_info: Dict[str, Any]  # 提取的結構化資訊
    raw_response: str  # 原始響應內容


class FloorPlanAnalyzer:
    """室內平面圖分析器"""
    
    # 平面圖專業提示詞範本
    FLOOR_PLAN_PROMPT = """
你是一位專業的建築/室內平面圖分析專家。請仔細分析這張室內平面佈置圖。

**用戶問題：**
{question}

**重要說明：**
- 這是一張室內平面佈置圖，包含房間、尺寸標註、家具佈置、動線等資訊
- 圖中尺寸單位通常為毫米(mm)或米(m)，請根據數值大小推斷
- 請仔細識別所有可見的房間、標註、符號和空間關係

**分析維度（根據用戶問題選擇性回答）：**

1. **房間/功能區識別**：
   - 識別所有房間名稱（客廳、臥室、廚房、衛生間、陽台等）
   - 標註每個房間的位置（左上/右下/中央等方位）
   - 識別特殊功能區（儲藏室、玄關、衣帽間等）

2. **尺寸與面積**：
   - 提取圖中所有可見尺寸標註
   - 推斷單位並統一換算為米(m)
   - 計算房間的長、寬、面積
   - 標註整體平面外牆尺寸

3. **符號與標註**：
   - 解釋符號含義（虛線、箭頭、紅點/紅線、軸線等）
   - 識別文字標註（房間編號、面積、備註）
   - 說明牆體類型、門窗位置和開啟方向

4. **家具佈局**：
   - 列出所有可見家具及其位置
   - 判斷空間利用率（擁擠/適中/空曠）
   - 識別家具尺寸

5. **動線與連通性**：
   - 標出主入口、次入口位置
   - 描述主要動線路徑（如："入口→玄關→客廳→..."）
   - 列出房間連通關係（哪些房間相連）
   - 判斷佈局類型（開放式/分隔式）

6. **設計評估**（如果問題涉及）：
   - 動線合理性、是否有繞行或死角
   - 採光/朝向分析
   - 空間最佳化建議

**回答方式：**
- 首先直接、簡潔地回答用戶的具體問題
- 然後提供相關的詳細資訊（如果用戶問某个房間，重點描述該房間）
- 如果用戶問整體佈局，提供全局分析
- 如果涉及尺寸計算，請說明推理過程（如："標註22720mm = 22.72m"）

**輸出格式（JSON）：**
{{
    "answer": "直接回答用戶問題的核心內容（簡潔明了）",
    "extracted_info": {{
        "total_dimensions": {{
            "length": 22.72,
            "width": 12.5,
            "unit": "m",
            "total_area": 284.0
        }},
        "rooms": [
            {{
                "name": "客廳",
                "position": "中央偏右",
                "dimensions": {{"{'length': 5.79, 'width': 4.2, 'area': 24.3, 'unit': 'm'}}",
                "furniture": ["沙發", "茶几"],
                "connected_to": ["餐廳", "臥室1"],
                "windows": 2,
                "doors": 1
            }}
        ],
        "annotations": [
            {{"type": "dimension", "value": "22720", "parsed_value": 22.72, "unit": "m", "description": "外牆總長"}}
        ],
        "symbols": [
            {{"type": "door", "count": 5, "positions": ["客廳-餐廳", "臥室1入口"]}}
        ],
        "circulation": {{
            "main_entrance": "底部中央",
            "main_path": "主入口 → 玄關 → 客廳 → 餐廳",
            "layout_type": "開放式客餐廳"
        }},
        "design_notes": ["主臥帶獨立衛生間", "動線流暢"]
    }}
}}

**注意事項：**
- 如果標註不清晰，標註為"不可讀"或給出估算值並說明
- 優先回答用戶的具體問題，不要羅列所有資訊
- 如果用戶問"有幾個臥室"，就重點回答臥室數量和位置
- 如果用戶問"客廳面積"，就重點回答客廳的尺寸和面積
- 保持答案簡潔、針對性強"""
    
    def __init__(self, client: OpenAI, model_name: str):
        """初始化分析器"""
        self.client = client
        self.model_name = model_name
    
    def load_image(self, image_path: str) -> Image.Image:
        """加載本地圖片"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"圖片文件不存在: {image_path}")
        
        image = Image.open(image_path)
        print(f"圖片加載成功: {image.size}")
        return image
    
    def image_to_base64(self, image: Image.Image, max_size: int = 2000) -> str:
        """將PIL Image轉換為base64字符串"""
        # 如果圖片過大，進行壓縮
        if image.width > max_size or image.height > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"圖片已壓縮到: {image.size}")
        
        # 轉換為JPEG格式的base64
        buffer = io.BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        print(f"圖片轉換為base64: {len(base64_str) / 1024:.1f} KB")
        return base64_str
    
    def analyze(self, image_path: str, question: str) -> AnalysisResult:
        """
        分析平面圖
        
        Args:
            image_path: 圖片路徑
            question: 用戶問題
            
        Returns:
            AnalysisResult對象
        """
        
        # 1. 加載圖片
        image = self.load_image(image_path)
        
        # 2. 轉換為base64
        image_base64 = self.image_to_base64(image)
        
        # 3. 構建提示詞
        prompt = self.FLOOR_PLAN_PROMPT.format(question=question)
        
        # 4. 調用VLM API
        print("正在調用VLM模型...")
        messages = [
            {
                "role": "system",
                "content": "你是一位專業的建築平面圖分析專家。請仔細分析圖像並按照要求的JSON格式返回結果。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=4096,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # 5. 解析響應
        content = response.choices[0].message.content
        parsed = self._parse_json_response(content)
        
        print(f"分析完成！Token使用: {response.usage.total_tokens}")
        print("="*60 + "\n")
        
        return AnalysisResult(
            answer=parsed.get('answer', ''),
            extracted_info=parsed.get('extracted_info', {}),
            raw_response=content
        )
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析JSON響應"""
        try:
            # 清理可能的markdown程式碼塊標記
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON解析失敗: {e}")
            return {
                'answer': content,
                'extracted_info': {}
            }

# 創建平面圖分析器
analyzer = FloorPlanAnalyzer(vlm_client, MODEL_NAME)
```
這段程式碼是平面圖分析系統的核心。讓我們理解幾個關鍵點：

1. **提示詞針對平面圖最佳化**
   平面圖的提示詞更加注重：
   - 房間識別：客廳、臥室、廚房等功能區
   - 尺寸推斷：自動判斷單位是mm還是m（如22720mm→22.72m）
   - 動線分析：入口→玄關→客廳的流動路徑
   - 空間關係：房間之間的連通性和位置關係

2. **結構化輸出的重要性**
   輸出的JSON結構包含了完整的戶型元資料：
   - `total_dimensions`：整體尺寸和總面積
   - `rooms`：每個房間的詳細資訊（名稱、位置、面積、家具）
   - `circulation`：動線設計和佈局類型
   - `design_notes`：設計特點和建議

   這些結構化資訊將成為後續智能問答的核心資料源！

## 5.4 測試平面圖分析

接下來讓我們測試分析器的功能。

提示：請準備一張平面圖（戶型圖），替換下面的路徑後運行。
```python
# ========== 測試平面圖分析 ========== 

# 指定平面圖路徑（請替換為你的平面圖路徑）
FLOOR_PLAN_PATH = "./test_data/house1.png"  # 示例路徑

# 用戶問題
USER_QUESTION = "請詳細分析這張平面圖，包括房間佈局、尺寸面積、動線設計等資訊。"

# 執行分析
result = analyzer.analyze(FLOOR_PLAN_PATH, USER_QUESTION)

# 顯示分析結果
print(result.answer)
print("\n【提取的結構化元資料】")
print(json.dumps(result.extracted_info, ensure_ascii=False, indent=2))
```
接下來，我們需要將這些分析結果存儲到向量資料庫中，以支援高效的檢索和問答。

## 5.5 構建向量資料庫存儲系統

在多模態RAG系統中，向量資料庫扮演著至關重要的角色。它不僅存儲了圖紙的文字描述，還保存了提取的結構化元資料，使得我們可以：
1. **語義檢索**：根據用戶問題的語義，而非關鍵詞匹配，找到相關圖紙
2. **元資料過濾**：基於結構化資訊（如尺寸、材料等）进行精确筛选
3. **高效索引**：即使有成千上萬張圖紙，也能毫秒級返回結果

向量資料庫的核心思想是將文字轉換為高維向量（Embedding），相似的文字在向量空間中距離較近。當用戶提問時，問題也被轉換為向量，然後通過計算距離找到最相關的文檔。
> 文字內容 → Embedding模型 → 向量表示 → 存儲到ChromaDB
> 用戶問題 → Embedding模型 → 問題向量 → 相似度檢索 → 返回Top-K結果

接下來，我們就來實現這個向量存儲系統。
```python
class VectorStoreManager:
    """向量資料庫管理器 - 基於ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db_floor_plan"):
        """初始化向量資料庫"""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化 Embedding 模型
        print("正在初始化 Qwen Embedding 模型...")
        self.embeddings = QwenEmbeddings(
            model="text-embedding-v4",
            api_key=os.getenv("DASHSCOPE_API_KEY")  # 從環境變數讀取
        )
        
        # 初始化 ChromaDB
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="floor_plans"
        )
        
        # 文字分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        print(f"向量資料庫初始化完成")
    
    def add_document(
        self,
        file_id: str,
        file_name: str,
        content: str,
        extracted_info: Dict[str, Any]
    ) -> int:
        """添加戶型文檔到向量庫"""
        print(f"\n添加文檔到向量庫: {file_name}")
        
        # 1. 分割文字
        chunks = self.text_splitter.split_text(content)
        print(f"  文字分割為 {len(chunks)} 個塊")
        
        # 2. 創建Document對象
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "file_id": file_id,
                "file_name": file_name,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "extracted_info_json": json.dumps(extracted_info, ensure_ascii=False)
            }
            
            # 提取關鍵字段到元資料頂層（便於過濾和問答）
            if "total_dimensions" in extracted_info:
                dims = extracted_info["total_dimensions"]
                metadata["total_area"] = float(dims.get("total_area", 0))
                metadata["total_length"] = float(dims.get("length", 0))
                metadata["total_width"] = float(dims.get("width", 0))
            
            if "rooms" in extracted_info:
                rooms = extracted_info["rooms"]
                metadata["room_count"] = len(rooms)
                # 統計臥室數量
                bedrooms = [r for r in rooms if "臥" in r.get("name", "")]
                metadata["bedroom_count"] = len(bedrooms)
            
            if "circulation" in extracted_info:
                circ = extracted_info["circulation"]
                metadata["layout_type"] = circ.get("layout_type", "")
            
            documents.append(Document(
                page_content=chunk,
                metadata=metadata
            ))
        
        # 3. 添加到向量庫
        ids = [f"{file_id}_chunk_{i}" for i in range(len(documents))]
        self.vector_store.add_documents(documents, ids=ids)
        
        print(f"文檔已添加，共 {len(documents)} 個文字塊")
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """向量檢索"""
        print(f"\n執行向量檢索: {query[:50]}...")
        
        # 執行相似度檢索
        results = self.vector_store.similarity_search_with_score(
            query, 
            k=top_k
        )
        
        # 格式化結果
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity": float(1 - score)
            })
        
        print(f"✓ 找到 {len(formatted_results)} 個相關結果")
        return formatted_results


# 創建向量資料庫管理器
vector_manager = VectorStoreManager()
print("✓ 向量資料庫管理器已就緒！")
```
## 5.6 智能問答功能

這裡我們將實現兩種問答模式：
1. **直接問答**：從元資料直接提取答案（如"有幾個臥室？"、"客廳多大？"）
2. **圖像檢索**：返回相關戶型列表（如"找3室2廳的戶型"）
```python
class IntelligentQA:
    """智能問答系統 - LLM驅動版本"""
    
    def __init__(self, vector_manager: VectorStoreManager, llm_client: OpenAI, model_name: str):
        self.vector_manager = vector_manager
        self.llm_client = llm_client
        self.model_name = model_name
    
    def direct_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """使用LLM基於元資料生成答案"""
        print(f"\n{'='*60}")
        print("LLM智能問答模式")
        print(f"   問題: {question}")
        print("="*60)
        
        # 1. 向量檢索
        results = self.vector_manager.search(question, top_k=top_k)
        
        if not results:
            return {
                "answer": "抱歉，沒有找到相關戶型資訊。",
                "sources": [],
                "mode": "direct_answer"
            }
        
        # 2. 收集所有相關的元資料
        context_parts = []
        for i, result in enumerate(results):
            metadata = result["metadata"]
            
            # 解析結構化資訊
            extracted_info = {}
            if "extracted_info_json" in metadata:
                try:
                    extracted_info = json.loads(metadata["extracted_info_json"])
                except:
                    pass
            
            context_parts.append(f"""
文檔 {i+1}：{metadata.get('file_name', '未知文件')}
VLM描述：{result['content']}
結構化資料：{json.dumps(extracted_info, ensure_ascii=False, indent=2)}
相似度：{result['similarity']:.2f}
"""
)
        
        # 3. 構建LLM提示詞
        context = "\n".join(context_parts)
        
        prompt = f"""你是一個專業的房產顧問，請根據提供的戶型資訊回答用戶問題。

用戶問題：{question}

可用的戶型資訊：
{context}

請根據以上資訊回答用戶問題，要求：
1. 直接、準確地回答問題
2. 如果涉及具體資料（面積、尺寸等），請引用準確數值
3. 如果問題涉及多個戶型，請進行對比
4. 保持回答簡潔明了
5. 在回答末尾註明資訊來源

回答："""

        # 4. 調用LLM生成答案
        print("正在調用LLM生成智能答案...")
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一個專業的房產顧問，擅長分析戶型資訊並回答客戶問題。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        print(f"LLM答案生成完成！")
        print("="*60)
        
        return {
            "answer": answer,
            "sources": [{
                "file_id": result["metadata"].get("file_id"),
                "file_name": result["metadata"].get("file_name"),
                "similarity": result["similarity"]
            } for result in results],
            "mode": "direct_answer"
        }
    
    def search_images(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        圖像檢索模式 - 也用LLM來生成更智能的檢索結果描述
        """
        print(f"\n{'='*60}")
        print("LLM智能檢索模式")
        print(f"   查詢: {query}")
        print("="*60)
        
        # 1. 向量檢索
        results = self.vector_manager.search(query, top_k=top_k * 2)
        
        if not results:
            return {
                "message": f"沒有找到與 '{query}' 相關的戶型。",
                "images": [],
                "mode": "search_images"
            }
        
        # 2. 按文件聚合（去重）
        file_map = {}
        for result in results:
            file_id = result["metadata"].get("file_id")
            if file_id not in file_map:
                # 解析結構化資訊
                extracted_info = {}
                if "extracted_info_json" in result["metadata"]:
                    try:
                        extracted_info = json.loads(result["metadata"]["extracted_info_json"])
                    except:
                        pass
                
                file_map[file_id] = {
                    "file_id": file_id,
                    "file_name": result["metadata"].get("file_name"),
                    "similarity": result["similarity"],
                    "content": result["content"],
                    "extracted_info": extracted_info,
                    "metadata": result["metadata"]
                }
            else:
                # 更新最高相似度
                if result["similarity"] > file_map[file_id]["similarity"]:
                    file_map[file_id]["similarity"] = result["similarity"]
        
        # 3. 按相似度排序
        sorted_files = sorted(
            file_map.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]
        
        # 4. 用LLM生成智能的檢索結果描述
        files_info = []
        for file_info in sorted_files:
            files_info.append({
                "file_name": file_info["file_name"],
                "similarity": file_info["similarity"],
                "description": file_info["content"],
                "details": file_info["extracted_info"]
            })
        
        search_prompt = f"""作為房產顧問，請根據檢索到的戶型資訊，回答用戶的查詢需求。

用戶查詢：{query}

檢索到的戶型：
{json.dumps(files_info, ensure_ascii=False, indent=2)}

請：
1. 總結找到了幾個相關戶型
2. 對每個戶型進行簡要介紹（戶型、面積、特點等）
3. 根據用戶查詢給出推薦意見
4. 保持專業和友好的語調

回答："""

        print("正在生成智能檢索結果...")
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是專業的房產顧問，擅長根據客戶需求推薦合適的戶型。"},
                {"role": "user", "content": search_prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        message = response.choices[0].message.content
        
        print(f"✓ 找到 {len(sorted_files)} 個相關戶型")
        print("="*60)
        
        return {
            "message": message,
            "images": [{
                "file_id": f["file_id"],
                "file_name": f["file_name"], 
                "similarity": f["similarity"]
            } for f in sorted_files],
            "mode": "search_images"
        }

    def ask(self, question: str, mode: str = "auto") -> Dict[str, Any]:
        """統一問答介面"""
        # 智能判斷模式
        if mode == "auto":
            search_keywords = ["找", "有沒有", "哪些", "查找", "搜索", "推薦", "比較"]
            if any(kw in question for kw in search_keywords):
                return self.search_images(question)
            else:
                return self.direct_answer(question)
        
        if mode == "direct_answer":
            return self.direct_answer(question)
        elif mode == "search_images":
            return self.search_images(question)
        else:
            raise ValueError(f"不支援的模式: {mode}")


# 重新創建問答系統
qa_system = IntelligentQA(vector_manager, vlm_client, MODEL_NAME)
print("LLM智能問答系統已就緒！")
```
根據實際的應用場景，我們實現了兩種互補的問答模式：
- **直接問答**：適合需要快速獲取精確資訊的場景（如查詢材料、尺寸等）
- **圖像檢索**：適合需要瀏覽和對比多個圖紙的場景

# 六、企業專案實戰：多模態RAG專案本地部署

本節內容，我們將詳細介紹如何部署和運行這個基於VLM的多模態RAG智能問答系統。該系統支援CAD圖紙、平面圖、架構圖、PDF文檔等多種格式的智能分析和問答。

## 6.1 專案結構詳解

專案採用模組化設計，核心結構如下：
```
pc_multimodal_rag/                    # 專案根目錄
├── 📁 backend/                       # 後端服務層
│   ├── main_service.py              # FastAPI主服務 - 多模態RAG API
│   ├── simple_vlm_analyzer.py       # VLM圖像分析器 (支援CAD/平面圖/架構圖)
│   ├── qwen_embeddings.py           # 通義千問Embedding模型封裝
│   ├── simple_logger.py             # 日誌記錄模組
│   ├── 📁 unified/                   # 統一PDF處理模組
│   │   └── unified_pdf_extraction_service.py  # PDF解析服務
│   ├── 📁 Information-Extraction/    # 資訊提取模組
│   ├── 📁 image_analysis/            # 圖像分析模組
│   └── 📁 chroma_db/                 # ChromaDB向量資料庫存儲
│
├── 📁 frontend/                      # 前端介面 (可選)
├── 📁 uploads/                       # 上傳文件存儲目錄
├── 📁 previews/                      # 文件預覽/縮略圖存儲
├── 📁 test_data/                     # 測試資料
└── .env                              # 環境配置文件
```
### 核心組件功能說明

| 層級 | 技術棧 | 主要功能 | 關鍵文件 |
| :--- | :--- | :--- | :--- |
| **API服務層** | FastAPI + Pydantic | RESTful API、文件上傳、智能問答 | main_service.py |
| **VLM分析層** | 多模態大模型 + 自定義提示詞 | 多模態理解、圖像分析、結構化提取 | simple_vlm_analyzer.py |
| **向量檢索層** | ChromaDB + Qwen/HuggingFace Embeddings | 語義檢索、相似度計算 | qwen_embeddings.py + ChromaDB |
| **文檔處理層** | PyMuPDF + PIL + 自定義解析器 | PDF解析、圖像預處理、格式轉換 | unified_pdf_extraction_service.py |
| **資料存儲層** | 文件系統 + 向量資料庫 | 原文件存儲、向量索引、元資料管理 | uploads/ + chroma_db/ |

## 6.2 環境要求與依賴安裝

系統基於Python 3.11+開發，需要確保環境滿足以下要求：

| 組件 | 版本要求 | 安裝方式 | 驗證命令 |
| :--- | :--- | :--- | :--- |
| Python | ≥ 3.10 | 官網下載或conda | `python --version` |
| pip | 最新版 | 隨Python安裝 | `pip --version` |

首先需要創建Python虛擬環境
```bash
# 使用conda創建環境（推薦）
conda create -n multimodal_rag python=3.11
conda activate multimodal_rag

# 或使用venv創建環境
python -m venv multimodal_rag
source multimodal_rag/bin/activate  # Linux/Mac
# multimodal_rag\Scripts\activate     # Windows
```
接下來一鍵安裝核心依賴
```bash
# 進入專案目錄
cd pc_multimodal_rag/backend

# 安裝核心依賴包
pip install -r requirements_service.txt
```
## 6.3 後端服務配置與啟動

完成依賴安裝後，需要配置API密鑰和啟動後端服務。

創建 `.env` 文件，配置必要的API密鑰：
```bash
# 在專案根目錄創建 .env 文件
touch .env
```
在 `.env` 文件中添加以下配置：
```
# 多模態 RAG 服務配置

# VLM 模型配置
VLM_MODEL_URL=https:/
VLM_API_KEY=sk-Y4o8DF6Iq2l8nFT
VLM_MODEL_NAME=gpt-4o

# 服務配置
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000

# 存儲配置
UPLOAD_DIR=./uploads
PREVIEW_DIR=./previews
VECTOR_DB_DIR=./chroma_db

# Embedding 模型配置
EMBEDDING_TYPE=qwen  # qwen 或 huggingface
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
DASHSCOPE_API_KEY=sk-bdccf7277a5
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 文字分割配置
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```
**重要提示**：
- `OPENAI_API_KEY`：必須配置，用於調用gpt-4o模型進行圖像分析
- `DASHSCOPE_API_KEY`：必須配置，用於調用Qwen embedding模型
- 如果使用HuggingFace模型，可以不配置`DASHSCOPE_API_KEY`

### 核心組件功能說明

| 介面路徑 | 方法 | 功能 | 說明 |
| :--- | :--- | :--- | :--- |
| `/upload/` | POST | 文件上傳 | 支援圖片、PDF等多種格式 |
| `/search/` | POST | 智能搜索 | 基於向量檢索的語義搜索 |
| `/intelligent_qa/` | POST | 智能問答 | 多模態問答，支援直接回答和圖像檢索 |
| `/files/` | GET | 文件列表 | 獲取已上傳的文件列表 |

接下來啟動後端服務
```bash
# 啟動FastAPI後端服務
python backend/main_service.py
```
啟動成功後，終端會顯示如下資訊：
```
INFO:     Started server process [12345]  
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
## 6.4 前端服務配置與啟動

最後，啟動前端服務，進入前端目錄，安裝Node.js依賴，啟動開發伺服器：
```bash
cd frontend
npm install
npm run dev
```
打開瀏覽器訪問 http://localhost:5173，就可以看到"賦範空間公開體驗課"的介面。

至此，多模態RAG系統部署完成！系統支援CAD圖紙、平面圖、PDF文檔的智能分析和問答，可根據實際需求進行功能擴展和定制開發。

我們下期公開課，再見！

```