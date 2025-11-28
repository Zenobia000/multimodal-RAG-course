# 🔍 Vector Search Strategies - Complete Documentation

## 📋 目錄

1. [基礎檢索策略](#基礎檢索策略)
   - [1.1 語義搜索](#11-語義搜索-semantic-search)
   - [1.2 元數據過濾](#12-元數據過濾-metadata-filtering)
   - [1.3 混合搜索](#13-混合搜索-hybrid-search)

2. [高級檢索策略](#高級檢索策略)
   - [2.1 查詢擴展](#21-查詢擴展-query-expansion)
   - [2.2 HyDE](#22-hyde-假設文檔嵌入)
   - [2.3 LLM 重排序](#23-llm-重排序-llm-re-ranking)

3. [RAG 2.0 進階策略](#rag-20-進階策略)
   - [3.1 自適應檢索](#31-自適應檢索-adaptive-retrieval)

4. [Qdrant 特有功能](#qdrant-特有功能)
   - [4.1 Discovery API](#41-discovery-api-探索性搜索)
   - [4.2 Recommendation API](#42-recommendation-api-推薦搜索)
   - [4.3 分組聚合搜索](#43-分組聚合搜索-group-by-aggregation)

---

## 基礎檢索策略

### 1.1 語義搜索 (Semantic Search)

#### 🎯 解決痛點
- **關鍵詞匹配失效**：傳統搜尋無法理解同義詞、相關概念
- **語義理解需求**：使用者輸入「transformer」也想找到「self-attention」相關內容
- **多語言支援**：需要跨語言語義搜尋能力

#### 📋 策略說明
使用向量 embedding 將查詢與文檔映射到高維語義空間，透過餘弦相似度找出語義最接近的結果。

**核心流程：**
```
Query → Embedding → Vector Search → Top-K Results
```

#### 🏭 應用場域
- ✅ **知識庫問答**：客服機器人、內部文檔檢索
- ✅ **學術論文搜尋**：研究者查找相關文獻
- ✅ **產品推薦**：根據使用者描述找相似商品
- ✅ **程式碼搜尋**：自然語言查詢程式碼片段

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 理解語義相似性 | ❌ 無法保證精確關鍵詞匹配 |
| ✅ 跨語言支援 | ❌ 依賴 embedding 模型品質 |
| ✅ 實作簡單、速度快 | ❌ 專有名詞可能失準 |
| ✅ 適合開放式問題 | ❌ 無法利用元數據過濾 |

**效能指標：**
- ⏱️ 延遲：~0.3s
- 📊 召回率：60-80%
- 🎯 精準度：65-85%

---

### 1.2 元數據過濾 (Metadata Filtering)

#### 🎯 解決痛點
- **結果範圍過廣**：語義搜尋返回不相關時間/來源的文檔
- **精準篩選需求**：使用者只想查特定年份、作者、類別的內容
- **合規性要求**：需要限制搜尋範圍（如只查內部文檔、特定權限）

#### 📋 策略說明
在語義搜尋基礎上，加入元數據條件過濾，先篩選符合條件的文檔子集，再進行向量相似度計算。

**核心流程：**
```
Query → Embedding → Metadata Filter → Vector Search → Filtered Results
```

**常用過濾條件：**
- 📅 時間範圍：`processed_date >= "2024-01-01"`
- 📁 文檔來源：`source_file == "GPT-3.pdf"`
- 📏 內容長度：`chunk_size BETWEEN 500 AND 1000`
- 🏷️ 標籤分類：`category IN ["NLP", "CV"]`

#### 🏭 應用場域
- ✅ **法律文檔檢索**：限定法律條文生效日期
- ✅ **電商搜尋**：價格區間、品牌、庫存狀態篩選
- ✅ **企業知識庫**：依部門、權限、版本過濾
- ✅ **學術研究**：指定期刊、年份、作者

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 大幅提升精準度 | ❌ 可能過度限縮結果 |
| ✅ 支援複雜業務邏輯 | ❌ 需要良好的元數據結構 |
| ✅ 符合合規要求 | ❌ 元數據維護成本高 |
| ✅ 減少無關結果 | ❌ 過濾條件設計需經驗 |

**效能指標：**
- ⏱️ 延遲：+0.05-0.1s (過濾開銷)
- 📊 召回率：降低 (因範圍限縮)
- 🎯 精準度：80-95%

---

### 1.3 混合搜索 (Hybrid Search)

#### 🎯 解決痛點
- **專有名詞失準**：語義搜尋對「GPT-4」、「BERT」等專有名詞效果差
- **精確匹配需求**：使用者想找包含特定術語的文檔
- **語義+關鍵詞並重**：需要同時考慮語義相似與關鍵詞出現

#### 📋 策略說明
結合向量搜尋（語義）與全文檢索（關鍵詞），透過加權融合兩種分數，取得平衡結果。

**核心流程：**
```
Query → [Semantic Search + Keyword Search] → Score Fusion → Hybrid Results
```

**融合策略：**
- **Reciprocal Rank Fusion (RRF)**：基於排名融合
- **線性加權**：`final_score = α * semantic_score + β * keyword_score`
- **最小值策略**：`min(semantic_score, keyword_score)` 確保兩者都相關

#### 🏭 應用場域
- ✅ **技術文檔搜尋**：API 名稱、函數名等專有術語
- ✅ **醫療資訊檢索**：藥品名稱、疾病代碼精確匹配
- ✅ **產品型號查詢**：「iPhone 15 Pro」等包含數字的查詢
- ✅ **合約審查**：特定條款文字精確匹配

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 兼顧語義與精確匹配 | ❌ 實作複雜度高 |
| ✅ 對專有名詞友好 | ❌ 需調整融合權重 |
| ✅ 提升整體召回率 | ❌ 延遲增加 2-3倍 |
| ✅ 適應性強 | ❌ 需維護兩套索引 |

**效能指標：**
- ⏱️ 延遲：~0.35s (比單純語義慢)
- 📊 召回率：75-90%
- 🎯 精準度：70-85%

**最佳實踐：**
- 根據查詢類型動態調整權重
- 專有名詞查詢提高關鍵詞權重
- 開放式問題提高語義權重

---

## 高級檢索策略

### 2.1 查詢擴展 (Query Expansion)

#### 🎯 解決痛點
- **查詢過於簡短**：使用者輸入「attention」無法涵蓋所有相關內容
- **表達方式單一**：同一概念有多種描述方式
- **召回率不足**：單一查詢無法觸及多樣化的相關文檔

#### 📋 策略說明
使用 LLM 將原始查詢擴展為多個語義相關的查詢變體，分別檢索後合併去重。

**核心流程：**
```
Query → LLM Expansion → [Query1, Query2, Query3] → Multi-Search → Merge & Deduplicate
```

**擴展策略：**
- **同義詞擴展**：「神經網絡」→「neural network」、「深度學習模型」
- **相關概念**：「transformer」→「self-attention」、「BERT」、「GPT」
- **技術細化**：「scaling laws」→「compute-optimal training」、「Chinchilla scaling」

#### 🏭 應用場域
- ✅ **學術研究**：查找相關論文時自動擴展主題詞
- ✅ **電商搜尋**：「手機」擴展為「iPhone」「Android」「智慧型手機」
- ✅ **法律檢索**：法條用語擴展為白話文表達
- ✅ **醫療診斷**：症狀描述擴展為專業醫學術語

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 大幅提升召回率 | ❌ LLM 調用延遲 (3-5s) |
| ✅ 發現隱藏相關內容 | ❌ 可能引入噪音 |
| ✅ 適應使用者表達習慣 | ❌ LLM API 成本高 |
| ✅ 減少查詢迭代次數 | ❌ 精準度可能下降 |

**效能指標：**
- ⏱️ 延遲：~4s (LLM + 多次檢索)
- 📊 召回率：80-95% (大幅提升)
- 🎯 精準度：70-85%

**最佳實踐：**
- 限制擴展查詢數量 (3-5個)
- 使用快速 LLM (gpt-3.5-turbo)
- 快取常見查詢的擴展結果

---

### 2.2 HyDE (假設文檔嵌入)

#### 🎯 解決痛點
- **查詢-文檔語言差異**：使用者查詢是問句，文檔是陳述句
- **長度不匹配**：短查詢 vs 長文檔，embedding 效果差
- **語言風格差異**：口語化查詢 vs 正式文檔

#### 📋 策略說明
先用 LLM 生成一個「假設性的答案文檔」，用這個生成文檔的 embedding 來檢索，而非直接用查詢。

**核心流程：**
```
Query → LLM Generate Hypothetical Doc → Embed Doc → Vector Search → Real Results
```

**範例：**
```
查詢：「什麼是 attention mechanism?」
↓ LLM 生成假設文檔
假設文檔：「Attention mechanism is a fundamental component in transformer 
architectures that allows models to selectively focus on different parts 
of the input sequence...」
↓ 用假設文檔檢索
結果：真實的 transformer 論文片段
```

#### 🏭 應用場域
- ✅ **問答系統**：問句風格查詢，找陳述句答案
- ✅ **技術支援**：「如何安裝 X?」找安裝手冊
- ✅ **學術研究**：「X 的影響是什麼?」找研究結論
- ✅ **產品文檔**：「如何使用 Y 功能?」找使用說明

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 跨越查詢-文檔語言鴻溝 | ❌ 極高延遲 (5-7s) |
| ✅ 提升檢索精準度 | ❌ 依賴 LLM 生成品質 |
| ✅ 適合問答場景 | ❌ 成本最高的策略 |
| ✅ 文檔風格一致性好 | ❌ 可能偏離原始意圖 |

**效能指標：**
- ⏱️ 延遲：~5.3s (LLM 生成 + 檢索)
- 📊 召回率：75-85%
- 🎯 精準度：85-95% (最高)

**最佳實踐：**
- 快取熱門查詢的假設文檔
- 使用流式生成減少感知延遲
- 搭配傳統檢索作為 fallback

---

### 2.3 LLM 重排序 (LLM Re-ranking)

#### 🎯 解決痛點
- **向量相似度不等於相關性**：高 cosine similarity ≠ 真正相關
- **排序品質不佳**：檢索結果順序不符合使用者期待
- **缺乏語義深度理解**：embedding 無法理解複雜邏輯關係

#### 📋 策略說明
先用快速方法（語義搜尋）召回候選文檔（Top-50），再用 LLM 深度理解後重新排序（Top-5）。

**核心流程：**
```
Query → Fast Retrieval (Top-50) → LLM Deep Re-ranking → Final Top-K
```

**LLM 判斷標準：**
- 與查詢的相關性程度
- 資訊完整性與可讀性
- 時效性與權威性
- 是否直接回答問題

#### 🏭 應用場域
- ✅ **問答系統**：確保最相關答案排在前面
- ✅ **搜尋引擎**：提升首頁結果品質
- ✅ **推薦系統**：個人化內容排序
- ✅ **內容審核**：品質評估與優先級排序

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 顯著提升排序品質 | ❌ 高延遲 (僅適用Top-K) |
| ✅ 理解複雜語義關係 | ❌ LLM API 成本 |
| ✅ 可融入業務邏輯 | ❌ 不穩定（LLM 變異性） |
| ✅ 使用者滿意度高 | ❌ 需要 prompt 工程 |

**效能指標：**
- ⏱️ 延遲：+2-3s (僅重排 Top-10)
- 📊 召回率：不變
- 🎯 精準度：85-95%
- 💰 成本：每次查詢 $0.001-0.005

**最佳實踐：**
- 只重排 Top-10 ~ Top-20
- 使用快速小模型（gpt-3.5-turbo）
- 設計清晰的評分 prompt
- 批次重排降低 API 調用次數

---

## RAG 2.0 進階策略

### 3.1 自適應檢索 (Adaptive Retrieval)

#### 🎯 解決痛點
- **一刀切策略低效**：簡單查詢用 HyDE 浪費資源
- **不同查詢需求差異大**：事實查詢 vs 分析查詢需要不同策略
- **無法動態優化**：無法根據查詢特徵選擇最佳方法

#### 📋 策略說明
先用 LLM 分析查詢複雜度、類型、意圖，再動態選擇最適合的檢索策略。

**核心流程：**
```
Query → LLM Analyze (complexity, type, intent) → Strategy Selector → Adaptive Retrieval
```

**查詢分類維度：**
- **複雜度**：simple / medium / complex
- **類型**：factual / conceptual / comparative / analytical
- **需求**：single-doc / multi-doc / reasoning

**策略映射：**
```python
if complexity == "simple":
    → Semantic Search (快速)
elif complexity == "medium" or type == "comparative":
    → Query Expansion (召回率優先)
elif complexity == "complex" or requires_multi_docs:
    → HyDE + Re-ranking (品質優先)
```

#### 🏭 應用場域
- ✅ **智能助理**：根據問題類型自動選擇策略
- ✅ **企業搜尋**：平衡效能與品質
- ✅ **研究平台**：簡單查詢快速返回，複雜查詢深度檢索
- ✅ **多租戶系統**：不同用戶等級使用不同策略

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 最佳成本效益比 | ❌ 需要 LLM 分析查詢 (+0.5s) |
| ✅ 自動適應查詢特徵 | ❌ 策略選擇邏輯複雜 |
| ✅ 兼顧速度與品質 | ❌ 需要大量調試優化 |
| ✅ 資源利用最優化 | ❌ 多種策略維護成本 |

**效能指標：**
- ⏱️ 延遲：0.3s ~ 5.5s (動態)
- 📊 召回率：70-90% (動態)
- 🎯 精準度：75-90% (動態)
- 💰 成本：根據查詢類型浮動

**最佳實踐：**
- 快取查詢分析結果
- 設定清晰的策略選擇規則
- 監控各策略使用率與效果
- A/B 測試優化策略映射

---

## Qdrant 特有功能

### 4.1 Discovery API (探索性搜索)

#### 🎯 解決痛點
- **不知道要找什麼**：使用者無法明確描述需求
- **探索未知領域**：想發現相關但不熟悉的內容
- **避免特定主題**：「要 A 但不要 B」的複雜需求

#### 📋 策略說明
提供正面例子（想要的）和負面例子（不想要的），系統在語義空間中找到「接近正例、遠離負例」的內容。

**核心流程：**
```
Positive Examples + Negative Examples → Discovery → Balanced Results
```

**範例：**
```python
positive = [
    "transformer scaling laws for NLP",
    "compute-efficient self-attention"
]
negative = [
    "CNN for image classification",
    "reinforcement learning robotics"
]
→ 結果：Transformer NLP 論文，排除 CV/RL 內容
```

**分數計算：**
```
Discovery Score = Base Similarity + Σ(Positive Similarity - Negative Similarity)
分數 > 3.0 表示在所有維度都高度匹配
```

#### 🏭 應用場域
- ✅ **內容推薦**：「喜歡 A 但不喜歡 B」的推薦
- ✅ **探索式研究**：發現相關但未知的研究方向
- ✅ **內容策展**：精準定義內容風格
- ✅ **相似商品推薦**：排除特定類別的相似商品

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 精準控制檢索方向 | ❌ 需要提供多個正負例 |
| ✅ 探索未知相關內容 | ❌ 正負例設計需經驗 |
| ✅ 多維度對比篩選 | ❌ 分數解釋較複雜 |
| ✅ 適合冷啟動場景 | ❌ 不適合精確查詢 |

**效能指標：**
- ⏱️ 延遲：~0.4s
- 📊 召回率：60-75%
- 🎯 精準度：80-90% (高品質)
- 🎚️ 分數範圍：-N ~ +N (N=context_pairs數量)

**最佳實踐：**
- 提供 3-5 個正負例
- 正負例對比要明確
- 適合「要什麼不要什麼」的場景
- 不適合精確關鍵詞搜尋

---

### 4.2 Recommendation API (推薦搜索)

#### 🎯 解決痛點
- **「相關推薦」需求**：使用者看了文檔 A，想找類似內容
- **基於行為的推薦**：根據瀏覽/點擊歷史推薦
- **內容延伸閱讀**：文章底部的「相關文章」功能

#### 📋 策略說明
基於已知文檔 ID，找到向量空間中最相似的其他文檔（自動排除輸入 ID）。

**核心流程：**
```
Viewed Doc IDs → Extract Vectors → Similarity Search → Recommended Docs
```

**與語義搜尋差異：**
- 語義搜尋：Query Text → Vector → Search
- 推薦搜尋：Doc ID → Vector (already stored) → Search

#### 🏭 應用場域
- ✅ **新聞網站**：文章底部「相關閱讀」
- ✅ **電商平台**：「看過此商品的人也看了」
- ✅ **視頻平台**：「相關推薦」列表
- ✅ **文檔系統**：「相關文件」推薦

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 無需文本輸入 | ❌ 僅基於向量相似 |
| ✅ 速度極快 (無 embedding) | ❌ 無法融入業務邏輯 |
| ✅ 適合內容推薦 | ❌ 冷啟動問題 |
| ✅ 實作簡單 | ❌ 可能推薦過於相似內容 |

**效能指標：**
- ⏱️ 延遲：~0.2s (最快)
- 📊 召回率：70-85%
- 🎯 精準度：75-85%

**最佳實踐：**
- 結合多個文檔 ID 提升推薦品質
- 添加負例排除已讀內容
- 融合元數據過濾（如排除舊內容）
- 追蹤點擊率優化推薦策略

---

### 4.3 分組聚合搜索 (Group-by Aggregation)

#### 🎯 解決痛點
- **結果來源單一**：所有結果都來自同一篇論文/文件
- **缺乏多樣性**：想看多個來源的觀點
- **重複內容過多**：同一文檔的相鄰段落重複出現

#### 📋 策略說明
按指定欄位（如來源文件）分組，每組只返回 Top-K 個結果，確保結果來源多樣化。

**核心流程：**
```
Query → Vector Search → Group by Field → Top-K per Group → Diverse Results
```

**範例：**
```python
query_points_groups(
    query="transformer architecture",
    group_by="metadata.source_file",
    limit=5,        # 5 個不同來源
    group_size=2    # 每個來源 2 段
)
→ 結果：
  - GPT-3 論文 (2段)
  - Transformer 論文 (2段)
  - BERT 論文 (2段)
  - ViT 論文 (2段)
  - CLIP 論文 (2段)
```

#### 🏭 應用場域
- ✅ **學術研究**：確保引用多個論文來源
- ✅ **新聞聚合**：每個媒體取一篇相關報導
- ✅ **產品比較**：每個品牌取幾個相關商品
- ✅ **多角度分析**：從不同來源獲取觀點

#### ⚖️ Trade-offs

| 優勢 | 劣勢 |
|------|------|
| ✅ 確保結果多樣性 | ❌ 可能犧牲最相關結果 |
| ✅ 避免單一來源壟斷 | ❌ 需要良好的分組欄位 |
| ✅ 適合多來源綜合 | ❌ 實作稍複雜 |
| ✅ 提升內容豐富度 | ❌ 總結果數量增加 |

**效能指標：**
- ⏱️ 延遲：~0.35s
- 📊 召回率：60-75% (因分組限制)
- 🎯 精準度：70-85%
- 🎨 多樣性：顯著提升

**最佳實踐：**
- 選擇有意義的分組欄位（source_file, category, author）
- 適當調整 group_size（通常 1-3）
- 後處理可進一步去重
- 適合「綜合多方觀點」的場景

---

## 📊 策略選擇決策樹

```
查詢類型判斷
├─ 簡單事實查詢 (What is X?)
│   └─ Semantic Search (0.3s, 低成本)
│
├─ 需要精確術語 (API名稱、產品型號)
│   └─ Hybrid Search (0.35s, 中成本)
│
├─ 需要特定來源/時間
│   └─ Metadata Filtering (0.35s, 低成本)
│
├─ 開放式探索 (想了解X領域)
│   └─ Query Expansion (4s, 高成本)
│
├─ 問答風格查詢 (How to X?)
│   └─ HyDE (5s, 最高成本)
│
├─ 需要最佳排序
│   └─ Semantic Search + LLM Re-ranking (3s, 高成本)
│
├─ 複雜多面向查詢
│   └─ Adaptive Retrieval (動態, 智能成本)
│
├─ 探索相關但排除特定主題
│   └─ Discovery API (0.4s, 中成本)
│
├─ 基於已知內容推薦
│   └─ Recommendation API (0.2s, 低成本)
│
└─ 需要多來源多樣性
    └─ Grouped Search (0.35s, 低成本)
```

---

## 🎯 效能比較總表

| 策略 | 延遲 | 召回率 | 精準度 | 成本 | 適用場景 |
|------|------|--------|--------|------|----------|
| Semantic Search | 0.3s | 60-80% | 65-85% | 低 | 通用查詢 |
| Metadata Filtering | 0.35s | 50-70% | 80-95% | 低 | 精準篩選 |
| Hybrid Search | 0.35s | 75-90% | 70-85% | 中 | 專有名詞 |
| Query Expansion | 4s | 80-95% | 70-85% | 高 | 提升召回 |
| HyDE | 5.3s | 75-85% | 85-95% | 最高 | 問答系統 |
| LLM Re-ranking | +2-3s | - | 85-95% | 高 | 排序優化 |
| Adaptive Retrieval | 動態 | 動態 | 動態 | 智能 | 智能系統 |
| Discovery API | 0.4s | 60-75% | 80-90% | 中 | 探索搜尋 |
| Recommendation API | 0.2s | 70-85% | 75-85% | 低 | 內容推薦 |
| Grouped Search | 0.35s | 60-75% | 70-85% | 低 | 多樣性 |

---

## 💡 最佳實踐建議

### 1. 生產環境組合策略
```
Fast Path (80% 查詢):
  Semantic Search → 直接返回

Medium Path (15% 查詢):
  Hybrid Search + Metadata Filter → 返回

Slow Path (5% 複雜查詢):
  Query Expansion + HyDE + Re-ranking → 返回
```

### 2. 成本優化
- 使用快取熱門查詢結果 (TTL: 5-10分鐘)
- LLM 調用批次化處理
- 非關鍵查詢使用較小模型 (gpt-3.5-turbo)
- 監控 API 用量設定預算上限

### 3. 品質監控
- 追蹤各策略的點擊率 (CTR)
- 使用者滿意度評分
- A/B 測試新策略
- 定期人工評估 Top-100 查詢結果

### 4. 漸進式採用
```
Phase 1: Semantic Search (建立 baseline)
  ↓
Phase 2: + Metadata Filtering (加入業務邏輯)
  ↓
Phase 3: + Hybrid Search (處理專有名詞)
  ↓
Phase 4: + Adaptive Retrieval (智能化)
  ↓
Phase 5: + LLM Re-ranking (極致品質)
```

---

## 📚 參考資源

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [RAG Survey Paper 2024](https://arxiv.org/abs/2312.10997)
- [Advanced RAG Techniques](https://www.anthropic.com/research/rag)
- [Vector Database Benchmark](https://ann-benchmarks.com/)

---

**文檔版本：** v1.0  
**最後更新：** 2025-11-28  
**作者：** AI Assistant  
**授權：** MIT License

