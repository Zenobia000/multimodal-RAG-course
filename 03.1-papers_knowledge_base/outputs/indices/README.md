# 論文知識庫索引 (統一輸出架構 v2.0)

## 統計信息

- **論文總數**: 5
- **總大小**: 0.00 GB
- **聚合文件**: 5 個
- **生成時間**: 2025-11-24T18:08:13.894965
- **架構版本**: 2.0

## 分類統計

- **基礎設施 (IF)**: 5 篇論文, 1.0 MB

## 目錄結構

- `outputs/processed_papers/`: 按分類組織的論文處理結果
- `outputs/aggregated_chunks/`: 所有論文的JSONL文件統一管理
- `outputs/indices/`: 全局索引文件（本目錄）

## 索引文件說明

- `master_index.json`: 主索引，包含所有論文的完整元數據
- `category_index.json`: 按分類組織的索引
- `search_index.json`: 搜索索引，支持按年份和關鍵詞查找
- `statistics.json`: 詳細統計信息

## 使用方式

```python
import json
from pathlib import Path

# 載入主索引
index_dir = Path("outputs/indices")
with open(index_dir / 'master_index.json', 'r') as f:
    master_index = json.load(f)

# 查找特定論文
papers = master_index['papers']
clip_papers = [p for p in papers if 'clip' in p['output_name'].lower()]

# 載入聚合文件
chunks_dir = Path("outputs/aggregated_chunks")
for paper in papers:
    chunk_file = chunks_dir / f"{paper['output_name']}.jsonl"
    if chunk_file.exists():
        # 讀取 JSONL 內容
        pass
```

## 聚合文件統計

- **總文件數**: 5
- **總大小**: 0.5 MB
