#!/usr/bin/env python3
"""
全局索引生成腳本 v2.0
生成統一輸出架構論文知識庫的全局索引和檢索文件
"""

import json
import os
from pathlib import Path
from datetime import datetime
import hashlib
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GlobalIndexGenerator:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.outputs_dir = self.base_dir / "outputs"
        self.index_dir = self.outputs_dir / "indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 統一輸出架構的目錄
        self.processed_papers_dir = self.outputs_dir / "processed_papers"
        self.aggregated_chunks_dir = self.outputs_dir / "aggregated_chunks"

        self.categories = {
            "MP": {"name": "模型範式", "name_en": "Model_Paradigm"},
            "IF": {"name": "基礎設施", "name_en": "Infrastructure"},
            "LM": {"name": "語言模型", "name_en": "Language_Models"},
            "MM": {"name": "多模態", "name_en": "Multimodal"}
        }

    def scan_processed_files(self):
        """掃描統一輸出架構中的所有已處理文件"""
        all_papers = []

        if not self.processed_papers_dir.exists():
            logger.warning(f"處理文件目錄不存在: {self.processed_papers_dir}")
            return all_papers

        # 掃描每個分類的處理結果
        for category_code, info in self.categories.items():
            category_path = self.processed_papers_dir / category_code
            if not category_path.exists():
                logger.warning(f"分類目錄不存在: {category_path}")
                continue

            # 掃描分類下的所有論文
            for paper_dir in category_path.iterdir():
                if paper_dir.is_dir():
                    paper_info = self.extract_paper_metadata(paper_dir, category_code, info)
                    if paper_info:
                        all_papers.append(paper_info)

        return all_papers

    def extract_paper_metadata(self, paper_dir, category_code, category_info):
        """提取單個論文的元數據"""
        try:
            paper_name = paper_dir.name

            # 查找對應的元數據文件
            metadata_file = paper_dir / f"{paper_name}_metadata.json"

            paper_info = {
                "id": self.generate_paper_id(paper_name),
                "output_name": paper_name,
                "category_code": category_code,
                "category_name": category_info["name"],
                "category_name_en": category_info["name_en"],
                "processed_path": str(paper_dir),
                "has_metadata": metadata_file.exists()
            }

            # 讀取元數據（如果存在）
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    paper_info.update(metadata)

            # 掃描處理結果文件
            paper_info["files"] = self.scan_paper_files(paper_dir)

            # 提取內容摘要
            paper_info["content_summary"] = self.extract_content_summary(paper_dir)

            # 檢查聚合文件是否存在
            aggregated_file = self.aggregated_chunks_dir / f"{paper_name}.jsonl"
            paper_info["has_aggregated_chunk"] = aggregated_file.exists()

            return paper_info

        except Exception as e:
            logger.error(f"提取元數據失敗 {paper_dir.name}: {e}")
            return None

    def generate_paper_id(self, name):
        """生成論文唯一ID"""
        return hashlib.md5(name.encode()).hexdigest()[:12]

    def scan_paper_files(self, paper_dir):
        """掃描論文處理後的文件"""
        files_info = {
            "jsonl_files": [],
            "workspace_files": [],
            "total_size": 0
        }

        # 掃描 JSONL 文件（直接在論文目錄下）
        for jsonl_file in paper_dir.glob("*.jsonl"):
            size = jsonl_file.stat().st_size
            files_info["jsonl_files"].append({
                "filename": jsonl_file.name,
                "size": size,
                "path": str(jsonl_file)
            })
            files_info["total_size"] += size

        # 掃描 workspace 文件
        workspace_dir = paper_dir / "workspace"
        if workspace_dir.exists():
            for file in workspace_dir.rglob("*"):
                if file.is_file():
                    size = file.stat().st_size
                    files_info["workspace_files"].append({
                        "filename": file.name,
                        "relative_path": str(file.relative_to(workspace_dir)),
                        "size": size
                    })
                    files_info["total_size"] += size

        return files_info

    def extract_content_summary(self, paper_dir):
        """提取內容摘要"""
        summary = {
            "has_content": False,
            "estimated_pages": 0,
            "content_preview": ""
        }

        # 查找主要 JSONL 文件
        jsonl_files = list(paper_dir.glob("*.jsonl"))
        if jsonl_files:
            main_jsonl = jsonl_files[0]  # 取第一個
            try:
                with open(main_jsonl, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    if line:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if text:
                            summary["has_content"] = True
                            summary["content_preview"] = text[:300] + "..." if len(text) > 300 else text
                            # 粗略估算頁數（每頁約2000字符）
                            summary["estimated_pages"] = max(1, len(text) // 2000)
            except Exception as e:
                logger.warning(f"讀取內容摘要失敗 {main_jsonl}: {e}")

        return summary

    def scan_aggregated_chunks(self):
        """掃描聚合的 JSONL 文件"""
        aggregated_info = {
            "total_files": 0,
            "total_size": 0,
            "files": []
        }

        if not self.aggregated_chunks_dir.exists():
            logger.warning(f"聚合文件目錄不存在: {self.aggregated_chunks_dir}")
            return aggregated_info

        for jsonl_file in self.aggregated_chunks_dir.glob("*.jsonl"):
            size = jsonl_file.stat().st_size
            aggregated_info["files"].append({
                "filename": jsonl_file.name,
                "size": size,
                "path": str(jsonl_file)
            })
            aggregated_info["total_files"] += 1
            aggregated_info["total_size"] += size

        return aggregated_info

    def generate_category_index(self, papers_by_category):
        """生成分類索引"""
        category_index = {}

        for category_code, papers in papers_by_category.items():
            category_info = self.categories.get(category_code, {"name": "未知分類"})

            category_index[category_code] = {
                "name": category_info["name"],
                "name_en": category_info.get("name_en", "Unknown"),
                "paper_count": len(papers),
                "total_size": sum(p["files"]["total_size"] for p in papers),
                "papers": [
                    {
                        "id": p["id"],
                        "output_name": p["output_name"],
                        "year": p.get("paper_info", {}).get("year", "UNKNOWN"),
                        "name": p.get("paper_info", {}).get("name", "UNKNOWN"),
                        "has_content": p["content_summary"]["has_content"],
                        "estimated_pages": p["content_summary"]["estimated_pages"],
                        "has_aggregated_chunk": p.get("has_aggregated_chunk", False)
                    }
                    for p in papers
                ]
            }

        return category_index

    def generate_search_index(self, all_papers):
        """生成搜索索引"""
        search_index = {
            "papers_by_year": {},
            "papers_by_keyword": {},
            "all_papers": []
        }

        for paper in all_papers:
            paper_info = paper.get("paper_info", {})
            year = paper_info.get("year", "UNKNOWN")
            name = paper_info.get("name", "UNKNOWN")

            # 按年份索引
            if year not in search_index["papers_by_year"]:
                search_index["papers_by_year"][year] = []
            search_index["papers_by_year"][year].append({
                "id": paper["id"],
                "name": name,
                "category": paper["category_code"],
                "output_name": paper["output_name"]
            })

            # 簡單的關鍵詞索引（基於論文名稱）
            keywords = name.lower().replace("_", " ").split()
            for keyword in keywords:
                if len(keyword) > 2:  # 忽略太短的詞
                    if keyword not in search_index["papers_by_keyword"]:
                        search_index["papers_by_keyword"][keyword] = []
                    search_index["papers_by_keyword"][keyword].append({
                        "id": paper["id"],
                        "name": name,
                        "category": paper["category_code"]
                    })

            # 所有論文列表
            search_index["all_papers"].append({
                "id": paper["id"],
                "name": name,
                "year": year,
                "category_code": paper["category_code"],
                "category_name": paper["category_name"],
                "output_name": paper["output_name"],
                "has_content": paper["content_summary"]["has_content"],
                "has_aggregated_chunk": paper.get("has_aggregated_chunk", False)
            })

        return search_index

    def generate_statistics(self, all_papers, aggregated_info):
        """生成統計信息"""
        stats = {
            "total_papers": len(all_papers),
            "by_category": {},
            "by_year": {},
            "total_size": 0,
            "aggregated_chunks": aggregated_info,
            "generation_time": datetime.now().isoformat(),
            "architecture_version": "2.0"
        }

        for paper in all_papers:
            # 分類統計
            category = paper["category_code"]
            if category not in stats["by_category"]:
                stats["by_category"][category] = {
                    "count": 0,
                    "size": 0,
                    "name": paper["category_name"],
                    "name_en": paper["category_name_en"]
                }
            stats["by_category"][category]["count"] += 1
            stats["by_category"][category]["size"] += paper["files"]["total_size"]

            # 年份統計
            year = paper.get("paper_info", {}).get("year", "UNKNOWN")
            if year not in stats["by_year"]:
                stats["by_year"][year] = 0
            stats["by_year"][year] += 1

            # 總大小
            stats["total_size"] += paper["files"]["total_size"]

        return stats

    def generate_all_indexes(self):
        """生成所有索引文件"""
        logger.info("開始生成統一輸出架構的全局索引...")

        # 掃描所有處理文件
        all_papers = self.scan_processed_files()
        logger.info(f"找到 {len(all_papers)} 篇論文")

        # 掃描聚合文件
        aggregated_info = self.scan_aggregated_chunks()
        logger.info(f"找到 {aggregated_info['total_files']} 個聚合文件")

        if not all_papers:
            logger.warning("沒有找到已處理的論文")
            return

        # 按分類分組
        papers_by_category = {}
        for paper in all_papers:
            category = paper["category_code"]
            if category not in papers_by_category:
                papers_by_category[category] = []
            papers_by_category[category].append(paper)

        # 生成各種索引
        category_index = self.generate_category_index(papers_by_category)
        search_index = self.generate_search_index(all_papers)
        statistics = self.generate_statistics(all_papers, aggregated_info)

        # 保存索引文件
        indexes = {
            "master_index.json": {
                "metadata": {
                    "total_papers": len(all_papers),
                    "categories": list(papers_by_category.keys()),
                    "generation_time": datetime.now().isoformat(),
                    "base_directory": str(self.base_dir),
                    "architecture_version": "2.0",
                    "outputs_structure": {
                        "processed_papers": str(self.processed_papers_dir),
                        "aggregated_chunks": str(self.aggregated_chunks_dir),
                        "indices": str(self.index_dir)
                    }
                },
                "papers": all_papers
            },
            "category_index.json": category_index,
            "search_index.json": search_index,
            "statistics.json": statistics
        }

        for filename, data in indexes.items():
            index_file = self.index_dir / filename
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"生成索引文件: outputs/indices/{filename}")

        # 生成簡化的 README
        self.generate_index_readme(statistics)

        logger.info("統一輸出架構全局索引生成完成!")
        return statistics

    def generate_index_readme(self, statistics):
        """生成索引說明文件"""
        readme_content = f"""# 論文知識庫索引 (統一輸出架構 v2.0)

## 統計信息

- **論文總數**: {statistics['total_papers']}
- **總大小**: {statistics['total_size'] / (1024*1024*1024):.2f} GB
- **聚合文件**: {statistics['aggregated_chunks']['total_files']} 個
- **生成時間**: {statistics['generation_time']}
- **架構版本**: {statistics['architecture_version']}

## 分類統計

"""

        for category_code, info in statistics['by_category'].items():
            readme_content += f"- **{info['name']} ({category_code})**: {info['count']} 篇論文, {info['size'] / (1024*1024):.1f} MB\n"

        readme_content += f"""
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
    chunk_file = chunks_dir / f"{{paper['output_name']}}.jsonl"
    if chunk_file.exists():
        # 讀取 JSONL 內容
        pass
```

## 聚合文件統計

- **總文件數**: {statistics['aggregated_chunks']['total_files']}
- **總大小**: {statistics['aggregated_chunks']['total_size'] / (1024*1024):.1f} MB
"""

        readme_file = self.index_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        logger.info("生成索引說明文件: outputs/indices/README.md")

def main():
    parser = argparse.ArgumentParser(description="生成論文知識庫全局索引 (統一輸出架構)")
    parser.add_argument("--base_dir",
                       default="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03.1-papers_knowledge_base",
                       help="知識庫基礎目錄")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error(f"基礎目錄不存在: {base_dir}")
        return

    generator = GlobalIndexGenerator(base_dir)
    statistics = generator.generate_all_indexes()

    if statistics:
        print("\n" + "=" * 60)
        print("🎉 統一輸出架構索引生成完成!")
        print(f"📊 論文總數: {statistics['total_papers']}")
        print(f"📁 索引目錄: {generator.index_dir}")
        print(f"💾 總大小: {statistics['total_size'] / (1024*1024*1024):.2f} GB")
        print(f"📄 聚合文件: {statistics['aggregated_chunks']['total_files']} 個")
        print("\n📋 分類統計:")
        for category_code, info in statistics['by_category'].items():
            print(f"  {info['name']} ({category_code}): {info['count']} 篇")

if __name__ == "__main__":
    main()