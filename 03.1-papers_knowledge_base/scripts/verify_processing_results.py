#!/usr/bin/env python3
"""
處理結果驗證腳本 v2.0
驗證統一輸出架構論文知識庫的完整性和品質
"""

import json
import os
from pathlib import Path
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessingVerifier:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.outputs_dir = self.base_dir / "outputs"
        self.source_papers_dir = Path("/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/papers")

        # 統一輸出架構的目錄
        self.processed_papers_dir = self.outputs_dir / "processed_papers"
        self.aggregated_chunks_dir = self.outputs_dir / "aggregated_chunks"
        self.indices_dir = self.outputs_dir / "indices"
        self.logs_dir = self.outputs_dir / "logs"

        self.categories = {
            "MP": {"name": "模型範式", "source": "01_model_paradigm"},
            "IF": {"name": "基礎設施", "source": "02_infrastructure"},
            "LM": {"name": "語言模型", "source": "03_language_models"},
            "MM": {"name": "多模態", "source": "04_multimodal"}
        }

        self.verification_results = {
            "timestamp": datetime.now().isoformat(),
            "base_directory": str(base_dir),
            "architecture_version": "2.0",
            "checks": {},
            "summary": {}
        }

    def check_directory_structure(self):
        """檢查統一輸出架構的目錄結構完整性"""
        logger.info("檢查統一輸出架構目錄結構...")

        structure_check = {
            "base_exists": self.base_dir.exists(),
            "outputs_structure": {},
            "category_dirs": {}
        }

        # 檢查統一的輸出目錄結構
        required_dirs = {
            "outputs": self.outputs_dir,
            "processed_papers": self.processed_papers_dir,
            "aggregated_chunks": self.aggregated_chunks_dir,
            "indices": self.indices_dir,
            "logs": self.logs_dir,
            "temp": self.outputs_dir / "temp",
            "embeddings": self.outputs_dir / "embeddings"
        }

        for dir_name, dir_path in required_dirs.items():
            structure_check["outputs_structure"][dir_name] = dir_path.exists()

        # 檢查分類目錄
        for category_code, info in self.categories.items():
            category_path = self.processed_papers_dir / category_code
            structure_check["category_dirs"][category_code] = {
                "exists": category_path.exists(),
                "name": info["name"]
            }

        self.verification_results["checks"]["directory_structure"] = structure_check
        return structure_check

    def check_source_coverage(self):
        """檢查源文件覆蓋率"""
        logger.info("檢查源文件覆蓋率...")

        coverage_check = {
            "categories": {},
            "overall": {},
            "aggregated_chunks": {}
        }

        total_source = 0
        total_processed = 0
        total_aggregated = 0

        for category_code, info in self.categories.items():
            source_dir = self.source_papers_dir / info["source"]
            processed_dir = self.processed_papers_dir / category_code

            # 計算源文件數量
            source_pdfs = list(source_dir.glob("*.pdf")) if source_dir.exists() else []
            processed_items = list(processed_dir.iterdir()) if processed_dir.exists() else []

            # 檢查聚合文件 - 修正檢查邏輯
            aggregated_files = []
            if self.aggregated_chunks_dir.exists():
                # 直接按分類代碼匹配聚合文件
                category_aggregated_files = list(self.aggregated_chunks_dir.glob(f"{category_code}_*.jsonl"))
                aggregated_files = category_aggregated_files

            coverage_check["categories"][category_code] = {
                "source_count": len(source_pdfs),
                "processed_count": len([item for item in processed_items if item.is_dir()]),
                "aggregated_count": len(aggregated_files),
                "coverage_rate": len(processed_items) / len(source_pdfs) if source_pdfs else 0,
                "source_files": [f.name for f in source_pdfs],
                "processed_files": [f.name for f in processed_items if f.is_dir()],
                "aggregated_files": [f.name for f in aggregated_files]
            }

            total_source += len(source_pdfs)
            total_processed += len([item for item in processed_items if item.is_dir()])
            total_aggregated += len(aggregated_files)

        coverage_check["overall"] = {
            "total_source": total_source,
            "total_processed": total_processed,
            "total_aggregated": total_aggregated,
            "processed_coverage": total_processed / total_source if total_source > 0 else 0,
            "aggregated_coverage": total_aggregated / total_source if total_source > 0 else 0
        }

        # 檢查聚合文件目錄
        if self.aggregated_chunks_dir.exists():
            all_aggregated = list(self.aggregated_chunks_dir.glob("*.jsonl"))
            coverage_check["aggregated_chunks"] = {
                "total_files": len(all_aggregated),
                "total_size": sum(f.stat().st_size for f in all_aggregated),
                "files": [f.name for f in all_aggregated]
            }

        self.verification_results["checks"]["source_coverage"] = coverage_check
        return coverage_check

    def check_file_integrity(self):
        """檢查文件完整性"""
        logger.info("檢查文件完整性...")

        integrity_check = {
            "categories": {},
            "aggregated_chunks": {},
            "issues": []
        }

        # 檢查處理文件完整性
        for category_code, info in self.categories.items():
            category_path = self.processed_papers_dir / category_code

            category_integrity = {
                "processed_items": [],
                "missing_metadata": [],
                "empty_items": [],
                "valid_items": [],
                "missing_aggregated": []
            }

            if category_path.exists():
                for item in category_path.iterdir():
                    if item.is_dir():
                        item_name = item.name
                        category_integrity["processed_items"].append(item_name)

                        # 檢查是否有對應的元數據
                        metadata_file = item / f"{item_name}_metadata.json"
                        if not metadata_file.exists():
                            category_integrity["missing_metadata"].append(item_name)
                            integrity_check["issues"].append(f"缺少元數據: {category_code}/{item_name}")

                        # 檢查是否為空目錄
                        if not any(item.iterdir()):
                            category_integrity["empty_items"].append(item_name)
                            integrity_check["issues"].append(f"空目錄: {category_code}/{item_name}")
                        else:
                            # 檢查是否有 JSONL 文件
                            if list(item.glob("*.jsonl")):
                                category_integrity["valid_items"].append(item_name)
                            else:
                                integrity_check["issues"].append(f"缺少 JSONL 文件: {category_code}/{item_name}")

                        # 檢查對應的聚合文件是否存在
                        aggregated_file = self.aggregated_chunks_dir / f"{item_name}.jsonl"
                        if not aggregated_file.exists():
                            category_integrity["missing_aggregated"].append(item_name)
                            integrity_check["issues"].append(f"缺少聚合文件: {item_name}.jsonl")

            integrity_check["categories"][category_code] = category_integrity

        # 檢查聚合文件完整性
        if self.aggregated_chunks_dir.exists():
            aggregated_files = list(self.aggregated_chunks_dir.glob("*.jsonl"))
            orphan_files = []

            for aggregated_file in aggregated_files:
                # 檢查是否有對應的處理文件
                found_processed = False
                for category_code in self.categories.keys():
                    processed_dir = self.processed_papers_dir / category_code / aggregated_file.stem
                    if processed_dir.exists():
                        found_processed = True
                        break

                if not found_processed:
                    orphan_files.append(aggregated_file.name)

            integrity_check["aggregated_chunks"] = {
                "total_files": len(aggregated_files),
                "orphan_files": orphan_files
            }

            if orphan_files:
                integrity_check["issues"].extend([f"孤立聚合文件: {f}" for f in orphan_files])

        self.verification_results["checks"]["file_integrity"] = integrity_check
        return integrity_check

    def check_content_quality(self):
        """檢查內容品質"""
        logger.info("檢查內容品質...")

        quality_check = {
            "categories": {},
            "aggregated_chunks": {},
            "overall_stats": {
                "total_papers": 0,
                "papers_with_content": 0,
                "total_content_size": 0,
                "average_content_size": 0
            }
        }

        total_papers = 0
        papers_with_content = 0
        total_content_size = 0

        # 檢查處理文件的內容品質
        for category_code, info in self.categories.items():
            category_path = self.processed_papers_dir / category_code

            category_quality = {
                "papers": [],
                "content_stats": {
                    "count": 0,
                    "with_content": 0,
                    "total_size": 0
                }
            }

            if category_path.exists():
                for item in category_path.iterdir():
                    if item.is_dir():
                        paper_quality = self.analyze_paper_quality(item)
                        category_quality["papers"].append(paper_quality)

                        # 更新統計
                        category_quality["content_stats"]["count"] += 1
                        total_papers += 1

                        if paper_quality["has_content"]:
                            category_quality["content_stats"]["with_content"] += 1
                            papers_with_content += 1

                        content_size = paper_quality["content_size"]
                        category_quality["content_stats"]["total_size"] += content_size
                        total_content_size += content_size

            quality_check["categories"][category_code] = category_quality

        # 檢查聚合文件的內容品質
        if self.aggregated_chunks_dir.exists():
            aggregated_files = list(self.aggregated_chunks_dir.glob("*.jsonl"))
            aggregated_quality = {
                "total_files": len(aggregated_files),
                "files_with_content": 0,
                "total_size": 0,
                "average_size": 0
            }

            files_with_content = 0
            total_aggregated_size = 0

            for aggregated_file in aggregated_files:
                try:
                    file_size = aggregated_file.stat().st_size
                    total_aggregated_size += file_size

                    with open(aggregated_file, 'r', encoding='utf-8') as f:
                        line = f.readline().strip()
                        if line:
                            data = json.loads(line)
                            text = data.get("text", "")
                            if text and len(text) > 100:
                                files_with_content += 1
                except Exception as e:
                    logger.warning(f"分析聚合文件失敗 {aggregated_file}: {e}")

            aggregated_quality["files_with_content"] = files_with_content
            aggregated_quality["total_size"] = total_aggregated_size
            aggregated_quality["average_size"] = total_aggregated_size / len(aggregated_files) if aggregated_files else 0

            quality_check["aggregated_chunks"] = aggregated_quality

        # 計算全局統計
        quality_check["overall_stats"] = {
            "total_papers": total_papers,
            "papers_with_content": papers_with_content,
            "total_content_size": total_content_size,
            "average_content_size": total_content_size / total_papers if total_papers > 0 else 0,
            "content_coverage_rate": papers_with_content / total_papers if total_papers > 0 else 0
        }

        self.verification_results["checks"]["content_quality"] = quality_check
        return quality_check

    def analyze_paper_quality(self, paper_dir):
        """分析單篇論文的品質"""
        quality_info = {
            "name": paper_dir.name,
            "has_content": False,
            "content_size": 0,
            "file_count": 0,
            "has_jsonl": False,
            "has_workspace": False,
            "has_metadata": False
        }

        # 計算文件數量
        quality_info["file_count"] = len(list(paper_dir.rglob("*")))

        # 檢查元數據文件
        metadata_file = paper_dir / f"{paper_dir.name}_metadata.json"
        quality_info["has_metadata"] = metadata_file.exists()

        # 檢查 JSONL 文件
        jsonl_files = list(paper_dir.glob("*.jsonl"))
        quality_info["has_jsonl"] = len(jsonl_files) > 0

        if jsonl_files:
            # 分析主要 JSONL 文件的內容
            main_jsonl = jsonl_files[0]
            try:
                with open(main_jsonl, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    if line:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if text and len(text) > 100:  # 至少100字符才算有內容
                            quality_info["has_content"] = True
                            quality_info["content_size"] = len(text)
            except Exception as e:
                logger.warning(f"分析內容失敗 {main_jsonl}: {e}")

        # 檢查 workspace
        workspace_dir = paper_dir / "workspace"
        quality_info["has_workspace"] = workspace_dir.exists()

        return quality_info

    def check_naming_consistency(self):
        """檢查檔名一致性"""
        logger.info("檢查檔名一致性...")

        naming_check = {
            "categories": {},
            "aggregated_chunks": {},
            "naming_violations": []
        }

        # 檢查處理文件的命名一致性
        for category_code, info in self.categories.items():
            category_path = self.processed_papers_dir / category_code

            category_naming = {
                "expected_prefix": category_code,
                "compliant_names": [],
                "non_compliant_names": []
            }

            if category_path.exists():
                for item in category_path.iterdir():
                    if item.is_dir():
                        item_name = item.name
                        if item_name.startswith(f"{category_code}_"):
                            category_naming["compliant_names"].append(item_name)
                        else:
                            category_naming["non_compliant_names"].append(item_name)
                            naming_check["naming_violations"].append(f"處理文件: {category_code}/{item_name}")

            naming_check["categories"][category_code] = category_naming

        # 檢查聚合文件的命名一致性
        if self.aggregated_chunks_dir.exists():
            aggregated_files = list(self.aggregated_chunks_dir.glob("*.jsonl"))
            aggregated_naming = {
                "compliant_names": [],
                "non_compliant_names": []
            }

            for aggregated_file in aggregated_files:
                filename = aggregated_file.stem
                # 檢查是否符合 {category}_{year}_{name}_{timestamp} 格式
                parts = filename.split("_", 3)
                if len(parts) >= 3 and parts[0] in self.categories:
                    aggregated_naming["compliant_names"].append(filename)
                else:
                    aggregated_naming["non_compliant_names"].append(filename)
                    naming_check["naming_violations"].append(f"聚合文件: {filename}")

            naming_check["aggregated_chunks"] = aggregated_naming

        self.verification_results["checks"]["naming_consistency"] = naming_check
        return naming_check

    def check_index_files(self):
        """檢查索引文件完整性"""
        logger.info("檢查索引文件...")

        index_check = {
            "indices_dir_exists": self.indices_dir.exists(),
            "required_files": {},
            "optional_files": {}
        }

        required_index_files = [
            "master_index.json",
            "category_index.json",
            "search_index.json",
            "statistics.json"
        ]

        optional_index_files = [
            "README.md"
        ]

        for filename in required_index_files:
            file_path = self.indices_dir / filename
            index_check["required_files"][filename] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0
            }

        for filename in optional_index_files:
            file_path = self.indices_dir / filename
            index_check["optional_files"][filename] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0
            }

        self.verification_results["checks"]["index_files"] = index_check
        return index_check

    def generate_verification_report(self):
        """生成驗證報告"""
        logger.info("生成驗證報告...")

        # 計算摘要統計
        checks = self.verification_results["checks"]

        summary = {
            "overall_status": "PASS",
            "total_issues": 0,
            "critical_issues": 0,
            "warnings": 0,
            "architecture_version": "2.0"
        }

        issues = []

        # 檢查目錄結構
        if "directory_structure" in checks:
            if not checks["directory_structure"]["base_exists"]:
                issues.append("CRITICAL: 基礎目錄不存在")
                summary["critical_issues"] += 1

            # 檢查重要的輸出目錄
            outputs_structure = checks["directory_structure"]["outputs_structure"]
            critical_dirs = ["outputs", "processed_papers", "aggregated_chunks"]
            for dir_name in critical_dirs:
                if not outputs_structure.get(dir_name, False):
                    issues.append(f"CRITICAL: 缺少重要目錄 {dir_name}")
                    summary["critical_issues"] += 1

        # 檢查覆蓋率
        if "source_coverage" in checks:
            processed_coverage = checks["source_coverage"]["overall"]["processed_coverage"]
            aggregated_coverage = checks["source_coverage"]["overall"]["aggregated_coverage"]

            if processed_coverage < 0.8:  # 少於80%覆蓋率
                issues.append(f"WARNING: 處理文件覆蓋率僅 {processed_coverage:.1%}")
                summary["warnings"] += 1

            if aggregated_coverage < 0.8:  # 少於80%覆蓋率
                issues.append(f"WARNING: 聚合文件覆蓋率僅 {aggregated_coverage:.1%}")
                summary["warnings"] += 1

        # 檢查文件完整性
        if "file_integrity" in checks:
            integrity_issues = len(checks["file_integrity"]["issues"])
            if integrity_issues > 0:
                issues.append(f"WARNING: 發現 {integrity_issues} 個文件完整性問題")
                summary["warnings"] += integrity_issues

        # 檢查內容品質
        if "content_quality" in checks:
            content_rate = checks["content_quality"]["overall_stats"]["content_coverage_rate"]
            if content_rate < 0.9:  # 少於90%有內容
                issues.append(f"WARNING: 內容覆蓋率僅 {content_rate:.1%}")
                summary["warnings"] += 1

        # 檢查索引文件
        if "index_files" in checks:
            if not checks["index_files"]["indices_dir_exists"]:
                issues.append("WARNING: 索引目錄不存在")
                summary["warnings"] += 1
            else:
                missing_indices = [name for name, info in checks["index_files"]["required_files"].items()
                                 if not info["exists"]]
                if missing_indices:
                    issues.append(f"WARNING: 缺少索引文件: {', '.join(missing_indices)}")
                    summary["warnings"] += len(missing_indices)

        summary["total_issues"] = summary["critical_issues"] + summary["warnings"]
        if summary["critical_issues"] > 0:
            summary["overall_status"] = "FAIL"
        elif summary["warnings"] > 0:
            summary["overall_status"] = "PASS_WITH_WARNINGS"

        summary["issues"] = issues

        self.verification_results["summary"] = summary

        # 保存驗證報告到統一的 logs 目錄
        report_file = self.logs_dir / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.verification_results, f, indent=2, ensure_ascii=False)

        return summary, report_file

    def run_all_checks(self):
        """執行所有檢查"""
        logger.info("開始執行統一輸出架構處理結果驗證...")

        # 執行各項檢查
        self.check_directory_structure()
        self.check_source_coverage()
        self.check_file_integrity()
        self.check_content_quality()
        self.check_naming_consistency()
        self.check_index_files()

        # 生成報告
        summary, report_file = self.generate_verification_report()

        return summary, report_file

def main():
    parser = argparse.ArgumentParser(description="驗證統一輸出架構論文知識庫處理結果")
    parser.add_argument("--base_dir",
                       default="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03.1-papers_knowledge_base",
                       help="知識庫基礎目錄")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error(f"基礎目錄不存在: {base_dir}")
        return

    verifier = ProcessingVerifier(base_dir)
    summary, report_file = verifier.run_all_checks()

    # 顯示結果
    print("\n" + "=" * 60)
    print("🔍 統一輸出架構處理結果驗證完成!")
    print(f"📋 總體狀態: {summary['overall_status']}")
    print(f"⚠️ 總問題數: {summary['total_issues']}")
    print(f"🔥 嚴重問題: {summary['critical_issues']}")
    print(f"⚡ 警告: {summary['warnings']}")
    print(f"📄 詳細報告: {report_file}")
    print(f"🏗️ 架構版本: {summary['architecture_version']}")

    if summary["issues"]:
        print("\n📋 發現的問題:")
        for issue in summary["issues"]:
            print(f"  - {issue}")

    # 顯示詳細統計
    checks = verifier.verification_results["checks"]
    if "source_coverage" in checks:
        coverage = checks["source_coverage"]["overall"]
        print(f"\n📊 處理統計:")
        print(f"  源文件: {coverage['total_source']}")
        print(f"  已處理: {coverage['total_processed']} ({coverage['processed_coverage']:.1%})")
        print(f"  聚合文件: {coverage['total_aggregated']} ({coverage['aggregated_coverage']:.1%})")

    if "content_quality" in checks:
        quality = checks["content_quality"]["overall_stats"]
        print(f"\n📝 內容品質:")
        print(f"  論文總數: {quality['total_papers']}")
        print(f"  有內容: {quality['papers_with_content']}")
        print(f"  內容覆蓋率: {quality['content_coverage_rate']:.1%}")
        print(f"  平均內容大小: {quality['average_content_size'] / 1024:.1f} KB")

        # 聚合文件統計
        if "aggregated_chunks" in checks["content_quality"]:
            aggregated = checks["content_quality"]["aggregated_chunks"]
            print(f"\n📄 聚合文件:")
            print(f"  文件總數: {aggregated['total_files']}")
            print(f"  有內容: {aggregated['files_with_content']}")
            print(f"  總大小: {aggregated['total_size'] / (1024*1024):.1f} MB")
            print(f"  平均大小: {aggregated['average_size'] / 1024:.1f} KB")

if __name__ == "__main__":
    main()