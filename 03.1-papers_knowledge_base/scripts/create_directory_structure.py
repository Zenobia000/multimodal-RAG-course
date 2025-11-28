#!/usr/bin/env python3
"""
論文知識庫目錄結構創建腳本 v2.0
根據統一輸出架構自動創建所有必要的目錄結構
"""

import os
from pathlib import Path
from datetime import datetime
import json

def create_directory_structure():
    """創建統一輸出架構的目錄結構"""

    # 基礎路徑
    base_path = Path("/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03.1-papers_knowledge_base")

    # 分類定義
    categories = {
        "MP": {
            "name_en": "Model_Paradigm",
            "name_zh": "模型範式",
            "source": "01_model_paradigm",
            "file_count": 10
        },
        "IF": {
            "name_en": "Infrastructure",
            "name_zh": "基礎設施",
            "source": "02_infrastructure",
            "file_count": 5
        },
        "LM": {
            "name_en": "Language_Models",
            "name_zh": "語言模型",
            "source": "03_language_models",
            "file_count": 8
        },
        "MM": {
            "name_en": "Multimodal",
            "name_zh": "多模態",
            "source": "04_multimodal",
            "file_count": 8
        }
    }

    print("🏗️ 創建論文知識庫統一輸出架構")
    print("=" * 60)

    # 創建基礎目錄
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ 基礎目錄: {base_path}")

    # 創建統一輸出目錄結構
    outputs_path = base_path / "outputs"
    outputs_path.mkdir(exist_ok=True)
    print(f"✅ 統一輸出目錄: outputs/")

    # 創建主要子目錄
    main_subdirs = {
        "processed_papers": "按分類組織的論文處理結果",
        "aggregated_chunks": "所有論文的JSONL文件統一管理",
        "embeddings": "向量嵌入文件",
        "indices": "全局索引文件",
        "logs": "統一處理日誌",
        "temp": "臨時處理工作區"
    }

    for subdir, description in main_subdirs.items():
        subdir_path = outputs_path / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"  ├── {subdir}/ ({description})")

    # 為每個分類創建 processed_papers 子目錄
    processed_papers_path = outputs_path / "processed_papers"
    for category_code, info in categories.items():
        category_processed_path = processed_papers_path / category_code
        category_processed_path.mkdir(exist_ok=True)
        print(f"    ├── {category_code}/ ({info['name_zh']})")

    # 創建 scripts 目錄（如果不存在）
    scripts_path = base_path / "scripts"
    scripts_path.mkdir(exist_ok=True)
    print(f"✅ 處理腳本目錄: scripts/")

    # 創建架構信息文件
    structure_info = {
        "architecture": "統一輸出架構 v2.0",
        "base_path": str(base_path),
        "outputs_structure": {
            "processed_papers": {
                "description": "按分類組織的論文處理結果",
                "categories": {code: info['name_zh'] for code, info in categories.items()}
            },
            "aggregated_chunks": {
                "description": "所有論文的JSONL文件統一管理",
                "naming_format": "{category_code}_{year}_{paper_name}_{timestamp}.jsonl"
            },
            "embeddings": {"description": "向量嵌入文件"},
            "indices": {"description": "全局索引文件"},
            "logs": {"description": "統一處理日誌"},
            "temp": {"description": "臨時處理工作區（自動清理）"}
        },
        "categories": categories,
        "total_pdf_files": sum(info['file_count'] for info in categories.values()),
        "created_time": datetime.now().isoformat(),
        "version": "2.0"
    }

    # 保存架構信息到 JSON 文件
    structure_info_file = base_path / "structure_info.json"
    with open(structure_info_file, 'w', encoding='utf-8') as f:
        json.dump(structure_info, f, indent=2, ensure_ascii=False)
    print(f"✅ 架構信息文件: structure_info.json")

    # 創建統計信息文件
    stats_file = base_path / "structure_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("統一輸出架構統計\n")
        f.write("=" * 30 + "\n")
        f.write(f"基礎目錄: {base_path}\n")
        f.write(f"架構版本: v2.0\n")
        f.write(f"分類數量: {len(categories)}\n")
        f.write(f"總PDF文件: {sum(info['file_count'] for info in categories.values())}\n")
        f.write(f"創建時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("統一輸出目錄結構:\n")
        f.write("outputs/\n")
        for subdir, description in main_subdirs.items():
            f.write(f"  ├── {subdir}/ - {description}\n")
            if subdir == "processed_papers":
                for category_code, info in categories.items():
                    f.write(f"    ├── {category_code}/ - {info['name_zh']} ({info['file_count']} files)\n")
        f.write("\n")

        f.write("分類詳細:\n")
        for category_code, info in categories.items():
            f.write(f"  {category_code}:\n")
            f.write(f"    中文名稱: {info['name_zh']}\n")
            f.write(f"    英文名稱: {info['name_en']}\n")
            f.write(f"    源目錄: {info['source']}\n")
            f.write(f"    文件數量: {info['file_count']}\n\n")

        # 計算總目錄數
        total_dirs = len(main_subdirs) + len(categories) + 2  # outputs + scripts + base
        f.write(f"總目錄數: {total_dirs}\n")

    print(f"✅ 統計文件: structure_stats.txt")

    # 創建 .gitignore 文件（忽略臨時和大文件）
    gitignore_content = """# 忽略臨時文件
outputs/temp/
*.log

# 忽略大型輸出文件（可選）
# outputs/processed_papers/
# outputs/aggregated_chunks/
# outputs/embeddings/

# 忽略系統文件
.DS_Store
__pycache__/
*.pyc
"""

    gitignore_file = base_path / ".gitignore"
    with open(gitignore_file, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print(f"✅ Git忽略文件: .gitignore")

    print("\n" + "=" * 60)
    print("🎉 統一輸出架構創建完成！")
    print(f"📁 基礎路徑: {base_path}")
    print(f"📊 分類數量: {len(categories)}")
    print(f"📈 預期處理: {sum(info['file_count'] for info in categories.values())} 個PDF文件")
    print("\n🗂️ 主要輸出目錄:")
    for subdir, description in main_subdirs.items():
        print(f"  • outputs/{subdir}/ - {description}")

    print("\n📋 下一步:")
    print("  1. 運行 batch_process_papers.py 處理PDF文件")
    print("  2. 使用 --max_workers 1 進行單線程處理")
    print("  3. 檢查 outputs/logs/ 中的處理日誌")
    print("  4. 運行 verify_processing_results.py 驗證結果")

def verify_structure():
    """驗證目錄結構是否正確創建"""
    base_path = Path("/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03.1-papers_knowledge_base")

    required_paths = [
        "outputs",
        "outputs/processed_papers",
        "outputs/aggregated_chunks",
        "outputs/logs",
        "outputs/temp",
        "outputs/embeddings",
        "outputs/indices",
        "scripts"
    ]

    print("🔍 驗證目錄結構...")
    all_exist = True

    for path_str in required_paths:
        path = base_path / path_str
        if path.exists():
            print(f"✅ {path_str}")
        else:
            print(f"❌ {path_str} - 不存在")
            all_exist = False

    if all_exist:
        print("\n🎉 所有必需目錄已正確創建！")
    else:
        print("\n⚠️ 部分目錄缺失，請重新運行 create_directory_structure()")

    return all_exist

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_structure()
    else:
        create_directory_structure()
        print("\n" + "="*30)
        verify_structure()