#!/usr/bin/env python3
"""
使用示例：展示如何使用 Embedding Pipeline
"""

import os
from embedding_pipeline import EmbeddingPipeline, EmbeddingConfig

def example_basic_usage():
    """基本使用示例"""
    print("📝 基本使用示例")
    print("-" * 30)

    # 檢查 API Key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  請先設定 OPENAI_API_KEY 環境變數")
        return False

    try:
        # 創建 pipeline 實例
        pipeline = EmbeddingPipeline()

        # 顯示配置
        config = pipeline.config
        print(f"🔧 配置檢查:")
        print(f"   Qdrant URL: {config.QDRANT_URL}")
        print(f"   集合名稱: {config.QDRANT_COLLECTION_NAME}")
        print(f"   文本塊大小: {config.CHUNK_SIZE}")
        print(f"   批次大小: {config.BATCH_SIZE}")

        # 檢查 OLMoCR 輸出路徑
        from pathlib import Path
        olmocr_path = Path(config.OLMOCR_OUTPUT_PATH)
        if olmocr_path.exists():
            print(f"✅ OLMoCR 路徑: {olmocr_path}")
        else:
            print(f"⚠️  OLMoCR 路徑不存在: {olmocr_path}")

        return True

    except Exception as e:
        print(f"❌ 配置檢查失敗: {e}")
        return False

def example_custom_config():
    """自定義配置示例"""
    print("\n📝 自定義配置示例")
    print("-" * 30)

    # 創建自定義配置
    config = EmbeddingConfig()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.BATCH_SIZE = 30

    print(f"🔧 自定義配置:")
    print(f"   文本塊大小: {config.CHUNK_SIZE}")
    print(f"   重疊大小: {config.CHUNK_OVERLAP}")
    print(f"   批次大小: {config.BATCH_SIZE}")

def example_check_dependencies():
    """檢查依賴套件"""
    print("\n📝 依賴檢查示例")
    print("-" * 30)

    dependencies = [
        'langchain',
        'langchain_openai',
        'langchain_qdrant',
        'qdrant_client',
        'openai',
        'pydantic',
        'python-dotenv'
    ]

    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} (需要安裝)")

def example_dry_run():
    """模擬執行示例（不實際處理）"""
    print("\n📝 模擬執行示例")
    print("-" * 30)

    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  請先設定 OPENAI_API_KEY 環境變數")
        return

    try:
        pipeline = EmbeddingPipeline()

        # 只載入和檢查 OLMoCR 結果，不實際處理
        processor = pipeline.processor
        results = processor.load_olmocr_results()

        print(f"📊 找到 {len(results)} 筆 OLMoCR 記錄")

        if results:
            # 顯示第一筆記錄的資訊
            first_result = results[0]
            print(f"📄 第一筆記錄:")
            print(f"   ID: {first_result.get('id', 'N/A')}")
            print(f"   源文件: {first_result.get('source_file', 'N/A')}")
            print(f"   內容長度: {len(first_result.get('text', ''))}")

            # 測試文檔處理（不寫入資料庫）
            documents = processor.process_documents(results[:1])  # 只處理第一筆
            print(f"🔄 處理結果: {len(documents)} 個文檔塊")

            if documents:
                first_doc = documents[0]
                print(f"📝 第一個文檔塊:")
                print(f"   ID: {first_doc.metadata.get('chunk_id', 'N/A')}")
                print(f"   大小: {first_doc.metadata.get('chunk_size', 'N/A')}")
                print(f"   內容預覽: {first_doc.page_content[:100]}...")

    except Exception as e:
        print(f"❌ 模擬執行失敗: {e}")

def main():
    """主函數"""
    print("🧪 Embedding Pipeline 使用示例")
    print("=" * 50)

    # 執行各種示例
    example_check_dependencies()
    example_basic_usage()
    example_custom_config()
    example_dry_run()

    print("\n" + "=" * 50)
    print("💡 使用提示:")
    print("   1. 設定環境變數: export OPENAI_API_KEY='your-key'")
    print("   2. 執行完整 pipeline: python3 run_pipeline.py")
    print("   3. 檢視詳細日誌: python3 embedding_pipeline.py")

if __name__ == "__main__":
    main()