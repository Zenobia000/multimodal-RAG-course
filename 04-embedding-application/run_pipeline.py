#!/usr/bin/env python3
"""
簡化的 Embedding Pipeline 執行腳本
適用於快速執行和測試
"""

import sys
import os
from pathlib import Path

# 添加當前目錄到 Python 路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from embedding_pipeline import EmbeddingPipeline

def main():
    """主執行函數"""
    print("🚀 快速執行 Embedding Pipeline")
    print("=" * 50)

    # 檢查環境變數
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ 請先設定 OPENAI_API_KEY 環境變數")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False

    try:
        # 創建並執行 pipeline
        pipeline = EmbeddingPipeline()

        print("📋 執行配置:")
        print(f"   - OLMoCR 路徑: {pipeline.config.OLMOCR_OUTPUT_PATH}")
        print(f"   - Qdrant URL: {pipeline.config.QDRANT_URL}")
        print(f"   - 集合名稱: {pipeline.config.QDRANT_COLLECTION_NAME}")
        print(f"   - 批次大小: {pipeline.config.BATCH_SIZE}")
        print("")

        # 執行 pipeline
        results = pipeline.run_pipeline()

        # 簡化結果顯示
        if results['success']:
            print("✅ 執行成功!")
            print(f"📊 處理: {results['processed_documents']} 個文檔")
            print(f"💾 新增: {results['added_documents']} 個向量")

            # 顯示集合統計
            stats = pipeline.get_collection_stats()
            if stats:
                print(f"🗄️ 總計: {stats.get('total_points', 'N/A')} 個向量")
        else:
            print("❌ 執行失敗")
            for error in results.get('errors', []):
                print(f"   {error}")

        return results['success']

    except Exception as e:
        print(f"❌ 執行出錯: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)