#!/bin/bash
# Embedding Application 設置腳本

echo "🚀 Embedding Application 環境設置"
echo "=" * 50

# 檢查 Python 版本
echo "🐍 檢查 Python 版本..."
python3 --version

# 創建必要目錄
echo "📁 創建目錄結構..."
mkdir -p logs

# 設置權限
echo "🔒 設置執行權限..."
chmod +x embedding_pipeline.py
chmod +x run_pipeline.py
chmod +x example_usage.py

# 檢查 .env 文件
if [ ! -f .env ]; then
    echo "📝 創建 .env 文件..."
    cp .env.example .env
    echo "⚠️  請編輯 .env 文件並設置你的 OPENAI_API_KEY"
else
    echo "✅ .env 文件已存在"
fi

# 檢查依賴
echo "📦 檢查 Python 依賴..."
python3 -c "
import sys
missing = []
required = ['langchain', 'qdrant_client', 'openai', 'pydantic']

for pkg in required:
    try:
        __import__(pkg.replace('-', '_'))
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'\\n📦 需要安裝: pip install {\" \".join(missing)}')
    print('或執行: pip install -r requirements.txt')
else:
    print('\\n✅ 所有依賴都已安裝')
"

# 檢查 Qdrant 連線
echo "🗄️ 檢查 Qdrant 服務..."
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant 服務運行正常"
else
    echo "❌ 無法連接 Qdrant 服務 (localhost:6333)"
    echo "   請確保 Qdrant 已啟動"
fi

# 檢查 OLMoCR 輸出路徑
OLMOCR_PATH="../03-advanced-tools/olmocr/output/workspace"
if [ -d "$OLMOCR_PATH" ]; then
    JSONL_COUNT=$(find "$OLMOCR_PATH" -name "output_*.jsonl" | wc -l)
    echo "✅ OLMoCR 輸出路徑存在，找到 $JSONL_COUNT 個 JSONL 文件"
else
    echo "⚠️  OLMoCR 輸出路徑不存在: $OLMOCR_PATH"
fi

echo ""
echo "🎯 設置完成！"
echo "📋 下一步："
echo "   1. 編輯 .env 文件設置 OPENAI_API_KEY"
echo "   2. 如需要，安裝依賴: pip install -r requirements.txt"
echo "   3. 測試執行: python3 example_usage.py"
echo "   4. 執行 pipeline: python3 run_pipeline.py"