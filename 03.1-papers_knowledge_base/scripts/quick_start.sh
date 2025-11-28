#!/bin/bash
# 論文知識庫建立 - 快速開始腳本

echo "🚀 論文知識庫建立 - 快速開始"
echo "=" * 60

# 設定路徑
BASE_DIR="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03.1-papers_knowledge_base"
SCRIPTS_DIR="$BASE_DIR/scripts"
OLMOCR_DIR="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03-advanced-tools/olmocr"
VENV_PATH="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03-advanced-tools/.venv"

# 檢查路徑
echo "🔍 檢查環境..."

if [ ! -d "$OLMOCR_DIR" ]; then
    echo "❌ OLMoCR 目錄不存在: $OLMOCR_DIR"
    exit 1
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 虛擬環境不存在: $VENV_PATH"
    exit 1
fi

echo "✅ 環境檢查通過"

# 進入基礎目錄
echo "📁 準備工作目錄..."
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# 設定執行權限
chmod +x "$SCRIPTS_DIR"/*.py

# 步驟 1: 創建目錄結構
echo "🏗️ 步驟 1: 創建目錄結構"
python3 "$SCRIPTS_DIR/create_directory_structure.py"

if [ $? -eq 0 ]; then
    echo "✅ 目錄結構創建完成"
else
    echo "❌ 目錄結構創建失敗"
    exit 1
fi

echo ""
echo "🎯 目錄結構已準備完成！"
echo ""
echo "📋 接下來的步驟（腳本將自動執行部分）："
echo ""
echo "1. 激活虛擬環境..."
source "$VENV_PATH/bin/activate"
echo "✅ 虛擬環境已激活"
echo ""

# 步驟 2: 批量處理論文 (僅處理 infrastructure 分類作為範例)
echo "🚀 步驟 2: 開始批量處理 'infrastructure' 分類的論文..."
python3 "$SCRIPTS_DIR/batch_process_papers.py" \
  --input_dir '/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/papers/02_infrastructure' \
  --output_dir "$BASE_DIR" \
  --category 'IF' \
  --max_workers 1

if [ $? -ne 0 ]; then
    echo "❌ 'infrastructure' 分類處理失敗"
    exit 1
fi
echo "✅ 'infrastructure' 分類處理完成"
echo ""

# 步驟 3: 生成全局索引
echo "🌐 步驟 3: 生成全局索引..."
python3 "$SCRIPTS_DIR/generate_global_index.py"
if [ $? -ne 0 ]; then
    echo "❌ 全局索引生成失敗"
    exit 1
fi
echo "✅ 全局索引生成完成"
echo ""

# 步驟 4: 驗證處理結果
echo "🔎 步驟 4: 驗證處理結果..."
python3 "$SCRIPTS_DIR/verify_processing_results.py"
if [ $? -ne 0 ]; then
    echo "❌ 結果驗證失敗"
    exit 1
fi
echo "✅ 結果驗證成功"
echo ""

echo "🎉 所有自動化步驟已成功完成！"
echo ""
echo "🔔 提醒:"
echo "   - 本快速腳本僅處理了 'infrastructure' 分類。"
echo "   - 如需處理其他分類，請參考本腳本中的命令手動執行。"
echo "   - 所有輸出位於 '$BASE_DIR/outputs' 目錄。"
echo ""
echo "如有問題，請查閱 logs/ 目錄中的日誌文件。"