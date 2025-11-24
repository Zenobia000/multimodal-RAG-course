#!/bin/bash

# 論文批量處理腳本 - 處理所有分類
# 使用單線程模式避免衝突

set -e

BASE_DIR="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview"
PAPERS_DIR="${BASE_DIR}/papers"
SCRIPT_DIR="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03.1-papers_knowledge_base"

cd "${SCRIPT_DIR}"

echo "🚀 開始批量處理所有論文分類"
echo "================================"
echo "論文目錄: ${PAPERS_DIR}"
echo "輸出目錄: ${SCRIPT_DIR}"
echo "處理模式: 單線程 (max_workers=1)"
echo ""

# 分類定義
declare -A categories=(
    ["01_model_paradigm"]="MP"
    ["02_infrastructure"]="IF"
    ["03_language_models"]="LM"
    ["04_multimodal"]="MM"
)

# 處理每個分類
for input_dir in "${!categories[@]}"; do
    category_code="${categories[$input_dir]}"
    full_input_path="${PAPERS_DIR}/${input_dir}"

    echo "📁 處理分類: ${category_code} (${input_dir})"
    echo "   輸入目錄: ${full_input_path}"

    # 檢查目錄是否存在
    if [ ! -d "${full_input_path}" ]; then
        echo "   ⚠️  目錄不存在，跳過: ${full_input_path}"
        continue
    fi

    # 檢查是否有PDF文件
    pdf_count=$(find "${full_input_path}" -name "*.pdf" | wc -l)
    if [ ${pdf_count} -eq 0 ]; then
        echo "   ⚠️  目錄中沒有PDF文件，跳過"
        continue
    fi

    echo "   📊 找到 ${pdf_count} 個PDF文件"
    echo "   🔄 開始處理..."

    # 執行處理
    python3 scripts/batch_process_papers.py \
        --input_dir "${full_input_path}" \
        --output_dir "." \
        --category "${category_code}" \
        --max_workers 1

    echo "   ✅ 分類 ${category_code} 處理完成"
    echo ""
done

echo "🎉 所有分類處理完成！"
echo ""
echo "📋 後續步驟:"
echo "1. 檢查處理結果: python3 scripts/verify_processing_results.py"
echo "2. 生成全局索引: python3 scripts/generate_global_index.py"
echo "3. 查看日誌: tail outputs/logs/batch_processing.log"