#!/bin/bash
# 論文知識庫處理 - 主啟動腳本
# 按照 SOP 架構執行完整的處理流程

echo "🚀 論文知識庫處理 - 主啟動腳本"
echo "依照 SOP_Papers_Knowledge_Base.md 架構設計"
echo "=" * 60

# 設定路徑
BASE_DIR="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/multimodel-RAG/03.1-papers_knowledge_base"
SCRIPTS_DIR="$BASE_DIR/scripts"
PAPERS_DIR="/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/RAG_full_tech_overview/papers"
OLMOCR_DIR="../03-advanced-tools/olmocr"
VENV_PATH="../03-advanced-tools/.venv"

# 顯示目錄結構驗證
echo "📁 目錄結構驗證:"
echo "  基礎目錄: $BASE_DIR"
echo "  腳本目錄: $SCRIPTS_DIR"
echo "  論文源目錄: $PAPERS_DIR"
echo "  OLMoCR工具: $OLMOCR_DIR"
echo "  虛擬環境: $VENV_PATH"
echo ""

# 檢查目錄結構完整性
echo "🔍 檢查目錄結構完整性..."

# 檢查分類目錄
categories=("01_MP_Model_Paradigm" "02_IF_Infrastructure" "03_LM_Language_Models" "04_MM_Multimodal")
missing_dirs=()

for category in "${categories[@]}"; do
    if [ ! -d "$BASE_DIR/$category" ]; then
        missing_dirs+=("$category")
    else
        # 檢查子目錄
        subdirs=("processed" "markdown" "metadata" "logs")
        for subdir in "${subdirs[@]}"; do
            if [ ! -d "$BASE_DIR/$category/$subdir" ]; then
                missing_dirs+=("$category/$subdir")
            fi
        done
    fi
done

# 檢查全局目錄
global_dirs=("scripts" "index" "logs")
for dir in "${global_dirs[@]}"; do
    if [ ! -d "$BASE_DIR/$dir" ]; then
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_dirs[@]} -eq 0 ]; then
    echo "✅ 目錄結構完整"
else
    echo "❌ 缺少以下目錄:"
    for dir in "${missing_dirs[@]}"; do
        echo "  - $dir"
    done
    echo "請先執行: cd $BASE_DIR && python3 scripts/create_directory_structure.py"
    exit 1
fi

# 檢查腳本存在
echo ""
echo "🔧 檢查處理腳本..."
required_scripts=("batch_process_papers.py" "generate_global_index.py" "verify_processing_results.py")
missing_scripts=()

for script in "${required_scripts[@]}"; do
    if [ ! -f "$SCRIPTS_DIR/$script" ]; then
        missing_scripts+=("$script")
    fi
done

if [ ${#missing_scripts[@]} -eq 0 ]; then
    echo "✅ 所有處理腳本就緒"
else
    echo "❌ 缺少以下腳本:"
    for script in "${missing_scripts[@]}"; do
        echo "  - $script"
    done
    exit 1
fi

# 檢查源文件
echo ""
echo "📄 檢查源論文文件..."
source_categories=("01_model_paradigm" "02_infrastructure" "03_language_models" "04_multimodal")
total_pdfs=0

for src_category in "${source_categories[@]}"; do
    src_dir="$PAPERS_DIR/$src_category"
    if [ -d "$src_dir" ]; then
        pdf_count=$(find "$src_dir" -name "*.pdf" | wc -l)
        echo "  $src_category: $pdf_count 個 PDF 文件"
        total_pdfs=$((total_pdfs + pdf_count))
    else
        echo "  ❌ $src_category: 目錄不存在"
    fi
done

echo "  總計: $total_pdfs 個 PDF 文件"

if [ $total_pdfs -eq 0 ]; then
    echo "❌ 沒有找到源 PDF 文件"
    exit 1
fi

# 環境檢查
echo ""
echo "🌍 環境檢查..."

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 虛擬環境不存在: $VENV_PATH"
    exit 1
else
    echo "✅ 虛擬環境: $VENV_PATH"
fi

if [ ! -f "$OLMOCR_DIR/demo.py" ]; then
    echo "❌ OLMoCR 工具不存在: $OLMOCR_DIR/demo.py"
    exit 1
else
    echo "✅ OLMoCR 工具: $OLMOCR_DIR/demo.py"
fi

# GPU 檢查（可選）
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU 可用"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️  GPU 不可用或未安裝 nvidia-smi"
fi

echo ""
echo "🎯 環境檢查完成，準備開始處理流程！"
echo ""
echo "📋 接下來您可以選擇執行模式："
echo ""
echo "1. 🧪 測試模式（建議先執行）"
echo "   處理 02_IF_Infrastructure（5個文件）來測試流程"
echo ""
echo "2. 🏃 分類處理模式"
echo "   依序處理各個分類"
echo ""
echo "3. 🚀 完整自動化模式"
echo "   自動處理所有分類（需要較長時間）"
echo ""

read -p "請選擇模式 (1/2/3): " mode

case $mode in
    1)
        echo ""
        echo "🧪 執行測試模式..."
        echo "處理 02_IF_Infrastructure (5個文件)..."

        source "$VENV_PATH/bin/activate"

        python3 "$SCRIPTS_DIR/batch_process_papers.py" \
            --input_dir "$PAPERS_DIR/02_infrastructure" \
            --output_dir "$BASE_DIR/02_IF_Infrastructure" \
            --category "IF" \
            --max_workers 2

        if [ $? -eq 0 ]; then
            echo "✅ 測試處理完成"
            echo "📊 執行驗證..."
            python3 "$SCRIPTS_DIR/verify_processing_results.py"
        else
            echo "❌ 測試處理失敗"
        fi
        ;;

    2)
        echo ""
        echo "🏃 進入分類處理模式..."
        echo "請手動執行以下命令序列:"
        echo ""
        echo "# 激活環境"
        echo "source $VENV_PATH/bin/activate"
        echo ""
        echo "# 處理各分類（建議按順序執行）"
        echo ""
        echo "# 1. 基礎設施 (5 files)"
        echo "python3 $SCRIPTS_DIR/batch_process_papers.py \\"
        echo "  --input_dir '$PAPERS_DIR/02_infrastructure' \\"
        echo "  --output_dir '$BASE_DIR/02_IF_Infrastructure' \\"
        echo "  --category 'IF' --max_workers 2"
        echo ""
        echo "# 2. 多模態 (8 files)"
        echo "python3 $SCRIPTS_DIR/batch_process_papers.py \\"
        echo "  --input_dir '$PAPERS_DIR/04_multimodal' \\"
        echo "  --output_dir '$BASE_DIR/04_MM_Multimodal' \\"
        echo "  --category 'MM' --max_workers 2"
        echo ""
        echo "# 3. 語言模型 (8 files)"
        echo "python3 $SCRIPTS_DIR/batch_process_papers.py \\"
        echo "  --input_dir '$PAPERS_DIR/03_language_models' \\"
        echo "  --output_dir '$BASE_DIR/03_LM_Language_Models' \\"
        echo "  --category 'LM' --max_workers 2"
        echo ""
        echo "# 4. 模型範式 (10 files)"
        echo "python3 $SCRIPTS_DIR/batch_process_papers.py \\"
        echo "  --input_dir '$PAPERS_DIR/01_model_paradigm' \\"
        echo "  --output_dir '$BASE_DIR/01_MP_Model_Paradigm' \\"
        echo "  --category 'MP' --max_workers 2"
        echo ""
        echo "# 生成全局索引"
        echo "python3 $SCRIPTS_DIR/generate_global_index.py"
        echo ""
        echo "# 最終驗證"
        echo "python3 $SCRIPTS_DIR/verify_processing_results.py"
        ;;

    3)
        echo ""
        echo "🚀 執行完整自動化模式..."
        echo "⚠️  這將需要幾個小時的時間"

        read -p "確定要繼續嗎？ (y/n): " confirm

        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "開始完整處理..."

            source "$VENV_PATH/bin/activate"

            # 處理順序：從小到大
            categories_to_process=(
                "02_infrastructure:02_IF_Infrastructure:IF"
                "04_multimodal:04_MM_Multimodal:MM"
                "03_language_models:03_LM_Language_Models:LM"
                "01_model_paradigm:01_MP_Model_Paradigm:MP"
            )

            for category_info in "${categories_to_process[@]}"; do
                IFS=':' read -ra ADDR <<< "$category_info"
                src_dir="${ADDR[0]}"
                output_dir="${ADDR[1]}"
                code="${ADDR[2]}"

                echo ""
                echo "🔄 處理分類: $code ($src_dir)"

                python3 "$SCRIPTS_DIR/batch_process_papers.py" \
                    --input_dir "$PAPERS_DIR/$src_dir" \
                    --output_dir "$BASE_DIR/$output_dir" \
                    --category "$code" \
                    --max_workers 2

                if [ $? -ne 0 ]; then
                    echo "❌ 處理失敗: $code"
                    break
                fi

                echo "✅ 完成: $code"
                sleep 5  # 短暫休息
            done

            echo ""
            echo "📊 生成全局索引..."
            python3 "$SCRIPTS_DIR/generate_global_index.py"

            echo ""
            echo "🔍 執行最終驗證..."
            python3 "$SCRIPTS_DIR/verify_processing_results.py"

            echo ""
            echo "🎉 完整處理流程結束！"

        else
            echo "取消自動化處理"
        fi
        ;;

    *)
        echo "無效的選擇"
        exit 1
        ;;
esac

echo ""
echo "📁 處理結果目錄結構:"
find . -type d | head -20 | sort

echo ""
echo "📋 更多詳細信息請查閱 SOP_Papers_Knowledge_Base.md"