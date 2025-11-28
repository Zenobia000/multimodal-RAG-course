#!/usr/bin/env python3
"""
olmOCR v0.4.6 PDF 處理工具 - 單一檔案輸入版本
適配新版本的 vLLM 後端，支持命令列指定 PDF 檔案
"""

import subprocess
import sys
import os
from pathlib import Path

# 配置：指定使用的 GPU 設備
GPU_DEVICE = 1

def convert_pdf_v046(pdf_path):
    """使用 olmOCR v0.4.6 轉換PDF，新版本使用 vLLM 替代 SGLang"""
    try:
        # 創建輸出目錄（workspace）
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # olmOCR 使用 workspace 目錄
        workspace_dir = output_dir / "workspace"
        workspace_dir.mkdir(exist_ok=True)

        print(f"📁 工作目錄: {workspace_dir}")
        print(f"📄 處理文件: {pdf_path}")
        print(f"🔄 olmOCR v0.4.6 - 使用 vLLM 後端")

        # olmOCR v0.4.6 pipeline 命令
        cmd = [
            sys.executable, "-m", "olmocr.pipeline",
            str(workspace_dir),
            "--pdfs", str(pdf_path),
            "--markdown",  # 生成 markdown 輸出
            "--max_page_error_rate", "0.3",  # 允許30%頁面錯誤率
            "--gpu_memory_utilization", "0.7",  # vLLM GPU 記憶體使用率
            "--max_model_len", "8192",  # 增加 context 長度以支持 8000 token 輸出
            "--tensor_parallel_size", "1",  # 單 GPU
            "--data_parallel_size", "1",  # 無 data parallelism
        ]

        print(f"🚀 執行命令: olmocr.pipeline {' '.join(cmd[3:])}")

        # 設置環境變量
        env = os.environ.copy()
        if GPU_DEVICE is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(GPU_DEVICE)
            print(f"🎯 使用 GPU: {GPU_DEVICE}")

        # 執行命令，設定45分鐘超時（增加超時時間）
        print("⏳ 開始處理...")
        result = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              timeout=2700,  # 45分鐘
                              env=env)

        print(f"📊 返回碼: {result.returncode}")

        if result.stdout:
            print(f"📤 標準輸出 (最後500字符):\n{result.stdout[-500:]}")

        if result.stderr:
            print(f"🔍 錯誤輸出 (最後500字符):\n{result.stderr[-500:]}")

        # 檢查結果
        if result.returncode == 0:
            # 查找生成的文件
            all_files = list(workspace_dir.rglob('*'))
            output_files = [f for f in all_files if f.is_file()]

            # 分類文件類型
            md_files = [f for f in output_files if f.suffix == '.md']
            json_files = [f for f in output_files if f.suffix in ['.json', '.jsonl']]

            print(f"✅ 處理完成！")
            print(f"📁 工作目錄: {workspace_dir}")
            print(f"📄 生成文件數量: {len(output_files)}")

            if md_files:
                print(f"📝 Markdown 文件 ({len(md_files)} 個):")
                for f in md_files:
                    size = f.stat().st_size
                    print(f"   - {f.relative_to(workspace_dir)} ({size} bytes)")

            if json_files:
                print(f"🗂️  JSON 文件 ({len(json_files)} 個):")
                for f in json_files:
                    size = f.stat().st_size
                    print(f"   - {f.relative_to(workspace_dir)} ({size} bytes)")

            if output_files:
                print("📝 其他文件:")
                other_files = [f for f in output_files if f.suffix not in ['.md', '.json', '.jsonl']]
                for f in other_files[:5]:  # 顯示前5個其他文件
                    size = f.stat().st_size
                    print(f"   - {f.relative_to(workspace_dir)} ({size} bytes)")
                if len(other_files) > 5:
                    print(f"   ... 還有 {len(other_files) - 5} 個文件")

            return {
                'success': True,
                'workspace': str(workspace_dir),
                'file_count': len(output_files),
                'markdown_files': len(md_files),
                'json_files': len(json_files),
                'files': [str(f.relative_to(workspace_dir)) for f in output_files[:10]]
            }
        else:
            return {
                'success': False,
                'error': f'olmOCR v0.4.6 處理失敗，返回碼: {result.returncode}',
                'stderr': result.stderr[-500:] if result.stderr else None,
                'stdout': result.stdout[-500:] if result.stdout else None
            }

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': '處理超時（超過45分鐘）'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    """主函數：支持命令列參數或使用預設檔案"""

    # 檢查命令列參數
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"❌ 錯誤：檔案不存在 - {pdf_path}")
            return
    else:
        # 使用預設檔案
        pdf_path = "/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/OCR-tool-comparsion/03-advanced-tools/test_pdfs/2015_ResNet.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ 錯誤：預設檔案不存在 - {pdf_path}")
            print("💡 使用方法: python demo.py <PDF檔案路徑>")
            return

    print("🧪 olmOCR v0.4.6 PDF 處理工具")
    print("=" * 60)
    print(f"📄 目標檔案: {pdf_path}")
    print(f"📏 檔案大小: {Path(pdf_path).stat().st_size / (1024*1024):.1f} MB")

    result = convert_pdf_v046(pdf_path)

    print("=" * 60)
    if result['success']:
        print("🎉 處理成功！")
        print(f"📁 結果位置: {result['workspace']}")
        if result.get('markdown_files', 0) > 0:
            print(f"📝 生成了 {result['markdown_files']} 個 Markdown 檔案")
        if result.get('json_files', 0) > 0:
            print(f"🗂️  生成了 {result['json_files']} 個 JSON 檔案")
    else:
        print(f"❌ 處理失敗: {result['error']}")

    print(f"🏁 最終結果: {result}")

if __name__ == "__main__":
    main()