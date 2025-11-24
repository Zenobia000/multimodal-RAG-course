#!/usr/bin/env python3
"""
論文批量處理腳本
使用 OLMoCR 批量處理指定分類的 PDF 文件
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading

# 設置日誌 - 動態設置日誌目錄
def setup_logging(output_base):
    log_dir = Path(output_base) / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "batch_processing.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
logger = logging.getLogger(__name__)

class PaperProcessor:
    def __init__(self, olmocr_path, output_base, category_code):
        self.olmocr_path = Path(olmocr_path)
        self.output_base = Path(output_base)
        self.category_code = category_code
        self.demo_script = self.olmocr_path / "demo.py"
        self.venv_path = Path("../03-advanced-tools/.venv")
        self.lock = threading.Lock()

        # 確保路徑存在
        if not self.demo_script.exists():
            raise FileNotFoundError(f"OLMoCR demo.py not found: {self.demo_script}")

        # 自動創建必要的目錄結構
        self._ensure_directory_structure()

    def _ensure_directory_structure(self):
        """自動檢測並創建必要的目錄結構"""
        logger.info("檢查並創建必要的目錄結構...")

        # 統一輸出架構的目錄
        required_dirs = [
            self.output_base / "outputs",
            self.output_base / "outputs" / "processed_papers",
            self.output_base / "outputs" / "processed_papers" / self.category_code,
            self.output_base / "outputs" / "aggregated_chunks",
            self.output_base / "outputs" / "logs",
            self.output_base / "outputs" / "temp",
            self.output_base / "outputs" / "indices",
            self.output_base / "outputs" / "embeddings"
        ]

        created_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))

        if created_dirs:
            logger.info(f"創建了 {len(created_dirs)} 個目錄:")
            for created_dir in created_dirs:
                logger.info(f"  ✓ {created_dir}")
        else:
            logger.info("所有必要目錄已存在")

    def extract_paper_info(self, pdf_path):
        """從 PDF 檔名提取論文資訊"""
        filename = Path(pdf_path).stem

        # 嘗試解析年份
        year = "UNKNOWN"
        name = filename

        # 常見格式: 2021_CLIP.pdf, 2017_Transformer.pdf
        if "_" in filename:
            parts = filename.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 4:
                year = parts[0]
                name = parts[1]

        return {
            "year": year,
            "name": name,
            "original_filename": filename
        }

    def generate_output_name(self, pdf_path):
        """生成標準化的輸出檔名"""
        paper_info = self.extract_paper_info(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d")

        return f"{self.category_code}_{paper_info['year']}_{paper_info['name']}_{timestamp}"

    def process_single_paper(self, pdf_path):
        """處理單個 PDF 文件"""
        pdf_path = Path(pdf_path)
        output_name = self.generate_output_name(pdf_path)
        
        # 為每個任務創建唯一的臨時工作區
        temp_workspace = self.output_base / "outputs" / "temp" / output_name
        temp_workspace.mkdir(parents=True, exist_ok=True)

        logger.info(f"開始處理: {pdf_path.name} -> {output_name}")

        try:
            # 使用鎖來保護對共享 OLMoCR 工具的調用
            with self.lock:
                logger.info(f"[{output_name}] 獲取鎖，開始執行 OLMoCR")
                # 激活虛擬環境並執行 OLMoCR
                activate_cmd = f"source {self.venv_path}/bin/activate"
                olmocr_cmd = f"cd {self.olmocr_path} && python3 demo.py '{pdf_path}'"
                full_cmd = f"{activate_cmd} && {olmocr_cmd}"

                # 執行處理
                result = subprocess.run(
                    ['/bin/bash', '-c', full_cmd],
                    capture_output=True,
                    text=True,
                    timeout=3600
                )

                if result.returncode != 0:
                    logger.error(f"[{output_name}] 處理失敗: {result.stderr}")
                    return {
                        "status": "failed", "pdf_path": str(pdf_path),
                        "output_name": output_name, "error": result.stderr
                    }

                # 立刻將結果從共享目錄移動到臨時獨立目錄
                olmocr_output = self.olmocr_path / "output" / "workspace"
                if olmocr_output.exists():
                    shutil.move(str(olmocr_output), str(temp_workspace / "workspace"))
                    logger.info(f"[{output_name}] 結果已移動到臨時工作區: {temp_workspace}")
                else:
                    logger.warning(f"[{output_name}] 未找到 OLMoCR 輸出目錄: {olmocr_output}")

                logger.info(f"[{output_name}] 釋放鎖")

            # 從臨時獨立目錄整理輸出文件
            self.organize_output(pdf_path, output_name, temp_workspace)

            logger.info(f"處理完成: {pdf_path.name}")
            return {
                "status": "success", "pdf_path": str(pdf_path),
                "output_name": output_name, "timestamp": datetime.now().isoformat()
            }

        except subprocess.TimeoutExpired:
            logger.error(f"處理超時: {pdf_path.name}")
            return {"status": "timeout", "pdf_path": str(pdf_path), "output_name": output_name}
        except Exception as e:
            logger.error(f"處理出錯 {pdf_path.name}: {e}")

            # 清理失敗的輸出文件
            processed_paper_dir = self.output_base / "outputs" / "processed_papers" / self.category_code / output_name
            aggregated_chunk_file = self.output_base / "outputs" / "aggregated_chunks" / f"{output_name}.jsonl"

            if processed_paper_dir.exists():
                shutil.rmtree(processed_paper_dir, ignore_errors=True)
                logger.info(f"已清理失敗的處理目錄: {processed_paper_dir}")

            if aggregated_chunk_file.exists():
                aggregated_chunk_file.unlink(missing_ok=True)
                logger.info(f"已清理失敗的聚合文件: {aggregated_chunk_file}")

            return {"status": "error", "pdf_path": str(pdf_path), "output_name": output_name, "error": str(e)}
        finally:
            # 清理臨時工作區
            if temp_workspace.exists():
                shutil.rmtree(temp_workspace, ignore_errors=True)
                logger.info(f"已清理臨時工作區: {temp_workspace}")

    def organize_output(self, pdf_path, output_name, temp_workspace):
        """整理來自臨時工作區的輸出文件到新的統一結構中"""
        source_workspace = temp_workspace / "workspace"
        outputs_dir = self.output_base / "outputs"

        # 新的統一輸出路徑
        # 1. 按論文分類的完整處理結果
        processed_paper_dir = outputs_dir / "processed_papers" / self.category_code / output_name
        # 2. 聚合所有論文的 JSONL 區塊
        aggregated_chunks_dir = outputs_dir / "aggregated_chunks"

        # 確保所有目標目錄存在
        processed_paper_dir.mkdir(parents=True, exist_ok=True)
        aggregated_chunks_dir.mkdir(parents=True, exist_ok=True)

        # 處理來自臨時工作區的輸出結果
        if source_workspace.exists():
            # 1. 複製完整 workspace 到 processed_paper_dir
            shutil.copytree(source_workspace, processed_paper_dir / "workspace", dirs_exist_ok=True)

            # 2. 處理 JSONL 結果文件
            results_dir = source_workspace / "results"
            if results_dir.exists():
                jsonl_files = list(results_dir.glob("*.jsonl"))
                if jsonl_files:
                    for result_file in jsonl_files:
                        # 複製到 processed_paper_dir
                        shutil.copy2(result_file, processed_paper_dir / f"{output_name}.jsonl")
                        # 複製到 aggregated_chunks_dir
                        shutil.copy2(result_file, aggregated_chunks_dir / f"{output_name}.jsonl")
                    logger.info(f"[{output_name}] 成功處理 {len(jsonl_files)} 個 JSONL 文件")
                else:
                    raise Exception(f"Results 目錄存在但沒有 JSONL 文件: {results_dir}")
            else:
                raise Exception(f"未找到 results 目錄，OLMoCR 處理失敗: {results_dir}")
        else:
            raise Exception(f"臨時工作區源目錄不存在，OLMoCR 處理失敗: {source_workspace}")

        # 創建元數據文件（只保存在 processed_paper_dir 中）
        metadata_path = processed_paper_dir / f"{output_name}_metadata.json"
        aggregated_chunk_path = aggregated_chunks_dir / f"{output_name}.jsonl"

        metadata = {
            "output_name": output_name,
            "original_pdf": str(pdf_path),
            "category_code": self.category_code,
            "processing_time": datetime.now().isoformat(),
            "paper_info": self.extract_paper_info(pdf_path),
            "output_locations": {
                "processed_paper_dir": str(processed_paper_dir),
                "aggregated_chunk_path": str(aggregated_chunk_path)
            }
        }

        # 寫入元數據文件
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="批量處理論文 PDF 文件")
    parser.add_argument("--input_dir", required=True, help="輸入 PDF 目錄")
    parser.add_argument("--output_dir", required=True, help="輸出目錄")
    parser.add_argument("--category", required=True, help="分類代碼 (MP/IF/LM/MM)")
    parser.add_argument("--max_workers", type=int, default=2, help="最大並行處理數")
    parser.add_argument("--olmocr_path",
                       default="../03-advanced-tools/olmocr",
                       help="OLMoCR 工具路徑")

    args = parser.parse_args()

    # 驗證參數
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"輸入目錄不存在: {input_dir}")
        return

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 設置日誌
    setup_logging(output_dir)

    # 找到所有 PDF 文件
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"在 {input_dir} 中沒有找到 PDF 文件")
        return

    logger.info(f"找到 {len(pdf_files)} 個 PDF 文件")
    logger.info(f"輸出目錄: {output_dir}")
    logger.info(f"分類代碼: {args.category}")
    logger.info(f"最大並行數: {args.max_workers}")

    # 創建處理器
    processor = PaperProcessor(args.olmocr_path, output_dir, args.category)

    # 開始批量處理
    results = []
    start_time = datetime.now()

    if args.max_workers == 1:
        # 單線程處理
        for pdf_file in pdf_files:
            result = processor.process_single_paper(pdf_file)
            results.append(result)
    else:
        # 多線程處理
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任務
            future_to_pdf = {
                executor.submit(processor.process_single_paper, pdf_file): pdf_file
                for pdf_file in pdf_files
            }

            # 處理完成的任務
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"處理任務失敗 {pdf_file}: {e}")
                    results.append({
                        "status": "error",
                        "pdf_path": str(pdf_file),
                        "error": str(e)
                    })

    # 生成處理報告
    end_time = datetime.now()
    processing_time = end_time - start_time

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    report = {
        "summary": {
            "total_files": len(pdf_files),
            "successful": len(successful),
            "failed": len(failed),
            "processing_time": str(processing_time),
            "category": args.category,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir)
        },
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

    # 保存報告
    report_file = output_dir / "logs" / f"processing_report_{args.category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 顯示摘要
    print("\n" + "=" * 60)
    print("🎉 批量處理完成!")
    print(f"📊 總文件數: {len(pdf_files)}")
    print(f"✅ 成功處理: {len(successful)}")
    print(f"❌ 失敗: {len(failed)}")
    print(f"⏱️  處理時間: {processing_time}")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"📋 處理報告: {report_file}")

    if failed:
        print("\n❌ 失敗的文件:")
        for result in failed:
            print(f"  - {Path(result['pdf_path']).name}: {result.get('error', result['status'])}")

if __name__ == "__main__":
    main()