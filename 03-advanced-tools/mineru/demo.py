#!/usr/bin/env python3
"""
MinerU PDF數據處理實現
"""

import subprocess
import tempfile
import time
import json
from pathlib import Path

def process_pdfs():
    """處理test_pdfs目錄下的PDF文件"""
    # 指向父目錄的 test_pdfs
    test_dir = Path(__file__).parent.parent / "test_pdfs"
    if not test_dir.exists():
        print("❌ test_pdfs目錄不存在")
        print(f"   預期路徑: {test_dir.absolute()}")
        return []

    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ 無PDF文件")
        return []

    print(f"📁 發現 {len(pdf_files)} 個PDF")
    results = []

    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024*1024)
        print(f"處理: {pdf.name} ({size_mb:.1f}MB)")

        start_time = time.time()
        result = convert_pdf(pdf)
        process_time = time.time() - start_time

        result_data = {
            'file': pdf.name,
            'size_mb': size_mb,
            'process_time': process_time,
            'success': result['success'],
            'output_size': result.get('output_size', 0),
            'error': result.get('error', None)
        }
        
        # 添加成功時的額外信息
        if result['success']:
            result_data['md_count'] = result.get('md_count', 0)
            result_data['json_count'] = result.get('json_count', 0)
            result_data['output_dir'] = result.get('output_dir', '')
            result_data['warning'] = result.get('warning', None)
            
            if result_data['md_count'] > 0:
                print(f"  ✅ 成功！生成 {result_data['md_count']} 個 Markdown 文件")
            elif result_data['json_count'] > 0:
                print(f"  ✅ 成功！生成 {result_data['json_count']} 個 JSON 文件")
            else:
                warning = result_data.get('warning', '')
                if warning:
                    print(f"  ⚠️  {warning}")
                else:
                    print(f"  ⚠️  處理完成，但未找到輸出文件")
        else:
            error_msg = result.get('error', '未知錯誤')
            # 截斷過長的錯誤信息
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"  ❌ 失敗: {error_msg}")
        
        results.append(result_data)

    return results

def convert_pdf(pdf_path):
    """使用mineru轉換PDF"""
    try:
        # 創建輸出目錄（在 mineru 目錄下）
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        # 為每個PDF創建單獨的輸出子目錄
        pdf_output_dir = output_dir / pdf_path.stem
        pdf_output_dir.mkdir(exist_ok=True)

        # 正確的命令格式：mineru -p <input_path> -o <output_path>
        # 使用 -m auto 自動選擇最佳方法（ocr 或 txt）
        cmd = [
            "mineru",
            "-p", str(pdf_path),  # 輸入文件路徑
            "-o", str(pdf_output_dir),  # 輸出目錄
            "-m", "auto"  # 自動選擇最佳方法
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # 檢查 stderr 中是否有錯誤（即使返回碼為 0，也可能有錯誤）
        error_output = result.stderr.strip() or result.stdout.strip()
        has_error = error_output and ('error' in error_output.lower() or 'not found' in error_output.lower() or 'traceback' in error_output.lower())
        
        # 檢查返回碼和輸出
        if result.returncode == 0 and not has_error:
            # 查找生成的 markdown 文件
            md_files = list(pdf_output_dir.rglob('*.md'))
            if md_files:
                output_size = sum(f.stat().st_size for f in md_files)
                return {'success': True, 'output_size': output_size, 'md_count': len(md_files), 'output_dir': str(pdf_output_dir)}
            else:
                # 也檢查其他可能的輸出格式
                json_files = list(pdf_output_dir.rglob('*.json'))
                if json_files:
                    output_size = sum(f.stat().st_size for f in json_files)
                    return {'success': True, 'output_size': output_size, 'json_count': len(json_files), 'output_dir': str(pdf_output_dir)}
                
                # 檢查是否有任何文件生成
                all_files = list(pdf_output_dir.rglob('*'))
                if all_files:
                    # 有文件但格式不對，返回信息
                    file_types = set(f.suffix for f in all_files if f.is_file())
                    return {
                        'success': True, 
                        'output_size': 0, 
                        'md_count': 0, 
                        'output_dir': str(pdf_output_dir),
                        'warning': f'生成了文件但未找到 .md 或 .json 格式，發現的文件類型: {file_types}'
                    }
                
                # 沒有生成任何文件，檢查錯誤輸出
                if error_output and ('error' in error_output.lower() or 'not found' in error_output.lower() or 'traceback' in error_output.lower()):
                    # 提取關鍵錯誤信息
                    error_lines = [line for line in error_output.split('\n') if 'error' in line.lower() or 'not found' in line.lower()]
                    if error_lines:
                        error_msg = error_lines[0][:300]  # 取第一行錯誤信息
                    else:
                        error_msg = error_output[:300]
                    return {'success': False, 'error': error_msg}
                
                return {
                    'success': True, 
                    'output_size': 0, 
                    'md_count': 0, 
                    'output_dir': str(pdf_output_dir),
                    'warning': '命令執行成功但未找到輸出文件'
                }
        else:
            # 命令失敗，組合錯誤信息
            if not error_output:
                error_msg = f"命令執行失敗，返回碼: {result.returncode}"
            else:
                # 提取關鍵錯誤信息
                error_lines = [line for line in error_output.split('\n') if 'error' in line.lower() or 'not found' in line.lower()]
                if error_lines:
                    error_msg = error_lines[0][:300]  # 取第一行錯誤信息
                else:
                    error_msg = error_output[:300]
            return {'success': False, 'error': error_msg}

    except FileNotFoundError:
        return {'success': False, 'error': 'mineru命令未找到，請確認已安裝 mineru'}
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': '處理超時（超過600秒）'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_results(results):
    """分析處理結果"""
    if not results:
        return

    total_files = len(results)
    success_count = sum(1 for r in results if r['success'])
    total_size = sum(r['size_mb'] for r in results)
    total_time = sum(r['process_time'] for r in results)

    print(f"\n📊 處理結果:")
    print(f"成功率: {success_count}/{total_files} ({success_count/total_files*100:.1f}%)")
    print(f"總大小: {total_size:.1f}MB")
    print(f"總時間: {total_time:.2f}秒")
    if total_time > 0:
        print(f"平均速度: {total_size/total_time:.2f}MB/秒")

    # 顯示錯誤
    for r in results:
        if not r['success']:
            print(f"❌ {r['file']}: {r['error']}")

    # 保存結果
    output_file = Path(__file__).parent / 'mineru_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"結果已保存到: {output_file}")

def check_mineru_command():
    """檢查 mineru 命令是否可用"""
    try:
        result = subprocess.run(
            ["mineru", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            print(f"✅ mineru 命令可用: {version}")
            return True
        else:
            print("⚠️  mineru 命令執行失敗")
            return False
    except FileNotFoundError:
        print("❌ mineru 命令未找到")
        print("   請確認已安裝 mineru: uv pip install -U 'mineru[core]'")
        return False
    except Exception as e:
        print(f"⚠️  檢查 mineru 命令時出錯: {e}")
        return False

def main():
    """執行PDF批量處理"""
    print("🚀 MinerU PDF處理")
    
    # 檢查 mineru 命令
    print("\n📋 檢查 mineru 命令...")
    mineru_ok = check_mineru_command()
    if not mineru_ok:
        print("\n❌ mineru 命令不可用，無法繼續處理")
        print("   請先安裝 mineru: uv pip install -U 'mineru[core]'")
        return
    
    results = process_pdfs()
    analyze_results(results)

if __name__ == "__main__":
    main()

