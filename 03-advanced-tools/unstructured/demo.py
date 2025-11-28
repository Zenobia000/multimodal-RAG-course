#!/usr/bin/env python3
"""
Unstructured PDF數據處理實現
"""

import time
import json
from pathlib import Path

def process_pdfs(output_dir=None):
    """處理test_pdfs目錄下的PDF文件"""
    # 設置輸出目錄
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    else:
        output_dir = Path(output_dir)
    
    # 確保輸出目錄存在
    output_dir.mkdir(exist_ok=True)

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
        result = convert_pdf(pdf, output_dir)
        process_time = time.time() - start_time

        results.append({
            'file': pdf.name,
            'size_mb': size_mb,
            'process_time': process_time,
            'success': result['success'],
            'output_size': result.get('output_size', 0),
            'output_file': result.get('output_file', None),
            'error': result.get('error', None)
        })

    return results

def elements_to_markdown(elements):
    """將 unstructured 元素轉換為 Markdown 格式"""
    markdown_lines = []
    last_was_title = False
    title_count = 0
    
    for element in elements:
        elem_type = type(element).__name__
        category = getattr(element, 'category', None)
        text = str(element).strip()
        
        if not text:
            continue
        
        # 根據元素類型和分類決定 Markdown 格式
        if elem_type == 'Title' or category == 'Title':
            # 標題處理
            # 第一個標題作為一級標題，後續標題作為二級標題
            title_count += 1
            if title_count == 1 and len(text) < 150:
                # 第一個較短的標題作為主標題
                markdown_lines.append(f"# {text}\n\n")
            else:
                # 其他標題作為二級標題
                # 如果上一行也是標題，不需要額外空行
                if last_was_title:
                    markdown_lines.append(f"## {text}\n\n")
                else:
                    if markdown_lines and not markdown_lines[-1].endswith('\n\n'):
                        markdown_lines.append("\n")
                    markdown_lines.append(f"## {text}\n\n")
            last_was_title = True
        elif elem_type == 'ListItem' or category == 'ListItem':
            # 列表項
            if not last_was_title and markdown_lines and not markdown_lines[-1].strip().startswith('-'):
                markdown_lines.append("\n")
            markdown_lines.append(f"- {text}\n")
            last_was_title = False
        elif elem_type == 'Table' or category == 'Table':
            # 表格（如果 unstructured 提供表格元素）
            if markdown_lines and not markdown_lines[-1].endswith('\n\n'):
                markdown_lines.append("\n\n")
            markdown_lines.append(f"```\n{text}\n```\n\n")
            last_was_title = False
        else:
            # 普通文本段落
            # 如果上一行是標題，已經有空行了，不需要再添加
            if last_was_title:
                pass  # 標題後已經有空行
            elif markdown_lines and markdown_lines[-1].strip() and not markdown_lines[-1].startswith(('-', '#' , '`')):
                # 連續的普通文本段落之間添加空行
                markdown_lines.append("\n")
            
            markdown_lines.append(f"{text}\n")
            last_was_title = False
    
    return "".join(markdown_lines)

def convert_pdf(pdf_path, output_dir):
    """使用Unstructured解析PDF並轉換為Markdown"""
    try:
        from unstructured.partition.auto import partition

        elements = partition(filename=str(pdf_path))
        
        # 轉換為 Markdown 格式
        markdown_content = elements_to_markdown(elements)
        # 保留原始文本用於預覽
        text_preview = "\n".join([str(element) for element in elements[:5]])

        # 保存提取的文本到 output 目錄
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 生成輸出文件名（PDF文件名 + .md）
        output_file = output_dir / f"{pdf_path.stem}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return {
            'success': True,
            'output_size': len(markdown_content.encode('utf-8')),
            'element_count': len(elements),
            'text_preview': text_preview[:200],
            'output_file': str(output_file.relative_to(output_dir))
        }

    except ImportError:
        return {'success': False, 'error': 'unstructured套件未安裝，請查看 README.md'}
    except (AttributeError, Exception) as e:
        error_msg = str(e)
        # 檢查是否是 NumPy 兼容性問題
        if '_ARRAY_API' in error_msg or ('NumPy' in error_msg and '2.' in error_msg):
            return {'success': False, 'error': 'NumPy 版本不兼容，請查看 README.md 安裝要求'}
        return {'success': False, 'error': error_msg}

def analyze_results(results, output_dir=None):
    """分析處理結果"""
    if not results:
        return

    # 如果未指定輸出目錄，使用默認的 output 目錄
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    else:
        output_dir = Path(output_dir)
    
    # 確保輸出目錄存在
    output_dir.mkdir(exist_ok=True)

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

    # 顯示成功處理的文件
    success_files = [r for r in results if r['success']]
    if success_files:
        print(f"\n✅ 成功處理 {len(success_files)} 個文件:")
        for r in success_files:
            if r.get('output_file'):
                print(f"   - {r['file']} → {r['output_file']}")

    # 保存結果到 output 目錄
    output_file = output_dir / 'unstructured_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果已保存到: {output_file}")

def main():
    """執行PDF批量處理"""
    print("🚀 Unstructured PDF處理")
    
    # 設置輸出目錄
    output_dir = Path(__file__).parent / "output"
    results = process_pdfs(output_dir=output_dir)
    analyze_results(results, output_dir=output_dir)

if __name__ == "__main__":
    main()

