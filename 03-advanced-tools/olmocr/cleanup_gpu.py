#!/usr/bin/env python3
"""
GPU 記憶體清理腳本 - 清理卡住的 vLLM 和 Python 進程
"""

import subprocess
import os
import signal
import time

def cleanup_gpu_processes():
    """清理 GPU 上的 Python 和 vLLM 進程"""
    print("🧹 開始清理 GPU 進程...")

    try:
        # 獲取 nvidia-smi 輸出
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ 無法執行 nvidia-smi")
            return False

        lines = result.stdout.strip().split('\n')
        killed_count = 0

        for line in lines:
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                pid = parts[0]
                process_name = parts[1]

                # 只殺掉 Python 相關進程，避免殺掉系統進程
                if any(name in process_name.lower() for name in ['python', 'vllm']):
                    try:
                        pid_int = int(pid)
                        # 檢查是否是我們自己的進程
                        if pid_int != os.getpid():
                            print(f"🔫 終止進程: PID {pid} ({process_name})")
                            os.kill(pid_int, signal.SIGTERM)
                            time.sleep(1)
                            # 如果還沒死，用 SIGKILL
                            try:
                                os.kill(pid_int, signal.SIGKILL)
                            except ProcessLookupError:
                                pass  # 進程已死亡
                            killed_count += 1
                    except (ValueError, ProcessLookupError, PermissionError) as e:
                        print(f"⚠️  無法終止 PID {pid}: {e}")

        print(f"✅ 清理完成，終止了 {killed_count} 個進程")

        # 等待 GPU 釋放記憶體
        print("⏳ 等待 GPU 記憶體釋放...")
        time.sleep(3)

        # 顯示清理後的 GPU 狀態
        print("\n📊 清理後 GPU 狀態:")
        subprocess.run(['nvidia-smi'], check=False)

        return True

    except Exception as e:
        print(f"❌ 清理過程出錯: {e}")
        return False

if __name__ == "__main__":
    cleanup_gpu_processes()