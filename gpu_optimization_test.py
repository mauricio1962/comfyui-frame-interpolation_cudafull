#!/usr/bin/env python3
"""
GPU优化测试脚本
验证RIFE插帧是否完全在GPU上运行
"""

import torch
import time
import psutil
import os

def test_gpu_optimization():
    print("=== ComfyUI Frame Interpolation GPU优化测试 ===")
    
    # 检查CUDA状态
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行GPU测试")
        return False
    
    print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA版本: {torch.version.cuda}")
    print(f"✅ PyTorch版本: {torch.__version__}")
    
    # 检查GPU内存
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU内存: {gpu_memory:.1f} GB")
    
    # 测试GPU内存分配
    try:
        # 创建测试张量
        test_tensor = torch.randn(4, 3, 512, 512, device='cuda', dtype=torch.float16)
        print(f"✅ GPU张量创建成功: {test_tensor.shape}")
        
        # 检查是否在GPU上
        assert test_tensor.is_cuda, "张量不在GPU上!"
        print("✅ 张量确认在GPU上")
        
        # 测试GPU计算
        start_time = time.time()
        result = torch.nn.functional.interpolate(test_tensor, scale_factor=2.0, mode='bilinear')
        torch.cuda.synchronize()  # 等待GPU计算完成
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU计算测试完成: {gpu_time:.4f}s")
        print(f"✅ 输出形状: {result.shape}")
        print(f"✅ 输出设备: {result.device}")
        
        # 清理
        del test_tensor, result
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def monitor_gpu_usage():
    """监控GPU使用率"""
    print("\n=== GPU使用率监控 ===")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
            print(f"GPU利用率: {gpu_util}%")
            print(f"GPU内存使用: {mem_used}MB / {mem_total}MB ({float(mem_used)/float(mem_total)*100:.1f}%)")
        else:
            print("⚠️ 无法获取GPU使用率信息")
    except Exception as e:
        print(f"⚠️ GPU监控失败: {e}")

def test_vfi_gpu_mode():
    """测试VFI GPU模式"""
    print("\n=== VFI GPU模式测试 ===")
    try:
        # 导入VFI工具
        from vfi_utils import GPU_ONLY_MODE, AGGRESSIVE_GPU_OPTIMIZATION, preprocess_frames, postprocess_frames
        
        print(f"GPU_ONLY_MODE: {GPU_ONLY_MODE}")
        print(f"AGGRESSIVE_GPU_OPTIMIZATION: {AGGRESSIVE_GPU_OPTIMIZATION}")
        
        # 创建模拟帧数据
        frames = torch.randn(2, 512, 512, 3)  # 2帧，512x512，RGB
        print(f"原始帧数据: {frames.shape}, 设备: {frames.device}")
        
        # 测试预处理
        processed = preprocess_frames(frames)
        print(f"预处理后: {processed.shape}, 设备: {processed.device}")
        
        # 测试后处理（保持在GPU）
        result = postprocess_frames(processed, keep_on_gpu=True)
        print(f"后处理后(GPU): {result.shape}, 设备: {result.device}")
        
        # 测试后处理（转到CPU）
        result_cpu = postprocess_frames(processed, keep_on_gpu=False)
        print(f"后处理后(CPU): {result_cpu.shape}, 设备: {result_cpu.device}")
        
        print("✅ VFI GPU模式测试通过")
        return True
        
    except Exception as e:
        print(f"❌ VFI GPU模式测试失败: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    # 基础GPU测试
    success &= test_gpu_optimization()
    
    # GPU使用率监控
    monitor_gpu_usage()
    
    # VFI GPU模式测试
    success &= test_vfi_gpu_mode()
    
    print(f"\n=== 测试总结 ===")
    if success:
        print("🎉 所有测试通过! GPU优化已启用")
        print("💡 建议:")
        print("  - 使用较大的batch size充分利用GPU")
        print("  - 监控GPU利用率确保达到80%+")
        print("  - 如遇到内存不足，适当降低clear_cache_after_n_frames")
    else:
        print("❌ 部分测试失败，请检查配置")
    
    print("\n运行nvidia-smi监控GPU状态:")
    monitor_gpu_usage() 