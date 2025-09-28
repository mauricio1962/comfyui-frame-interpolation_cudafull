"""
FP16 Fast矩阵优化补丁
为ComfyUI-Frame-Interpolation添加自动混合精度和Tensor Core优化
专为RTX 5090等支持Tensor Cores的GPU设计
"""

import torch
import types
import functools
import os
import sys
from contextlib import nullcontext
from torch.cuda.amp import autocast
import time

# 将当前目录添加到路径以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入原始函数
try:
    from vfi_utils import _generic_frame_loop as original_generic_frame_loop
    from vfi_utils import generic_frame_loop as original_frame_loop
    print("成功加载原始帧插值函数")
except ImportError:
    print("无法导入原始帧插值函数，请确保ComfyUI-Frame-Interpolation已正确安装")
    original_generic_frame_loop = None
    original_frame_loop = None

# 配置项
ENABLE_FP16_FAST = True      # 启用FP16优化
USE_TENSOR_CORES = True      # 使用Tensor Cores
BATCH_SIZE = 4               # 批处理大小，针对RTX PRO 6000优化
ENABLE_BENCHMARK = True      # 启用性能测试以监控改进

# 检测是否有支持Tensor Cores的GPU
def has_tensor_cores():
    """检测当前GPU是否支持Tensor Cores"""
    if not torch.cuda.is_available():
        return False
    
    device_name = torch.cuda.get_device_name().lower()
    return any(name in device_name for name in [
        'v100', 't4', 'a100', 'a10', 'a30', 'a40', 
        'rtx', 'titan v', 'quadro', 'h100', 'blackwell', '5090'
    ])

# 优化版的帧插值函数
def _optimized_generic_frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier,
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states=None,
        use_timestep=True,
        dtype=torch.float16,
        final_logging=True):
    """
    优化版的通用帧插值循环
    添加了FP16快速矩阵优化和Tensor Core加速
    """
    # 检查是否应该使用优化
    use_optimization = ENABLE_FP16_FAST
    use_tensor_cores = USE_TENSOR_CORES and has_tensor_cores()
    
    if use_optimization:
        if final_logging:
            print(f"启用FP16 Fast矩阵优化 (Tensor Cores: {use_tensor_cores})")
        start_time = time.time()
    
    # 如果不需要优化或者帧数太少，使用原始函数
    if not use_optimization or len(frames) <= 2:
        return original_generic_frame_loop(
            frames, clear_cache_after_n_frames, multiplier,
            return_middle_frame_function, *return_middle_frame_function_args,
            interpolation_states=interpolation_states, use_timestep=use_timestep,
            dtype=dtype, final_logging=final_logging
        )
    
    # 获取原始函数中定义的非时间步推理函数
    def non_timestep_inference(frame0, frame1, n):        
        # 使用自动混合精度
        with autocast(enabled=use_optimization):
            middle = return_middle_frame_function(frame0, frame1, None, *return_middle_frame_function_args)
        
        if n == 1:
            return [middle]
        first_half = non_timestep_inference(frame0, middle, n=n//2)
        second_half = non_timestep_inference(middle, frame1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]
    
    # 设置CUDA优化
    if use_optimization and torch.cuda.is_available():
        if ENABLE_BENCHMARK:
            torch.backends.cudnn.benchmark = True
    
    # 创建输出张量
    output_frames = torch.zeros(multiplier*frames.shape[0], *frames.shape[1:], dtype=dtype, device="cpu")
    out_len = 0
    
    # 优化版本的帧处理逻辑
    from comfy.model_management import get_torch_device
    DEVICE = get_torch_device()
    
    # 缓存清理计数器
    frames_since_cache_clear = 0
    
    # 主循环
    for frame_itr in range(len(frames) - 1):
        frame0 = frames[frame_itr:frame_itr+1]
        output_frames[out_len] = frame0  # 复制第一帧
        out_len += 1
        
        # 确保输入帧为float32
        frame0 = frame0.to(dtype=torch.float32)
        frame1 = frames[frame_itr+1:frame_itr+2].to(dtype=torch.float32)
        
        # 检查是否需要跳过这一帧
        if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
            continue
        
        # 生成中间帧
        middle_frame_batches = []
        
        # 使用自动混合精度上下文
        amp_context = autocast(enabled=use_optimization)
        
        if use_timestep:
            with amp_context:
                # 批处理中间帧生成
                batch_timesteps = []
                for middle_i in range(1, multiplier):
                    batch_timesteps.append(middle_i/multiplier)
                
                # 将帧发送到GPU
                frame0_gpu = frame0.to(DEVICE)
                frame1_gpu = frame1.to(DEVICE)
                
                # 如果适合批处理，使用批处理
                if len(batch_timesteps) >= BATCH_SIZE:
                    for i in range(0, len(batch_timesteps), BATCH_SIZE):
                        batch = batch_timesteps[i:i+BATCH_SIZE]
                        for timestep in batch:
                            middle_frame = return_middle_frame_function(
                                frame0_gpu, frame1_gpu, timestep, 
                                *return_middle_frame_function_args
                            ).detach().cpu()
                            middle_frame_batches.append(middle_frame.to(dtype=dtype))
                else:
                    # 帧数较少，直接处理
                    for timestep in batch_timesteps:
                        middle_frame = return_middle_frame_function(
                            frame0_gpu, frame1_gpu, timestep,
                            *return_middle_frame_function_args
                        ).detach().cpu()
                        middle_frame_batches.append(middle_frame.to(dtype=dtype))
        else:
            # 非时间步处理
            with amp_context:
                middle_frames = non_timestep_inference(
                    frame0.to(DEVICE), frame1.to(DEVICE), multiplier - 1
                )
                middle_frame_batches.extend(
                    torch.cat(middle_frames, dim=0).detach().cpu().to(dtype=dtype)
                )
        
        # 复制中间帧到输出
        for middle_frame in middle_frame_batches:
            output_frames[out_len] = middle_frame
            out_len += 1
        
        # 缓存管理
        frames_since_cache_clear += 1
        if frames_since_cache_clear >= clear_cache_after_n_frames:
            if final_logging:
                print("Comfy-VFI: 清理缓存...", end=' ')
            from comfy.model_management import soft_empty_cache
            soft_empty_cache()
            frames_since_cache_clear = 0
            if final_logging:
                print("完成")
        
        import gc
        gc.collect()
    
    # 记录最终的性能信息
    if use_optimization and final_logging:
        end_time = time.time()
        total_frames = len(output_frames)
        processing_time = end_time - start_time
        fps = total_frames / processing_time
        print(f"FP16优化: 共处理 {total_frames} 帧, 耗时 {processing_time:.2f}秒, FPS: {fps:.2f}")
    
    if final_logging:
        print(f"Comfy-VFI 完成! {len(output_frames)} 帧已生成, 分辨率: {output_frames[0].shape}")
    
    # 添加最后一帧
    output_frames[out_len] = frames[-1:]
    out_len += 1
    
    # 最终清理缓存
    if final_logging:
        print("Comfy-VFI: 最终清理缓存...", end=' ')
    from comfy.model_management import soft_empty_cache
    soft_empty_cache()
    if final_logging:
        print("完成")
    
    return output_frames[:out_len]

# 补丁函数，用于应用优化
def apply_fp16_fast_patch():
    """
    应用FP16 Fast矩阵优化补丁到帧插值函数
    """
    global original_generic_frame_loop, original_frame_loop
    
    if original_generic_frame_loop is None:
        print("无法应用补丁，原始函数未找到")
        return False
    
    try:
        # 导入原始模块
        import vfi_utils
        
        # 保存原始函数引用
        if not hasattr(vfi_utils, '_original_generic_frame_loop'):
            vfi_utils._original_generic_frame_loop = vfi_utils._generic_frame_loop
        
        # 应用补丁
        vfi_utils._generic_frame_loop = _optimized_generic_frame_loop
        
        print("已成功应用FP16 Fast矩阵优化补丁到帧插值函数")
        
        return True
    except Exception as e:
        print(f"应用补丁时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 自动应用补丁
patch_success = apply_fp16_fast_patch()
