import torch
import time
import os
from torch.utils.cpp_extension import load

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sources = [os.path.join(current_dir, "shadow_kernel.cu")]

# 加载CUDA扩展
try:
    shadow_extension = load(
        name="shadow_extension",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", 
            "--use_fast_math",
            "-Xcompiler", "-fopenmp",
            "-DNDEBUG"
        ],
        verbose=True
    )
    print("Successfully loaded shadow_extension module.")
except Exception as e:
    print(f"Error loading shadow_extension: {e}")
    # 创建虚拟模块以便调试
    class DummyModule:
        @staticmethod
        def calculate_shadows_cuda(*args, **kwargs):
            print("Dummy function called - CUDA extension not loaded")
            return torch.ones(args[0].size(0), device=args[0].device)
    
    shadow_extension = DummyModule()

def calculate_shadows_ignore_first_hits(
    points_gpu: torch.Tensor,
    L: torch.Tensor,
    distance_to_light: torch.Tensor,
    opacity: torch.Tensor,
    enable_shadows: bool = True,
    shadow_batch_size : int = 1024,
    shadow_epsilon: float = 1e-5,
    alignment_threshold: float = 0.99,
    num_surface_points: int = 5
) -> torch.Tensor:
    N = points_gpu.shape[0]
    _device = points_gpu.device
    dtype = points_gpu.dtype
    
    if not enable_shadows or N <= 1 or num_surface_points < 0:
        return torch.ones(N, 1, device=_device, dtype=dtype)

    # 生成有效遮挡点索引
    occluder_mask = (opacity > 0).squeeze(1)
    occluder_points = points_gpu[occluder_mask]
    M = occluder_points.shape[0]
    
    if M == 0:
        return torch.ones(N, 1, device=_device, dtype=dtype)

    print(f"高性能CUDA阴影计算 (目标点={N}, 遮挡点={M})...")
    start_time = time.time()
    
    # 计算光源距离平方
    dist_light_sq = distance_to_light.square().contiguous()
    
    # 调用CUDA扩展
    shadow_factors = shadow_extension.calculate_shadows_cuda(
        points_gpu.contiguous(),
        occluder_points.contiguous(),
        L.contiguous(),
        dist_light_sq.contiguous(),
        shadow_epsilon,
        alignment_threshold,
        num_surface_points
    )
    
    # 转换为原始形状
    shadow_factors = shadow_factors.view(-1, 1)
    
    print(f"CUDA阴影计算完成, 耗时: {time.time() - start_time:.2f}s")
    return shadow_factors