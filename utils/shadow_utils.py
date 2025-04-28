import torch
from collections import defaultdict
import time

# --- 辅助函数: calculate_distances_to_point_cuda_concise ---
# (保持不变, 输入输出都在 CUDA)
def calculate_distances_to_point_cuda_concise(points_tensor, target_point_tensor):
  """计算点到目标点的距离和向量 (在 CUDA 上)"""
  if not points_tensor.is_cuda or not target_point_tensor.is_cuda:
      raise ValueError("输入张量必须在 CUDA 上")
  if target_point_tensor.ndim == 1:
      target_point_tensor = target_point_tensor.unsqueeze(0) # 确保 target_point 是 (1, 3)
  # 计算向量
  vectors = points_tensor - target_point_tensor # CUDA
  # 计算距离 (L2 范数)
  distances = torch.linalg.norm(vectors, ord=2, dim=1) # CUDA
  return distances, vectors # 同时返回距离和向量 (都在 CUDA)

# --- 辅助函数: bind_point2_imgcoord_combined ---
# (输入 point_yx 在 CUDA, 返回 img_coords_int 在 CUDA, 但字典在 CPU)
def bind_point2_imgcoord_combined(point_yx_cuda: torch.Tensor):
    """
    将 CUDA 上的投影点绑定到图像坐标。
    返回整数坐标 (CUDA) 和 CPU 上的邻域索引字典。
    """
    if not torch.is_tensor(point_yx_cuda) or point_yx_cuda.ndim != 2 or point_yx_cuda.shape[1] != 2:
        raise ValueError("输入 point_yx 必须是形状为 (N, 2) 的 PyTorch 张量。")
    if not point_yx_cuda.is_cuda:
        raise ValueError("输入 point_yx_cuda 必须在 CUDA 设备上。")

    # 在 CUDA 上计算整数坐标
    img_coords_int_cuda = torch.round(point_yx_cuda).long()

    # --- CPU 部分：构建字典 ---
    # 将整数坐标移到 CPU 以便在 Python 字典中使用和迭代
    img_coords_int_cpu = img_coords_int_cuda.cpu()
    num_points = point_yx_cuda.shape[0]
    coord_to_indices = defaultdict(list)
    coord_to_indices_3x3 = defaultdict(list)

    # 在 CPU 上循环构建字典
    for i in range(num_points):
        center_y = img_coords_int_cpu[i, 0].item()
        center_x = img_coords_int_cpu[i, 1].item()
        center_coord_key = (center_y, center_x)
        original_index = i # 使用原始索引

        coord_to_indices[center_coord_key].append(original_index)

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                pixel_y = center_y + dy
                pixel_x = center_x + dx
                neighbor_coord_key = (pixel_y, pixel_x)
                coord_to_indices_3x3[neighbor_coord_key].append(original_index)

    # 去重 (在 CPU 上)
    for key in coord_to_indices_3x3:
        coord_to_indices_3x3[key] = list(dict.fromkeys(coord_to_indices_3x3[key]))

    # 将字典转换为普通 dict (仍然在 CPU)
    coord_to_indices_cpu = dict(coord_to_indices)
    coord_to_indices_3x3_cpu = dict(coord_to_indices_3x3)
    # --- /CPU 部分 ---

    # 返回 CUDA 上的整数坐标 和 CPU 上的字典
    return img_coords_int_cuda, coord_to_indices_cpu, coord_to_indices_3x3_cpu
# --- /辅助函数 ---


def calculate_occlusion_map_light_dist_angle_cuda(
    point_xyz_cuda: torch.Tensor,
    point_yx_proj_cuda: torch.Tensor,
    light_pos_xyz_cuda: torch.Tensor,
    angle_threshold: float = 0.99): # 余弦相似度阈值
    """
    (优化版: 核心计算在 CUDA)
    基于点到光源的距离和方向向量夹角，结合3x3投影邻域信息判断遮挡。

    Args:
        point_xyz_cuda (torch.Tensor): 原始 3D 点坐标 (N, 3)，在 CUDA 上。
        point_yx_proj_cuda (torch.Tensor): 点的 2D 投影坐标 (N, 2)，在 CUDA 上。
        light_pos_xyz_cuda (torch.Tensor): 光源的 3D 坐标 (3,)，在 CUDA 上。
        angle_threshold (float): 向量夹角的余弦相似度阈值 (越接近1表示角度越小)。

    Returns:
        torch.Tensor: 形状为 (N,) 的布尔张量，在 CPU 上。
                      True 表示该点被遮挡。
    """
    start_time = time.time()
    device = point_xyz_cuda.device # 获取 CUDA 设备

    # --- 1. 输入检查 ---
    if not (point_xyz_cuda.is_cuda and point_yx_proj_cuda.is_cuda and light_pos_xyz_cuda.is_cuda):
        raise ValueError("所有输入张量必须在 CUDA 设备上。")
    if point_xyz_cuda.shape[0] != point_yx_proj_cuda.shape[0]:
        raise ValueError("点数必须相同。")
    if point_xyz_cuda.shape[1] != 3 or light_pos_xyz_cuda.shape[0] != 3:
         raise ValueError("3D 坐标维度必须是 3。")
    if not (0 <= angle_threshold <= 1):
        raise ValueError("angle_threshold 必须在 [0, 1] 之间。")

    num_points = point_xyz_cuda.shape[0]
    print(f"输入点数: {num_points}, 设备: {device}, Angle Threshold: {angle_threshold}")

    # --- 2. 投影和分组 ---
    # 调用修改后的函数，获取 CUDA 坐标和 CPU 字典
    print("步骤 1: 绑定点到图像坐标 (获取 CUDA 坐标和 CPU 字典)...")
    # img_coords_int_cuda 在 CUDA, coord_to_indices_3x3_cpu 在 CPU
    img_coords_int_cuda, _, coord_to_indices_3x3_cpu = bind_point2_imgcoord_combined(point_yx_proj_cuda)
    print(f"  完成投影和分组，耗时: {time.time() - start_time:.4f} 秒")
    step_start_time = time.time()

    # --- 3. 计算所有点到光源的距离和向量 (CUDA) ---
    print("步骤 2: 计算所有点到光源的距离和向量 (CUDA)...")
    distances_to_light_cuda, vectors_to_light_cuda = calculate_distances_to_point_cuda_concise(point_xyz_cuda, light_pos_xyz_cuda)
    # distances_to_light_cuda (N,), vectors_to_light_cuda (N, 3) 都在 CUDA
    print(f"  完成距离和向量计算，耗时: {time.time() - step_start_time:.4f} 秒")
    step_start_time = time.time()

    # --- 4. 比较邻域距离和角度并确定遮挡状态 (核心计算在 CUDA) ---
    print("步骤 3: 比较邻域距离和角度并更新遮挡状态 (CUDA)...")
    # 初始化遮挡状态张量 (在 CUDA 上)
    point_occlusion_status_cuda = torch.zeros(num_points, dtype=torch.bool, device=device)
    epsilon = 1e-9 # 用于防止除以零

    # --- CPU 循环遍历像素邻域 ---
    # 这个循环本身在 CPU 上，因为它迭代 Python 字典
    processed_pixels = 0
    for pixel_coord, neighbor_indices_list in coord_to_indices_3x3_cpu.items():
        k = len(neighbor_indices_list)
        if k <= 1:
            continue
        processed_pixels += 1

        # --- GPU 计算开始 ---
        # 1. 将邻居索引列表从 CPU 传输到 CUDA
        neighbor_indices_tensor = torch.tensor(neighbor_indices_list, device=device, dtype=torch.long)

        # 2. 在 CUDA 上获取邻居的向量和距离
        V_neighbors = vectors_to_light_cuda[neighbor_indices_tensor] # (k, 3) CUDA
        D_neighbors = distances_to_light_cuda[neighbor_indices_tensor] # (k,) CUDA

        # 3. 计算 Pairwise 条件 (完全在 CUDA 上)
        # 3a. 距离比较矩阵
        Closer_Matrix = D_neighbors.unsqueeze(1) < D_neighbors.unsqueeze(0) # (k, k) CUDA

        # 3b. 余弦相似度矩阵
        Dot_Matrix = V_neighbors @ V_neighbors.T # (k, k) CUDA
        Dist_Prod_Matrix = D_neighbors.unsqueeze(1) * D_neighbors.unsqueeze(0) # (k, k) CUDA
        Cos_Matrix = Dot_Matrix / (Dist_Prod_Matrix + epsilon) # (k, k) CUDA
        Cos_Matrix.clamp_(-1.0, 1.0) # CUDA

        # 3c. 角度阈值矩阵
        Angle_OK_Matrix = Cos_Matrix > angle_threshold # (k, k) CUDA

        # 4. 合并条件 (CUDA)
        Occluding_Matrix = Closer_Matrix & Angle_OK_Matrix # (k, k) CUDA

        # 5. 确定哪些点在此邻域被遮挡 (CUDA)
        is_occluded_flags = torch.any(Occluding_Matrix, dim=1) # (k,) CUDA

        # 6. 更新全局遮挡状态 (直接在 CUDA 上操作)
        # 使用 CUDA 上的索引张量和布尔标志进行更新
        point_occlusion_status_cuda[neighbor_indices_tensor] = point_occlusion_status_cuda[neighbor_indices_tensor] | is_occluded_flags
        # --- GPU 计算结束 ---

    # --- CPU 循环结束 ---
    print(f"  完成遮挡状态更新 (处理了 {processed_pixels} 个非空邻域)，耗时: {time.time() - step_start_time:.4f} 秒")
    step_start_time = time.time()

    # --- 5. 将最终结果移回 CPU ---
    print("步骤 4: 将最终结果移回 CPU...")
    point_occlusion_status_cpu = point_occlusion_status_cuda.cpu()
    print(f"  完成 CPU 传输，耗时: {time.time() - step_start_time:.4f} 秒")
    print(f"总耗时: {time.time() - start_time:.4f} 秒")

    return point_occlusion_status_cpu

