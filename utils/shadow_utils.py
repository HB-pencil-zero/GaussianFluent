import torch
from collections import defaultdict
import time
import math
import numpy as np

# --- 辅助函数: calculate_distances_to_point_cuda_concise ---
# (保持不变)
def calculate_distances_to_point_cuda_concise(points_tensor, target_point_tensor):
  """计算点到目标点的距离和向量 (在 CUDA 上)"""
  if not points_tensor.is_cuda or not target_point_tensor.is_cuda:
      raise ValueError("输入张量必须在 CUDA 上")
  if target_point_tensor.ndim == 1:
      target_point_tensor = target_point_tensor.unsqueeze(0) # 确保 target_point 是 (1, 3)
  vectors = points_tensor - target_point_tensor # CUDA
  distances = torch.linalg.norm(vectors, ord=2, dim=1) # CUDA
  return distances, vectors # 同时返回距离和向量 (都在 CUDA)

# --- 辅助函数: calculate_distances_and_unit_vectors_cuda ---
# (保持不变)
def calculate_distances_and_unit_vectors_cuda(points_tensor, target_point_tensor, epsilon=1e-9):
  """计算点到目标点的距离和归一化方向向量 (在 CUDA 上)"""
  if not points_tensor.is_cuda or not target_point_tensor.is_cuda:
      raise ValueError("输入张量必须在 CUDA 上")
  if target_point_tensor.ndim == 1:
      target_point_tensor = target_point_tensor.unsqueeze(0) # 确保 target_point 是 (1, 3)
  vectors = points_tensor - target_point_tensor # CUDA (N, 3)
  distances = torch.linalg.norm(vectors, ord=2, dim=1) # CUDA (N,)
  unit_vectors = vectors / (distances.unsqueeze(1) + epsilon) # CUDA (N, 3)
  return distances, unit_vectors # 返回距离和单位向量 (都在 CUDA)

# --- 辅助函数: bind_point2_imgcoord_combined ---
# (保持不变)
def bind_point2_imgcoord_combined(point_yx_cuda: torch.Tensor):
    """
    将 CUDA 上的投影点绑定到图像坐标。
    返回整数坐标 (CUDA) 和 CPU 上的邻域索引字典。
    """
    if not torch.is_tensor(point_yx_cuda) or point_yx_cuda.ndim != 2 or point_yx_cuda.shape[1] != 2:
        raise ValueError("输入 point_yx 必须是形状为 (N, 2) 的 PyTorch 张量。")
    if not point_yx_cuda.is_cuda:
        raise ValueError("输入 point_yx_cuda 必须在 CUDA 设备上。")
    img_coords_int_cuda = torch.round(point_yx_cuda).long()
    img_coords_int_cpu = img_coords_int_cuda.cpu()
    num_points = point_yx_cuda.shape[0]
    coord_to_indices = defaultdict(list)
    coord_to_indices_3x3 = defaultdict(list)
    for i in range(num_points):
        center_y = img_coords_int_cpu[i, 0].item()
        center_x = img_coords_int_cpu[i, 1].item()
        center_coord_key = (center_y, center_x)
        original_index = i
        coord_to_indices[center_coord_key].append(original_index)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                pixel_y = center_y + dy
                pixel_x = center_x + dx
                neighbor_coord_key = (pixel_y, pixel_x)
                coord_to_indices_3x3[neighbor_coord_key].append(original_index)
    for key in coord_to_indices_3x3:
        coord_to_indices_3x3[key] = list(dict.fromkeys(coord_to_indices_3x3[key]))
    coord_to_indices_cpu = dict(coord_to_indices)
    coord_to_indices_3x3_cpu = dict(coord_to_indices_3x3)
    return img_coords_int_cuda, coord_to_indices_cpu, coord_to_indices_3x3_cpu
# --- /辅助函数 ---


def bind_point2_imgcoord_combined_gpu(point_yx_cuda: torch.Tensor):
    """
    (GPU优化版) 将投影点绑定到图像坐标，完全在GPU上生成邻域索引
    返回: 
        img_coords_int_cuda: (N, 2) CUDA坐标
        coord_to_indices_3x3: 邻域字典 (CPU端)
    """
    if not point_yx_cuda.is_cuda:
        raise ValueError("输入必须在CUDA上")

    device = point_yx_cuda.device
    N = point_yx_cuda.shape[0]

    # --- 步骤1: 生成整数坐标 ---
    img_coords_int_cuda = torch.round(point_yx_cuda).long()  # (N, 2)

    # --- 步骤2: 生成3x3邻域坐标 ---
    # 生成3x3邻域偏移量
    dy = torch.tensor([-1, 0, 1], device=device)
    dx = torch.tensor([-1, 0, 1], device=device)
    delta_y, delta_x = torch.meshgrid(dy, dx, indexing='ij')
    deltas = torch.stack([delta_y.flatten(), delta_x.flatten()], dim=1)  # (9, 2)

    # 扩展每个点的坐标到9个邻域
    neighbor_coords = img_coords_int_cuda.unsqueeze(1) + deltas.unsqueeze(0)  # (N, 9, 2)
    neighbor_coords = neighbor_coords.view(-1, 2)  # (N*9, 2)

    # --- 步骤3: 生成对应的点索引 ---
    point_indices = torch.arange(N, device=device).repeat_interleave(9)  # (N*9,)

    # --- 步骤4: 合并相同坐标并分组索引 ---
    # 将坐标转换为唯一键 (使用int64防止溢出)
    keys = neighbor_coords[:, 0].long() * (1 << 32) + neighbor_coords[:, 1].long()  # (N*9,)

    # 排序并找到唯一键
    sorted_keys, sorted_indices = torch.sort(keys)
    unique_keys, inverse_indices, counts = torch.unique(
        sorted_keys, return_inverse=True, return_counts=True
    )

    # --- 步骤5: 构建字典 (仅在最后转CPU) ---
    # 将数据转移到CPU进行最终字典构建
    sorted_point_indices = point_indices[sorted_indices].cpu().numpy()
    unique_keys_cpu = unique_keys.cpu().numpy()
    counts_cpu = counts.cpu().numpy()

    coord_to_indices_3x3 = defaultdict(list)
    ptr = 0
    for key, count in zip(unique_keys_cpu, counts_cpu):
        # 解码坐标
        y = np.int32(key >> 32)
        x = np.int32(key & 0xFFFFFFFF)
        # 收集索引并去重
        indices = np.unique(sorted_point_indices[ptr:ptr+count])
        coord_to_indices_3x3[(y.item(), x.item())] = indices.tolist()
        ptr += count

    return img_coords_int_cuda, dict(), dict(coord_to_indices_3x3)



def calculate_occlusion_map_light_dist_angle_cuda(
    point_xyz_cuda: torch.Tensor,
    point_yx_proj_cuda: torch.Tensor,
    light_pos_xyz_cuda: torch.Tensor,
    angle_threshold: float = 0.9999998, # 余弦相似度阈值
    dist_eps: float = 1e-4):       # 最小有效遮挡距离阈值
    """
    (优化版: 核心计算在 CUDA, 使用单位向量, 增加最小距离判断)
    基于点到光源的距离、方向单位向量夹角以及点间距离，结合3x3投影邻域信息判断遮挡。

    Args:
        point_xyz_cuda (torch.Tensor): 原始 3D 点坐标 (N, 3)，在 CUDA 上。
        point_yx_proj_cuda (torch.Tensor): 点的 2D 投影坐标 (N, 2)，在 CUDA 上。
        light_pos_xyz_cuda (torch.Tensor): 光源的 3D 坐标 (3,)，在 CUDA 上。
        angle_threshold (float): 向量夹角的余弦相似度阈值 (越接近1表示角度越小)。
        dist_eps (float): 两个点之间的最小 3D 距离，大于此距离才可能发生遮挡。

    Returns:
        torch.Tensor: 形状为 (N,) 的布尔张量，在 CPU 上。
                      True 表示该点被照亮 (未被遮挡)。
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
    if dist_eps < 0:
        raise ValueError("dist_eps 必须是非负数。")

    num_points = point_xyz_cuda.shape[0]
    print(f"输入点数: {num_points}, 设备: {device}, Angle Threshold: {angle_threshold}, Dist Eps: {dist_eps}")

    # --- 2. 投影和分组 ---
    print("步骤 1: 绑定点到图像坐标 (获取 CUDA 坐标和 CPU 字典)...")
    img_coords_int_cuda, _, coord_to_indices_3x3_cpu = bind_point2_imgcoord_combined_gpu(point_yx_proj_cuda)
    print(f"  完成投影和分组，耗时: {time.time() - start_time:.4f} 秒")
    step_start_time = time.time()

    # --- 3. 计算所有点到光源的距离和单位向量 (CUDA) ---
    print("步骤 2: 计算所有点到光源的距离和单位向量 (CUDA)...")
    # 调用修改后的辅助函数
    distances_to_light_cuda, unit_vectors_to_light_cuda = calculate_distances_and_unit_vectors_cuda(point_xyz_cuda, light_pos_xyz_cuda)
    print(f"  完成距离和单位向量计算，耗时: {time.time() - step_start_time:.4f} 秒")
    step_start_time = time.time()

    # --- 4. 比较邻域距离和角度并确定遮挡状态 (核心计算在 CUDA) ---
    print("步骤 3: 优化遮挡计算 (分块批处理+邻域剪枝)...")
    occlusion_status = torch.zeros(num_points, device=device, dtype=torch.bool)

    # 参数设置
    BATCH_NEIGHBORHOODS = 20000  # 每次处理的邻域数量（根据显存调整）
    neighborhoods = list(coord_to_indices_3x3_cpu.values())
    
    # 预处理：按邻域大小排序，优先处理大邻域
    neighborhoods.sort(key=lambda x: len(x), reverse=True)

    # 分块处理
    for i in range(0, len(neighborhoods), BATCH_NEIGHBORHOODS):
        batch = neighborhoods[i:i+BATCH_NEIGHBORHOODS]
        
        # 生成当前批次的所有i<j对
        all_pairs = []
        for indices in batch:
            if len(indices) < 2:
                continue
            indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            pairs = torch.combinations(indices_tensor, 2)  # 生成i<j对
            all_pairs.append(pairs)
        
        if not all_pairs:
            continue
            
        all_pairs = torch.cat(all_pairs, dim=0)
        all_pairs = torch.unique(all_pairs, dim=0)  # 去重

        # 分割i和j索引
        i_indices, j_indices = all_pairs[:, 0], all_pairs[:, 1]

        # 条件1: 距离比较
        D_i = distances_to_light_cuda[i_indices]
        D_j = distances_to_light_cuda[j_indices]
        mask = (D_j < D_i) | (D_i < D_j)  # 排除D_i == D_j的情况

        # 筛选有效对
        valid_pairs = mask.nonzero(as_tuple=True)[0]
        if valid_pairs.numel() == 0:
            continue
            
        i_indices = i_indices[valid_pairs]
        j_indices = j_indices[valid_pairs]
        D_i = D_i[valid_pairs]
        D_j = D_j[valid_pairs]

        # 动态确定检查方向
        check_i = D_j < D_i  # 需要检查i是否被j遮挡
        check_j = ~check_i   # 需要检查j是否被i遮挡

        # 分情况处理 -----------------------------------------------------------------
        # 情况1: j遮挡i
        if torch.any(check_i):
            idx_i = check_i.nonzero(as_tuple=True)[0]
            
            # 计算单位向量点积
            unit_i = unit_vectors_to_light_cuda[i_indices[idx_i]]
            unit_j = unit_vectors_to_light_cuda[j_indices[idx_i]]
            cos_sim = (unit_i * unit_j).sum(dim=1)
            
            # 计算点间距
            pos_i = point_xyz_cuda[i_indices[idx_i]]
            pos_j = point_xyz_cuda[j_indices[idx_i]]
            dist = torch.norm(pos_i - pos_j, dim=1)
            
            # 合并条件
            valid = (cos_sim > angle_threshold) & (dist > dist_eps)
            occlusion_status[i_indices[idx_i[valid]]] = True

        # 情况2: i遮挡j
        if torch.any(check_j):
            idx_j = check_j.nonzero(as_tuple=True)[0]
            
            unit_i = unit_vectors_to_light_cuda[i_indices[idx_j]]
            unit_j = unit_vectors_to_light_cuda[j_indices[idx_j]]
            cos_sim = (unit_i * unit_j).sum(dim=1)
            
            pos_i = point_xyz_cuda[i_indices[idx_j]]
            pos_j = point_xyz_cuda[j_indices[idx_j]]
            dist = torch.norm(pos_i - pos_j, dim=1)
            
            valid = (cos_sim > angle_threshold) & (dist > dist_eps)
            occlusion_status[j_indices[idx_j[valid]]] = True

        # 及时释放中间变量
        del all_pairs, i_indices, j_indices, D_i, D_j, mask, valid_pairs
        torch.cuda.empty_cache()
    print(f"  完成遮挡状态更新 ，耗时: {time.time() - step_start_time:.4f} 秒")

    # 返回未被遮挡的点 (亮=True)
    return ~occlusion_status


