import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import open3d as o3d
from plyfile import PlyData, PlyElement
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
from shadow_extension.shadow_extension import calculate_shadows_ignore_first_hits

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# --- PyTorch向量操作函数 ---
def normalize_batch(vectors):
    """批量归一化向量 (PyTorch实现)"""
    magnitudes = torch.sqrt(torch.sum(vectors**2, dim=1, keepdim=True))
    # 避免除以零
    magnitudes = torch.clamp(magnitudes, min=1e-10)
    return vectors / magnitudes

def dot_product_batch(v1, v2):
    """批量计算点积 (PyTorch实现)"""
    return torch.sum(v1 * v2, dim=1)

def reflect_vectors_batch(incident, normal):
    """批量计算反射向量 (PyTorch实现)"""
    dot_nl = dot_product_batch(normal, incident).unsqueeze(1)
    reflected = 2.0 * dot_nl * normal - incident
    return reflected




def calculate_phong_colors_attenuated_shadowed_internal_batch(
    points, # 完整的点云坐标 (N, 3) - NumPy or Tensor
    normals, # 对应的法向量 (N, 3) - NumPy or Tensor
    light,
    material,
    view_position,
    output_folder,
    point_rgb_colors, # 点的原始 RGB 颜色 (N, 3) - NumPy or Tensor
    device='cuda',       # 'cuda' or 'cpu'
    enable_shadows=True,
    shadow_epsilon=1e-4,
    alignment_threshold=0.999,
    attenuation_constant=5.0,
    shadow_batch_size=10000, # !!! 新增: 控制阴影计算的内部批大小 !!!
    return_cpu=True ,     # 是否将最终结果返回到CPU
    opacity  = None,
    ignore_first_n_hits = 1

    ):
    """
    计算点云的Phong着色 (带衰减和阴影)，阴影部分内部批处理。

    接收完整的点云数据，但阴影计算步骤在内部按 shadow_batch_size 分批执行，
    以管理内存消耗。其他计算步骤一次性完成。

    Args:
        points (Tensor or ndarray): 完整的点云坐标 (N, 3).
        normals (Tensor or ndarray): 对应的法向量 (N, 3).
        light (dict): 光源参数.
        material (dict): 材质参数.
        view_position (list or tuple): 观察者位置 (3,).
        point_rgb_colors (Tensor or ndarray): 点的原始 RGB 颜色 (N, 3).
        device (str or torch.device): 计算设备 ('cuda' or 'cpu').
        enable_shadows (bool): 是否启用阴影计算.
        shadow_epsilon (float): 阴影计算中用于避免自遮挡的距离阈值.
        alignment_threshold (float): 阴影计算中判断射线对齐的阈值.
        attenuation_constant (float): 光强衰减公式中的常数.
        shadow_batch_size (int): 阴影计算的内部批处理大小.
        return_cpu (bool): 是否将最终结果从 GPU 移回 CPU (如果使用了 GPU).

    Returns:
        Tensor or ndarray: 计算得到的颜色 (N, 3)，根据 return_cpu 决定类型。
    """
    overall_start_time = time.time()
    print(f"开始 Phong 计算 (总点数 N={points.shape[0]})")

    # --- 确定设备 ---
    if isinstance(device, str):
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: 请求 CUDA 但不可用，将使用 CPU。")
            _device = torch.device("cpu")
        elif device not in ['cuda', 'cpu']:
             print(f"警告: 无效的设备 '{device}'，将使用 CPU。")
             _device = torch.device("cpu")
        else:
            _device = torch.device(device)
    elif isinstance(device, torch.device):
        _device = device
    else:
        print("警告: 无效的设备对象，将使用 CPU。")
        _device = torch.device("cpu")
    print(f"使用设备: {_device}")

    # --- 将所有输入数据转换为指定设备上的 Tensor ---
    print("准备数据并传输到设备...")
    transfer_start_time = time.time()
    if not isinstance(points, torch.Tensor):
        points_gpu = torch.tensor(points, dtype=torch.float32, device=_device)
    elif points.device != _device:
        points_gpu = points.to(_device)
    else:
        points_gpu = points

    if not isinstance(normals, torch.Tensor):
        normals_gpu = torch.tensor(normals, dtype=torch.float32, device=_device)
    elif normals.device != _device:
        normals_gpu = normals.to(_device)
    else:
        normals_gpu = normals

    if not isinstance(point_rgb_colors, torch.Tensor):
        colors_input_gpu = torch.tensor(point_rgb_colors, dtype=torch.float32, device=_device)
    elif point_rgb_colors.device != _device:
        colors_input_gpu = point_rgb_colors.to(_device)
    else:
        colors_input_gpu = point_rgb_colors

    N = points_gpu.shape[0] # 点云总点数

    # 光源参数
    light_pos = torch.tensor(light['position'], dtype=torch.float32, device=_device)
    light_ambient = torch.tensor(light['ambient'], dtype=torch.float32, device=_device)
    light_diffuse = torch.tensor(light['diffuse'], dtype=torch.float32, device=_device)
    light_specular = torch.tensor(light['specular'], dtype=torch.float32, device=_device)

    # 材质参数
    mat_specular = torch.tensor(material['specular'], dtype=torch.float32, device=_device)
    mat_shininess = material['shininess']

    view_pos = torch.tensor(view_position, dtype=torch.float32, device=_device)
    print(f"数据准备和传输耗时: {time.time() - transfer_start_time:.4f} 秒")

    # --- 计算光照基础分量 (对所有 N 个点一次性计算) ---
    print("计算基础光照分量...")
    calc_start_time = time.time()
    # 1. 归一化法向量
    N_norm = normalize_batch(normals_gpu) # (N, 3)

    # 2. 计算光照方向向量 (从点指向光源) 和距离
    L_vec = light_pos.unsqueeze(0) - points_gpu # (N, 3)
    distance_sq = torch.sum(L_vec**2, dim=1, keepdim=True) # (N, 1)
    distance_sq = torch.clamp(distance_sq, min=1e-6)
    distance_to_light = torch.sqrt(distance_sq) # (N, 1)
    L = normalize_batch(L_vec) # (N, 3)

    # 3. 计算衰减因子
    attenuation = attenuation_constant / distance_sq # (N, 1)

    # 4. 计算视线方向向量
    V_vec = view_pos.unsqueeze(0) - points_gpu # (N, 3)
    V = normalize_batch(V_vec) # (N, 3)

    # 5. 计算反射向量
    R = reflect_vectors_batch(L, N_norm) # (N, 3)
    R = normalize_batch(R) # (N, 3)

    # 6. 计算环境光分量
    ambient_term = colors_input_gpu * light_ambient # (N, 3)

    # --- 7. 计算阴影因子 (Shadow Factor) - 内部按批处理 ---

    shadow_factors = torch.ones(N, 1, device=_device) # 初始化为1 (不在阴影中)
    if enable_shadows and N_norm.shape[0] > 1: # 使用 N_norm.shape[0] 代替 N
        print(f"开始计算阴影 (内部批大小 B={shadow_batch_size})... 这可能仍然较慢...")
        shadow_calc_start_time = time.time()
        # --- 硬编码 ignore_first_n_hits ---
        
        print("calculate 阴影")
        shadow_factors = calculate_shadows_ignore_first_hits(
            points_gpu, L, distance_to_light, opacity ,enable_shadows, shadow_batch_size,
            shadow_epsilon, alignment_threshold, ignore_first_n_hits) # 传递参数

        # --- 清理显存 (可选) ---
        # ... (之前的清理代码保持不变) ...

        shadow_calc_end_time = time.time()
        print(f"阴影计算耗时: {shadow_calc_end_time - shadow_calc_start_time:.2f} 秒")

        dot_nl_initial = torch.sum(N_norm * L, dim=1) # (N,)

        # 使用 > 0.5 而不是 == 1.0 来允许浮点数误差
        is_lit = (shadow_factors.squeeze() > 0.5)
        needs_flip = (dot_nl_initial < 0.0)
        flip_mask = is_lit & needs_flip # 布尔掩码 (N,)

        # 3. 创建修正后的法线 N_corrected
        N_corrected = torch.where(flip_mask.unsqueeze(1), -N_norm, N_norm) # (N, 3)

        # 可选：打印被翻转法线的点的数量
        num_flipped = torch.sum(flip_mask).item()
        if num_flipped > 0:
            print(f"信息: {num_flipped} 个被照亮点的法线因点积为负而被翻转。")

        # --- 创建包含参数的文件名 ---
        # 将浮点数中的 '.' 替换为 'p' 以避免文件名问题
        eps_str = str(shadow_epsilon).replace('.', 'p')
        align_str = str(alignment_threshold).replace('.', 'p')

        # 构建文件名后缀
        filename_suffix = (
            f"_shdw{int(enable_shadows)}" # 1 if True, 0 if False
            f"_bs{shadow_batch_size}"
            f"_eps{eps_str}"
            f"_align{align_str}"
            f"_ign{ignore_first_n_hits}"
        )
        # 构建完整文件名
        shadow_factors_filename = f"shadow_factors{filename_suffix}.npy"
        N_norm_corrected_filename = f"N_norm_corrected{filename_suffix}.npy" # 文件名也修正一下

        # --- 保存文件 ---
        print(f"正在保存阴影因子到: {shadow_factors_filename}")
        np.save( os.path.join(output_folder, shadow_factors_filename), shadow_factors.detach().cpu().numpy())

        print(f"正在保存修正后的法线到: {N_norm_corrected_filename}")
        # 注意：这里应该保存修正后的法线 N_corrected，而不是原始的 N_norm
        np.save(os.path.join(output_folder,N_norm_corrected_filename), N_corrected.detach().cpu().numpy())

        # --- 更新 N_norm 以便后续代码使用修正后的版本 ---
        # 如果后续代码期望 N_norm 是修正后的，则执行此操作
        N_norm = N_corrected
        print("注意: 变量 N_norm 已更新为修正后的法线。")

    else:
        eps_str = str(shadow_epsilon).replace('.', 'p')
        align_str = str(alignment_threshold).replace('.', 'p')

        # 构建文件名后缀
        filename_suffix = (
            f"_shdw{int(1)}" # 1 if True, 0 if False
            f"_bs{shadow_batch_size}"
            f"_eps{eps_str}"
            f"_align{align_str}"
            f"_ign{ignore_first_n_hits}"
        )

        # 构建完整文件名
        shadow_factors_filename =   os.path.join(  output_folder,f"shadow_factors{filename_suffix}.npy" )
        N_norm_corrected_filename = os.path.join(  output_folder,f"N_norm_corrected{filename_suffix}.npy" )# 文件名也修正一下
        
        # --- 尝试加载阴影因子和修正后的法线 ---
        print(f"尝试从文件加载阴影因子: {shadow_factors_filename}")
        try:
            shadow_factors = torch.from_numpy(np.load(shadow_factors_filename)).to(device)
            print("成功加载阴影因子。")
        except:
            print("无法加载阴影因子文件,将使用全1阴影因子。")
            shadow_factors = torch.ones(points.shape[0], 1, device=device)
            
        print(f"尝试从文件加载修正后的法线: {N_norm_corrected_filename}")    
        try:
            N_norm = torch.from_numpy(np.load(N_norm_corrected_filename)).to(device)
            print("成功加载修正后的法线。")
        except:
            print("无法加载修正后的法线文件,将使用原始法线。")
        
        # 如果不计算阴影或只有一个点，N_norm 保持不变
        print("跳过阴影计算或法线翻转。")
        # 如果需要，也可以在这里保存原始 N_norm
        # np.save("N_norm_original.npy", N_norm.detach().cpu().numpy())
    
    # ambient_term[shadow_factors.squeeze(1) < 0.1] *= 0.8

    # # 2. 找出需要修正法线的点：被照亮 (shadow_factor > 0.5) 且 原始点积为负
    # #    使用 > 0.5 而不是 == 1.0 来允许浮点数误差
    # is_lit = (shadow_factors.squeeze() > 0.5)
    # needs_flip = (dot_nl_initial < 0.0)
    # flip_mask = is_lit & needs_flip # 布尔掩码 (N,)

    # # 3. 创建修正后的法线 N_corrected
    # #    对于 flip_mask 为 True 的点，使用 -N_norm，否则使用 N_norm
    # N_norm = torch.where(flip_mask.unsqueeze(1), -N_norm, N_norm) # (N, 3)

    # # 可选：打印被翻转法线的点的数量
    # num_flipped = torch.sum(flip_mask).item()
    # if num_flipped > 0:
    #     print(f"信息: {num_flipped} 个被照亮点的法线因点积为负而被翻转。")
    
    
    
    
    # 8. 计算漫反射基础贡献
    diffuse_intensity = torch.clamp(dot_product_batch(N_norm, L), min=0.0) # (N,)
    diffuse_base = colors_input_gpu * light_diffuse * diffuse_intensity.unsqueeze(1) # (N, 3)

    # 9. 计算镜面反射基础贡献
    dot_rv = torch.clamp(dot_product_batch(R, V), min=0.0) # (N,)
    if mat_shininess > 0:
        specular_intensity = torch.pow(dot_rv, mat_shininess) # (N,)
    else:
        specular_intensity = torch.zeros_like(dot_rv)
    specular_base = mat_specular * light_specular * specular_intensity.unsqueeze(1) # (N, 3)
    print(f"基础光照计算耗时: {time.time() - calc_start_time:.4f} 秒")





    # --- 10. 应用衰减和阴影因子 (对所有 N 个点一次性计算) ---
    print("应用衰减和阴影...")
    apply_start_time = time.time()
    diffuse_term = diffuse_base * attenuation * shadow_factors # (N, 3)
    specular_term = specular_base * attenuation * shadow_factors # (N, 3)

    # 11. 合并所有光照分量
    colors_final_gpu = ambient_term + diffuse_term + specular_term # (N, 3)

    # 12. 将颜色限制在[0,1]范围内
    colors_final_gpu = torch.clamp(colors_final_gpu, 0.0, 1.0)
    print(f"应用衰减和阴影耗时: {time.time() - apply_start_time:.4f} 秒")

    overall_end_time = time.time()
    print(f"总计算耗时 (不含最终数据传输): {overall_end_time - overall_start_time:.4f} 秒")

    # --- 返回结果 ---
    if return_cpu and _device != torch.device('cpu'):
        print("将结果传输回 CPU...")
        final_colors_cpu = colors_final_gpu.cpu().numpy()
        print("完成。")
        return final_colors_cpu
    elif return_cpu and _device == torch.device('cpu'):
        print("计算在 CPU 上完成，返回 NumPy 数组。")
        return colors_final_gpu.numpy() # 如果在 CPU 计算，直接转 NumPy
    else:
        print("完成。结果保留在 GPU 上。")
        return colors_final_gpu # 返回 GPU Tenso



def calculate_shadows_ignore_first_hits__(
    points_gpu: torch.Tensor,
    L: torch.Tensor,
    distance_to_light: torch.Tensor,
    opacity: torch.Tensor,
    enable_shadows: bool = True,
    shadow_batch_size: int = 1024,
    shadow_epsilon: float = 1e-5,
    alignment_threshold: float = 0.99,
    num_surface_points: int = 5
) -> torch.Tensor:
    N = points_gpu.shape[0]
    _device = points_gpu.device
    shadow_factors = torch.ones(N, 1, device=_device, dtype=points_gpu.dtype)

    if not enable_shadows or N <= 1 or num_surface_points < 0:
        return shadow_factors

    # ==================== 关键修改1：opacity处理 ====================
    # 生成有效遮挡点索引（保持原索引顺序）
    occluder_mask = (opacity > 0).squeeze(1)          # (N,)
    occluder_points = points_gpu[occluder_mask]       # (M,3)
    M = occluder_points.shape[0]
    if M == 0:
        return shadow_factors

    print(f"计算阴影 (有效遮挡点={M}/{N}, 批大小={shadow_batch_size})...")
    shadow_calc_start_time = time.time()
    num_shadow_batches = math.ceil(N / shadow_batch_size)
    
    # ==================== 关键修改2：遮挡点维度调整 ====================
    # 仅使用有效遮挡点构建张量（原索引自动映射）
    points_expanded_occluder = occluder_points.unsqueeze(0)  # (1,M,3)

    for i in tqdm(range(num_shadow_batches)):
        batch_start_idx = i * shadow_batch_size
        batch_end_idx = min((i+1)*shadow_batch_size, N)
        current_batch_indices = torch.arange(batch_start_idx, batch_end_idx, device=_device)
        current_active_batch_size = len(current_batch_indices)
        
        if current_active_batch_size == 0:
            continue

        # ================ 保持原索引结构 ================
        batch_points = points_gpu[current_batch_indices]        # (B,3)
        batch_L = L[current_batch_indices]                      # (B,3)
        batch_dist_light_sq = distance_to_light[current_batch_indices].square().view(-1,1)  # (B,1)

        # ================ 关键修改3：向量差计算 ================
        # 目标点维度: (B,1,3) | 遮挡点维度: (1,M,3)
        vec_diff = points_expanded_occluder - batch_points.unsqueeze(1)  # (B,M,3)
        
        # 计算点积（保持与原逻辑相同的物理意义）
        dot_products = torch.einsum('bmd,bd->bm', vec_diff, batch_L)  # (B,M)
        dist_sq = torch.sum(vec_diff.pow(2), dim=2)                   # (B,M)

        # ================ 关键修改4：条件判断修正 ================
        valid_dist_mask = (dist_sq > shadow_epsilon**2) & \
                        (dist_sq < batch_dist_light_sq)   # 正确广播 (B,1) vs (B,M)
        positive_dot_mask = dot_products > 0
        alignment_mask = (dot_products.pow(2) > (alignment_threshold**2 * dist_sq)) & positive_dot_mask
        is_occluder = valid_dist_mask & alignment_mask

        # ================ 保持原索引映射 ================
        occluder_counts = is_occluder.sum(dim=1)
        shadow_mask = occluder_counts > num_surface_points
        shadow_indices = current_batch_indices[shadow_mask]
        if shadow_indices.numel() > 0:
            shadow_factors[shadow_indices] = 0.0

    print(f"阴影计算耗时: {time.time() - shadow_calc_start_time:.2f}s")
    return shadow_factors




def optimize_normals_consistency(points, normals, k=10):
    """使用Open3D优化法向量一致性"""
    print("正在优化法向量一致性...")
    start_time = time.time()

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 使用一致性传播算法优化法向量方向
    pcd.orient_normals_consistent_tangent_plane(k=k)

    # 获取优化后的法向量
    optimized_normals = np.asarray(pcd.normals)

    end_time = time.time()
    print(f"法向量一致性优化耗时: {end_time - start_time:.2f}秒")

    return optimized_normals


import open3d as o3d
import numpy as np
import time

# def optimize_normals_consistency(
#     points: np.ndarray,
#     normals: np.ndarray,
#     k: int = 10,
#     force_continuity: bool = True
# ) -> np.ndarray:
#     """优化法向量一致性，内存布局优化版
    
#     Args:
#         points: 点云坐标 (N, 3)
#         normals: 初始法向量 (N, 3)
#         k: 邻域点数，建议范围 5-20
#         force_continuity: 强制内存连续布局以加速转换
    
#     Returns:
#         优化后的法向量 (N, 3)
#     """
#     # 确保输入数据为连续内存布局
#     if force_continuity:
#         points = np.ascontiguousarray(points)
#         normals = np.ascontiguousarray(normals)

#     # 提前验证数据有效性
#     assert points.shape == normals.shape, "点云与法向量维度不匹配"
#     assert points.ndim == 2 and points.shape[1] == 3, "点云应为(N,3)数组"
    
#     # 创建点云对象（复用对象可提升约5%性能）
#     pcd = o3d.geometry.PointCloud()
    
#     # 使用内存视图避免完整拷贝（需Open3D>=0.14支持）
#     if hasattr(o3d.utility, 'Vector3dVector'):
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.normals = o3d.utility.Vector3dVector(normals)
#     else:  # 兼容旧版本
#         pcd.points = o3d.utility.Vector3dVector(points.copy())
#         pcd.normals = o3d.utility.Vector3dVector(normals.copy())

#     # 执行优化（Open3D内部使用多线程）
#     pcd.orient_normals_consistent_tangent_plane(k=k)

#     # 直接访问底层数据避免二次拷贝
#     return np.asarray(pcd.normals, dtype=np.float32)


# --- 主程序 ---
if __name__ == "__main__":
    start_total = time.time()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--opacity_path", type=str, required=True)
    parser.add_argument("--valid_indice_path", type=str, required=True)
    parser.add_argument("--shs_path", type=str, required=True)
    parser.add_argument("--color_path", type=str, required=True)
    args = parser.parse_args()
    output_folder = args.output_folder 
    

    # --- 文件路径定义 ---
    # 新的 PLY 文件，包含位置和法向量
    combined_ply_path = args.npy_path 
    # 原始的 PLY 文件，用于读取颜色 (f_dc 特征)
    # original_ply_path = "/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model/watermelon/point_cloud/iteration_30000/point_cloud.ply"
    shs_path = args.shs_path
    color_path = args.color_path
    valid_indices = np.load(args.valid_indice_path)
    opacity_tensor = torch.load(args.opacity_path, weights_only=True).cuda()[torch.from_numpy(valid_indices)]

    points = None
    normals = None
    point_rgb_colors_from_ply = None

    # --- 1. 从新的组合 PLY 文件读取位置和法向量 ---
    print(f"正在从 '{combined_ply_path}' 读取位置和法向量...")
    start_time = time.time()
    try:
        pcd_combined = o3d.io.read_point_cloud(combined_ply_path)

        if not pcd_combined.has_points():
            print(f"错误: 文件 '{combined_ply_path}' 不包含点数据。")
            exit()
        points = np.asarray(pcd_combined.points)
        print(f"从组合文件读取的位置 (points) 形状: {points.shape}")
        # 注意：这里的 points 中，原先是 NaN 的坐标现在是 0.0

        if not pcd_combined.has_normals():
            print(f"警告: 文件 '{combined_ply_path}' 不包含法向量数据。")
            # 可以选择生成随机法线或退出
            normals = np.random.rand(len(points), 3) * 2 - 1
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            print("使用了随机法向量。")
        else:
            normals = np.asarray(pcd_combined.normals)
            print(f"从组合文件读取的法向量 (normals) 形状: {normals.shape}")
            # 检查点数和法向量数是否匹配
            if points.shape[0] != normals.shape[0]:
                 print(f"警告：读取的位置点数 ({points.shape[0]}) 与法向量数 ({normals.shape[0]}) 不匹配！可能文件已损坏。")
                 # 可以选择退出或尝试继续，但后续处理可能出错
                 # exit()

    except FileNotFoundError:
        print(f"错误: 找不到组合 PLY 文件 '{combined_ply_path}'")
        exit()
    except Exception as e:
        print(f"读取组合 PLY 文件 '{combined_ply_path}' 时出错: {e}")
        exit()

    end_time = time.time()
    print(f"读取位置和法向量耗时: {end_time - start_time:.2f}秒")

    # --- 2. 从原始 PLY 文件读取颜色信息 ---
    start_time = time.time()
    if color_path: 
        point_rgb_colors_from_ply = torch.load(color_path, weights_only=True).detach().numpy()[valid_indices]        
    else :
        print(f"正在从 '{shs_path}' 提取颜色 (f_dc)...")
        start_time = time.time()
        features_dc = torch.load(shs_path, weights_only=True)[:, 0].detach().numpy()
        C0 = 0.28209479177387814 # 球谐函数的零阶系数
        # 将SH系数转换为[0, 1]范围的RGB值

        point_rgb_colors_from_ply = np.clip(features_dc * C0 + 0.5, 0.0, 1.0)[valid_indices]

    print(f"从原始文件提取的颜色 (point_rgb_colors_from_ply) 形状: {point_rgb_colors_from_ply.shape}")
    end_time = time.time()
    print(f"提取颜色耗时: {end_time - start_time:.2f}秒")


    # --- 3. 可选：处理法向量 ---
    if normals is not None:
        print("正在处理法向量 (优化一致性, 方向反转)...")
        start_time = time.time()
        # 可选：优化法向量一致性
        # normals = optimize_normals_consistency(points, normals, k=30)
        # normals = optimize_normals_consistency(points, normals, k=30)  #两次结果是一样的
        # 根据需要反转法线方向 (这里的反转是示例，你需要根据你的场景决定是否需要)
        # normals = -normals
        print("法向量处理完成。")
        end_time = time.time()
        print(f"处理法向量耗时: {end_time - start_time:.2f}秒")
    else:
        print("跳过法向量处理，因为法向量未能成功加载。")

    # --- 结束 ---
    end_total = time.time()
    print(f"\n总处理时间: {end_total - start_total:.2f}秒")

    # --- 检查最终结果 ---
    if points is not None:
        print(f"最终 Points 形状: {points.shape}")
    else:
        print("最终 Points 未加载")
    if normals is not None:
        print(f"最终 Normals 形状: {normals.shape}")
    else:
        print("最终 Normals 未加载或生成")
    if point_rgb_colors_from_ply is not None:
        print(f"最终 Colors 形状: {point_rgb_colors_from_ply.shape}")
    else:
        print("最终 Colors 未加载或生成")


    end_time = time.time()
    print(f"读取或处理法向量耗时: {end_time - start_time:.2f}秒")
    print(f"法向量形状: {normals.shape}")

    # 确保点、颜色和法向量数量匹配
    # min_len = min(len(points), len(normals), len(point_rgb_colors_from_ply))
    # if len(points) != min_len or len(normals) != min_len or len(point_rgb_colors_from_ply) != min_len:
    #     print(f"警告：点、法向量或颜色数量不匹配！将截断为最小长度: {min_len}")
    #     points = points[:min_len]
    #     normals = normals[:min_len]
    #     point_rgb_colors_from_ply = point_rgb_colors_from_ply[:min_len]

    # 定义光源
    light_source = {
        'position': [0.0, 0.0, 0],   # 光源位置
        'ambient': [0.7, 0.7, 0.7],   # 环境光强度（调低一点，因为物体自带环境色）
        'diffuse': [0.4, 0.4, 0.4],   # 漫反射光强度
        'specular': [0.0, 0.0, 0.0]   # 镜面反射光强度
    }

    # 定义材质 (现在只关心镜面反射和高光)
    material = {
        # 'ambient': [0.1, 0.6, 0.1], # 不再使用
        # 'diffuse': [0.1, 0.8, 0.1], # 不再使用
        'specular': [1, 1, 1], # 镜面反射颜色/系数 (白色高光)
        'shininess': 3.0             # 高光指数 (控制高光大小和锐度)
    }

    # 观察者位置 (与光源位置相同，模拟头灯效果)
    viewer_pos = [0.0, 0.0, 5.0]    

    # 使用PyTorch CUDA批量计算所有点的颜色
    print("正在使用PyTorch CUDA计算Phong着色...")
    phong_colors = calculate_phong_colors_attenuated_shadowed_internal_batch(
        points,
        normals,
        light_source,
        material,
        viewer_pos,
        output_folder,
        point_rgb_colors_from_ply, # 传入每个点的RGB颜色
        device='cuda',
        enable_shadows=True,
        shadow_batch_size=200,
        shadow_epsilon= 1e-3,
        alignment_threshold=0.999,
        attenuation_constant=9,
        opacity = opacity_tensor ,
        ignore_first_n_hits=1
    )
    # phong_colors = calculate_phong_colors_batch_attenuated(points, normals, light_source, material, viewer_pos, point_rgb_colors_from_ply)

    np.save(os.path.join(output_folder, "phong_colors.npy"), phong_colors)
