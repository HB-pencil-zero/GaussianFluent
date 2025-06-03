import numpy as np
import torch
# from sklearn.neighbors import KDTree
import time
import open3d as o3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback # 用于打印详细错误
from scipy.spatial import KDTree

def compute_point_cloud_normals_remove_nan(points, k=30, orientation_k=None,
                                           batch_size=204800, device='cuda',
                                           orientation_method='neighbor_vector'):
    """
    计算点云的法向量，删除包含 NaN 的点，并确保方向一致性。

    参数:
        points: np.ndarray, 形状为 (n, 3) 的原始点云数据，可能包含 NaN。
        k: int, 用于法向量估计的邻近点数量，默认为30。
        orientation_k: int, 用于方向调整的邻近点数量，默认与k相同。
        batch_size: int, 批处理大小，用于控制内存使用。
        device: str, 'cuda' 或 'cpu'，用于计算的设备。
        orientation_method: str, 法向量方向调整方法
            - 'neighbor_vector': 使用邻域距离向量方法。
            - 'viewpoint': 使用视点方法(视点为原点)。
            - 'none': 不调整方向。

    返回:
        valid_normals: np.ndarray, 形状为 (n_valid, 3) 的单位法向量数组，仅包含有效点的法向量。
        valid_indices: np.ndarray, 形状为 (n_valid,) 的整数数组，包含有效点在原始 `points` 数组中的索引。
        nan_indices: np.ndarray, 形状为 (n_nan,) 的整数数组，包含被删除的 NaN 点在原始 `points` 数组中的索引。
    """
    n_points = points.shape[0]
    original_indices = np.arange(n_points)

    # --- NaN 处理开始 ---
    # 1. 识别 NaN 点
    nan_mask = np.isnan(points).any(axis=1)
    valid_mask = ~nan_mask

    valid_indices = original_indices[valid_mask] # 有效点的原始索引
    nan_indices = original_indices[nan_mask]   # NaN 点的原始索引

    n_valid_points = valid_indices.shape[0]
    n_nan_points = nan_indices.shape[0]

    if n_nan_points > 0:
        print(f"信息: 输入点云包含 {n_nan_points} 个含有 NaN 坐标的点。这些点将被移除，不计算法向量。")

    if n_valid_points == 0:
        print("错误: 所有点都包含 NaN 或点云为空，无法计算法向量。返回空结果。")
        # 返回空的法向量数组、空的有效索引和包含所有原始索引的 NaN 索引
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=int), original_indices
    # --- NaN 处理结束 ---

    # 仅处理有效点
    valid_points = points[valid_mask]
    # 初始化有效点的法向量数组
    valid_normals = np.zeros((n_valid_points, 3), dtype=np.float32)

    # 设置用于方向调整的邻近点数量
    if orientation_k is None:
        orientation_k = min(k, 10000000) # 限制方向调整的邻居数，避免过大

    print(f"计算有效点云法向量，有效点数: {n_valid_points}，邻域大小: k={k}")
    start_time = time.time()

    # 检查设备可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU替代")
        device = 'cpu'

    print(f"使用设备: {device}")

    # 3. 构建 KD 树 (仅使用有效点)
    print("正在为有效点构建 KD 树...")
    kdtree_build_start = time.time()
    try:
        kdtree = KDTree(valid_points)
        print(f"KD 树构建完成，用时: {time.time() - kdtree_build_start:.2f} 秒")
    except Exception as e:
        print(f"错误: 构建 KD 树失败: {e}")
        print("返回空结果。")
        traceback.print_exc()
        # 返回部分计算结果可能意义不大，直接返回空
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=int), nan_indices


    # 批处理计算有效点的法向量
    total_batches = int(np.ceil(n_valid_points / batch_size))

    calculation_successful = True # 标记计算是否成功
    for batch_idx, batch_start in enumerate(range(0, n_valid_points, batch_size)):
        batch_end = min(batch_start + batch_size, n_valid_points)
        current_batch_size = batch_end - batch_start
        batch_valid_points = valid_points[batch_start:batch_end] # 当前批次的有效点

        # 打印进度
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f"处理有效点批次 {batch_idx+1}/{total_batches} ({(batch_idx+1)/total_batches*100:.1f}%)...")

        # 查找 k 个最近邻点 (在有效点中查找)
        # 确保 k 不超过有效点总数减一
        k_query = min(k + 1, n_valid_points)
        if k_query <= 1: # 如果有效点太少，无法形成邻域
             print(f"警告: 批次 {batch_idx+1} 有效点不足 ({n_valid_points})，无法计算邻域 k={k}。将为这些点设置默认法线 [0, 0, 1]。")
             # 为这个批次的点设置默认法线
             valid_normals[batch_start:batch_end] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
             continue # 继续处理下一个批次

        # 在有效点KD树中查询当前批次的邻居
        try:
            _, nn_indices_in_valid = kdtree.query(batch_valid_points, k=k_query)
        except Exception as e:
             print(f"错误: 批次 {batch_idx+1} KD树查询失败: {e}")
             print("将为这个批次的点设置默认法线 [0, 0, 1]。")
             traceback.print_exc()
             valid_normals[batch_start:batch_end] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
             calculation_successful = False # 标记计算中出现问题
             continue # 继续处理下一个批次


        # 将邻域点转换为张量 (邻域点来自 valid_points)
        neighbors_list = []
        actual_k_used = k_query - 1 # 实际用于计算的邻居数
        for i in range(current_batch_size):
            # nn_indices_in_valid 包含相对于 valid_points 的索引
            # 排除点本身 (通常是第一个最近邻)
            neighbor_indices_in_valid = nn_indices_in_valid[i, 1:k_query] # 使用 k_query 确保索引有效
            neighbors_list.append(valid_points[neighbor_indices_in_valid])

        # 处理邻居数量不足的情况
        if actual_k_used < 2:
             print(f"警告: 批次 {batch_idx+1} 中部分点邻居数不足 ({actual_k_used})，法线可能不准确。将尝试计算或使用默认值。")
             # 如果邻居数少于2，SVD会失败或无意义，可以提前处理
             # 这里我们让SVD尝试，并在失败时回退

        try:
            neighbors = np.array(neighbors_list, dtype=np.float32) # 形状 (current_batch_size, actual_k_used, 3)
            neighbors_tensor = torch.tensor(neighbors, device=device)

            # 计算邻域点的中心
            centroids = torch.mean(neighbors_tensor, dim=1, keepdim=True)

            # 将点集中心化
            centered_neighbors = neighbors_tensor - centroids

            # 计算协方差矩阵
            if actual_k_used > 0:
                covariance_matrices = torch.bmm(
                    centered_neighbors.transpose(1, 2),
                    centered_neighbors
                ) / max(1, actual_k_used) # 除以实际邻居数，至少为1
            else:
                 # 没有邻居，协方差无意义，设为零矩阵
                 covariance_matrices = torch.zeros(current_batch_size, 3, 3, device=device)

            # 对每个协方差矩阵执行SVD分解
            try:
                # 添加小的对角扰动以提高数值稳定性
                # covariance_matrices += torch.eye(3, device=device) * 1e-6
                U, S, V = torch.linalg.svd(covariance_matrices)
                # 法向量是对应最小奇异值的右奇异向量
                batch_normals = -V[:, 2, :] # 
            except torch.linalg.LinAlgError as e: # 捕获特定的线性代数错误
                print(f"警告: 批次 {batch_idx+1} 批量 SVD 失败 ({e})，切换到逐个 SVD。")
                batch_normals = torch.zeros((current_batch_size, 3), device=device)
                fallback_count = 0
                default_normal_tensor = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
                for i in range(current_batch_size):
                    try:
                        # 确保协方差矩阵是有效的
                        if torch.isnan(covariance_matrices[i]).any() or torch.isinf(covariance_matrices[i]).any():
                             batch_normals[i] = default_normal_tensor
                             fallback_count += 1
                             continue

                        # 检查邻居数是否足够进行SVD
                        # if neighbors_list[i].shape[0] < 2: # 检查单个点的邻居数
                        #     batch_normals[i] = default_normal_tensor
                        #     fallback_count += 1
                        #     continue

                        U_i, S_i, V_i = torch.linalg.svd(covariance_matrices[i])
                        batch_normals[i] = V_i[:, -1] # 使用 -1 索引
                    except Exception as svd_err:
                        # 如果单个SVD也失败，使用默认法向量
                        if fallback_count < 5: # 限制打印次数
                            original_idx = valid_indices[batch_start + i]
                            print(f"警告: 点 {original_idx} 的 SVD 失败 ({svd_err})，使用默认法向量 [0,0,1]")
                        batch_normals[i] = default_normal_tensor
                        fallback_count += 1
                if fallback_count > 0:
                    print(f"批次 {batch_idx+1} 中 {fallback_count} 个点 SVD 失败或邻居不足，使用了默认法向量。")
                calculation_successful = False # 标记计算中出现问题

            # 确保法向量是单位向量
            norms = torch.norm(batch_normals, dim=1, keepdim=True)
            # 只有在 norm > 1e-10 时才进行归一化，否则保持原样 (可能是零向量或默认向量)
            valid_norm_mask = norms > 1e-10
            batch_normals = torch.where(valid_norm_mask, batch_normals / torch.clamp(norms, min=1e-10), batch_normals)
            # 对于归一化失败的向量（norm <= 1e-10），也设置为默认值
            batch_normals = torch.where(~valid_norm_mask, torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32), batch_normals)


            # 将计算结果移回CPU并转换为NumPy数组
            batch_normals_np = batch_normals.cpu().numpy()

            # 存储有效点的法向量
            valid_normals[batch_start:batch_end] = batch_normals_np

        except Exception as batch_proc_err:
            print(f"错误: 处理批次 {batch_idx+1} 时发生意外错误: {batch_proc_err}")
            print("将为这个批次的点设置默认法线 [0, 0, 1]。")
            traceback.print_exc()
            valid_normals[batch_start:batch_end] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            calculation_successful = False # 标记计算中出现问题
            continue # 继续处理下一个批次


        # 清理GPU内存
        if device == 'cuda' and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    print(f"有效点初始法向量计算完成，用时: {time.time() - start_time:.2f} 秒")
    if not calculation_successful:
        print("警告: 法向量计算过程中部分点使用了默认值或跳过，结果可能不完整或不准确。")




    print(f"总计算时间: {time.time() - start_time:.2f} 秒")

    # 返回计算出的有效点法向量、它们的原始索引和 NaN 点的原始索引
    return valid_normals, valid_indices, nan_indices


import torch
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


# --- 辅助函数 orient_normals_by_neighbor_vectors 和 orient_normals_by_viewpoint ---

def orient_normals_by_neighbor_vectors(points, normals, kdtree=None, k=30):
    """
    使用邻域距离向量确保法向量朝外 (处理传入的 points 和 normals)

    参数:
        points: np.ndarray, 形状为 (n_valid, 3) 的有效点云数据
        normals: np.ndarray, 形状为 (n_valid, 3) 的有效点初始法向量数据
        kdtree: KDTree对象，应基于传入的 points 构建
        k: int, 用于计算邻域距离向量的邻近点数量
    """
    print(f"使用邻域距离向量方法调整法向量方向，邻域大小 k={k}...")

    n_points = points.shape[0] # 这是有效点的数量
    if n_points == 0:
        print("没有点需要调整方向。")
        return normals

    oriented_normals = normals.copy()

    # 如果没有提供KDTree，则构建一个 (理论上应该由主函数传入)
    if kdtree is None:
        print("警告: 未提供 KDTree，正在重新构建...")
        try:
            kdtree = KDTree(points)
        except Exception as e:
            print(f"错误: 重建 KD 树失败: {e}")
            traceback.print_exc()
            print("无法调整方向，返回原始法向量。")
            return oriented_normals


    # 查找每个点的k个最近邻
    k_query = min(k + 1, n_points)
    if k_query <= 1:
         print("警告: 点数不足，无法执行邻域方向调整。")
         return oriented_normals

    try:
        _, nn_indices = kdtree.query(points, k=k_query)
    except Exception as e:
        print(f"错误: 邻域查询失败: {e}")
        traceback.print_exc()
        print("无法调整方向，返回原始法向量。")
        return oriented_normals


    # 计算邻域距离向量并调整法向量方向
    flipped_count = 0
    for i in range(n_points):
        # 获取邻域点(排除点本身)
        neighbors = points[nn_indices[i, 1:k_query]]

        if neighbors.shape[0] == 0: continue # 没有邻居，无法调整

        # 计算当前点到邻域点的向量
        vectors_to_neighbors = neighbors - points[i]

        # 计算平均距离向量
        avg_vector = np.mean(vectors_to_neighbors, axis=0)

        # 如果平均距离向量不是零向量
        avg_norm = np.linalg.norm(avg_vector)
        if avg_norm > 1e-10:
            # 计算法向量与平均距离向量的点积
            # 如果点积为正，说明法向量指向"内部"，需要翻转
            # (假设点云表面局部是凸的，平均向量指向外部)
            if np.dot(oriented_normals[i], avg_vector) > 0:
                oriented_normals[i] = -oriented_normals[i]
                flipped_count += 1
        # else: # 如果平均向量是零向量，无法判断，保持原样


    print(f"邻域向量方法：共翻转了 {flipped_count}/{n_points} 个有效点法向量 ({flipped_count/n_points*100:.2f}%)")
    return oriented_normals


# --- 更新保存和可视化函数，使其接收有效点和法向量 ---

def save_point_cloud_with_normals(points, normals, filename="point_cloud_with_normals.ply"):
    """
    保存【有效】点云及其对应的法向量到PLY文件。
    假设输入的 points 和 normals 已经是匹配的有效数据。
    """
    n_points = points.shape[0]
    if n_points == 0:
        print("错误: 点云为空，无法保存。")
        return None
    if points.shape[0] != normals.shape[0]:
        print(f"错误: 点 ({points.shape[0]}) 和法向量 ({normals.shape[0]}) 数量不匹配，无法保存。")
        return None
    if np.isnan(points).any() or np.isnan(normals).any():
         print(f"警告: 输入到保存函数的数据仍包含 NaN，这不符合预期。将尝试替换为0保存。")
         points = np.nan_to_num(points, nan=0.0)
         normals = np.nan_to_num(normals, nan=0.0)


    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 尝试保存文件
    try:
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True) # 使用 ASCII 格式
        print(f"有效点云 (共 {n_points} 个点) 已保存至: {filename}")
        return pcd
    except Exception as e:
        print(f"保存点云到 {filename} 时出错: {e}")
        traceback.print_exc()
        print("尝试不带法线保存...")
        try:
             pcd_no_normals = o3d.geometry.PointCloud()
             pcd_no_normals.points = o3d.utility.Vector3dVector(points)
             alt_filename = filename.replace(".ply", "_no_normals.ply")
             o3d.io.write_point_cloud(alt_filename, pcd_no_normals, write_ascii=True)
             print(f"仅坐标的点云已尝试保存至: {alt_filename}")
        except Exception as e2:
             print(f"不带法线保存也失败: {e2}")
        return None

# (之前的代码，包括 imports, compute_point_cloud_normals_remove_nan,
#  orient_normals_by_neighbor_vectors, orient_normals_by_viewpoint,
#  save_point_cloud_with_normals)
# ...

# --- 更新示例函数以使用新函数和返回结构 ---


# --- 修改后的主函数 ---
def example_from_file(  output_folder, points_npy_path, ply_file_path=None ):
    """
    加载点云 (优先NPY)，计算法向量 (移除NaN)，保存有效点云和法向量，
    并生成可视化结果。所有输出保存在实际加载的输入文件所在的目录下。

    Args:
        ply_file_path (str): PLY文件的路径。
        points_npy_path (str, optional): NPY文件的路径。如果提供且存在，将优先加载此文件。
                                        Defaults to None.
    """
    points_input = None
    source_info = ""
    actual_input_path = "" # 用于存储实际加载的文件路径

    try:
        # 确定要加载的文件路径和来源信息
        if points_npy_path and os.path.exists(points_npy_path):
            actual_input_path = points_npy_path
            source_info = f"NumPy 文件 '{os.path.basename(points_npy_path)}'"
        elif os.path.exists(ply_file_path):
            actual_input_path = ply_file_path
            source_info = f"PLY 文件 '{os.path.basename(ply_file_path)}'"
        else:
            # 根据调用前的检查，这里理论上不应该发生，但作为健壮性检查
            print(f"错误: 输入文件均不存在。PLY: '{ply_file_path}', NPY: '{points_npy_path}'")
            return

        print(f"尝试从 {source_info} 加载点云...")
        if actual_input_path.endswith('.npy'):
            points_input = np.load(actual_input_path)
            # 确保数据类型是 float32
            if points_input.dtype != np.float32:
                 points_input = points_input.astype(np.float32)
        elif actual_input_path.endswith('.pt'):
            points_input = torch.load(actual_input_path)
            points_input = np.array(points_input)
            # 确保数据类型是 float32
            if points_input.dtype != np.float32:
                 points_input = points_input.astype(np.float32)
        elif actual_input_path.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(actual_input_path)
            points_input = np.asarray(pcd.points, dtype=np.float32)
        else:
             print(f"错误: 不支持的文件类型 '{actual_input_path}'") # 理论上不会到这里
             return

        print(f"成功从 {source_info} 加载点云，原始点数: {len(points_input)}")

        if points_input is None or points_input.shape[0] == 0:
             print("错误: 加载的点云为空。")
             return
        if points_input.ndim != 2 or points_input.shape[1] != 3:
             print(f"错误: 加载的点云形状不正确: {points_input.shape}，应为 (n, 3)。")
             return

        n_original_points = len(points_input)

        # --- 检查并报告 NaN (在计算前) ---
        initial_nan_mask = np.isnan(points_input).any(axis=1)
        initial_nan_count = np.sum(initial_nan_mask)
        if initial_nan_count > 0:
            print(f"加载的点云中包含 {initial_nan_count} 个含有 NaN 的点。")
        # ---

        # 计算法向量 (移除 NaN 点)
        # 注意：确保你的 compute_point_cloud_normals_remove_nan 函数存在且可用
        valid_normals, valid_indices, nan_indices_computed = compute_point_cloud_normals_remove_nan(
            points_input,
            k=200,
            orientation_k=30,
            orientation_method='neighbor_vector',
            batch_size= int(5e5),
            device='cuda' # 确保你的函数支持这个设备
        )

        # 获取有效点
        # np.save(os.path.join(output_folder, "valid_normals.npy"), valid_normals)
        # np.save(os.path.join(output_folder, "valid_indices.npy"), valid_indices)
        valid_points = points_input[valid_indices]
        n_valid_points = len(valid_points)
        # 注意：nan_indices_computed 是相对于原始 points_input 的索引
        n_removed_points = len(nan_indices_computed)

        print(f"法向量计算完成。有效点数: {n_valid_points}, 移除的 NaN 点数: {n_removed_points}")

        if n_valid_points > 0:
            # --- 确定输出路径 ---
            # 获取实际加载文件的目录
            output_directory = os.path.dirname(actual_input_path)
            # 获取实际加载文件的基本名称（无扩展名）
            output_base_filename = os.path.splitext(os.path.basename(actual_input_path))[0]

            # 构建输出文件的完整路径
            output_file_ply = os.path.join(output_folder, output_base_filename + "_valid_with_normals.ply")
            output_file_png = os.path.join(output_folder, output_base_filename + "_valid_normals.png")
            output_dir_views = os.path.join(output_directory, output_base_filename + "_valid_views")
            output_indices = os.path.join(output_folder, output_base_filename + "_valid_indice.npy")
            np.save(output_indices, valid_indices)


            print(f"输出将保存在目录: {output_directory}")
            print(f"  - 有效点云PLY: {os.path.basename(output_file_ply)}")
            print(f"  - 单视图PNG: {os.path.basename(output_file_png)}")
            print(f"  - 多视图目录: {os.path.basename(output_dir_views)}")

            # 保存带法向量的【有效】点云
            # 注意：确保你的 save_point_cloud_with_normals 函数存在且可用
            np.save("valid_normals.npy", valid_normals)
            saved_pcd = save_point_cloud_with_normals(valid_points, valid_normals, output_file_ply)

            # if saved_pcd: # 只有成功保存才进行可视化
            #     # 使用Matplotlib可视化【有效】点云并保存图片
            #     # 注意：确保你的 visualize_point_cloud_matplotlib 函数存在且可用
            #     visualize_point_cloud_matplotlib(valid_points, valid_normals, sample_ratio=0.01,
            #                                     output_path=output_file_png, # 使用新路径
            #                                     title=f"点云法向量 - {os.path.basename(output_base_filename)} (移除NaN后)",
            #                                     original_total_points=n_original_points,
            #                                     removed_nan_count=n_removed_points)

            #     # 生成多视角图像 (使用有效点)
            #     # 注意：确保你的 visualize_point_cloud_multi_views 函数存在且可用
            #     visualize_point_cloud_multi_views(valid_points, valid_normals, output_dir=output_dir_views, # 使用新路径
            #                                      sample_ratio=0.01, n_views=6,
            #                                      original_total_points=n_original_points,
            #                                      removed_nan_count=n_removed_points)
            # else:
            #     print("未能成功保存有效点云，跳过可视化。")
        else:
            print("没有有效的点可以保存或可视化。")

        print(f"处理来自 {source_info} 的点云完成。")

    except FileNotFoundError:
        print(f"错误: 无法找到或访问文件 '{actual_input_path}'。请检查路径和权限。")
        traceback.print_exc()
    except ImportError as ie:
         print(f"错误：缺少必要的库。请确保安装了 numpy, open3d 等。错误信息：{ie}")
         traceback.print_exc()
    except Exception as e:
        print(f"处理文件 '{actual_input_path or ply_file_path or points_npy_path}' 时发生严重错误: {e}")
        traceback.print_exc() # 打印详细错误堆栈
        
# 主函数
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()
    # --- 运行文件示例 ---
    print("--- 运行文件示例 ---")
    # !!! 修改为你实际的文件路径 !!!
    # 指定 PLY 文件路径 (作为备用)
    ply_file = ""

    npy_file = args.npy_path
    output_folder = args.output_folder
    
    # 检查文件是否存在，避免因路径错误导致不运行
    if (npy_file and os.path.exists(npy_file)) or os.path.exists(ply_file):
        example_from_file(output_folder, npy_file)
    else:
        print(f"错误: 示例文件路径无效，请修改 'ply_file' 或 'npy_file' 的值。")
        print(f"PLY 路径: {ply_file}")
        print(f"NPY 路径: {npy_file}")

    print("\n所有示例运行完毕。")

