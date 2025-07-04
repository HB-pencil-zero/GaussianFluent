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
import torch.nn.functional as F
from pytorch3d.ops import knn_points
import time
import traceback


import torch.nn.functional as F
from pytorch3d.ops import knn_points
from scipy.spatial import KDTree

def compute_point_cloud_normals_remove_nan(
    points, k=30, orientation_k=None, batch_size=204800, device='cuda',
    orientation_method='neighbor_vector'
):
    """
    优化版本的点云法向量计算函数，使用PyTorch3d加速计算，并包含详细计时。
    
    参数和返回值与原函数相同。
    """
    # ==================== 1. 总计时开始 ====================
    total_start_time = time.time()
    
    n_points = points.shape[0]
    original_indices = np.arange(n_points)

    # --- NaN 处理 ---
    nan_handling_start = time.time()
    nan_mask = np.isnan(points).any(axis=1)
    valid_mask = ~nan_mask
    valid_indices = original_indices[valid_mask]
    nan_indices = original_indices[nan_mask]
    n_valid_points = valid_indices.shape[0]
    nan_handling_time = time.time() - nan_handling_start

    if n_valid_points == 0:
        print(f"法向量计算总耗时: {time.time() - total_start_time:.4f}s (无有效点)")
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=int), original_indices

    valid_points = points[valid_mask]
    valid_normals = np.zeros((n_valid_points, 3), dtype=np.float32)

    # --- 数据准备和设备检查 ---
    device_setup_start = time.time()
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，自动切换到CPU。")
        device = 'cpu'
    
    # 将点云转换为PyTorch张量
    points_tensor = torch.tensor(valid_points, dtype=torch.float32, device=device)
    device_setup_time = time.time() - device_setup_start
    
    # ==================== 2. 核心计算循环计时开始 ====================
    computation_start_time = time.time()
    
    # 使用PyTorch3d进行批处理KNN搜索
    total_batches = int(np.ceil(n_valid_points / batch_size))
    
    print(f"开始计算 {n_valid_points} 个点的法向量，共 {total_batches} 个批次...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_valid_points)
        current_batch = points_tensor[start_idx:end_idx]
        
        # 扩展维度以符合PyTorch3d的输入要求
        current_batch_gpu = current_batch.unsqueeze(0)  # [1, B, 3]
        all_points_gpu = points_tensor.unsqueeze(0)     # [1, N, 3]
        
        # 计算KNN - PyTorch3d优化实现
        knn_result = knn_points(current_batch_gpu, all_points_gpu, K=k+1)
        
        # 获取邻居点索引和坐标
        nn_idx = knn_result.idx[0, :, 1:]  # 排除点本身
        nn_points = all_points_gpu[0][nn_idx]   # [B, k, 3]
        
        # 计算中心点
        centroids = torch.mean(nn_points, dim=1, keepdim=True)
        
        # 中心化邻居点
        centered = nn_points - centroids
        
        # 计算协方差矩阵
        covariances = torch.bmm(centered.transpose(1, 2), centered) / k
        
        # 批量SVD计算
        try:
            # PyTorch 1.8+ 推荐使用 torch.linalg.svd
            _, _, V = torch.linalg.svd(covariances)
            # V的形状是 [B, 3, 3]，法向量是最后一个右奇异向量
            # batch_normals = V[:, :, 2]
            batch_normals = V[:, 2, :]
        except Exception as e:
            # 兼容性或失败时的备用方法
            # print(f"警告: 批处理SVD失败 ({e}), 正在使用逐个计算的备用方案...")
            batch_normals = torch.zeros((current_batch_gpu.shape[1], 3), device=device)
            for i in range(current_batch_gpu.shape[1]):
                try:
                    _, _, V_single = torch.linalg.svd(covariances[i])
                    batch_normals[i] = V_single[:, 2]
                except:
                    # 如果单个SVD也失败，则提供一个默认法线
                    batch_normals[i] = torch.tensor([0.0, 0.0, 1.0], device=device)

        # 归一化法向量
        norms = torch.norm(batch_normals, dim=1, keepdim=True)
        # 修正：确保 valid_mask 和 batch_normals 的设备一致
        default_normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        batch_normals = torch.where(norms > 1e-7, batch_normals / norms, default_normal)

        # 存储结果
        valid_normals[start_idx:end_idx] = batch_normals.cpu().numpy()
        
        # 定期清理GPU缓存
        if device == 'cuda' and batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    # ==================== 核心计算循环计时结束 ====================
    computation_time = time.time() - computation_start_time

    # ==================== 3. 方向调整计时开始 ====================
    orientation_start_time = time.time()
    
    # 法向量方向一致性调整 (可选)
    if orientation_method != 'none':
        # 先不要加法向量调整，之后再说
        # 这里可以添加方向一致性调整的实现
        # 例如使用最小生成树或视点方法
        pass
    
    orientation_time = time.time() - orientation_start_time
    
    # ==================== 4. 总计时结束并打印报告 ====================
    total_time = time.time() - total_start_time
    
    print("\n--- 法向量计算耗时报告 ---")
    print(f"总耗时: {total_time:.4f}s")
    print(f"  - NaN处理: {nan_handling_time:.4f}s")
    print(f"  - 设备与数据准备: {device_setup_time:.4f}s")
    print(f"  - 核心计算 (KNN+SVD): {computation_time:.4f}s")
    print(f"  - 方向调整: {orientation_time:.4f}s")
    print("--------------------------\n")

    return valid_normals, valid_indices, nan_indices



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
            points_input = torch.load(actual_input_path, weights_only=True)
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
            batch_size= int(3e5),
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
    # ply_file = "/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model/watermelon/point_cloud/iteration_30000/point_cloud.ply" # 例如: "/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model/watermelon/point_cloud/iteration_30000/point_cloud.ply"
    # 指定优先加载的 NPY 文件路径
    # npy_file = "/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/watermelon_frame/frame_20/pos.npy" # 例如: "/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/pos_15.npy"
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

