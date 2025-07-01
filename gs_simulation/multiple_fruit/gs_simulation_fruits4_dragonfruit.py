import sys

sys.path.append("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/gaussian-splatting")
sys.path.append("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian")

import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
from tqdm import tqdm
import subprocess
# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov
from utils.shadow_utils import *
# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.lighting_utils import *


import torch

def filter_gaussian_points_by_xyz(tensor,
                                  x_threshold=None, y_threshold=None, z_threshold=None,
                                  x_greater=True, y_greater=True, z_greater=True,
                                  cov=None, cal_bias=True, delta=0.02):
    """
    筛选出 x, y 或 z 坐标大于或小于指定阈值的高斯点，并可进行基于协方差的调整。
    同时返回被排除的点。

    参数:
        tensor (torch.Tensor): 输入的 n,3 形状的 Tensor,其中每行是一个 3D 点 (x, y, z)。
        x_threshold (float or None): X 轴的阈值。如果为 None，则不进行 X 轴筛选。
        y_threshold (float or None): Y 轴的阈值。如果为 None，则不进行 Y 轴筛选。
        z_threshold (float or None): Z 轴的阈值。如果为 None，则不进行 Z 轴筛选。
        x_greater (bool): 如果 x_threshold 不为 None，True 表示筛选 x > x_threshold 的点，False 表示筛选 x < x_threshold 的点。
        y_greater (bool): 如果 y_threshold 不为 None，True 表示筛选 y > y_threshold 的点，False 表示筛选 y < y_threshold 的点。
        z_greater (bool): 如果 z_threshold 不为 None，True 表示筛选 z > z_threshold 的点，False 表示筛选 z < z_threshold 的点。
        cov (torch.Tensor or None): 如果提供，将对点进行基于协方差矩阵的调整。形状为 (n, 6)，表示每个点的协方差矩阵的上三角部分
                                    (cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz)。
        cal_bias (bool): 是否计算并应用协方差偏置调整（仅当提供了 x, y 或 z 阈值时且可解释为超平面时适用）。
        delta (float): 超平面偏置的调整量，筛选满足 threshold-delta < coordinate < threshold+delta 的点（当 cal_bias 为 True 时）。

    返回:
        torch.Tensor: 筛选后的点 Tensor (原始坐标)。
        torch.Tensor: 保留下来的点的索引。
        torch.Tensor: 被排除的点的索引。
    """
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (n, 3)")

    valid_indices = torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device)
    tensor_adjusted = tensor.clone() # 用于计算的调整后坐标

    if cov is not None and cal_bias and (x_threshold is not None or y_threshold is not None or z_threshold is not None):
        if cov.shape[0] != tensor.shape[0]:
            raise ValueError("Tensor and Covariance matrix must have the same number of points (batch size).")
        if cov.shape[1] != 6:
            raise ValueError("Covariance matrix must have shape (n, 6).")

        batch_size = cov.shape[0]
        Sigma = torch.zeros((batch_size, 3, 3), device=tensor.device, dtype=tensor.dtype)

        # 填充协方差矩阵 (对称)
        # cov is (cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz)
        Sigma[:, 0, 0] = cov[:, 0]  # Σ_xx
        Sigma[:, 0, 1] = cov[:, 1]  # Σ_xy
        Sigma[:, 0, 2] = cov[:, 2]  # Σ_xz
        Sigma[:, 1, 0] = cov[:, 1]  # Σ_yx = Σ_xy
        Sigma[:, 1, 1] = cov[:, 3]  # Σ_yy
        Sigma[:, 1, 2] = cov[:, 4]  # Σ_yz
        Sigma[:, 2, 0] = cov[:, 2]  # Σ_zx = Σ_xz
        Sigma[:, 2, 1] = cov[:, 4]  # Σ_zy = Σ_yz
        Sigma[:, 2, 2] = cov[:, 5]  # Σ_zz

        bias_adjustment = torch.zeros_like(tensor, dtype=tensor.dtype) # 累积偏置

        if x_threshold is not None:
            w_x = torch.tensor([1.0, 0.0, 0.0], device=tensor.device, dtype=tensor.dtype).unsqueeze(0).repeat(batch_size, 1) # (batch_size, 3)
            # Sigma: (batch_size, 3, 3), w_x.unsqueeze(-1): (batch_size, 3, 1)
            Sigma_w_x = torch.bmm(Sigma, w_x.unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
            # w_x.unsqueeze(1): (batch_size, 1, 3)
            wT_Sigma_w_x = torch.bmm(w_x.unsqueeze(1), Sigma_w_x.unsqueeze(-1)).squeeze(-1).squeeze(-1) # (batch_size)

            # 避免除以零或非常小的值
            valid_wT_Sigma_w_x = wT_Sigma_w_x > 1e-8
            if torch.any(valid_wT_Sigma_w_x):
                bias_x_component = torch.zeros_like(Sigma_w_x)
                sqrt_wT_Sigma_w_x_valid = torch.sqrt(wT_Sigma_w_x[valid_wT_Sigma_w_x])
                bias_x_component[valid_wT_Sigma_w_x] = Sigma_w_x[valid_wT_Sigma_w_x] / sqrt_wT_Sigma_w_x_valid.unsqueeze(-1)
                bias_adjustment = bias_adjustment + bias_x_component


        if y_threshold is not None:
            w_y = torch.tensor([0.0, 1.0, 0.0], device=tensor.device, dtype=tensor.dtype).unsqueeze(0).repeat(batch_size, 1)
            Sigma_w_y = torch.bmm(Sigma, w_y.unsqueeze(-1)).squeeze(-1)
            wT_Sigma_w_y = torch.bmm(w_y.unsqueeze(1), Sigma_w_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

            valid_wT_Sigma_w_y = wT_Sigma_w_y > 1e-8
            if torch.any(valid_wT_Sigma_w_y):
                bias_y_component = torch.zeros_like(Sigma_w_y)
                sqrt_wT_Sigma_w_y_valid = torch.sqrt(wT_Sigma_w_y[valid_wT_Sigma_w_y])
                bias_y_component[valid_wT_Sigma_w_y] = Sigma_w_y[valid_wT_Sigma_w_y] / sqrt_wT_Sigma_w_y_valid.unsqueeze(-1)
                bias_adjustment = bias_adjustment + bias_y_component


        if z_threshold is not None:
            w_z = torch.tensor([0.0, 0.0, 1.0], device=tensor.device, dtype=tensor.dtype).unsqueeze(0).repeat(batch_size, 1)
            Sigma_w_z = torch.bmm(Sigma, w_z.unsqueeze(-1)).squeeze(-1)
            wT_Sigma_w_z = torch.bmm(w_z.unsqueeze(1), Sigma_w_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)

            valid_wT_Sigma_w_z = wT_Sigma_w_z > 1e-8
            if torch.any(valid_wT_Sigma_w_z):
                bias_z_component = torch.zeros_like(Sigma_w_z)
                sqrt_wT_Sigma_w_z_valid = torch.sqrt(wT_Sigma_w_z[valid_wT_Sigma_w_z])
                bias_z_component[valid_wT_Sigma_w_z] = Sigma_w_z[valid_wT_Sigma_w_z] / sqrt_wT_Sigma_w_z_valid.unsqueeze(-1)
                bias_adjustment = bias_adjustment + bias_z_component

        tensor_adjusted = tensor - bias_adjustment # 应用总偏置

    # 使用调整后的坐标进行筛选
    current_delta_x = delta if cal_bias and cov is not None and x_threshold is not None else 0
    current_delta_y = delta if cal_bias and cov is not None and y_threshold is not None else 0
    current_delta_z = delta if cal_bias and cov is not None and z_threshold is not None else 0

    if x_threshold is not None:
        x_coords = tensor_adjusted[:, 0]
        if x_greater:
            valid_x = x_coords > (x_threshold - current_delta_x)
        else:
            valid_x = x_coords < (x_threshold + current_delta_x)
        valid_indices &= valid_x

    if y_threshold is not None:
        y_coords = tensor_adjusted[:, 1]
        if y_greater:
            valid_y = y_coords > (y_threshold - current_delta_y)
        else:
            valid_y = y_coords < (y_threshold + current_delta_y)
        valid_indices &= valid_y

    if z_threshold is not None:
        z_coords = tensor_adjusted[:, 2]
        if z_greater:
            valid_z = z_coords > (z_threshold - current_delta_z)
        else:
            valid_z = z_coords < (z_threshold + current_delta_z)
        valid_indices &= valid_z

    valid_indices_positions = torch.nonzero(valid_indices, as_tuple=False).squeeze(-1)
    invalid_indices = ~valid_indices
    invalid_indices_positions = torch.nonzero(invalid_indices, as_tuple=False).squeeze(-1)

    # 返回原始坐标的点，但基于调整后坐标的筛选结果
    return tensor[valid_indices_positions], valid_indices_positions, invalid_indices_positions




def filter_tensor_by_hyperplanes_delta(tensor, hyperplanes=None,  pos=None, cov=None, cal_bias=True, delta=0.02 ):
    """
    筛选位于超平面正负delta范围内的点，通过检查点是否满足 b-delta < wx < b+delta。

    参数:
        tensor (torch.Tensor): 输入的 n,3 形状的 Tensor,其中每行是一个 3D 点。
        delta (float): 超平面偏置的调整量，筛选满足 b-delta < wx < b+delta 的点。
        hyperplanes (list[tuple] or None): 每个超平面由 (w, b) 组成，w 是形状为 (3,) 的法向量，b 是偏置。
                                        默认值为 [(torch.tensor([1.0, 0.0, 0.0]), 0.0)]。
        cov (torch.Tensor or None): 如果提供，将对点进行基于协方差矩阵的调整。
        cal_bias (bool): 是否计算并应用协方差偏置调整。

    返回:
        torch.Tensor: 筛选后的点 Tensor。
        torch.Tensor: 保留下来的点的索引。
    """
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (n, 3)")

    # 设置默认值
    if hyperplanes is None:
        hyperplanes = [(torch.tensor([1.0, 0.0, 0.0]), 0.0)]

    # 初始化所有点均为有效
    valid_indices = torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device)

    for w, b in hyperplanes:
        w = w.to(tensor.device, dtype=tensor.dtype)
        b = torch.tensor(b, device=tensor.device, dtype=tensor.dtype)

        tensor_adjusted = tensor

        if cov is not None and cal_bias:
            batch_size = cov.shape[0]
            Sigma = torch.zeros((batch_size, 3, 3), device=tensor.device)

            # 填充下三角部分
            Sigma[:, 0, 0] = cov[:, 0]  # Σ_00
            Sigma[:, 0, 1] = cov[:, 1]  # Σ_01
            Sigma[:, 0, 2] = cov[:, 2]  # Σ_02
            Sigma[:, 1, 0] = cov[:, 1]  # Σ_10 = Σ_01
            Sigma[:, 1, 1] = cov[:, 3]  # Σ_11
            Sigma[:, 1, 2] = cov[:, 4]  # Σ_12
            Sigma[:, 2, 0] = cov[:, 2]  # Σ_20 = Σ_02
            Sigma[:, 2, 1] = cov[:, 4]  # Σ_21 = Σ_12
            Sigma[:, 2, 2] = cov[:, 5]  # Σ_22

            # 计算 Σ @ w
            Sigma_w = Sigma @ w.unsqueeze(-1)  # 形状为 [n, 3, 1]
            Sigma_w = Sigma_w.squeeze(-1)  # 形状为 [n, 3]

            # 计算 w^T @ Σ @ w
            wT_Sigma_w = (w.unsqueeze(0) @ Sigma @ w.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # 标量
            bias = Sigma_w / torch.sqrt(wT_Sigma_w).unsqueeze(-1)
            tensor_adjusted = tensor - bias  # 应用偏置调整

        # 计算超平面条件 wx
        hyperplane_values = torch.matmul(tensor_adjusted, w)

        # 筛选满足 b-delta < wx < b+delta 的点
        valid_indices &= (hyperplane_values > (b - delta)) & (hyperplane_values < (b + delta))

    # 获取满足条件的原始索引
    valid_indices_positions = torch.nonzero(valid_indices, as_tuple=True)[0]

    # 返回筛选后的点和它们的索引
    return tensor[valid_indices_positions], valid_indices_positions
 
        






wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=2.0, random_seed=42)

import sys # 导入 sys 模块以使用 sys.stdout.flush()

def run_command_realtime(command_to_run):
    """
    执行一个 shell 命令并实时打印其标准输出和标准错误。

    Args:
        command_to_run (str): 要执行的 shell 命令。

    Returns:
        int: 子进程的退出码。
    """
    print(f"--- 开始执行命令: {command_to_run} ---")
    try:
        # 使用 Popen 启动子进程
        # 将 stderr 重定向到 stdout (2>&1)
        # text=True 使 stdout/stderr 成为文本流
        # bufsize=1 设置为行缓冲模式（如果可能）
        # encoding='utf-8' 明确指定编码
        process = subprocess.Popen(
            ["bash", "-c", f"{command_to_run} 2>&1"], # 将 stderr 合并到 stdout
            stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE, # 不再需要单独处理 stderr
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace' # 处理潜在的解码错误
        )

        # 实时读取 stdout 流
        if process.stdout:
            # 使用 iter 和 readline 逐行读取，直到流结束
            for line in iter(process.stdout.readline, ''):
                print(line, end='') # 打印读取到的行，end='' 避免额外换行
                sys.stdout.flush() # 强制刷新缓冲区，确保立即显示

        # 等待子进程结束
        process.wait()

        # 检查子进程的退出码
        return_code = process.returncode
        print(f"\n--- 命令执行完毕，退出码: {return_code} ---")
        if return_code != 0:
            print(f"警告：命令执行可能出错，退出码为 {return_code}")

        return return_code

    except FileNotFoundError:
        print(f"错误：无法找到 'bash' 命令或指定的程序。请检查路径和环境。")
        return -1
    except Exception as e:
        print(f"执行命令时发生错误: {e}")
        return -1


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_from_saved", action="store_true", help="Load simulation data from saved .pt files instead of running the simulation.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        # 复制config文件到输出目录
        import shutil
        config_name = os.path.basename(args.config)
        shutil.copy2(args.config, os.path.join(args.output_path, config_name))

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    
    gaussians._xyz = gaussians._xyz[mask, :]
    gaussians._features_dc = gaussians._features_dc[mask, :]
    gaussians._features_rest = gaussians._features_rest[mask, :]
    gaussians._opacity = gaussians._opacity[mask, :]
    gaussians._scaling = gaussians._scaling[mask, :]
    gaussians._rotation = gaussians._rotation[mask, :]

    mask_opa = mask
    
    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]
    

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    biases = [0.25, 0.19 , 0.16, 0.16 ]  #pineple
    frames = [60,  60 ,60, 90] 
    # biases = [ 0.12 ] 
    # frames = [100] 
    select_id = torch.tensor([], dtype=torch.int).cuda()
    frame_sum = 0
    frame_num = 0 
    select_id__= []
    # select_id__.append(filter_gaussian_points_by_xyz(tensor = mpm_init_pos, y_greater=True, x_threshold=1.0, z_greater=False, z_threshold=1.0)[1] )
    # select_id__.append(filter_gaussian_points_by_xyz(tensor = mpm_init_pos, x_greater=True, x_threshold=0.0, z_greater=False, z_threshold=1.0)[1] )

    select_id__.append(filter_gaussian_points_by_xyz(tensor = mpm_init_pos, y_greater=True, x_threshold=1.03)[1] )
    select_id__.append(filter_gaussian_points_by_xyz(tensor = mpm_init_pos, x_greater=True, x_threshold=0.97)[1] )
    select_id__.append(filter_gaussian_points_by_xyz(tensor = mpm_init_pos, x_greater=True, x_threshold=0)[1] )
    select_id__.append(filter_gaussian_points_by_xyz(tensor = mpm_init_pos, x_greater=True, x_threshold=0)[1] )
    mpm_init_pos[:, 2] += 0.3
    for k in range(len(biases)):
        pos_flag = True
        cal_bias = False
        x_tag = 1 
        y_tag = 0
        base_bias = 1
        bias = biases[k]
        frame_sum += frame_num
        frame_num = frames[k]
        # _ , select_id__ = filter_tensor_by_hyperplanes_delta(mpm_init_pos, [(torch.tensor([x_tag, y_tag, 0.0]), base_bias + bias)] , cov=mpm_init_cov, pos=[pos_flag], cal_bias=cal_bias, delta=0.05)
        # select_id = torch.concat([select_id, select_id__])
        
        
        select_id = torch.concat([select_id, select_id__[k]])
        select_id = torch.unique(select_id)






        # set up the mpm solver
        mpm_solver = MPM_Simulator_WARP(10)
        
        # mpm_init_pos[:, 2] = mpm_init_pos[:, 2] - 0.5
        # scale_factor =  1 / 3 
        # mpm_init_cov *= scale_factor**2
        # mpm_init_pos =  mpm_init_pos.mean(dim = 0) + scale_factor *(mpm_init_pos -  mpm_init_pos.mean(dim = 0))

        mpm_solver.load_initial_data_from_torch(
            mpm_init_pos[select_id],
            mpm_init_vol[select_id],
            mpm_init_cov[select_id],
            n_grid=material_params["n_grid"],
            grid_lim=material_params["grid_lim"],
        )
        mpm_solver.set_parameters_dict(material_params)
        
        # beta = mpm_solver.mpm_model.beta.numpy()
        # mask1 = (gaussians._features_dc < -0.5).all(axis=2).cpu().numpy().squeeze()
        # mask2 = (gaussians._features_dc > -2.5).all(axis=2).cpu().numpy().squeeze()
        # mask_ = mask1 & mask2
        # selected_xyz = gaussians._xyz[mask_]
        # mask1 = (gaussians._features_dc > 1.0).all(axis=2).cpu().numpy().squeeze()
        # mask2 = (gaussians._features_dc < 4).all(axis=2).cpu().numpy().squeeze()
        # mask__ = mask1 & mask2
        
        

        # # 2. 创建KNN搜索器
        # all_xyz_np = gaussians._xyz.detach().cpu().numpy()
        # knn = NearestNeighbors(radius=0.03, algorithm='ball_tree')
        # knn.fit(all_xyz_np)

        # # 3. 查找距离小于0.05的所有点的索引
        # selected_xyz_np = selected_xyz.detach().cpu().numpy()
        # neighbors_indices = knn.radius_neighbors(selected_xyz_np, 0.03, return_distance=False)

        # # 4. 将所有邻居点的索引展平并去重
        # all_neighbors = np.unique(np.concatenate(neighbors_indices))

        # # 5. 创建新的mask，包含所有邻近点
        # new_mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device)
        # new_mask[all_neighbors] = True
        
        
        # beta[new_mask.cpu().numpy()] = 3000000000
        # beta[mask__] = 2
        
        # mpm_solver.mpm_model.beta.assign(beta)
        
        
        
        # Note: boundary conditions may depend on mass, so the order cannot be changed!
        set_boundary_conditions(mpm_solver, bc_params, time_params)

        mpm_solver.finalize_mu_lam()

        mpm_solver.import_particle_v_from_torch(torch.zeros(mpm_init_pos.shape[0], 3, device='cuda').add_(torch.tensor([0.0, 0.0, 0.0], device='cuda')))
        # camera setting
        mpm_space_viewpoint_center = (
            torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
        )
        mpm_space_vertical_upward_axis = (
            torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
            .reshape((1, 3))
            .cuda()
        )
        (
            viewpoint_center_worldspace,
            observant_coordinates,
        ) = get_center_view_worldspace_and_observant_coordinate(
            mpm_space_viewpoint_center,
            mpm_space_vertical_upward_axis,
            rotation_matrices,
            scale_origin,
            original_mean_pos,
        )

        # run the simulation
        if args.output_ply or args.output_h5:
            directory_to_save = os.path.join(args.output_path, "simulation_ply")
            if not os.path.exists(directory_to_save):
                os.makedirs(directory_to_save)

            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                0,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )
        
        dx = material_params["grid_lim"] / material_params['n_grid']
        substep_dt = time_params["substep_dt"]
        E = material_params['E']
        nu = material_params['nu']
        rho = material_params['density']
        def evaluate_sound_speed_linear_elasticity_analysis(E, nu, rho):
            return np.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho))
        cfl = 0.6
        substep_dt = cfl * dx / evaluate_sound_speed_linear_elasticity_analysis(E, nu, rho)
        frame_dt = time_params["frame_dt"]
        # frame_num = time_params["frame_num"]
        step_per_frame = int(frame_dt / substep_dt)
        opacity_render = opacity
        shs_render = shs
        height = None
        width = None
        ti.reset()
        # torch.cuda.empty_cache()
        color_flag = False
        # color_flag = True
        light_flag = True
        end_frame = 1000

        
        azimuth_list , elvation_list, radius_list, center_list = generate_and_append_ellipse_path(start_azimuth=470, azimuth_max_delta= 20, start_elevation=10, path_radius=7.5, path_center=np.array([2.7797, -1.9686, -0.6975]), elevation_max_delta=10, stage_num=10)    

        azimuth_list , elvation_list, radius_list, center_list = linear_transition_and_append(
        start_pose=(470, 10, 7.5, (2.7797, -1.9686, -0.6975)),
        target_pose=(470, 30, 12.5, (2, 1, -1.2)),
        stage_num=30,
        existing_azimuths=azimuth_list , existing_elevations=elvation_list, existing_radii=radius_list, existing_centers=center_list )

        azimuth_list , elvation_list, radius_list, center_list = linear_transition_and_append(
        start_pose=(110, 30, 12.5, (2, 1, -1.2)),
        target_pose=(290, 30, 12.5, (2, 1, -1.2)),
        stage_num=90,
        existing_azimuths=azimuth_list , existing_elevations=elvation_list, existing_radii=radius_list, existing_centers=center_list )

        azimuth_list , elvation_list, radius_list, center_list = linear_transition_and_append(
        start_pose=(290, 30, 12.5, (2, 1, -1.2)),
        target_pose=(470, 30, 12.5, (2, 1, -1.2)),
        stage_num=90,
        existing_azimuths=azimuth_list , existing_elevations=elvation_list, existing_radii=radius_list, existing_centers=center_list )

        pos2, cov3D2, rot2, opacity2, shs2 = load_prop_dict("gs_simulation/multiple_fruit/save_tensor/pineapple.pt")
        for frame in tqdm(range(frame_num)):
            # frame = 16 + frame
            frame = frame + frame_sum
            
            try :
                start_id = 0
                if frame >= start_id:
                    camera_params['init_azimuthm'] = azimuth_list[frame - start_id]
                    camera_params['init_elevation'] = elvation_list[frame - start_id]
                    camera_params['init_radius'] = radius_list[frame - start_id]
                    viewpoint_center_worldspace = center_list[frame - start_id]
            except Exception as e:
                camera_params['init_azimuthm'] = azimuth_list[-1]
                camera_params['init_elevation'] = elvation_list[-1]
                camera_params['init_radius'] = radius_list[-1]
                viewpoint_center_worldspace = center_list[-1]
                
            current_camera = get_camera_view(
                model_path,
                default_camera_index=camera_params["default_camera_index"],
                center_view_world_space= viewpoint_center_worldspace, #np.array([2.7797, -1.9686, -0.6975]),  # np.array([2, 1, -1.2]), #  np.array([2.8, -2, -1.2])
                observant_coordinates=observant_coordinates,
                show_hint=camera_params["show_hint"],
                init_azimuthm=camera_params["init_azimuthm"],
                init_elevation=camera_params["init_elevation"],
                init_radius=camera_params["init_radius"],
                move_camera=camera_params["move_camera"],
                current_frame=frame,
                delta_a=camera_params["delta_a"],
                delta_e=camera_params["delta_e"],
                delta_r=camera_params["delta_r"],
            )
            rasterize = initialize_resterize(
                current_camera, gaussians, pipeline, background
            )
            
            if not args.load_from_saved :
                pass
                # for step in range(step_per_frame):
                #     mpm_solver.p2g2p(step, substep_dt, device=device, flip_pic_ratio=material_params['flip_pic_ratio'])

            if args.output_ply or args.output_h5:
                save_data_at_frame(
                    mpm_solver,
                    directory_to_save,
                    frame + 1,
                    save_to_ply=args.output_ply,
                    save_to_h5=args.output_h5,
                )
                

            indice = select_id
            if args.render_img:
                # Define a new base directory within args.output_path for detailed tensor data
                per_frame_tensor_output_base_dir = os.path.join(args.output_path, "gaussian_frame_data")
                os.makedirs(per_frame_tensor_output_base_dir, exist_ok=True)

                # Create a subdirectory for the current frame's tensors
                current_frame_tensor_dir = os.path.join(per_frame_tensor_output_base_dir, f"frame_{frame:05d}") # e.g., frame_00000, frame_00001
                os.makedirs(current_frame_tensor_dir, exist_ok=True)
                # 获取初始数据
                if not args.load_from_saved :
                    pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                    cov3D = mpm_solver.export_particle_cov_to_torch()
                    rot = mpm_solver.export_particle_R_to_torch()
                    cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
                    rot = rot.view(-1, 3, 3)[:gs_num].to(device)



                    mpm_init_pos[indice] = pos
                    mpm_init_cov[indice] = cov3D
                    identity_matrix_3x3 = torch.eye(3, dtype=mpm_init_pos.dtype, device=mpm_init_pos.device)
                    identity_flat_9d = identity_matrix_3x3.flatten()
                    mpm_init_rot = identity_flat_9d.unsqueeze(0).expand(mpm_init_pos.shape[0], -1).reshape(-1, 3, 3).clone()
                    mpm_init_rot[indice] = rot
                    pos = mpm_init_pos
                    cov3D = mpm_init_cov
                    rot = mpm_init_rot

                    # 应用变换
                    pos = apply_inverse_rotations(
                        undotransform2origin(
                            undoshift2center111(pos), scale_origin, original_mean_pos
                        ),
                        rotation_matrices,
                    )
                    cov3D = cov3D / (scale_origin * scale_origin)
                    cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
                    opacity = opacity_render
                    shs = shs_render



                    # 如果有sim_area,添加未选择的点
                    if preprocessing_params["sim_area"] is not None:
                        pos = torch.cat([pos, unselected_pos], dim=0)
                        cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                        opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                        shs = torch.cat([shs_render, unselected_shs], dim=0)
                    
                else:
                    # 从当前帧的目录加载保存的张量数据
                    tensors_to_load = {
                        "pos.pt": None,
                        "rot.pt": None,
                        "cov3D.pt": None, 
                        "shs.pt": None,
                        "opacity.pt": None
                    }

                    # 加载每个张量
                    for filename, _ in tensors_to_load.items():
                        load_path = os.path.join(current_frame_tensor_dir, filename)
                        if os.path.exists(load_path):
                            tensors_to_load[filename] = torch.load(load_path, weights_only=True).to(device)
                        else:
                            print(f"警告: 找不到文件 {filename} ,帧 {frame}")

                    # 将加载的张量赋值给对应变量
                    pos = tensors_to_load["pos.pt"]
                    rot = tensors_to_load["rot.pt"]
                    cov3D = tensors_to_load["cov3D.pt"]
                    shs = tensors_to_load["shs.pt"]
                    opacity = tensors_to_load["opacity.pt"]
                    
                light_output_base_dir = os.path.join(args.output_path, "normal_and_light")
                os.makedirs(light_output_base_dir, exist_ok=True)
                
                current_frame_light_dir = os.path.join(light_output_base_dir, f"frame_{frame:05d}") # e.g., frame_00000, frame_00001
                os.makedirs(current_frame_light_dir, exist_ok=True)
                
                if color_flag:
                    if  light_flag : 
                                        

                        npy_path = os.path.join(current_frame_tensor_dir, 'pos.pt')
                        opacity_path = os.path.join(current_frame_tensor_dir, 'opacity.pt')
                        output_folder = current_frame_light_dir
                        # np.save("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/watermelon_frame/frame_20/pos.npy", pos.detach().cpu().numpy())
                        command = f'cd /root/autodl-tmp/debug_physgaussian/cdmpmGaussian/ && source $(conda info --base)/etc/profile.d/conda.sh && conda activate PhysGaussian && python normal_vector_proc_nan.py --npy_path {npy_path} --output_folder {output_folder}'
                        run_command_realtime(command)

                        normal_path = os.path.join(output_folder, 'pos_valid_with_normals.ply')
                        valid_indice_path = os.path.join(output_folder, "pos_valid_indice.npy")
                        command = f'cd /root/autodl-tmp/debug_physgaussian/cdmpmGaussian/ && source $(conda info --base)/etc/profile.d/conda.sh && conda activate PhysGaussian && python phong_model_wm_shs_15.py \
                                    --npy_path {normal_path} --output_folder {output_folder}  --opacity_path {opacity_path} --valid_indice_path {valid_indice_path}'
                        run_command_realtime(command)

                    valid_indice = torch.from_numpy(np.load(os.path.join(output_folder, "pos_valid_indice.npy"))).to("cuda")
                    colors = torch.from_numpy(np.load(os.path.join(output_folder, "phong_colors.npy"))).to("cuda").reshape(-1 , 3).float()

                
                # pos_ = pos
                # cov3D_ = cov3D
                # rot_ = rot
                # opacity_ =  opacity
                # shs_ = shs
                
                
                    
                pos_ = torch.concat([pos  + torch.Tensor([2.8, -2,-1.2]).cuda(), pos2],dim =0 ) 
                cov3D_ = torch.concat([cov3D, cov3D2],dim =0 )
                rot_ = torch.concat([rot, rot2],dim =0 )
                opacity_ = torch.concat([opacity , opacity2],dim =0 )
                shs_ = torch.concat([shs, shs2],dim =0 )


                # pos_, cov3D_, rot_, opacity_, shs_ = load_and_concat_prop_dict("gs_simulation/multiple_fruit/save_tensor/pineapple.pt", pos_, cov3D_, rot_, opacity_, shs_, [2.8, -2,-1.2])
                colors_precomp = convert_SH(shs_, current_camera, gaussians, pos_, rot_)
                if color_flag:
                    colors_precomp[ valid_indice ]  = colors.clone()


            
                
                
                rendering, raddi, point_xy = rasterize(
                    means3D=pos_,
                    means2D=init_screen_points,
                    shs=None,
                    colors_precomp=colors_precomp.float(),
                    opacities=opacity_, 
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D_,
                )
                


                # Tensors to save for the current frame.
                # Ensure these variables hold the correct data for the *current frame* at this point.
 
                cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                if height is None or width is None:
                    height = cv2_img.shape[0] // 2 * 2
                    width = cv2_img.shape[1] // 2 * 2
                assert args.output_path is not None
                cv2.imwrite(
                    os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                    255 * cv2_img,
                )
                if frame > end_frame:
                    light_flag = False

                    # 获取初始数据
                    pos = mpm_solver.export_particle_x_to_torch().to(device)
                    cov3D = mpm_solver.export_particle_cov_to_torch()
                    rot = mpm_solver.export_particle_R_to_torch()
                    cov3D = cov3D.view(-1, 6).to(device)
                    rot = rot.view(-1, 3, 3).to(device)


                    mpm_init_pos[indice] = pos
                    mpm_init_cov[indice] = cov3D
                    identity_matrix_3x3 = torch.eye(3, dtype=mpm_init_pos.dtype, device=mpm_init_pos.device)
                    identity_flat_9d = identity_matrix_3x3.flatten()
                    mpm_init_rot = identity_flat_9d.unsqueeze(0).expand(mpm_init_pos.shape[0], -1).reshape(-1, 3, 3).clone()
                    mpm_init_rot[indice] = rot
                    pos = mpm_init_pos
                    cov3D = mpm_init_cov
                    rot = mpm_init_rot

                                
                    # 应用变换
                    pos = apply_inverse_rotations(
                        undotransform2origin(
                            undoshift2center111(pos), scale_origin, original_mean_pos
                        ),
                        rotation_matrices,
                    )
                    
                    
                    cov3D = cov3D / (scale_origin * scale_origin)
                    cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)




                    # 如果有sim_area,添加未选择的点
                    if preprocessing_params["sim_area"] is not None:
                        pos = torch.cat([pos, unselected_pos], dim=0)
                        cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                        opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                        shs = torch.cat([shs_render, unselected_shs], dim=0)

                    # opacity = opacity_render[select_id]
                    # shs = shs_render[select_id]
                    
                    opacity = opacity_render
                    shs = shs_render

                    pos_ = torch.concat([pos, pos2],dim =0 )
                    cov3D_ = torch.concat([cov3D, cov3D2],dim =0 )
                    rot_ = torch.concat([rot, rot2],dim =0 )
                    opacity_ = torch.concat([opacity_render, opacity_render2],dim =0 )
                    shs_ = torch.concat([shs_render, shs_render2],dim =0 )


                    colors_precomp = convert_SH(shs_, current_camera, gaussians, pos_, rot_)

                    

                    
                    
                    rendering, raddi, point_xy = rasterize(
                        means3D=pos_,
                        means2D=init_screen_points,
                        shs=None,
                        colors_precomp=colors_precomp.float(),
                        opacities=opacity_, 
                        scales=None,
                        rotations=None,
                        cov3D_precomp=cov3D_,
                    )
                    
                    
                    per_frame_tensor_output_base_dir = os.path.join(args.output_path, "gaussian_frame_data")
                    os.makedirs(per_frame_tensor_output_base_dir, exist_ok=True)

                    # Create a subdirectory for the current frame's tensors
                    current_frame_tensor_dir = os.path.join(per_frame_tensor_output_base_dir, f"frame_{frame:05d}") # e.g., frame_00000, frame_00001
                    os.makedirs(current_frame_tensor_dir, exist_ok=True)

                    
                    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                    if height is None or width is None:
                        height = cv2_img.shape[0] // 2 * 2
                        width = cv2_img.shape[1] // 2 * 2
                    assert args.output_path is not None
                    cv2.imwrite(
                        os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                        255 * cv2_img,
                    )
                    if frame > end_frame:
                        light_flag = False
    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )

