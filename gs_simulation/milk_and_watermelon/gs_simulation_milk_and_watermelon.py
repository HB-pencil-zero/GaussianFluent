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
from typing import Optional
# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.lighting_utils import *
from utils.normal_utils import *

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0, random_seed=42)

import sys # 导入 sys 模块以使用 sys.stdout.flush()


def save_core_init_render_vars(
    filepath: str,
    mpm_init_pos: Optional[torch.Tensor], # <--- 修改这里
    mpm_init_vol: Optional[torch.Tensor], # <--- 修改这里
    mpm_init_cov: Optional[torch.Tensor], # <--- 修改这里
    opacity_render: Optional[torch.Tensor], # <--- 修改这里
    shs_render: Optional[torch.Tensor], # <--- 修改这里
):
    """
    将核心的 MPM 初始化位置/体积/协方差和渲染用的不透明度/SHs 保存到文件。
    自动处理 Tensor 的 .cpu().detach()。兼容 Python 3.9 类型提示。
    """
    print(f"\n准备保存核心初始化和渲染变量至: {filepath}")

    # 辅助函数，用于安全地准备 Tensor 进行保存
    def prep_tensor(t):
        # 检查 t 是否是 Tensor，因为 Optional[Tensor] 意味着 t 也可能是 None
        return t.cpu().detach().clone() if isinstance(t, torch.Tensor) else t

    # 创建只包含指定变量的保存字典
    state_to_save = {
        "mpm_init_pos": prep_tensor(mpm_init_pos),
        "mpm_init_vol": prep_tensor(mpm_init_vol),
        "mpm_init_cov": prep_tensor(mpm_init_cov),
        "opacity_render": prep_tensor(opacity_render),
        "shs_render": prep_tensor(shs_render),
    }
    torch.save(state_to_save, filepath)
    print(f"核心初始化和渲染变量已成功保存至: {filepath}")


def load_core_init_render_vars(
    filepath: str,
    map_location: Optional[str] = 'cpu' # 默认加载到 CPU，更安全
) :
    """
    从指定文件加载核心初始化和渲染变量。

    Args:
        filepath: 保存变量的 .pt 文件路径。
        map_location: 指定加载张量的设备 ('cpu', 'cuda', 'cuda:0' 等)。
                      默认为 'cpu'，以避免在没有 GPU 的机器上出错。

    Returns:
        一个包含加载变量的字典 ('mpm_init_pos', 'mpm_init_vol', 'mpm_init_cov',
        'opacity_render', 'shs_render')，如果文件不存在或加载失败则返回 None。
        注意：字典中的值可能是 Tensor 或 None（如果保存时是 None）。
    """
    print(f"\n尝试从以下路径加载核心初始化和渲染变量: {filepath}")

    if not os.path.exists(filepath):
        print(f"错误：文件未找到 - {filepath}")
        return None

    try:
        # 使用 torch.load 加载数据
        # map_location 参数确保张量被加载到指定的设备
        loaded_data = torch.load(filepath, map_location=map_location, weights_only=True)

        # 验证加载的数据是否是字典（可选但推荐）
        if not isinstance(loaded_data, dict):
            print(f"错误：加载的文件内容不是预期的字典格式 - {filepath}")
            return None

        # 验证是否包含预期的键（可选但推荐）
        expected_keys = {"mpm_init_pos", "mpm_init_vol", "mpm_init_cov", "opacity_render", "shs_render"}
        if not expected_keys.issubset(loaded_data.keys()):
            print(f"警告：加载的字典缺少部分预期键。文件路径: {filepath}")
            # 你可以选择仍然返回字典，或者返回 None，取决于你的需求
            # return None

        print(f"核心初始化和渲染变量已成功加载自: {filepath}")
        # 返回加载的字典
        return loaded_data

    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return None


def calculate_minimum_bounding_box_torch(
    positions: torch.Tensor,
    epsilon: float = 1e-7
) :
    """
    计算包含所有给定 3D 点的最小轴对齐包围盒 (AABB) 的参数，使用 PyTorch Tensor。

    它会检查是否有轴的尺寸接近零，并打印警告。

    Args:
        positions: 一个 PyTorch Tensor，形状为 (n, 3)，包含 n 个 3D 点的坐标。
                   Tensor 可以在任何设备上 (CPU or CUDA)。
        epsilon: 用于检查尺寸是否接近零的阈值。

    Returns:
        一个元组 (point, size):
        - point: 包围盒中心的 3D 坐标 (PyTorch Tensor, shape (3,), 同输入设备)。
        - size: 包围盒沿 x, y, z 轴的半尺寸 (PyTorch Tensor, shape (3,), 同输入设备)。
        如果输入 positions 为空，则返回 (None, None)。
    """
    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"输入必须是 PyTorch Tensor，但收到了 {type(positions)}")
    if positions.shape[0] == 0:
        print("警告：输入的 positions Tensor 为空。")
        return None, None
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"输入 Tensor 'positions' 的形状应为 (n, 3)，但收到了 {positions.shape}")

    # 1. 找到所有点在 x, y, z 轴上的最小值和最大值
    # torch.min/max 返回一个包含值和索引的元组，我们只需要值 [0]
    # 这步已经隐式地“检测”了每个轴的范围
    min_coords = torch.min(positions, dim=0)[0]  # shape (3,)
    max_coords = torch.max(positions, dim=0)[0]  # shape (3,)

    # 2. 计算包围盒的中心点 (point)
    point = (min_coords + max_coords) / 2.0

    # 3. 计算包围盒的半尺寸 (size)
    size = (max_coords - min_coords) / 2.0

    # 4. 检查每个轴的尺寸是否过小 (接近零)
    # full_size = max_coords - min_coords
    # zero_size_axes = torch.where(full_size < epsilon)[0] # 找到尺寸小于epsilon的轴的索引
    # 使用半尺寸 size 检查更直接
    near_zero_size_axes = torch.where(size < epsilon)[0] # 找到半尺寸小于epsilon的轴的索引

    if len(near_zero_size_axes) > 0:
        axis_names = ['x', 'y', 'z']
        problematic_axes = [axis_names[i] for i in near_zero_size_axes.tolist()]
        print(f"警告：计算出的包围盒在以下轴上的半尺寸小于 {epsilon}: {', '.join(problematic_axes)}")
        print(f"   - 最小坐标: {min_coords.tolist()}")
        print(f"   - 最大坐标: {max_coords.tolist()}")
        print(f"   - 计算的半尺寸: {size.tolist()}")
        # 注意：这里只是打印警告，没有修改 size。
        # 如果需要确保 size 不为零，可以在这里处理，例如：
        # size = torch.clamp(size, min=epsilon)
        # print(f"   - (如果应用钳位) 调整后的半尺寸: {size.tolist()}")

    return point, size




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

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    

    
    
    mpm_init_pos[:, 0] += 1.5
    mpm_init_pos[:, 1] += 1.5
    
    scale = 2
    mpm_init_pos = (mpm_init_pos - mpm_init_pos.mean(dim = 0)) * scale + mpm_init_pos.mean(dim = 0)
    mpm_init_cov =  mpm_init_cov  * (scale**2)
    mpm_init_vol =  mpm_init_vol * (scale**3)

    save_filepath = 'watermelon_new.pt'
    
    loaded_data = load_core_init_render_vars(save_filepath, map_location='cuda')
    
    pos2 = (loaded_data['mpm_init_pos'] - original_mean_pos) * scale_origin   
    pos2[:, :2] =  pos2[:, :2] - pos2[:, :2].mean(dim=0) + mpm_init_pos[:, :2].mean(dim=0) 
    pos2[:, 2] += 1.05
    cov2 =  scale_origin * scale_origin * loaded_data["mpm_init_cov"]
    vol2 = get_particle_volume(
        pos2,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device) 

    mpm_init_pos = torch.concat([mpm_init_pos, pos2], dim=0 )
    mpm_init_cov = torch.concat([mpm_init_cov, cov2], dim=0 )
    mpm_init_vol= torch.concat([mpm_init_vol, vol2], dim=0 )

    mpm_init_pos[:, 2] += 0.2
    gaussians3 = load_checkpoint("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model/garden")
    transform_matrix = torch.from_numpy(np.loadtxt("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model/garden/transform_matrix.txt")).to(device).float()
    pos3 = gaussians3._xyz.detach()
    pos3 = (pos3  @ transform_matrix[:3, :3].T  + transform_matrix[:3, 3])*3
    pos3[:, 2] -= 2.6
    pos3[:, 0] += 9.8 #- 3
    pos3[:, 1] += 8.8 #- 2
    cov3D3 = (rotate_flat_covariance(gaussians3.get_covariance(), transform_matrix[:3, :3])*3**2) * (scale_origin**2)
    rot3 = torch.tensor(transform_matrix[:3, :3], dtype=torch.float32, device="cuda").detach().clone().unsqueeze(0).expand(gaussians3._xyz.shape[0], 3, 3)
    opacity_render3 = gaussians3.get_opacity
    shs_render3 = 1.0 * gaussians3.get_features

    pos3_select = (pos3 - original_mean_pos) * scale_origin
    pos3_mask = torch.ones(pos3.shape[0], dtype=torch.bool).to(device="cuda")
    boundary_pos3  = [0, 5, 0, 5, -1.2, 3.8 ]
    for i in range(3):
        pos3_mask = torch.logical_and(pos3_mask, pos3_select[:, i] > boundary_pos3[2 * i])
        pos3_mask = torch.logical_and(pos3_mask, pos3_select[:, i] < boundary_pos3[2 * i + 1])
    pos3_select = pos3_select[pos3_mask]
    pos3_select[:, 2] += 1.2
    cov3D3_select = cov3D3[pos3_mask]
    vol3 = get_particle_volume(
        pos3_select,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device) 
    rot3_select = rot3[pos3_mask]
    opacity_render3_select = opacity_render3[pos3_mask]
    shs_render3_select = shs_render3[pos3_mask]

    mpm_init_pos = torch.concat([mpm_init_pos, pos3_select], dim=0 )
    mpm_init_cov = torch.concat([mpm_init_cov, cov3D3_select], dim=0 )
    mpm_init_vol= torch.concat([mpm_init_vol, vol3], dim=0 )

    mpm_init_pos[:, 2] += 0.5

    pos3_unselect = pos3[~pos3_mask] + torch.tensor(apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(torch.tensor([0,0,1.7]).cuda().unsqueeze(0)), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            ))
    cov3D3_unselect = cov3D3[~pos3_mask] / (scale_origin * scale_origin)
    rot3_unselect = rot3[~pos3_mask]
    opacity_render3_unselect = opacity_render3[~pos3_mask]
    shs_render3_unselect = shs_render3[~pos3_mask]

    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)
    
    beta = mpm_solver.mpm_model.beta.numpy()
    # mask1 = (gaussians._features_dc > 0.5).all(axis=2).cpu().numpy().squeeze()
    # mask2 = (gaussians._features_dc > -2.5).all(axis=2).cpu().numpy().squeeze()
    # mask_ = mask1 & mask2
    
    mask1 = (gaussians._features_dc > 1.0).all(axis=2).cpu().numpy().squeeze()
    # mask2 = (gaussians._features_dc < 4).all(axis=2).cpu().numpy().squeeze()
    # mask__ = mask1 & mask2
    selected_xyz = gaussians._xyz[mask1]
    
    

    # 2. 创建KNN搜索器
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
    # # beta[mask] = 2
    # mpm_solver.mpm_model.beta.assign(beta)
    
    
    
    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    mpm_solver.finalize_mu_lam()

    mpm_solver.import_particle_v_from_torch(torch.zeros(mpm_init_pos.shape[0], 3, device='cuda').add_(torch.tensor([0.0, 0.0, -4.0], device='cuda')))  # -6.0
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
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    
    opacity_render = torch.concat([opacity_render, loaded_data['opacity_render'], opacity_render3_select], dim=0)
    shs_render =  torch.concat([shs_render, loaded_data['shs_render'], shs_render3_select], dim=0)
    
    height = None
    width = None
    ti.reset()
    # torch.cuda.empty_cache()
    color_flag = False
    # color_flag = True
    light_flag = True
    end_frame = 22


    gs_num = mpm_init_pos.shape[0]


    # gaussians2 = load_checkpoint("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model/garden_ours")
    # pos2 =  gaussians2._xyz.detach() * 5
    # pos2[:, :2] += 4.3
    # pos2[:, 2] -= 1
    # cov3D2 = gaussians2.get_covariance() * 25
    # rot2 = torch.eye(3, device="cuda").expand(gaussians2._xyz.shape[0], 3, 3)
    # opacity_render2 = gaussians2.get_opacity
    # shs_render2 = 1 * gaussians2.get_features



    gaussians_watermenlon = load_checkpoint("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model/watermelon")
    mask = loaded_data['mask']
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=-1,
            center_view_world_space=viewpoint_center_worldspace,
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
        

        for step in range(step_per_frame):
            mpm_solver.p2g2p(step, substep_dt, device=device, flip_pic_ratio=material_params['flip_pic_ratio'])

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )


        
        if args.render_img:
            # 获取初始数据
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)
            


            
            #opacity_render
            # shs_render

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



            



            rot[-rot3_select.shape[0]:] = rot3_select
            pos = torch.concat([pos, pos3_unselect],dim =0 )
            cov3D = torch.concat([cov3D, cov3D3_unselect],dim =0 )
            rot = torch.concat([rot, rot3_unselect],dim =0 )
            opacity = torch.concat([opacity_render, opacity_render3_unselect],dim =0 )
            shs = torch.concat([shs_render, shs_render3_unselect],dim =0 )


            # gaussians._xyz = pos - torch.tensor(viewpoint_center_worldspace).cuda().unsqueeze(0)
            # gaussians._scaling = torch.concat([gaussians._scaling, gaussians_watermenlon._scaling[mask], gaussians2._scaling], dim=0)
            # gaussians._rotation = torch.concat([gaussians._rotation, gaussians_watermenlon._rotation[mask], gaussians2._rotation], dim=0)
            # gaussians._features_dc = shs[:,:1]
            # gaussians._features_rest = shs[:, 1:]
            # gaussians._opacity = torch.logit(opacity, eps=1e-10)
            # gaussians.save_ply("model/garden_with_milk_and_watermelon/point_cloud/iteration_30000/point_cloud.ply")


            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            if color_flag:
                if  light_flag : 
                    np.save("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/watermelon_frame/frame_20/pos.npy", pos.detach().cpu().numpy())
                    # np.save("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/watermelon_frame/frame_20/pos.npy", pos.detach().cpu().numpy())
                    command = 'cd /root/autodl-tmp/debug_physgaussian/cdmpmGaussian/ && source $(conda info --base)/etc/profile.d/conda.sh && conda activate PhysGaussian && python normal_vector_proc_nan.py'
                    run_command_realtime(command)


                    command = 'cd /root/autodl-tmp/debug_physgaussian/cdmpmGaussian/ && source $(conda info --base)/etc/profile.d/conda.sh && conda activate PhysGaussian && python phong_model_wm_shs_15.py'
                    run_command_realtime(command)


                pos = torch.from_numpy(np.load("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/watermelon_frame/frame_20/pos.npy")).to("cuda")
                valid_indice = torch.from_numpy(np.load("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/watermelon_frame/frame_20/pos_valid_indice.npy")).to("cuda")
                colors = torch.from_numpy(np.load("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/phong_colors.npy")).to("cuda").reshape(-1 , 3).float()
                colors_precomp = colors

            

            
            
            rendering, raddi, point_xy = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp.float(),
                opacities=opacity, 
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )

            

            # Apply the combined mask to get the filtered points
            # filtered_points = point_xy[combined_mask]
            
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
        fps = int(1.0 / time_params["frame_dt"] /1.2 )
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )

