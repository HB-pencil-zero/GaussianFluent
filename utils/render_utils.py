import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torchvision
import torch
import os
import numpy as np
import json
import copy
from tqdm import tqdm

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov
from utils.render_utils import * 

def initialize_resterize(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterize = GaussianRasterizer(raster_settings=raster_settings)
    return rasterize


def load_params_from_gs(
    pc: GaussianModel, pipe, scaling_modifier=1.0, override_color=None
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.

    return {
        "pos": means3D,
        "screen_points": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
    }


def convert_SH(
    shs_view,
    viewpoint_camera,
    pc: GaussianModel,
    position: torch.tensor,
    rotation: torch.tensor = None,
):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        n = rotation.shape[0]
        dir_pp[:n] = torch.matmul(rotation, dir_pp[:n].unsqueeze(2)).squeeze(2)
        # dir_pp[:n] = torch.matmul( dir_pp[:n].unsqueeze(1), rotation).squeeze(1)

    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # valid_indice = torch.from_numpy(np.load("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/watermelon_frame/frame_20/pos_valid_indice.npy")).to("cuda")
    # colors = torch.from_numpy(np.load("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/phong_colors.npy")).to("cuda").reshape(-1 , 3).float()
    # colors_precomp[valid_indice] = colors
    return colors_precomp


def filter_points_verbose(pos2: torch.Tensor, thresold: float = 30 ) -> torch.Tensor:
    """
    筛选 tensor (详细步骤版)。
    """
    # 步骤 1: 分别为 x 和 y 坐标创建条件
    # pos2[:, 0] 是所有行的第0列 (x坐标)
    condition_x = torch.abs(pos2[:, 0]) < thresold
    # pos2[:, 1] 是所有行的第1列 (y坐标)
    condition_y = torch.abs(pos2[:, 1]) < thresold
    
    # 步骤 2: 将两个条件用逻辑“与” (&) 组合成最终的掩码
    # 只有当 condition_x 和 condition_y 在同一位置都为 True 时，结果才为 True
    combined_mask = condition_x & condition_y
    
    # 步骤 3: 使用组合后的掩码进行筛选
    return combined_mask



# def normalize_torch(v_arr):
#     """向量或向量数组归一化 (PyTorch 版本)"""
#     norm = torch.linalg.norm(v_arr, dim=-1, keepdim=True)
#     # 使用 torch.where 来安全地处理零向量
#     return torch.where(norm > 1e-8, v_arr / norm, torch.zeros_like(v_arr))


# def blinn_phong_shading_with_transmittance_torch(
#     point_3d_single, normal_single, base_color_single,
#     light_pos_3d_global, camera_pos_3d_global,
#     transmittance, # 这是个标量
#     ka=0.1, kd=0.7, ks=0.5, shininess=32.0,
#     ambient_light_color_global=None,
#     light_color_intensity_global=None
# ):
#     """
#     为单个点计算Blinn-Phong着色颜色 (PyTorch 版本)。
#     漫反射和镜面反射强度乘以传入的透过率，并根据距离光源的平方进行衰减。
#     """
#     # 确保光照颜色是张量
#     device = point_3d_single.device
#     dtype = point_3d_single.dtype
#     if ambient_light_color_global is None:
#         ambient_light_color_global = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=dtype)
#     if light_color_intensity_global is None:
#         light_color_intensity_global = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)

#     # 环境光部分
#     ambient = ka * ambient_light_color_global * base_color_single

#     # 距离平方衰减
#     vec_to_light = light_pos_3d_global - point_3d_single
#     distance_sq = torch.dot(vec_to_light, vec_to_light)
#     attenuation = 1.0 / (distance_sq + 1e-7)

#     # 光照方向
#     light_dir = normalize_torch(vec_to_light)

#     # 漫反射部分
#     diffuse_intensity = torch.abs(torch.dot(normal_single, light_dir))
#     diffuse = kd * diffuse_intensity * base_color_single * light_color_intensity_global

#     # 镜面反射部分
#     view_dir = normalize_torch(camera_pos_3d_global - point_3d_single)
#     half_vector = normalize_torch(light_dir + view_dir)
#     specular_intensity_base = torch.abs(torch.dot(normal_single, half_vector))
#     specular = ks * (specular_intensity_base ** shininess) * light_color_intensity_global

#     # 最终颜色
#     return ambient + (diffuse + specular) * attenuation * transmittance


# def calculate_colors_per_point_with_accumulation_torch(
#     yx2d_all: torch.Tensor,
#     points_3d_all: torch.Tensor,
#     normal_vector_all: torch.Tensor,
#     valid_indice_mask: torch.Tensor,
#     base_colors_all: torch.Tensor,
#     light_source_3d_global:  torch.Tensor,
#     point_base_opacity_all: torch.Tensor,
#     camera_position_3d_global:  torch.Tensor,
#     w=800, h=800,
#     ka=0.1, kd=0.7, ks=0.5, shininess=32.0,
#     ambient_light_color_global=torch.tensor([0.1, 0.1, 0.1]),
#     light_color_intensity_global=torch.tensor([1.0, 1.0, 1.0])
# ):
#     """
#     为每个输入点计算其最终颜色 (PyTorch版本)。
#     输入应全部为 torch.Tensor。
#     """
#     # 0. 确定设备和数据类型，并初始化输出
#     device = points_3d_all.device
#     dtype = points_3d_all.dtype
#     num_points = len(points_3d_all)
#     output_colors = torch.zeros_like(base_colors_all)
#     original_indices = torch.arange(num_points, device=device)

#     # 将所有输入张量分离计算图，并确保它们在正确的设备上
#     yx2d_all = yx2d_all.detach()
#     points_3d_all = points_3d_all.detach()
#     normal_vector_all = normal_vector_all.detach()
#     valid_indice_mask = valid_indice_mask.detach()
#     base_colors_all = base_colors_all.detach()
#     point_base_opacity_all = point_base_opacity_all.detach()
    
#     # 将全局参数转换为张量并放到正确的设备
#     light_source_3d_global = torch.as_tensor(light_source_3d_global, device=device, dtype=dtype)
#     camera_position_3d_global = torch.as_tensor(camera_position_3d_global, device=device, dtype=dtype)
#     ambient_light_color_global = torch.as_tensor(ambient_light_color_global, device=device, dtype=dtype)
#     light_color_intensity_global = torch.as_tensor(light_color_intensity_global, device=device, dtype=dtype)

#     # 1. 根据 valid_indice_mask 过滤数据
#     yx2d_valid = yx2d_all[valid_indice_mask]
#     points_3d_valid = points_3d_all[valid_indice_mask]
#     normal_vector_valid = normal_vector_all.detach().clone()
#     base_colors_valid = base_colors_all[valid_indice_mask]
#     point_base_opacity_valid = point_base_opacity_all[valid_indice_mask]
#     original_indices_valid = original_indices[valid_indice_mask]

#     # 2. 保留位于图像边界内的点
#     in_bounds_mask = (yx2d_valid[:, 0] >= 0) & (yx2d_valid[:, 0] < h) & \
#                      (yx2d_valid[:, 1] >= 0) & (yx2d_valid[:, 1] < w)

#     yx2d_bounded = yx2d_valid[in_bounds_mask]
#     points_3d_bounded = points_3d_valid[in_bounds_mask]
#     normal_vector_bounded = normal_vector_valid[in_bounds_mask]
#     base_colors_bounded = base_colors_valid[in_bounds_mask]
#     point_base_opacity_bounded = point_base_opacity_valid[in_bounds_mask]
#     original_indices_bounded = original_indices_valid[in_bounds_mask]

#     if len(yx2d_bounded) == 0:
#         print("警告: 过滤后没有点在图像边界内。")
#         return output_colors

#     # 3. 将yx坐标四舍五入到整数像素坐标
#     pixel_coords_rounded = torch.round(yx2d_bounded).to(torch.long)
#     pixel_coords_rounded[:, 0] = torch.clamp(pixel_coords_rounded[:, 0], 0, h - 1)
#     pixel_coords_rounded[:, 1] = torch.clamp(pixel_coords_rounded[:, 1], 0, w - 1)

#     # 4. 将点按其所属像素分组
#     pixel_data_map = defaultdict(list)
#     # 将张量移到CPU进行循环，因为在GPU上逐元素访问非常慢
#     pixel_coords_cpu = pixel_coords_rounded.cpu().numpy()
    
#     for i in  tqdm(range(len(pixel_coords_cpu))):
#         py, px = pixel_coords_cpu[i, 0], pixel_coords_cpu[i, 1]
#         p3d = points_3d_bounded[i]
#         dist_to_light = torch.linalg.norm(p3d - light_source_3d_global)
        
#         pixel_data_map[(py, px)].append({
#             'dist_to_light': dist_to_light.item(), # 转换为Python标量用于排序
#             'point_3d': p3d,
#             'normal': normal_vector_bounded[i],
#             'base_color': base_colors_bounded[i],
#             'opacity': point_base_opacity_bounded[i].item(), # 转换为Python标量
#             'original_index': original_indices_bounded[i].item() # 转换为Python标量
#         })

#     # 5. 对每个像素分组进行处理：排序、计算透过率和颜色
#     for (py, px), points_in_this_pixel in tqdm(pixel_data_map.items()):
#         points_in_this_pixel.sort(key=lambda p_data: p_data['dist_to_light'])
#         accumulated_opacity = 0.0

#         for point_data in points_in_this_pixel:
#             transmittance = 1.0 - accumulated_opacity
            
#             shaded_color = blinn_phong_shading_with_transmittance_torch(
#                 point_data['point_3d'], point_data['normal'], point_data['base_color'],
#                 light_source_3d_global, camera_position_3d_global,
#                 transmittance,
#                 ka, kd, ks, shininess,
#                 ambient_light_color_global, light_color_intensity_global
#             )
            
#             original_idx = point_data['original_index']
#             output_colors[original_idx] = shaded_color
#             accumulated_opacity += point_data['opacity'] * transmittance

#     return output_colors


# import torch
# import torch.nn.functional as F

# def normalize_torch(v_arr):
#     """向量或向量数组归一化 (PyTorch 版本)"""
#     norm = torch.linalg.norm(v_arr, dim=-1, keepdim=True)
#     return torch.where(norm > 1e-8, v_arr / norm, torch.zeros_like(v_arr))

# def blinn_phong_shading_batch_torch(
#     points_3d, normals, base_colors,
#     light_pos_3d_global, camera_pos_3d_global,
#     transmittances,  # (n,) 透过率向量
#     ka=0.1, kd=0.7, ks=0.5, shininess=32.0,
#     ambient_light_color_global=None,
#     light_color_intensity_global=None
# ):
#     """
#     批量计算Blinn-Phong着色 (PyTorch 版本)
#     """
#     device = points_3d.device
#     dtype = points_3d.dtype
#     n_points = points_3d.shape[0]
    
#     # 设置默认光照
#     if ambient_light_color_global is None:
#         ambient_light_color_global = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=dtype)
#     if light_color_intensity_global is None:
#         light_color_intensity_global = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
    
#     # 环境光部分 (n, 3)
#     ambient = ka * ambient_light_color_global * base_colors

#     # 向量化计算光照方向
#     vec_to_light = light_pos_3d_global - points_3d  # (n, 3)
#     distance_sq = torch.sum(vec_to_light**2, dim=1, keepdim=True)  # (n, 1)
#     attenuation = 1.0 / (distance_sq + 1e-7)  # (n, 1)
#     light_dir = normalize_torch(vec_to_light)  # (n, 3)

#     # 漫反射部分 (n, 3)
#     diffuse_intensity = torch.sum(normals * light_dir, dim=1, keepdim=True).abs()  # (n, 1)
#     diffuse = kd * diffuse_intensity * base_colors * light_color_intensity_global

#     # 镜面反射部分 (n, 3)
#     view_dir = normalize_torch(camera_pos_3d_global - points_3d)  # (n, 3)
#     half_vector = normalize_torch(light_dir + view_dir)  # (n, 3)
#     specular_intensity_base = torch.sum(normals * half_vector, dim=1, keepdim=True).abs()  # (n, 1)
#     specular = ks * (specular_intensity_base ** shininess) * light_color_intensity_global

#     # 最终颜色 (n, 3)
#     return ambient + (diffuse + specular) * attenuation * transmittances.view(-1, 1)  * 500

# def calculate_colors_per_point_optimized_torch(
#     yx2d_all: torch.Tensor,
#     points_3d_all: torch.Tensor,
#     normal_vector_all: torch.Tensor,
#     valid_indice_mask: torch.Tensor,
#     base_colors_all: torch.Tensor,
#     light_source_3d_global: torch.Tensor,
#     point_base_opacity_all: torch.Tensor,
#     camera_position_3d_global: torch.Tensor,
#     w=800, h=800,
#     ka=0.1, kd=0.7, ks=0.5, shininess=32.0,
#     ambient_light_color_global=torch.tensor([0.1, 0.1, 0.1]),
#     light_color_intensity_global=torch.tensor([1.0, 1.0, 1.0])
# ):
#     """
#     优化的点着色计算 (PyTorch 并行版本) - 修复维度问题
#     """
#     device = points_3d_all.device
#     dtype = points_3d_all.dtype
#     num_points = len(points_3d_all)
#     output_colors = torch.zeros_like(base_colors_all)
    
#     # 0. 确保不透明度是1D张量
#     point_base_opacity_all = point_base_opacity_all.flatten()
    
#     # 1. 过滤有效点
#     if not valid_indice_mask.any():
#         return output_colors
    
#     # 过滤有效点并确保在图像边界内
#     yx2d_valid = yx2d_all[valid_indice_mask]
#     points_3d_valid = points_3d_all[valid_indice_mask]
#     normal_vector_valid = normal_vector_all.clone() # 直接使用，不需要额外处理
#     base_colors_valid = base_colors_all[valid_indice_mask]
#     point_base_opacity_valid = point_base_opacity_all[valid_indice_mask]
    
#     # 使用arange代替nonzero - 更高效
#     original_indices = torch.arange(num_points, device=device)[valid_indice_mask]
    
#     # 边界检查
#     in_bounds_mask = (yx2d_valid[:, 0] >= 0) & (yx2d_valid[:, 0] < h) & \
#                      (yx2d_valid[:, 1] >= 0) & (yx2d_valid[:, 1] < w)
    
#     if not in_bounds_mask.any():
#         return output_colors
    
#     yx2d_bounded = yx2d_valid[in_bounds_mask]
#     points_3d_bounded = points_3d_valid[in_bounds_mask]
#     normal_vector_bounded = normal_vector_valid[in_bounds_mask]
#     base_colors_bounded = base_colors_valid[in_bounds_mask]
#     point_base_opacity_bounded = point_base_opacity_valid[in_bounds_mask]
#     original_indices_bounded = original_indices[in_bounds_mask]
    
#     # 2. 计算像素ID和距离光源距离
#     pixel_coords_rounded = torch.round(yx2d_bounded).to(torch.long)
#     pixel_coords_rounded[:, 0] = torch.clamp(pixel_coords_rounded[:, 0], 0, h - 1)
#     pixel_coords_rounded[:, 1] = torch.clamp(pixel_coords_rounded[:, 1], 0, w - 1)
#     pixel_ids = pixel_coords_rounded[:, 0] * w + pixel_coords_rounded[:, 1]  # (n,)
    
#     # 计算每个点到光源的距离 (用于排序)
#     vec_to_light = light_source_3d_global - points_3d_bounded
#     dist_to_light = torch.linalg.norm(vec_to_light, dim=1)  # (n,)
    
#     # 3. 按像素ID分组并排序
#     unique_pixel_ids, inverse_indices, counts = torch.unique(
#         pixel_ids, return_inverse=True, return_counts=True, sorted=True
#     )
    
#     # 为每个点创建排序键 (像素ID + 距离)
#     sort_keys = pixel_ids * 1e10 + dist_to_light  # 确保距离变化不影响像素分组
    
#     # 全局排序
#     sorted_indices = torch.argsort(sort_keys)
#     sorted_points = points_3d_bounded[sorted_indices]
#     sorted_normals = normal_vector_bounded[sorted_indices]
#     sorted_base_colors = base_colors_bounded[sorted_indices]
#     sorted_opacities = point_base_opacity_bounded[sorted_indices]
#     sorted_original_indices = original_indices_bounded[sorted_indices]
    
#     # 4. 批量计算透过率
#     # 计算每个像素组的起始索引 - 移动到CPU用于tensor_split
#     if counts.numel() > 1:
#         split_indices = torch.cumsum(counts, dim=0)[:-1].cpu()
#     else:
#         split_indices = []  # 只有一个组时不需要分割
    
#     # 分割为像素组列表
#     points_per_pixel = torch.tensor_split(sorted_points, split_indices)
#     normals_per_pixel = torch.tensor_split(sorted_normals, split_indices)
#     base_colors_per_pixel = torch.tensor_split(sorted_base_colors, split_indices)
#     opacities_per_pixel = torch.tensor_split(sorted_opacities, split_indices)
#     original_indices_per_pixel = torch.tensor_split(sorted_original_indices, split_indices)
    
#     # 5. 并行处理每个像素组
#     for points, normals, base_colors, opacities, indices in tqdm(zip(
#         points_per_pixel, normals_per_pixel, base_colors_per_pixel, 
#         opacities_per_pixel, original_indices_per_pixel
#     )):
#         n = points.shape[0]
#         if n == 0:
#             continue
        
#         # 确保不透明度是1D
#         opacities = opacities.flatten()
        
#         # 计算透过率 (使用向量化操作)
#         one_minus_opacity = 1 - opacities.clamp(0, 1)
        
#         # 构建累积透过率数组 - 修复维度问题
#         transmittances = torch.ones(n, device=device, dtype=dtype)
        
#         if n > 1:
#             # 计算累积乘积 (n-1个元素)
#             cum_prod = torch.cumprod(one_minus_opacity[:-1], dim=0)
            
#             # 直接赋值避免维度问题
#             transmittances[1:] = cum_prod
        
#         # 批量计算着色
#         shaded_colors = blinn_phong_shading_batch_torch(
#             points, normals, base_colors,
#             light_source_3d_global, camera_position_3d_global,
#             transmittances,
#             ka, kd, ks, shininess,
#             ambient_light_color_global, light_color_intensity_global
#         )
        
#         # 保存结果
#         output_colors[indices] = shaded_colors
    
#     return output_colors




def filter_gaussian_points_by_plane(tensor,
                                    plane_w,
                                    plane_b=0.0,
                                    plane_greater=True,
                                    cov=None,
                                    cal_bias=False,
                                    delta=0.02):
    """
    沿任意自定义平面 w·x = b 切割，返回上半部分 (w·x > b) 或下半部分 (w·x < b) 的点。

    参数:
        tensor (torch.Tensor): [n, 3]，每行一个 3D 点。
        plane_w (array-like | torch.Tensor): 平面法向量 w，形状为 (3,)。
        plane_b (float): 平面偏置 b，使平面满足 w·x = b。
        plane_greater (bool): True 取上半部分 (w·x > b)，False 取下半部分 (w·x < b)。
        cov (torch.Tensor or None): 若提供则形状为 [n, 6]，为各点的协方差上三角 (xx, xy, xz, yy, yz, zz)。
        cal_bias (bool): 是否进行基于协方差的偏置矫正。
        delta (float): 当 cal_bias 且提供 cov 时，对阈值进行 +/- delta 的边界放宽。

    返回:
        torch.Tensor: 筛选后的点（原始坐标）。
        torch.Tensor: 保留下来的点的索引。
        torch.Tensor: 被排除的点的索引。
    """
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (n, 3)")

    if isinstance(plane_w, (list, tuple, np.ndarray)):
        w = torch.tensor(plane_w, device=tensor.device, dtype=tensor.dtype)
    elif isinstance(plane_w, torch.Tensor):
        w = plane_w.to(device=tensor.device, dtype=tensor.dtype)
    else:
        raise TypeError("plane_w must be a list/tuple/ndarray/torch.Tensor of shape (3,)")

    if w.numel() != 3:
        raise ValueError("plane_w must have 3 elements")
    b = torch.as_tensor(plane_b, device=tensor.device, dtype=tensor.dtype)

    adjusted_tensor = tensor  - torch.tensor([1.0, 1.0, 1.0], device=tensor.device, dtype=tensor.dtype)

    # 协方差偏置矫正: x' = x - Sigma w / sqrt(w^T Sigma w)
    if cov is not None and cal_bias:
        if cov.shape[0] != tensor.shape[0] or cov.shape[1] != 6:
            raise ValueError("Covariance must have shape (n, 6) matching tensor.shape[0]")
        n = cov.shape[0]
        Sigma = torch.zeros((n, 3, 3), device=tensor.device, dtype=tensor.dtype)
        Sigma[:, 0, 0] = cov[:, 0]
        Sigma[:, 0, 1] = cov[:, 1]
        Sigma[:, 0, 2] = cov[:, 2]
        Sigma[:, 1, 0] = cov[:, 1]
        Sigma[:, 1, 1] = cov[:, 3]
        Sigma[:, 1, 2] = cov[:, 4]
        Sigma[:, 2, 0] = cov[:, 2]
        Sigma[:, 2, 1] = cov[:, 4]
        Sigma[:, 2, 2] = cov[:, 5]

        Sigma_w = torch.matmul(Sigma, w.view(3, 1)).squeeze(-1)  # [n, 3]
        wT_Sigma_w = torch.matmul(Sigma_w, w)  # [n]

        bias = torch.zeros_like(tensor)
        valid = wT_Sigma_w > 1e-8
        if torch.any(valid):
            bias_valid = Sigma_w[valid] / torch.sqrt(wT_Sigma_w[valid]).unsqueeze(-1)
            bias[valid] = bias_valid
        adjusted_tensor = tensor - bias

    # delta 放宽（与 xyz 函数保持一致，仅在 cal_bias 且提供 cov 时生效）
    current_delta = delta if (cal_bias and cov is not None) else 0.0
    wx = torch.matmul(adjusted_tensor, w)

    if plane_greater:
        keep_mask = wx > (b - current_delta)
    else:
        keep_mask = wx < (b + current_delta)

    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    drop_idx = torch.nonzero(~keep_mask, as_tuple=False).squeeze(-1)
    return tensor[keep_idx], keep_idx, drop_idx


def filter_gaussian_points_by_sphere(tensor,
                                     sphere_center,
                                     sphere_radius,
                                     sphere_greater=True,
                                     cov=None,
                                     cal_bias=False,
                                     delta=0.02):
    """
    沿球形截面切割，返回球外部分 (|x-c| > r) 或球内部分 (|x-c| < r) 的点。

    参数:
        tensor (torch.Tensor): [n, 3]，每行一个 3D 点。
        sphere_center (array-like | torch.Tensor): 球心坐标 c，形状为 (3,)。
        sphere_radius (float): 球半径 r。
        sphere_greater (bool): True 取球外部分 (|x-c| > r)，False 取球内部分 (|x-c| < r)。
        cov (torch.Tensor or None): 若提供则形状为 [n, 6]，为各点的协方差上三角 (xx, xy, xz, yy, yz, zz)。
        cal_bias (bool): 是否进行基于协方差的偏置矫正。
        delta (float): 当 cal_bias 且提供 cov 时，对阈值进行 +/- delta 的边界放宽。

    返回:
        torch.Tensor: 筛选后的点（原始坐标）。
        torch.Tensor: 保留下来的点的索引。
        torch.Tensor: 被排除的点的索引。
    """
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (n, 3)")

    if isinstance(sphere_center, (list, tuple, np.ndarray)):
        c = torch.tensor(sphere_center, device=tensor.device, dtype=tensor.dtype)
    elif isinstance(sphere_center, torch.Tensor):
        c = sphere_center.to(device=tensor.device, dtype=tensor.dtype)
    else:
        raise TypeError("sphere_center must be a list/tuple/ndarray/torch.Tensor of shape (3,)")

    if c.numel() != 3:
        raise ValueError("sphere_center must have 3 elements")
    
    r = torch.as_tensor(sphere_radius, device=tensor.device, dtype=tensor.dtype)

    # 计算点到球心的距离
    adjusted_tensor = tensor - c
    distances = torch.norm(adjusted_tensor, dim=1)  # [n]

    # 协方差偏置矫正: 对于球形截面，我们需要考虑径向方向的协方差
    if cov is not None and cal_bias:
        if cov.shape[0] != tensor.shape[0] or cov.shape[1] != 6:
            raise ValueError("Covariance must have shape (n, 6) matching tensor.shape[0]")
        
        n = cov.shape[0]
        Sigma = torch.zeros((n, 3, 3), device=tensor.device, dtype=tensor.dtype)
        Sigma[:, 0, 0] = cov[:, 0]
        Sigma[:, 0, 1] = cov[:, 1]
        Sigma[:, 0, 2] = cov[:, 2]
        Sigma[:, 1, 0] = cov[:, 1]
        Sigma[:, 1, 1] = cov[:, 3]
        Sigma[:, 1, 2] = cov[:, 4]
        Sigma[:, 2, 0] = cov[:, 2]
        Sigma[:, 2, 1] = cov[:, 4]
        Sigma[:, 2, 2] = cov[:, 5]

        # 计算径向方向的单位向量
        radial_dirs = adjusted_tensor / (distances.unsqueeze(-1) + 1e-8)  # [n, 3]
        
        # 计算径向方向的协方差: radial_dir^T @ Sigma @ radial_dir
        Sigma_radial = torch.matmul(Sigma, radial_dirs.unsqueeze(-1)).squeeze(-1)  # [n, 3]
        radial_var = torch.sum(radial_dirs * Sigma_radial, dim=1)  # [n]

        # 计算偏置调整
        bias = torch.zeros_like(tensor)
        valid = radial_var > 1e-8
        if torch.any(valid):
            bias_valid = Sigma_radial[valid] / torch.sqrt(radial_var[valid]).unsqueeze(-1)
            bias[valid] = bias_valid
        
        # 应用偏置调整
        adjusted_tensor = adjusted_tensor - bias
        distances = torch.norm(adjusted_tensor, dim=1)  # 重新计算距离

    # delta 放宽（与 xyz 函数保持一致，仅在 cal_bias 且提供 cov 时生效）
    current_delta = delta if (cal_bias and cov is not None) else 0.0

    if sphere_greater:
        # 取球外部分: |x-c| > r
        keep_mask = distances > (r - current_delta)
    else:
        # 取球内部分: |x-c| < r
        keep_mask = distances < (r + current_delta)

    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    drop_idx = torch.nonzero(~keep_mask, as_tuple=False).squeeze(-1)
    
    return tensor[keep_idx], keep_idx, drop_idx


def filter_gaussian_points_by_ellipsoid(tensor,
                                        ellipsoid_center,
                                        ellipsoid_axes,
                                        ellipsoid_greater=True,
                                        cov=None,
                                        cal_bias=False,
                                        delta=0.02):
    """
    沿椭球形截面切割，返回椭球外部分或椭球内部分的点。

    参数:
        tensor (torch.Tensor): [n, 3]，每行一个 3D 点。
        ellipsoid_center (array-like | torch.Tensor): 椭球心坐标，形状为 (3,)。
        ellipsoid_axes (array-like | torch.Tensor): 椭球三个轴的长度 [a, b, c]，形状为 (3,)。
        ellipsoid_greater (bool): True 取椭球外部分，False 取椭球内部分。
        cov (torch.Tensor or None): 若提供则形状为 [n, 6]，为各点的协方差上三角。
        cal_bias (bool): 是否进行基于协方差的偏置矫正。
        delta (float): 当 cal_bias 且提供 cov 时，对阈值进行 +/- delta 的边界放宽。

    返回:
        torch.Tensor: 筛选后的点（原始坐标）。
        torch.Tensor: 保留下来的点的索引。
        torch.Tensor: 被排除的点的索引。
    """
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (n, 3)")

    if isinstance(ellipsoid_center, (list, tuple, np.ndarray)):
        c = torch.tensor(ellipsoid_center, device=tensor.device, dtype=tensor.dtype)
    elif isinstance(ellipsoid_center, torch.Tensor):
        c = ellipsoid_center.to(device=tensor.device, dtype=tensor.dtype)
    else:
        raise TypeError("ellipsoid_center must be a list/tuple/ndarray/torch.Tensor of shape (3,)")

    if isinstance(ellipsoid_axes, (list, tuple, np.ndarray)):
        axes = torch.tensor(ellipsoid_axes, device=tensor.device, dtype=tensor.dtype)
    elif isinstance(ellipsoid_axes, torch.Tensor):
        axes = ellipsoid_axes.to(device=tensor.device, dtype=tensor.dtype)
    else:
        raise TypeError("ellipsoid_axes must be a list/tuple/ndarray/torch.Tensor of shape (3,)")

    if c.numel() != 3 or axes.numel() != 3:
        raise ValueError("ellipsoid_center and ellipsoid_axes must have 3 elements each")

    # 计算标准化坐标: (x-c)/a, (y-c)/b, (z-c)/z
    tensor = tensor - torch.tensor([1.0, 1.0, 1.0], device=tensor.device, dtype=tensor.dtype)
    adjusted_tensor = tensor - c
    normalized_coords = adjusted_tensor / (axes.unsqueeze(0) + 1e-8)  # [n, 3]
    
    # 计算椭球函数值: (x/a)^2 + (y/b)^2 + (z/c)^2
    ellipsoid_values = torch.sum(normalized_coords ** 2, dim=1)  # [n]

    # 协方差偏置矫正
    if cov is not None and cal_bias:
        if cov.shape[0] != tensor.shape[0] or cov.shape[1] != 6:
            raise ValueError("Covariance must have shape (n, 6) matching tensor.shape[0]")
        
        n = cov.shape[0]
        Sigma = torch.zeros((n, 3, 3), device=tensor.device, dtype=tensor.dtype)
        Sigma[:, 0, 0] = cov[:, 0]
        Sigma[:, 0, 1] = cov[:, 1]
        Sigma[:, 0, 2] = cov[:, 2]
        Sigma[:, 1, 0] = cov[:, 1]
        Sigma[:, 1, 1] = cov[:, 3]
        Sigma[:, 1, 2] = cov[:, 4]
        Sigma[:, 2, 0] = cov[:, 2]
        Sigma[:, 2, 1] = cov[:, 4]
        Sigma[:, 2, 2] = cov[:, 5]

        # 计算椭球法向量方向的协方差
        # 椭球法向量: [2x/a^2, 2y/b^2, 2z/c^2]
        normal_dirs = 2.0 * normalized_coords / (axes.unsqueeze(0) + 1e-8)  # [n, 3]
        
        # 归一化法向量
        normal_norms = torch.norm(normal_dirs, dim=1, keepdim=True)  # [n, 1]
        normal_dirs = normal_dirs / (normal_norms + 1e-8)  # [n, 3]
        
        # 计算法向量方向的协方差
        Sigma_normal = torch.matmul(Sigma, normal_dirs.unsqueeze(-1)).squeeze(-1)  # [n, 3]
        normal_var = torch.sum(normal_dirs * Sigma_normal, dim=1)  # [n]

        # 计算偏置调整
        bias = torch.zeros_like(tensor)
        valid = normal_var > 1e-8
        if torch.any(valid):
            bias_valid = Sigma_normal[valid] / torch.sqrt(normal_var[valid]).unsqueeze(-1)
            bias[valid] = bias_valid
        
        # 应用偏置调整
        adjusted_tensor = adjusted_tensor - bias
        normalized_coords = adjusted_tensor / (axes.unsqueeze(0) + 1e-8)
        ellipsoid_values = torch.sum(normalized_coords ** 2, dim=1)

    # delta 放宽
    current_delta = delta if (cal_bias and cov is not None) else 0.0

    if ellipsoid_greater:
        # 取椭球外部分: (x/a)^2 + (y/b)^2 + (z/c)^2 > 1
        keep_mask = ellipsoid_values > (1.0 - current_delta)
    else:
        # 取椭球内部分: (x/a)^2 + (y/b)^2 + (z/c)^2 < 1
        keep_mask = ellipsoid_values < (1.0 + current_delta)

    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    drop_idx = torch.nonzero(~keep_mask, as_tuple=False).squeeze(-1)
    
    return tensor[keep_idx], keep_idx, drop_idx


def filter_tensor_by_hyperplanes_delta(tensor, hyperplanes=None,  pos=None, cov=None, cal_bias=True, delta=0.01 ):
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
 
    

