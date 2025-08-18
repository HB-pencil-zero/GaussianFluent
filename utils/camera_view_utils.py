import os
import json
import numpy as np
import torch
from scene.cameras import Camera as GSCamera
from utils.graphics_utils import focal2fov


def generate_camera_rotation_matrix(camera_to_object, object_vertical_downward):
    camera_to_object = camera_to_object / np.linalg.norm(
        camera_to_object
    )  # last column
    # the second column of rotation matrix is pointing toward the downward vertical direction
    camera_y = (
        object_vertical_downward
        - np.dot(object_vertical_downward, camera_to_object) * camera_to_object
    )
    camera_y = camera_y / np.linalg.norm(camera_y)  # second column
    first_column = np.cross(camera_y, camera_to_object)
    R = np.column_stack((first_column, camera_y, camera_to_object))
    return R


# supply vertical vector in world space
def generate_local_coord(vertical_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:
        horizontal_1 = np.array([0.72, 0.37, -0.67])
    # gram schimit
    horizontal_1 = (
        horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    )
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)

    return vertical_vector, horizontal_1, horizontal_2


# scalar (in degrees), scalar (in degrees), scalar, vec3, mat33 = [horizontal_1; horizontal_2; vertical];  -> vec3
def get_point_on_sphere(azimuth, elevation, radius, center, observant_coordinates):
    canonical_coordinates = (
        np.array(
            [
                np.cos(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(elevation / 180.0 * np.pi),
            ]
        )
        * radius
    )

    return center + observant_coordinates @ canonical_coordinates


def get_camera_position_and_rotation(
    azimuth, elevation, radius, view_center, observant_coordinates
):
    # get camera position
    position = get_point_on_sphere(
        azimuth, elevation, radius, view_center, observant_coordinates
    )
    # get rotation matrix
    R = generate_camera_rotation_matrix(
        view_center - position, -observant_coordinates[:, 2]
    )
    return position, R


def get_current_radius_azimuth_and_elevation(
    camera_position, view_center, observesant_coordinates
):
    center2camera = -view_center + camera_position
    radius = np.linalg.norm(center2camera)
    dot_product = np.dot(center2camera, observesant_coordinates[:, 2])
    cosine = dot_product / (
        np.linalg.norm(center2camera) * np.linalg.norm(observesant_coordinates[:, 2])
    )
    elevation = np.rad2deg(np.pi / 2.0 - np.arccos(cosine))
    proj_onto_hori = center2camera - dot_product * observesant_coordinates[:, 2]
    dot_product2 = np.dot(proj_onto_hori, observesant_coordinates[:, 0])
    cosine2 = dot_product2 / (
        np.linalg.norm(proj_onto_hori) * np.linalg.norm(observesant_coordinates[:, 0])
    )

    if np.dot(proj_onto_hori, observesant_coordinates[:, 1]) > 0:
        azimuth = np.rad2deg(np.arccos(cosine2))
    else:
        azimuth = -np.rad2deg(np.arccos(cosine2))
    return radius, azimuth, elevation


def get_camera_view(
    model_path,
    default_camera_index=0,
    center_view_world_space=None,
    observant_coordinates=None,
    show_hint=False,
    init_azimuthm=None,
    init_elevation=None,
    init_radius=None,
    move_camera=False,
    current_frame=0,
    delta_a=0,
    delta_e=0,
    delta_r=0,
    scales = 1.1 ,
):
    """Load one of the default cameras for the scene."""
    cam_path = os.path.join(model_path, "cameras.json")
    with open(cam_path) as f:
        data = json.load(f)

        if show_hint:
            if default_camera_index < 0:
                default_camera_index = 0
            r, a, e = get_current_radius_azimuth_and_elevation(
                data[default_camera_index]["position"],
                center_view_world_space,
                observant_coordinates,
            )
            print("Default camera ", default_camera_index, " has")
            print("azimuth:    ", a)
            print("elevation:  ", e)
            print("radius:     ", r)
            print("Now exit program and set your own input!")
            exit()

        if default_camera_index > -1:
            raw_camera = data[default_camera_index]

        else:
            raw_camera = data[0]  # get data to be modified

            assert init_azimuthm is not None
            assert init_elevation is not None
            assert init_radius is not None

            if move_camera:
                assert delta_a is not None
                assert delta_e is not None
                assert delta_r is not None
                position, R = get_camera_position_and_rotation(
                    init_azimuthm + current_frame * delta_a,
                    init_elevation + current_frame * delta_e,
                    init_radius + current_frame * delta_r,
                    center_view_world_space,
                    observant_coordinates,
                )
            else:
                position, R = get_camera_position_and_rotation(
                    init_azimuthm,
                    init_elevation,
                    init_radius,
                    center_view_world_space,
                    observant_coordinates,
                )
            raw_camera["rotation"] = R.tolist()
            raw_camera["position"] = position.tolist()

        tmp = np.zeros((4, 4))
        tmp[:3, :3] = raw_camera["rotation"]
        tmp[:3, 3] = raw_camera["position"]
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]


        width = 1280    
        height = 720    
        # width = int(raw_camera["width"])  
        # height = int(raw_camera["height"] )  

        # width = 700  
        # height = 700
        
        fovx = focal2fov( scales *raw_camera["fx"], width)
        fovy = focal2fov( scales * raw_camera["fy"], height)

        return GSCamera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros((3, height, width)),  # fake
            gt_alpha_mask=None,
            image_name="fake",
            uid=0,
        )



def reconstruct_cov_from_flat(uncertainty_flat):
    """
    将 (n, 6) 的扁平化协方差表示转换回 (n, 3, 3) 的完整协方差矩阵。
    uncertainty_flat 的元素对应: [c00, c01, c02, c11, c12, c22]
    """
    n = uncertainty_flat.shape[0]
    # 确保在与 uncertainty_flat 相同的设备和数据类型上创建
    cov_matrices = torch.zeros((n, 3, 3), dtype=uncertainty_flat.dtype, device=uncertainty_flat.device)

    cov_matrices[:, 0, 0] = uncertainty_flat[:, 0]
    cov_matrices[:, 0, 1] = uncertainty_flat[:, 1]
    cov_matrices[:, 1, 0] = uncertainty_flat[:, 1] # 对称性
    cov_matrices[:, 0, 2] = uncertainty_flat[:, 2]
    cov_matrices[:, 2, 0] = uncertainty_flat[:, 2] # 对称性
    cov_matrices[:, 1, 1] = uncertainty_flat[:, 3]
    cov_matrices[:, 1, 2] = uncertainty_flat[:, 4]
    cov_matrices[:, 2, 1] = uncertainty_flat[:, 4] # 对称性
    cov_matrices[:, 2, 2] = uncertainty_flat[:, 5]
    return cov_matrices

def flatten_cov_to_flat(cov_matrices):
    """
    将 (n, 3, 3) 的完整协方差矩阵转换回 (n, 6) 的扁平化表示。
    输出的扁平化表示对应: [c00, c01, c02, c11, c12, c22]
    """
    n = cov_matrices.shape[0]
    # 确保在与 cov_matrices 相同的设备和数据类型上创建
    uncertainty_flat = torch.zeros((n, 6), dtype=cov_matrices.dtype, device=cov_matrices.device)

    uncertainty_flat[:, 0] = cov_matrices[:, 0, 0]
    uncertainty_flat[:, 1] = cov_matrices[:, 0, 1]
    uncertainty_flat[:, 2] = cov_matrices[:, 0, 2]
    uncertainty_flat[:, 3] = cov_matrices[:, 1, 1]
    uncertainty_flat[:, 4] = cov_matrices[:, 1, 2]
    uncertainty_flat[:, 5] = cov_matrices[:, 2, 2]
    return uncertainty_flat

def rotate_flat_covariance(uncertainty_flat, R_batch):
    """
    对以 (n, 6) 格式表示的一批协方差施加旋转。

    参数:
        uncertainty_flat (torch.Tensor): 形状为 (n, 6) 的张量，表示 n 个协方差矩阵的
                                         上三角元素 [c00, c01, c02, c11, c12, c22]。
        R_batch (torch.Tensor): 旋转矩阵。可以是：
                                - (3, 3): 应用于所有协方差的单个旋转矩阵。
                                - (n, 3, 3): 一批旋转矩阵，每个协方差对应一个。

    返回:
        torch.Tensor: 形状为 (n, 6) 的张量，表示旋转后的协方差，格式与输入相同。
    """
    n = uncertainty_flat.shape[0]

    # 1. 从 (n, 6) 重构为 (n, 3, 3)
    cov_matrices = reconstruct_cov_from_flat(uncertainty_flat) # Shape: (n, 3, 3)

    # 2. 应用旋转: Cov_rot = R @ Cov @ R.T
    if R_batch.ndim == 2: # 单个 (3,3) 旋转矩阵
        if R_batch.shape != (3, 3):
            raise ValueError(f"单个旋转矩阵 R_batch 必须是 (3,3)，得到 {R_batch.shape}")
        # R_batch (3,3), cov_matrices (n,3,3)
        # R_batch @ cov_matrices -> (n,3,3) (利用matmul的广播规则)
        # (n,3,3) @ R_batch.T (3,3) -> (n,3,3)
        R_T_batch = R_batch.T
        rotated_cov_matrices = R_batch @ cov_matrices @ R_T_batch
    elif R_batch.ndim == 3: # 一批 (n,3,3) 旋转矩阵
        if R_batch.shape != (n, 3, 3):
            raise ValueError(f"批量旋转矩阵 R_batch 必须是 ({n},3,3)，得到 {R_batch.shape}")
        # R_batch (n,3,3), cov_matrices (n,3,3)
        # R_batch @ cov_matrices -> (n,3,3) (逐元素批次矩阵乘法)
        # (n,3,3) @ R_batch.transpose(-2, -1) (n,3,3) -> (n,3,3)
        R_T_batch = R_batch.transpose(-2, -1) # 转置最后两个维度
        rotated_cov_matrices = R_batch @ cov_matrices @ R_T_batch
    else:
        raise ValueError("R_batch 必须是形状为 (3,3) 或 (n,3,3) 的张量")

    # 3. 将旋转后的 (n, 3, 3) 协方差矩阵转换回 (n, 6) 格式
    rotated_uncertainty_flat = flatten_cov_to_flat(rotated_cov_matrices)

    return rotated_uncertainty_flat