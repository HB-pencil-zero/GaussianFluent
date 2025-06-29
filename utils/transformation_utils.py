import torch
import numpy as np
from utils.camera_view_utils import *
from typing import Optional

def save_core_init_render_vars(
    filepath: str,
    mpm_init_pos: Optional[torch.Tensor], # <--- 修改这里
    mpm_init_vol: Optional[torch.Tensor], # <--- 修改这里
    mpm_init_cov: Optional[torch.Tensor], # <--- 修改这里
    opacity_render: Optional[torch.Tensor], # <--- 修改这里
    shs_render: Optional[torch.Tensor], # <--- 修改这里
    mask : torch.Tensor,
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
        "mask": prep_tensor(mask),
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
        'opacity_render', 'shs_render', 'mask')，如果文件不存在或加载失败则返回 None。
        注意：字典中的值可能是 Tensor 或 None(如果保存时是 None)。
    """
    print(f"\n尝试从以下路径加载核心初始化和渲染变量: {filepath}")

    if not os.path.exists(filepath):
        print(f"错误：文件未找到 - {filepath}")
        return None

    try:
        # 使用 torch.load 加载数据
        # map_location 参数确保张量被加载到指定的设备
        loaded_data = torch.load(filepath, map_location=map_location)

        # 验证加载的数据是否是字典（可选但推荐）
        if not isinstance(loaded_data, dict):
            print(f"错误：加载的文件内容不是预期的字典格式 - {filepath}")
            return None

        # 验证是否包含预期的键（可选但推荐）
        expected_keys = {"mpm_init_pos", "mpm_init_vol", "mpm_init_cov", "opacity_render", "shs_render", "mask"}
        if not expected_keys.issubset(loaded_data.keys()):
            print(f"警告：加载的字典缺少部分预期键。文件路径: {filepath}")
            # 你可以选择仍然返回字典，或者返回 None，取决于你的需求

        print(f"核心初始化和渲染变量已成功加载自: {filepath}")
        return loaded_data

    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return None


# 在文件开头添加这行
import torch
torch.backends.cuda.preferred_linalg_library('cusolver')  # 或者 'magma'

# 或者强制使用 CPU 进行线性代数运算
def decompose_covariance_to_scaling_rotation(symm, scaling_modifier):
    n = symm.shape[0]
    device = symm.device
    
    # 重建协方差矩阵
    cov = torch.zeros((n, 3, 3), device=device)
    cov[:, 0, 0] = symm[:, 0]
    cov[:, 0, 1] = cov[:, 1, 0] = symm[:, 1]
    cov[:, 0, 2] = cov[:, 2, 0] = symm[:, 2]
    cov[:, 1, 1] = symm[:, 3]
    cov[:, 1, 2] = cov[:, 2, 1] = symm[:, 4]
    cov[:, 2, 2] = symm[:, 5]
    
    # 转到 CPU 进行线性代数运算
    cov_cpu = cov.cpu()
    
    # 特征值分解
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_cpu)
    eigenvalues = torch.clamp(eigenvalues, min=1e-6)
    
    # 重构 L
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    L = eigenvectors * sqrt_eigenvalues.unsqueeze(-2)
    
    # SVD 分解
    U, S, Vh = torch.linalg.svd(L)
    
    # 确保旋转矩阵的行列式为正
    det = torch.det(U @ Vh)
    U = torch.where(det.unsqueeze(-1).unsqueeze(-1) < 0,
                    torch.cat([U[:, :, :-1], -U[:, :, -1:]], dim=-1), U)
    S = torch.where(det.unsqueeze(-1) < 0,
                    torch.cat([S[:, :-1], -S[:, -1:]], dim=-1), S)
    
    # 转回原设备
    R = (U @ Vh).to(device)
    scaling = (S / scaling_modifier).to(device)
    rotation = matrix_to_quaternion(R)
    
    return scaling, rotation


def matrix_to_quaternion(R):
    """旋转矩阵转四元数"""
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    q = torch.zeros((R.shape[0], 4), device=R.device)
    
    # w分量
    q[:, 0] = torch.sqrt(torch.clamp(1 + trace, min=0)) / 2
    
    # xyz分量
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * q[:, 0])
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * q[:, 0])
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * q[:, 0])
    
    return q


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





def transform2origin(position_tensor):
    min_pos = torch.min(position_tensor, 0)[0]
    max_pos = torch.max(position_tensor, 0)[0]
    max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    scale = 1.0 / max_diff
    original_mean_pos = original_mean_pos.to(device="cuda")
    scale = scale.to(device="cuda")
    new_position_tensor = (position_tensor - original_mean_pos) * scale

    return new_position_tensor, scale, original_mean_pos


def undotransform2origin(position_tensor, scale, original_mean_pos):
    return original_mean_pos + position_tensor / scale


def generate_rotation_matrix(degree, axis):
    cos_theta = torch.cos(degree / 180.0 * 3.1415926)
    sin_theta = torch.sin(degree / 180.0 * 3.1415926)
    if axis == 0:
        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:
        rotation_matrix = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    elif axis == 2:
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError("Invalid axis selection")
    return rotation_matrix.cuda()


def generate_rotation_matrices(degrees, axises):
    assert len(degrees) == len(axises)

    matrices = []

    for i in range(len(degrees)):
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))

    return matrices


def apply_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix.T)
    return rotated


def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated


def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device="cuda")
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)


def get_uppder_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device="cuda")
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat


def apply_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        position_tensor = apply_rotation(position_tensor, rotation_matrices[i])
    return position_tensor


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices[i])
    return get_uppder_from_mat(cov_tensor)


def shift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor + tensor111


def undoshift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor - tensor111


def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated


def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor


def apply_inverse_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        cov_tensor = apply_cov_rotation(cov_tensor, R.T)
    return get_uppder_from_mat(cov_tensor)


# input must be (n,3) tensor on cuda
def undo_all_transforms(input, rotation_matrices, scale_origin, original_mean_pos):
    return apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )


def get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
):
    viewpoint_center_worldspace = undo_all_transforms(
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos
    )
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center
    worldspace_up = undo_all_transforms(
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace
    viewpoint_center_worldspace = np.squeeze(
        viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0
    )
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )
    observant_coordinates = np.column_stack((h1, h2, vertical))

    return viewpoint_center_worldspace, observant_coordinates
