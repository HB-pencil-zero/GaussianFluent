import torch
import numpy as np
from utils.camera_view_utils import *
from typing import Optional
from particle_filling.filling import *
import sys
sys.path.append("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/gaussian-splatting")
from utils.sh_utils import eval_sh



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


def save_prop_dict(
        save_file: str,
        pos: torch.Tensor,
        cov3D: torch.Tensor,
        rot: torch.Tensor,
        opacity: torch.Tensor,
        shs: torch.Tensor
): 
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建一个字典来存储所有属性
    # .detach(): 将张量从当前的计算图中分离出来。
    #            这样，保存的张量就不再需要梯度信息，变得更轻量。
    # .cpu():    将张量从GPU转移到CPU。
    #            这是一种好的实践，可以确保保存的文件在没有GPU的机器上也能被加载。
    prop_dict = {
        'pos': pos.detach().cpu(),
        'cov3D': cov3D.detach().cpu(),
        'rot': rot.detach().cpu(),
        'opacity': opacity.detach().cpu(),
        'shs': shs.detach().cpu()
    }

    # 使用 torch.save 保存字典
    torch.save(prop_dict, save_file)
    print(f"属性字典已成功保存到: {save_file}")


def load_and_concat_prop_dict(
        load_file: str,
        pos: torch.Tensor,
        cov3D: torch.Tensor,
        rot: torch.Tensor,
        opacity: torch.Tensor,
        shs: torch.Tensor, # 修正了 Tnesor -> Tensor
        bias: list = None,
):
    """
    从文件加载属性字典，并将其中的张量与传入的现有张量沿第一个维度拼接。

    Args:
        load_file (str): 要加载的属性字典文件路径。
        pos, cov3D, rot, opacity, shs (torch.Tensor): 已存在于内存中的张量，
                                                     将与从文件中加载的张量进行拼接。

    """
    if not os.path.exists(load_file):
        print(f"错误: 文件不存在 -> {load_file}")
        return None

    # 从内存中的张量推断出目标设备
    target_device = pos.device


        # 1. 安全地加载文件内容到CPU
    loaded_dict = torch.load(load_file, map_location='cpu')

    if not isinstance(loaded_dict, dict):
        print(f"错误: 文件 '{load_file}' 的内容不是一个字典。")
        return None

    # 2. 将内存中的现有张量组织成一个字典，方便按键名访问
    input_tensors = {
        'pos': pos, 'cov3D': cov3D, 'rot': rot,
        'opacity': opacity, 'shs': shs
    }

    concatenated_dict = {}
    if bias is not None: 
        bias = torch.tensor(bias).to(pos.device)
        input_tensors['pos'] += bias 
    # 3. 遍历每个属性键，进行拼接
    for key in input_tensors.keys():

        # 获取文件中的张量和内存中的张量
        tensor_from_file = loaded_dict[key]
        tensor_from_memory = input_tensors[key]

        # 4. 将从文件加载的张量移动到目标设备，然后使用 torch.cat 拼接
        #    torch.cat 的输入是一个张量列表。dim=0 表示沿着第一个维度（通常是数量维度）拼接。
        concatenated_dict[key] = torch.cat([
            tensor_from_memory,
            tensor_from_file.to(target_device)
        ], dim=0)

    return concatenated_dict['pos'], concatenated_dict['cov3D'], concatenated_dict['rot'], concatenated_dict['opacity'], concatenated_dict['shs']


def azimith_round_array(max_delta,  stage_num , start_azimith):
    list1 = [ max_delta * i / stage_num for i in range(stage_num)]
    list2 = [ max_delta -  max_delta * i / stage_num for i in range(2*stage_num) ]
    list4 = [-max_delta + max_delta * i / stage_num for i in range(stage_num) ]
    return start_azimith + np.array(list1 + list2 + list4)

def  elevation_round_array(max_delta,  stage_num , start_elevation): 
    list1 = [max_delta* i / stage_num  for i in range(2*stage_num)]
    list2 = [2 * max_delta - max_delta * i / stage_num  for i in range(2*stage_num)]
    return start_elevation + np.array(list1 + list2)


def azimith_and_elvation_array(
    start_azimith,      # 起始方位角，也是路径的第一个方位角
    azimith_max_delta,  # 方位角方向的半径
    start_elevation,    # 起始俯仰角，路径的第一个俯仰角，也是椭圆的最低点
    elevation_max_delta, # 俯仰角方向的半径 (椭圆将从 start_elevation 向上扩展)
    stage_num
):
    """
    生成圆形（或椭圆形）采样路径的方位角和俯仰角数组。
    此版本中，(start_azimith, start_elevation) 是路径的第一个点，
    并且是椭圆路径的最低点。

    参数:
        start_azimith (float): 起始方位角。路径的第一个方位角值。
        azimith_max_delta (float): 方位角方向的半径。
        start_elevation (float): 起始俯仰角。路径的第一个俯仰角值，
                                 同时也是椭圆路径的最低俯仰角。
        elevation_max_delta (float): 俯仰角方向的半径。椭圆的最高点将比最低点高 2*elevation_max_delta。
                                     假定为非负值。
        stage_num (int): 用于确定椭圆路径上的点数。总点数将是 4 * stage_num。
                         如果 stage_num <= 0, 只返回起始点。

    返回:
        一个元组 (azimuth_array, elevation_array)。
    """
    num_total_points = 4 * stage_num

    if num_total_points <= 0:
        # 如果 stage_num <= 0, 返回起始点本身。
        return np.array([start_azimith]), np.array([start_elevation])

    # 确定椭圆的中心
    # 方位角中心与 start_azimith 一致
    ellipse_center_az = start_azimith
    # 俯仰角中心在 start_elevation 上方 elevation_max_delta 处，
    # 这样当 sin(t) = -1 时，俯仰角为 start_elevation。
    ellipse_center_el = start_elevation + elevation_max_delta

    # 生成角度参数 t。
    # 为了使 (start_azimith, start_elevation) 成为路径的第一个点和最低点，
    # 我们需要 t 的初始值使得：
    # cos(t_initial) 对于方位角部分在特定情况下为0 (如果椭圆的最低点对应方位角中轴线)
    # sin(t_initial) = -1 对于俯仰角部分。
    # 这对应于角度 -np.pi / 2 (或 3 * np.pi / 2)。
    # 我们从 -np.pi / 2 开始，生成一个完整的 2*pi 周期。
    
    # 参数 t 的范围从 -pi/2 到 -pi/2 + 2pi (不包含端点)
    # 这确保了当 t = -pi/2 时，我们得到最低点。
    # cos(-pi/2) = 0
    # sin(-pi/2) = -1
    start_angle = -np.pi / 2
    t_values = np.linspace(
        start_angle,
        start_angle + 2 * np.pi,
        num_total_points,
        endpoint=False  # 不包括周期的结束点，以避免与起点重复
    )

    # 计算椭圆路径上的方位角和俯仰角值
    # 方位角: ellipse_center_az + azimith_max_delta * cos(t)
    # 当 t = -pi/2 (第一个点):
    # az = ellipse_center_az + azimith_max_delta * 0 = ellipse_center_az = start_azimith
    azimuth_values = ellipse_center_az + azimith_max_delta * np.cos(t_values)
    
    # 俯仰角: ellipse_center_el + elevation_max_delta * sin(t)
    # 当 t = -pi/2 (第一个点):
    # el = ellipse_center_el + elevation_max_delta * (-1)
    #    = (start_elevation + elevation_max_delta) - elevation_max_delta
    #    = start_elevation
    elevation_values = ellipse_center_el + elevation_max_delta * np.sin(t_values)

    return azimuth_values, elevation_values


def uniform_linear_transition_az_el(
    start_azimith,
    target_azimith,
    start_elevation,
    target_elevation,
    stage_num  # 在此函数中，这代表过渡序列中的总点数
):
    """
    生成从起始方位角/俯仰角到目标方位角/俯仰角的均匀线性过渡序列。

    参数:
        start_azimith (float): 起始方位角。
        target_azimith (float): 目标方位角。
        start_elevation (float): 起始俯仰角。
        target_elevation (float): 目标俯仰角。
        stage_num (int):      生成的过渡点数量（包括起始点和目标点）。
                              - 如果 stage_num = 1, 返回包含起始点的数组。
                              - 如果 stage_num = 0, 返回空数组。
                              - stage_num 必须是非负整数。

    返回:
        一个元组 (azimuth_array, elevation_array)，分别包含方位角和俯仰角的 NumPy 数组。
    """
    if not isinstance(stage_num, int) or stage_num < 0:
        raise ValueError("stage_num 必须是一个非负整数。")

    if stage_num == 0:
        return np.array([]), np.array([])
    
    # np.linspace 在 stage_num=1 时会返回包含起始值的数组，
    # 在 stage_num > 1 时会返回包含起始点和目标点的 stage_num 个均匀分布的点。
    azimuth_values = np.linspace(start_azimith, target_azimith, stage_num)
    elevation_values = np.linspace(start_elevation, target_elevation, stage_num)
    
    return azimuth_values, elevation_values

