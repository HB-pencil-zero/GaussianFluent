import torch
import torch.nn.functional as F # 需要 F 来 normalize
import time

# --- 假设存在 eval_sh 函数 ---
# 你需要确保你有可用的 eval_sh 函数，例如从 3D Gaussian Splatting 库中导入
# from utils.sh_utils import eval_sh
# 这里放一个占位符，你需要替换成实际的实现
def eval_sh(deg, sh, dirs):
    # Placeholder implementation - replace with actual SH evaluation
    # Assumes sh has shape (N, 3, (deg+1)**2) and dirs has shape (N, 3)
    # Returns (N, 3)
    C0 = 0.28209479177387814
    if deg < 0:
         return torch.zeros_like(dirs) # Or handle appropriately
    # Simplistic: Return only the DC component (degree 0) scaled
    # A real implementation uses higher order SH basis functions
    return C0 * sh[:, :, 0] # Returns shape (N, 3)

# --- PyTorch向量操作函数 (复用之前的) ---
def normalize_batch(vectors):
    """批量归一化向量 (PyTorch实现)"""
    # Use F.normalize for potentially better stability/performance
    return F.normalize(vectors, p=2, dim=1)
    # magnitudes = torch.sqrt(torch.sum(vectors**2, dim=1, keepdim=True))
    # magnitudes = torch.clamp(magnitudes, min=1e-10)
    # return vectors / magnitudes

def dot_product_batch(v1, v2):
    """批量计算点积 (PyTorch实现)"""
    return torch.sum(v1 * v2, dim=1)

def reflect_vectors_batch(incident, normal):
    """批量计算反射向量 (PyTorch实现)"""
    # incident should point FROM the light TO the surface for standard reflection
    # L in Phong usually points FROM surface TO light, so we might need -L here
    # Let's assume incident is the vector pointing towards the surface (-L)
    dot_nl = dot_product_batch(normal, incident).unsqueeze(1)
    reflected = incident - 2.0 * dot_nl * normal # Standard reflection formula: I - 2 * dot(N, I) * N
    return reflected


import torch # 确保 torch 已导入

# --- 假设这些辅助函数已定义 ---
def normalize_batch(x):
    # Placeholder: 实现批量归一化
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def dot_product_batch(a, b):
    # Placeholder: 实现批量点积
    return torch.sum(a * b, dim=-1)


# --- 假设这些辅助函数已定义 ---
def normalize_batch(x):
    # Placeholder: 实现批量归一化
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def dot_product_batch(a, b):
    # Placeholder: 实现批量点积
    return torch.sum(a * b, dim=-1)

def reflect_vectors_batch(incident, normal):
    # Placeholder: 实现批量反射向量计算 R = I - 2 * (N dot I) * N
    # 注意：这里的 incident 通常是光线方向 L，但公式常用 I
    # R = L - 2 * dot_product_batch(normal, L).unsqueeze(1) * normal
    # 如果 L 是从点指向光源，入射向量应该是 -L
    neg_L = -incident
    dot_nl = dot_product_batch(normal, neg_L)
    # 确保法线和入射向量指向同一侧时才反射 (dot_nl > 0)
    # 或者更标准的做法是不检查，直接用公式
    reflected = neg_L - 2 * dot_nl.unsqueeze(1) * normal
    return reflected


# --- 修改后的函数，接收 is_lit_mask 并添加衰减 ---
def apply_phong_lighting_to_gaussians_with_mask(
    gaussian_model,             # GaussianModel 对象实例
    viewpoint_camera,           # 相机对象，需要 .camera_center
    is_lit_mask: torch.Tensor,  # 布尔张量 (N,)，True 表示被照亮
    # 可选：覆盖从模型获取的法线
    normals_override: torch.Tensor = None,
    light_source: dict = {
        'position': [0.0, 0.0, 3.0],   # 光源位置
        'ambient': [0.6, 0.6, 0.6],   # 环境光强度
        'diffuse': [0.4, 0.4, 0.4],   # 漫反射光强度
        'specular': [0.2, 0.2, 0.2]   # 镜面反射光强度
    },
    material_phong: dict = {
        'specular_color': [0.8, 0.8, 0.8], # 镜面反射颜色/系数 (白色高光)
        'shininess': 32.0             # 高光指数
    },
    attenuation_constant: float = 20.0, # 光强衰减公式中的常数 (参考代码默认5.0)
    rotation: torch.Tensor = None,     # 可选的旋转
    mask = None,
    return_cpu: bool = False           # 是否将最终结果返回到CPU (默认不返回)
):
    """
    从 GaussianModel 获取数据，计算基础颜色，并应用带衰减和外部阴影遮罩的Phong光照模型。

    Args:
        gaussian_model: GaussianModel 实例，需要提供 ._features_dc, .get_xyz(), .get_normals()。
        viewpoint_camera: 包含相机中心信息的相机对象。
        is_lit_mask: 布尔张量 (N,)，标记每个点是否被光源照亮 (True=亮, False=阴影)。
                     必须与 gaussian_model 中的点数匹配。
        normals_override: 可选，如果提供，则使用此法线张量代替从 gaussian_model 获取的法线。
        light_source: 包含光源属性的字典。
        material_phong: 包含镜面材质属性的字典。
        attenuation_constant: 光照距离衰减的常数。设置为 0 或负数可禁用衰减。
        rotation: 可选的旋转矩阵。
        return_cpu: 是否将最终结果从 GPU 移回 CPU (如果使用了 GPU)。

    Returns:
        Tensor or ndarray: 计算得到的颜色 (N, 3)，根据 return_cpu 决定类型。

    Raises:
        AttributeError: 如果 gaussian_model 缺少必要的属性或方法。
        ValueError: 如果获取的数据形状或 is_lit_mask 形状不兼容。
        TypeError: 如果输入类型不正确。
    """
    overall_start_time = time.time()
    print("开始 Phong 计算 (使用外部光照遮罩)...")

    # --- 从 gaussian_model 获取数据 ---
    try:
        if mask is None :
            mask = torch.arange(start=0, end=gaussian_model._xyz.shape[0]).cuda()
        features_dc_input = gaussian_model._features_dc[mask]
        position = gaussian_model.get_xyz[mask]
        # 获取法线 - 优先使用 override，否则从模型获取
        if normals_override is not None:
            normals = normals_override
        elif hasattr(gaussian_model, 'get_normals'):
             normals = gaussian_model.get_normals()
             if not isinstance(normals, torch.Tensor):
                 raise TypeError("gaussian_model.get_normals() did not return a torch.Tensor")
        elif hasattr(gaussian_model, 'normals'):
             normals = gaussian_model.normals
             if not isinstance(normals, torch.Tensor):
                 raise TypeError("gaussian_model.normals is not a torch.Tensor")
        else:
            raise AttributeError("gaussian_model does not have 'get_normals()' method or 'normals' attribute, and normals_override was not provided.")

    except AttributeError as e:
        raise AttributeError(f"gaussian_model is missing required attribute/method: {e}")

    # --- 检查获取的数据类型 ---
    if not isinstance(features_dc_input, torch.Tensor):
        raise TypeError("gaussian_model._features_dc must be a torch.Tensor")
    if not isinstance(position, torch.Tensor):
        raise TypeError("gaussian_model.get_xyz must be a torch.Tensor")
    if not isinstance(normals, torch.Tensor):
         raise TypeError("Normals must be a torch.Tensor")
    if not isinstance(is_lit_mask, torch.Tensor):
        raise TypeError("is_lit_mask must be a torch.Tensor")
    if is_lit_mask.dtype != torch.bool:
        raise TypeError("is_lit_mask must be a boolean tensor (torch.bool)")

    num_points = position.shape[0]
    device = position.device # 获取张量所在的设备
    print(f"使用设备: {device}, 总点数 N={num_points}")

    # --- 检查 is_lit_mask 形状 ---
    if is_lit_mask.shape != (num_points,):
        raise ValueError(f"is_lit_mask shape ({is_lit_mask.shape}) must match number of points ({num_points}). Expected shape ({num_points},).")

    # --- 确保所有必要张量在同一设备 ---
    print("准备数据并传输到设备 (如果需要)...")
    transfer_start_time = time.time()
    features_dc_input = features_dc_input.to(device)
    normals = normals.to(device)
    is_lit_mask = is_lit_mask.to(device) # 移动 mask 到设备
    if rotation is not None: rotation = rotation.to(device)
    print(f"数据准备和传输耗时: {time.time() - transfer_start_time:.4f} 秒")

    # 1. 计算基础颜色 (来自 DC 特征)
    if features_dc_input.shape[0] != num_points:
         raise ValueError(f"features_dc_input ({features_dc_input.shape[0]}) and position ({num_points}) point counts from gaussian_model do not match.")

    if features_dc_input.ndim == 3 and features_dc_input.shape[1] == 1 and features_dc_input.shape[2] == 3:
        features_dc = features_dc_input.squeeze(1) # (N, 1, 3) -> (N, 3)
    elif features_dc_input.ndim == 2 and features_dc_input.shape[1] == 3:
        features_dc = features_dc_input # Already (N, 3)
    else:
        raise ValueError(f"Incompatible features_dc shape from gaussian_model: {features_dc_input.shape}. Expected (N, 1, 3) or (N, 3).")

    C0 = 0.28209479177387814
    base_color = torch.clamp(features_dc * C0 + 0.5, 0.0, 1.0) # (N, 3)

    # 2. 准备 Phong 计算所需的向量和参数
    print("计算基础光照向量和参数...")
    calc_start_time = time.time()
    N_norm_original = normalize_batch(normals) # (N, 3) - 存储原始归一化法线

    # 光源参数
    light_pos = torch.tensor(light_source['position'], dtype=torch.float32, device=device)
    light_ambient = torch.tensor(light_source['ambient'], dtype=torch.float32, device=device)
    light_diffuse = torch.tensor(light_source['diffuse'], dtype=torch.float32, device=device)
    light_specular = torch.tensor(light_source['specular'], dtype=torch.float32, device=device)

    # 材质参数
    mat_specular = torch.tensor(material_phong['specular_color'], dtype=torch.float32, device=device)
    mat_shininess = float(material_phong['shininess'])

    # 计算光照方向向量 (从点指向光源) 和距离
    L_vec = light_pos.unsqueeze(0) - position # (N, 3)
    distance_sq = torch.sum(L_vec**2, dim=1, keepdim=True) # (N, 1)
    distance_sq = torch.clamp(distance_sq, min=1e-6) # 避免除零
    # distance_to_light = torch.sqrt(distance_sq) # (N, 1) - 如果需要实际距离
    L = normalize_batch(L_vec) # (N, 3)

    # 计算视线方向向量
    cam_center_tensor = viewpoint_camera.camera_center
    if not isinstance(cam_center_tensor, torch.Tensor):
        cam_center_tensor = torch.tensor(cam_center_tensor, dtype=position.dtype, device=device)
    else:
        cam_center_tensor = cam_center_tensor.to(device=device, dtype=position.dtype)

    if cam_center_tensor.ndim == 1: cam_center_tensor = cam_center_tensor.unsqueeze(0)
    if cam_center_tensor.shape[0] == 1 and num_points > 1:
         view_dir = cam_center_tensor.repeat(num_points, 1) - position
    elif cam_center_tensor.shape[0] == num_points:
         view_dir = cam_center_tensor - position
    else:
         raise ValueError(f"viewpoint_camera.camera_center shape ({cam_center_tensor.shape}) incompatible with position ({position.shape}).")

    if rotation is not None:
        n_rot = rotation.shape[0]
        if n_rot > num_points: n_rot = num_points
        view_dir[:n_rot] = torch.matmul(rotation, view_dir[:n_rot].unsqueeze(2)).squeeze(2)

    V = normalize_batch(view_dir)      # (N, 3)

    # --- 3. 法线翻转 (基于 is_lit_mask) ---
    # 仅对被照亮的点，如果其原始法线背离光源，则翻转法线用于后续计算
    dot_nl_initial = torch.sum(N_norm_original * L, dim=1) # (N,)
    needs_flip = (dot_nl_initial < 0.0)                   # (N,)
    flip_mask = is_lit_mask & needs_flip                  # (N,) 布尔掩码

    N_norm = torch.where(flip_mask.unsqueeze(1), -N_norm_original, N_norm_original) # (N, 3)

    num_flipped = torch.sum(flip_mask).item()
    if num_flipped > 0:
        print(f"信息: {num_flipped} 个被照亮点的法线因点积为负而被翻转以进行光照计算。")

    # --- 4. 计算 Phong 光照分量 (使用可能修正过的 N_norm) ---
    # 计算反射向量 R (使用修正后的 N_norm)
    R = reflect_vectors_batch(L, N_norm) # (N, 3)
    R = normalize_batch(R) # 确保 R 是单位向量

    # 环境光 (不受阴影和衰减影响)
    ambient_term = base_color * light_ambient # (N, 3)

    # 漫反射基础贡献
    diffuse_intensity = torch.clamp(dot_product_batch(N_norm, L), min=0.0) # (N,)
    diffuse_base = base_color * light_diffuse * diffuse_intensity.unsqueeze(1) # (N, 3)

    # 镜面反射基础贡献
    dot_rv = torch.clamp(dot_product_batch(R, V), min=0.0) # (N,)
    if mat_shininess > 0:
        specular_intensity = torch.pow(dot_rv, mat_shininess) # (N,)
    else:
        specular_intensity = torch.zeros_like(dot_rv)
    specular_base = mat_specular * light_specular * specular_intensity.unsqueeze(1) # (N, 3)
    print(f"基础光照计算耗时: {time.time() - calc_start_time:.4f} 秒")

    # --- 5. 计算衰减和阴影因子 ---
    print("应用衰减和阴影遮罩...")
    apply_start_time = time.time()
    # 计算衰减因子
    if attenuation_constant > 0:
        attenuation = attenuation_constant / distance_sq # (N, 1)
    else:
        attenuation = torch.ones_like(distance_sq) # 禁用衰减

    # 从 is_lit_mask 创建阴影因子 (1.0 for lit, 0.0 for shadow)
    shadow_factors = is_lit_mask.float().unsqueeze(1) # (N, 1)

    # --- 6. 应用衰减和阴影因子 ---
    diffuse_term = diffuse_base * attenuation * shadow_factors # (N, 3)
    specular_term = specular_base * attenuation * shadow_factors # (N, 3)

    # --- 7. 合并所有光照分量 ---
    colors_final_gpu = ambient_term + diffuse_term + specular_term # (N, 3)

    # --- 8. 将颜色限制在[0,1]范围内 ---
    colors_final_gpu = torch.clamp(colors_final_gpu, 0.0, 1.0)
    print(f"应用衰减和阴影耗时: {time.time() - apply_start_time:.4f} 秒")

    overall_end_time = time.time()
    print(f"总计算耗时 (不含最终数据传输): {overall_end_time - overall_start_time:.4f} 秒")

    # --- 返回结果 ---
    if return_cpu:
        if device != torch.device('cpu'):
            print("将结果传输回 CPU...")
            final_colors_cpu = colors_final_gpu.cpu().numpy()
            print("完成。")
            return final_colors_cpu
        else:
            print("计算在 CPU 上完成，返回 NumPy 数组。")
            return colors_final_gpu.numpy() # 如果在 CPU 计算，直接转 NumPy
    else:
        print("完成。结果保留在计算设备上。")
        return colors_final_gpu # 返回 GPU/CPU Tensor