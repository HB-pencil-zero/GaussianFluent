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

# --- 修改后的 convert_SH 函数 ---
def convert_SH_with_lighting(
    shs_view,                   # 输入的 SH 系数 (N, C, num_coeffs) 或类似形状
    viewpoint_camera,           # 相机对象，需要 .camera_center
    pc,                         # GaussianModel 对象，需要 .max_sh_degree, .active_sh_degree, 并且 *假设* 有法线
    position: torch.Tensor,     # 点的位置 (N, 3)
    normals: torch.Tensor,      # 点的法向量 (N, 3) - **必须提供**
    light_source: dict,         # 光源参数 (包含 position, ambient, diffuse, specular)
    material_phong: dict,       # 材质参数 (包含 specular_color, shininess)
    rotation: torch.Tensor = None, # 可选的旋转
):
    """
    使用球谐函数计算基础颜色，并应用Phong光照模型。

    Args:
        shs_view: 球谐系数张量。
        viewpoint_camera: 包含相机中心信息的相机对象。
        pc: GaussianModel 实例。
        position: 高斯点中心位置 (N, 3)。
        normals: 每个高斯点的法向量 (N, 3)，必须在同一设备上。
        light_source: 包含光源属性的字典。
        material_phong: 包含镜面材质属性的字典。
        rotation: 可选的旋转矩阵。

    Returns:
        最终的 Phong 着色颜色 (N, 3)，范围在 [0, 1]。
    """
    num_points = position.shape[0]
    device = position.device # 获取张量所在的设备

    # 1. 计算基础颜色 (来自 SH) - 作为 ka 和 kd
    # 调整 shs_view 形状以匹配 eval_sh 的期望输入
    # 原始代码: shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    # 假设 shs_view 已经是 (N, C, num_coeffs) 或类似形式， C=3
    # 确保形状是 (N, 3, num_coeffs)
    if shs_view.shape[1] != 3:
         # 如果通道不在第二个维度，尝试调整
         # 例如，如果输入是 (N, num_coeffs, 3)，则转置
         if shs_view.shape[2] == 3:
             shs_view = shs_view.transpose(1, 2)
         else:
             raise ValueError(f"无法处理的 shs_view 形状: {shs_view.shape}")

    # 确保 shs_view 覆盖了正确的点数
    if shs_view.shape[0] != num_points:
         # 可能需要根据 position 的子集来索引 shs_view
         # 或者检查输入是否匹配
         raise ValueError(f"shs_view ({shs_view.shape[0]}) 和 position ({num_points}) 的点数不匹配")


    # 计算视线方向 (从点指向相机)
    view_dir = viewpoint_camera.camera_center.repeat(num_points, 1) - position
    # --- 处理可选旋转 ---
    if rotation is not None:
        # 假设 rotation 作用于 view_dir (或者根据具体语义调整)
        # 原始代码是作用于 dir_pp = position - center，这里我们作用于 view_dir = center - position
        n_rot = rotation.shape[0]
        if n_rot > num_points: n_rot = num_points # 确保索引不越界
        # 注意：原始代码的旋转逻辑可能需要根据具体坐标系和旋转含义调整
        # 这里假设旋转应用于世界坐标下的 view_dir
        view_dir[:n_rot] = torch.matmul(rotation, view_dir[:n_rot].unsqueeze(2)).squeeze(2)

    view_dir_normalized = normalize_batch(view_dir) # V in Phong (points to camera)

    # 使用 eval_sh 计算基础颜色
    # 注意：eval_sh 需要的方向通常是从相机指向点，即 -view_dir_normalized
    # 或者，如果 eval_sh 期望的是从点指向相机的方向，则直接用 view_dir_normalized
    # 假设 eval_sh 期望的是从相机指向点的方向 (dir_pp_normalized in original code)
    dir_pp_normalized = -view_dir_normalized
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    base_color = torch.clamp_min(sh2rgb + 0.5, 0.0) # (N, 3) - 这将用作 Ka 和 Kd

    # 2. 准备 Phong 计算所需的向量和参数
    N = normalize_batch(normals) # 归一化法向量 (N, 3)
    V = view_dir_normalized      # 视线向量 (N, 3)

    light_pos = torch.tensor(light_source['position'], dtype=torch.float32, device=device)
    light_ambient_color = torch.tensor(light_source['ambient'], dtype=torch.float32, device=device) # Ia
    light_diffuse_color = torch.tensor(light_source['diffuse'], dtype=torch.float32, device=device) # Id
    light_specular_color = torch.tensor(light_source['specular'], dtype=torch.float32, device=device) # Is

    mat_ambient_color = base_color  # Ka (N, 3)
    mat_diffuse_color = base_color  # Kd (N, 3)
    mat_specular_color = torch.tensor(material_phong['specular_color'], dtype=torch.float32, device=device) # Ks (3,)
    mat_shininess = material_phong['shininess'] # n (scalar)

    # 计算光照方向向量 (从点指向光源)
    L_vec = light_pos.unsqueeze(0) - position # (N, 3)
    L = normalize_batch(L_vec) # (N, 3)

    # 3. 计算 Phong 光照分量

    # 环境光
    ambient_term = mat_ambient_color * light_ambient_color # (N, 3) * (3,) -> (N, 3)

    # 漫反射
    dot_nl = torch.clamp(dot_product_batch(N, L), min=0.0) # max(0, N dot L)
    diffuse_term = mat_diffuse_color * light_diffuse_color * dot_nl.unsqueeze(1) # (N, 3) * (3,) * (N, 1) -> (N, 3)

    # 镜面反射
    # R = reflect_vectors_batch(L, N) # R = reflect(-L, N)
    # Phong Blinn 优化：使用半程向量 H = normalize(L + V)
    H = normalize_batch(L + V) # (N, 3)
    dot_nh = torch.clamp(dot_product_batch(N, H), min=0.0) # max(0, N dot H)

    # 检查 shininess 是否大于 0
    if mat_shininess > 0:
        specular_intensity = torch.pow(dot_nh, mat_shininess) # (N,)
    else:
        # 处理 shininess 为 0 或负数的情况 (例如，禁用镜面反射)
        specular_intensity = torch.zeros_like(dot_nh) # (N,)

    specular_term = mat_specular_color * light_specular_color * specular_intensity.unsqueeze(1) # (3,) * (3,) * (N, 1) -> (N, 3)

    # 4. 合并光照分量
    final_colors = ambient_term + diffuse_term + specular_term

    # 5. 钳制到 [0, 1] 范围
    final_colors = torch.clamp(final_colors, 0.0, None)

    return final_colors

# --- 示例用法 (需要填充 GaussianModel 和 Camera) ---
if __name__ == '__main__':
    # --- 假设你已经加载了 GaussianModel (pc) 和 viewpoint_camera ---
    class MockGaussianModel:
        def __init__(self, num_points, device):
            self.max_sh_degree = 3
            self.active_sh_degree = 3
            # **关键**: 假设模型有法线属性，或者你需要从其他地方加载
            # self.normals = torch.randn(num_points, 3, device=device)
            # self.normals = F.normalize(self.normals, p=2, dim=1)
            print(f"警告: MockGaussianModel 使用随机 SH 系数和法线。")
            # 假设 SH 系数已经提取并放到了 GPU
            num_sh_coeffs = (self.max_sh_degree + 1)**2
            self.shs_view = torch.randn(num_points, 3, num_sh_coeffs, device=device) * 0.1 # 模拟 SH 系数

    class MockCamera:
        def __init__(self, pos, device):
            self.camera_center = torch.tensor(pos, dtype=torch.float32, device=device)

    # --- 模拟数据 ---
    N_POINTS = 10000
    mock_pc = MockGaussianModel(N_POINTS, device)
    mock_camera = MockCamera([0.0, 0.0, 5.0], device) # 观察者位置

    # 模拟点的位置和法线 (确保在 GPU 上)
    mock_positions = torch.randn(N_POINTS, 3, device=device) * 0.5 # 假设点云在原点附近
    mock_normals = torch.randn(N_POINTS, 3, device=device) # 随机法线
    mock_normals = normalize_batch(mock_normals) # 归一化

    # --- 定义光源和材质 ---
    light = {
        'position': [2.0, 3.0, 4.0],   # 光源位置
        'ambient': [0.1, 0.1, 0.1],   # 环境光强度/颜色
        'diffuse': [0.8, 0.8, 0.8],   # 漫反射光强度/颜色
        'specular': [1.0, 1.0, 1.0]   # 镜面反射光强度/颜色
    }
    material = {
        'specular_color': [0.9, 0.9, 0.9], # 镜面反射颜色 (Ks)
        'shininess': 32.0             # 高光系数 (n)
    }

    # --- 调用函数 ---
    start_time = time.time()
    print("正在计算带光照的颜色...")
    final_point_colors = convert_SH_with_lighting(
        mock_pc.shs_view, # 从模拟模型获取 SH
        mock_camera,
        mock_pc,
        mock_positions,
        mock_normals,      # 传入模拟的法线
        light,
        material
    )
    end_time = time.time()
    print(f"计算完成，耗时: {end_time - start_time:.4f} 秒")
    print("输出颜色张量形状:", final_point_colors.shape) # 应为 (N_POINTS, 3)

    # --- 可视化 (可选) ---
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        print("正在准备可视化...")
        points_np = mock_positions.cpu().numpy()
        colors_np = final_point_colors.cpu().numpy()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors_np, marker='.', s=5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Phong Shaded Points (from SH + Lighting)")
        ax.set_facecolor('black')
        # 设置一个大致的范围，基于模拟数据
        lim = 1.5
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        plt.show()
    except ImportError:
        print("Matplotlib 未安装，跳过可视化。")
    except Exception as e:
        print(f"可视化时出错: {e}")

