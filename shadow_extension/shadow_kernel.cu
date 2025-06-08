#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int MAX_SHARED_MEMORY = 48 * 1024;  // 48KB shared memory

__device__ __forceinline__ float dot_product(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__global__ void shadow_kernel(
    const float* points,          // [N, 3]
    const float* occluders,       // [M, 3]
    const float* L,               // [N, 3]
    const float* dist_light_sq,   // [N]
    int N, int M,
    float shadow_epsilon_sq,
    float alignment_threshold_sq,
    int num_surface_points,
    float* shadow_factors         // [N] 输出阴影因子 (0或1)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 point = make_float3(points[idx*3], points[idx*3+1], points[idx*3+2]);
    float3 light_dir = make_float3(L[idx*3], L[idx*3+1], L[idx*3+2]);
    float max_dist_sq = dist_light_sq[idx];
    int occluder_count = 0;

    for (int j = 0; j < M; j++) {
        float3 occluder = make_float3(
            occluders[j*3], occluders[j*3+1], occluders[j*3+2]
        );
        
        // 计算向量差
        float3 diff = make_float3(
            occluder.x - point.x,
            occluder.y - point.y,
            occluder.z - point.z
        );
        
        // 计算距离平方
        float dist_sq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
        
        // 检查距离条件
        if (dist_sq <= shadow_epsilon_sq || dist_sq >= max_dist_sq) 
            continue;
        
        // 计算点积
        float dot_product_val = diff.x*light_dir.x + diff.y*light_dir.y + diff.z*light_dir.z;
        
        // 检查方向条件
        if (dot_product_val <= 0) continue;
        
        // 检查对齐条件 (使用平方比较避免开方)
        float dot_sq = dot_product_val * dot_product_val;
        if (dot_sq <= alignment_threshold_sq * dist_sq) 
            continue;
        
        // 符合条件的遮挡点
        occluder_count++;
        
        // 提前终止
        if (occluder_count > num_surface_points) 
            break;
    }
    
    // 关键修复：正确设置阴影因子
    if (occluder_count > num_surface_points) {
        shadow_factors[idx] = 0.0f;  // 在阴影中
    } else {
        shadow_factors[idx] = 1.0f;  // 不在阴影中
    }
}

torch::Tensor calculate_shadows_cuda(
    torch::Tensor points,
    torch::Tensor occluders,
    torch::Tensor L,
    torch::Tensor dist_light_sq,
    float shadow_epsilon,
    float alignment_threshold,
    int num_surface_points
) {
    CHECK_INPUT(points);
    CHECK_INPUT(occluders);
    CHECK_INPUT(L);
    CHECK_INPUT(dist_light_sq);
    
    int N = points.size(0);
    int M = occluders.size(0);
    
    // 关键修复：初始化为全1
    torch::Tensor shadow_factors = torch::ones(
        {N}, 
        torch::device(points.device()).dtype(torch::kFloat32)
    );
    
    if (N == 0 || M == 0) {
        return shadow_factors;
    }
    
    float shadow_epsilon_sq = shadow_epsilon * shadow_epsilon;
    float alignment_threshold_sq = alignment_threshold * alignment_threshold;
    
    // 配置CUDA内核
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    // 调用内核
    shadow_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        occluders.data_ptr<float>(),
        L.data_ptr<float>(),
        dist_light_sq.data_ptr<float>(),
        N, M,
        shadow_epsilon_sq,
        alignment_threshold_sq,
        num_surface_points,
        shadow_factors.data_ptr<float>()
    );
    
    // 同步确保内核执行完成
    cudaDeviceSynchronize();
    
    return shadow_factors;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_shadows_cuda", &calculate_shadows_cuda, "Calculate shadows with CUDA");
}