import warp as wp
from warp_utils import *
import numpy as np
import math



# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tau = 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)
    return tau


@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)



@wp.func
def kirchoff_stress_neoHookeanBoarden(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float, kappa: float, state: MPMStateStruct
):
    # 计算B = F*F^T
    B = F * wp.transpose(F)
    
    # 计算B的迹
    B_trace = B[0, 0] + B[1, 1] + B[2, 2]
    
    # 计算偏差部分 devB = B - I * (1/3) * trace(B)
    devB_00 = B[0, 0] - B_trace / 3.0
    devB_11 = B[1, 1] - B_trace / 3.0
    devB_22 = B[2, 2] - B_trace / 3.0
    devB_01 = B[0, 1]
    devB_02 = B[0, 2]
    devB_10 = B[1, 0]
    devB_12 = B[1, 2]
    devB_20 = B[2, 0]
    devB_21 = B[2, 1]
    
    # 计算偏差应力 tau_dev = mu * J^(-2/3) * devB
    scale_dev = mu * J**(-2.0/3.0)
    tau_dev = wp.mat33(
        scale_dev * devB_00, scale_dev * devB_01, scale_dev * devB_02,
        scale_dev * devB_10, scale_dev * devB_11, scale_dev * devB_12,
        scale_dev * devB_20, scale_dev * devB_21, scale_dev * devB_22
    )
    
    # 计算体积应力 tau_vol = J * prime * I，其中 prime = kappa/2 * (J - 1/J)
    prime = kappa / 2.0 * (J - 1.0/J)
    scale_vol = J * prime
    tau_vol = wp.mat33(
        scale_vol, 0.0, 0.0,
        0.0, scale_vol, 0.0,
        0.0, 0.0, scale_vol
    )
    
    # 根据J的值选择不同的组合方式
    tau = tau_dev + tau_vol
    
    return tau




@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, dt: float
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.func
def NonAssociativeCamClay_return_mapping(
    F_trial: wp.mat33,
    state: MPMStateStruct,
    model: MPMModelStruct,
    p: int
):
    # SVD分解
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma = wp.vec3(0.0)
    wp.svd3(F_trial, U, sigma, V)

    # 从state和model读取参数
    logJp = state.particle_Jp[p]
    mu = model.mu[p]
    kappa = model.kappa[p]
    M = model.M
    beta = model.beta[p]
    xi = model.xi
    hardeningOn = model.hardening

    # 防止NaN，设置最小阈值
    threshold = 0.0
    sigma[0] = wp.max(sigma[0], threshold)
    sigma[1] = wp.max(sigma[1], threshold)
    sigma[2] = wp.max(sigma[2], threshold)

    # 计算p0 - 屈服面参数
    p0 = kappa * (0.00001 + wp.sinh(xi * wp.max(-logJp, 0.0)))

    # 计算J和相关量
    J = sigma[0] * sigma[1] * sigma[2]

    # 计算B_hat_trial
    B_hat_trial_0 = sigma[0] * sigma[0]
    B_hat_trial_1 = sigma[1] * sigma[1]
    B_hat_trial_2 = sigma[2] * sigma[2]

    # 计算B_hat_trial的平均值
    B_hat_trial_mean = (B_hat_trial_0 + B_hat_trial_1 + B_hat_trial_2) / 3.0

    # 计算偏应力
    J_pow = wp.pow(J, -2.0/3.0)
    s_hat_trial_0 = mu * J_pow * (B_hat_trial_0 - B_hat_trial_mean)
    s_hat_trial_1 = mu * J_pow * (B_hat_trial_1 - B_hat_trial_mean)
    s_hat_trial_2 = mu * J_pow * (B_hat_trial_2 - B_hat_trial_mean)

    # 计算压力
    prime = kappa / 2.0 * (J - 1.0/J)
    p_trial = -prime * J

    # 计算屈服函数y
    dim = 3.0  # 三维情况
    y_s_half_coeff = (6.0 - dim) / 2.0 * (1.0 + 2.0 * beta)
    y_p_half = M * M * (p_trial + beta * p0) * (p_trial - p0)

    # 计算s_hat_trial的平方和
    s_hat_trial_sq_sum = s_hat_trial_0 * s_hat_trial_0 + s_hat_trial_1 * s_hat_trial_1 + s_hat_trial_2 * s_hat_trial_2
    y = y_s_half_coeff * s_hat_trial_sq_sum + y_p_half

    # 初始化最终的sigma值为原始值
    sigma_final_0 = sigma[0]
    sigma_final_1 = sigma[1]
    sigma_final_2 = sigma[2]

    # 更新logJp的初始值
    logJp_new = logJp

    # 投影到顶点：处理p_trial > p0的情况
    p_min = beta * p0
    if p_trial > p0:
        Je_new = wp.sqrt(-2.0 * p0 / kappa + 1.0)
        sigma_final_0 = wp.pow(Je_new, 1.0/3.0)
        sigma_final_1 = sigma_final_0
        sigma_final_2 = sigma_final_0
        
        # 更新硬化参数
        if hardeningOn > 0.5:
            logJp_new = logJp + wp.log(J / Je_new)

    # 投影到顶点：处理p_trial < -p_min的情况
    elif p_trial < -p_min:
        Je_new = wp.sqrt(2.0 * p_min / kappa + 1.0)
        sigma_final_0 = wp.pow(Je_new, 1.0/3.0)
        sigma_final_1 = sigma_final_0
        sigma_final_2 = sigma_final_0
        
        # 更新硬化参数
        if hardeningOn > 0.5:
            logJp_new = logJp + wp.log(J / Je_new)

    # 处理屈服面上的点
    elif y >= 1e-4:
        # 计算s_hat_trial的范数
        s_hat_trial_norm = wp.sqrt(s_hat_trial_sq_sum)
        s_hat_trial_norm = wp.max(s_hat_trial_norm, 1e-10)  # 避免除以零
        
        # 计算新的B_hat
        sqrt_factor = wp.sqrt(-y_p_half / y_s_half_coeff)
        J_pow_2 = wp.pow(J, 2.0/3.0)
        scale_factor = J_pow_2 / mu * sqrt_factor / s_hat_trial_norm
        
        B_hat_new_0 = scale_factor * s_hat_trial_0 + B_hat_trial_mean
        B_hat_new_1 = scale_factor * s_hat_trial_1 + B_hat_trial_mean
        B_hat_new_2 = scale_factor * s_hat_trial_2 + B_hat_trial_mean
        
        # 计算新的sigma
        sigma_final_0 = wp.sqrt(B_hat_new_0)
        sigma_final_1 = wp.sqrt(B_hat_new_1)
        sigma_final_2 = wp.sqrt(B_hat_new_2)
        
        # 硬化处理
        if hardeningOn > 0.5 and p0 > 1e-4 and p_trial < p0 - 1e-4 and p_trial > 1e-4 - p_min:
            p_center = (p0 - p_min) * 0.5
            q_trial = wp.sqrt((6.0 - dim) / 2.0) * s_hat_trial_norm
            
            # 计算方向向量
            direction_p = p_center - p_trial
            direction_q = 0.0 - q_trial
            direction_norm = wp.sqrt(direction_p * direction_p + direction_q * direction_q)
            direction_norm = wp.max(direction_norm, 1e-10)
            direction_p = direction_p / direction_norm
            
            # 计算二次方程系数
            C = M * M * (p_center + beta * p0) * (p_center - p0)
            B = M * M * direction_p * (2.0 * p_center - p0 + beta * p0)
            A = M * M * direction_p * direction_p + (1.0 + 2.0 * beta) * direction_q * direction_q
            
            # 解二次方程
            discriminant = B * B - 4.0 * A * C
            # discriminant = wp.max(discriminant, 0.0)  # 确保判别式非负
            
            l1 = (-B + wp.sqrt(discriminant)) / (2.0 * A)
            l2 = (-B - wp.sqrt(discriminant)) / (2.0 * A)
            
            p1 = p_center + l1 * direction_p
            p2 = p_center + l2 * direction_p
            
            # 选择正确的解
            diff_trial = p_trial - p_center
            diff1 = p1 - p_center
            
            # 计算条件：(p_trial - p_center) * (p1 - p_center) > 0
            if diff_trial * diff1 > 0.0:
                p_fake = p1
            else:
                p_fake = p2
            
            Je_new_fake = wp.sqrt(wp.abs(-2.0 * p_fake / kappa + 1.0))
            
            # 更新硬化参数
            if Je_new_fake > 1e-4  and hardeningOn > 0.5:
                logJp_new = logJp + wp.log(J / Je_new_fake)

    # 更新state中的塑性参数
    state.particle_Jp[p] = logJp_new

    # 构建对角矩阵
    sigma_diag = wp.mat33(
        sigma_final_0, 0.0, 0.0,
        0.0, sigma_final_1, 0.0,
        0.0, 0.0, sigma_final_2
    )
    
    # 计算新的F
    F_new = U * sigma_diag * wp.transpose(V)
    return F_new



@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (2.0 * (1.0 + model.nu[p]))
    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )
    model.kappa[p] = 2.0* model.mu[p] /3.0 + model.lam[p]

@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (
                        wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    ) * model.dx
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    C = state.particle_C[p]
                    # if model.rpic = 0, standard apic
                    C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                        C - wp.transpose(C)
                    )
                    if model.rpic_damping < -0.001:
                        # standard pic
                        C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    elastic_force = -state.particle_vol[p] * stress * dweight
                    v_in_add = (
                        weight
                        * state.particle_mass[p]
                        * (state.particle_v[p] + C * dpos)
                        + dt * elastic_force
                    )
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )

@wp.kernel
def p2g_flip_pic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    elastic_force = -state.particle_vol[p] * stress * dweight 
                    v_in_add = weight * state.particle_mass[p] * state.particle_v[p] 
                    v_out_add = weight * dt * elastic_force 
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(state.grid_v_out, ix, iy, iz, v_out_add)
                    wp.atomic_add(state.grid_m, ix, iy, iz, weight * state.particle_mass[p])






# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()
    force_momentum =  state.grid_v_out[grid_x, grid_y, grid_z]
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = (state.grid_v_in[grid_x, grid_y, grid_z] + force_momentum ) * (
            1.0 / state.grid_m[grid_x, grid_y, grid_z]
        )
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration 
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = new_v
        state.particle_x[p] = state.particle_x[p] + dt * new_v
        state.particle_C[p] = new_C
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, new_F, dt)



@wp.kernel
def g2p_flip(state: MPMStateStruct, model: MPMModelStruct, dt: float , flip_pic_ratio: float): 
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        old_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    old_v = old_v + state.grid_v_in[ix, iy, iz] * weight  / state.grid_m[ix, iy, iz]
                    grid_v = state.grid_v_out[ix, iy, iz] 
                    new_v = new_v + grid_v * weight
                    
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = state.particle_v[p] * flip_pic_ratio
        state.particle_v[p] = state.particle_v[p] + new_v - flip_pic_ratio * old_v  # flip_pic_ratio 为 0.99 
        state.particle_x[p] = state.particle_x[p] + dt * new_v
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, new_F, dt)


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        # apply return mapping
        if model.material[p] == 1:  # metal
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p
            )
        elif model.material[p] == 2:  # sand
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], state, model, p
            )
        elif model.material[p] == 3:  # visplas, with StVk+VM, no thickening
            state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                state.particle_F_trial[p], model, p, dt
            )
        elif model.material[p] == 5:
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )
        elif model.material[p] == 7:
            state.particle_F[p] = NonAssociativeCamClay_return_mapping(
                state.particle_F_trial[p], state, model, p
            )
        else:  # elastic
            state.particle_F[p] = state.particle_F_trial[p]

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wp.svd3(state.particle_F[p], U, sig, V)
        kappa = model.kappa[p]
        if model.material[p] == 0 or model.material[p] == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
            # print(stress)
            # wp.printf("tmp: %s\n", tmp)
        if model.material[p] == 7:
            stress = kirchoff_stress_neoHookeanBoarden(
                state.particle_F[p], U, V, J, sig, model.mu[p], model.lam[p], kappa, state
            )
            
        if model.material[p] == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material[p] == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material[p] == 3:
            # temporarily use stvk, subject to change
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )

        stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
        state.particle_stress[p] = stress


@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_init_cov[p * 6]
    init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)

    state.particle_cov[p * 6] = cov[0, 0]
    state.particle_cov[p * 6 + 1] = cov[0, 1]
    state.particle_cov[p * 6 + 2] = cov[0, 2]
    state.particle_cov[p * 6 + 3] = cov[1, 1]
    state.particle_cov[p * 6 + 4] = cov[1, 2]
    state.particle_cov[p * 6 + 5] = cov[2, 2]


@wp.kernel
def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    # polar svd decomposition
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    state.particle_R[p] = wp.transpose(R)


@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[grid_x, grid_y, grid_z] = (
        state.grid_v_out[grid_x, grid_y, grid_z] * scale
    )


@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        model.E[p] = params_modifier.E
        model.nu[p] = params_modifier.nu
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(
    state: MPMStateStruct, impulse_modifier: Impulse_modifier
):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
    ):
        impulse_modifier.mask[p] = 1
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_cylinder(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(
        offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal
    )
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0
