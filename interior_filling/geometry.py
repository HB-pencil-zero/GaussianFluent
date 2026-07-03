import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_SPLATTING_PATH = PROJECT_ROOT / "gaussian-splatting"
if GAUSSIAN_SPLATTING_PATH.exists():
    sys.path.append(str(GAUSSIAN_SPLATTING_PATH))


class PipelineParamsNoparse:
    """Minimal gaussian-splatting pipeline params used for covariance export."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    from scene.gaussian_model import GaussianModel
    from utils.system_utils import searchForMaxIteration

    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians, iteration, checkpt_path


def _mask_gaussian_tensors(gaussians, mask):
    gaussians._xyz = gaussians._xyz[mask, :]
    gaussians._features_dc = gaussians._features_dc[mask, :]
    gaussians._features_rest = gaussians._features_rest[mask, :]
    gaussians._opacity = gaussians._opacity[mask, :]
    gaussians._scaling = gaussians._scaling[mask, :]
    gaussians._rotation = gaussians._rotation[mask, :]


def _select_sim_area(rotated_pos, init_pos, init_cov, init_opacity, init_shs, boundary):
    if boundary is None:
        return rotated_pos, init_cov, init_opacity, init_shs, None

    if len(boundary) != 6:
        raise ValueError("sim_area must contain six values: xmin xmax ymin ymax zmin zmax")

    mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool, device=rotated_pos.device)
    for i in range(3):
        mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
        mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

    unselected = {
        "pos": init_pos[~mask, :],
        "cov": init_cov[~mask, :],
        "opacity": init_opacity[~mask, :],
        "shs": init_shs[~mask, :],
    }
    return (
        rotated_pos[mask, :],
        init_cov[mask, :],
        init_opacity[mask, :],
        init_shs[mask, :],
        unselected,
    )


def run_geometry_filling(
    model_path,
    config_path,
    output_path,
    iteration=-1,
    sh_degree=3,
    save_debug_ply=False,
):
    from mpm_solver_warp.engine_utils import particle_position_tensor_to_ply
    from particle_filling.filling import (
        fill_particles,
        get_particle_volume,
        init_filled_particles,
    )
    from utils.decode_param import decode_param_json
    from utils.render_utils import load_params_from_gs
    from utils.transformation_utils import (
        apply_cov_rotations,
        apply_rotations,
        generate_rotation_matrices,
        shift2center111,
        transform2origin,
    )

    material_params, _, _, preprocessing_params, _ = decode_param_json(config_path)

    gaussians, resolved_iteration, checkpoint_path = load_checkpoint(
        model_path, sh_degree=sh_degree, iteration=iteration
    )
    pipeline = PipelineParamsNoparse()
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    opacity_mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[opacity_mask, :]
    init_cov = init_cov[opacity_mask, :]
    init_opacity = init_opacity[opacity_mask, :]
    init_shs = init_shs[opacity_mask, :]
    _mask_gaussian_tensors(gaussians, opacity_mask)

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)
    rotated_pos, init_cov, init_opacity, init_shs, unselected = _select_sim_area(
        rotated_pos,
        init_pos,
        init_cov,
        init_opacity,
        init_shs,
        preprocessing_params["sim_area"],
    )

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    filling_params = preprocessing_params["particle_filling"]
    gs_num = transformed_pos.shape[0]
    if filling_params is not None:
        filled_pos = fill_particles(
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
        ).to(device="cuda")
    else:
        filled_pos = transformed_pos.to(device="cuda")

    if filling_params is not None and filling_params["visualize"]:
        shs, opacity, cov = init_filled_particles(
            filled_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            filled_pos[gs_num:],
        )
    else:
        cov = torch.zeros((filled_pos.shape[0], 6), device=filled_pos.device)
        cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    volume = get_particle_volume(
        filled_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=filled_pos.device)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint_path": checkpoint_path,
        "iteration": resolved_iteration,
        "gs_num": gs_num,
        "scale_origin": float(scale_origin),
        "original_mean_pos": original_mean_pos.detach().cpu(),
        "rotation_matrices": rotation_matrices.detach().cpu(),
        "pos": filled_pos.detach().cpu(),
        "cov": cov.detach().cpu(),
        "opacity": opacity.detach().cpu(),
        "shs": shs.detach().cpu(),
        "volume": volume.detach().cpu(),
        "unselected": {
            key: value.detach().cpu() for key, value in unselected.items()
        } if unselected is not None else None,
    }
    torch.save(result, output_dir / "filled_gaussians.pt")

    metadata = {
        "checkpoint_path": checkpoint_path,
        "iteration": resolved_iteration,
        "surface_gaussians": int(gs_num),
        "total_particles": int(filled_pos.shape[0]),
        "filled_particles": int(filled_pos.shape[0] - gs_num),
        "config_path": str(config_path),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if save_debug_ply:
        particle_position_tensor_to_ply(filled_pos, str(output_dir / "filled_particles.ply"))

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Fill Gaussian interiors for simulation.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--save_debug_ply", action="store_true")
    args = parser.parse_args()

    metadata = run_geometry_filling(
        model_path=args.model_path,
        config_path=args.config,
        output_path=args.output_path,
        iteration=args.iteration,
        sh_degree=args.sh_degree,
        save_debug_ply=args.save_debug_ply,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
