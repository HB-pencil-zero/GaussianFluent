import argparse
import glob
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_SPLATTING_PATH = PROJECT_ROOT / "gaussian-splatting"
if GAUSSIAN_SPLATTING_PATH.exists():
    sys.path.append(str(GAUSSIAN_SPLATTING_PATH))

from interior_filling.geometry import load_checkpoint


def load_projection(path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        data = {"sh_dc": data}
    if "sh_dc" not in data:
        raise ValueError(f"{path} does not contain 'sh_dc'")
    sh_dc = data["sh_dc"].float()
    if sh_dc.ndim != 2 or sh_dc.shape[1] != 3:
        raise ValueError(f"{path} sh_dc must have shape Nx3")
    valid = data.get("valid")
    if valid is None:
        valid = torch.ones(sh_dc.shape[0], dtype=torch.bool)
    else:
        valid = valid.bool()
    return sh_dc, valid


def expand_projection_paths(paths=None, pattern=None):
    result = []
    for path in paths or []:
        result.append(Path(path))
    if pattern:
        result.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not result:
        raise ValueError("no projection files provided")
    return result


def merge_projection_files(projection_paths, total_points, mode="average"):
    if mode not in ("average", "last"):
        raise ValueError("merge mode must be 'average' or 'last'")

    accum = torch.zeros((total_points, 3), dtype=torch.float32)
    counts = torch.zeros((total_points,), dtype=torch.float32)
    last_valid = torch.zeros((total_points,), dtype=torch.bool)

    for path in projection_paths:
        sh_dc, valid = load_projection(path)
        if sh_dc.shape[0] != total_points:
            raise ValueError(
                f"{path} has {sh_dc.shape[0]} points, expected {total_points}"
            )
        if mode == "average":
            accum[valid] += sh_dc[valid]
            counts[valid] += 1
        else:
            accum[valid] = sh_dc[valid]
            last_valid |= valid

    if mode == "average":
        merged_valid = counts > 0
        accum[merged_valid] = accum[merged_valid] / counts[merged_valid, None]
    else:
        merged_valid = last_valid
    return accum, merged_valid


def apply_projection_to_checkpoint(
    model_path,
    output_ply,
    projection_paths=None,
    projection_glob=None,
    iteration=-1,
    sh_degree=3,
    start_index=0,
    end_index=None,
    merge_mode="average",
    zero_rest=True,
):
    gaussians, resolved_iteration, checkpoint_path = load_checkpoint(
        model_path,
        sh_degree=sh_degree,
        iteration=iteration,
    )
    total = gaussians._features_dc.shape[0]
    if end_index is None:
        end_index = total
    if start_index < 0 or end_index > total or start_index > end_index:
        raise ValueError("invalid start_index/end_index")

    paths = expand_projection_paths(projection_paths, projection_glob)
    sh_dc, valid = merge_projection_files(paths, total, mode=merge_mode)
    range_mask = torch.zeros(total, dtype=torch.bool)
    range_mask[start_index:end_index] = True
    update_mask = valid & range_mask
    update_indices = torch.nonzero(update_mask, as_tuple=False).flatten()

    device = gaussians._features_dc.device
    with torch.no_grad():
        gaussians._features_dc[update_indices.to(device), 0] = sh_dc[
            update_indices
        ].to(device=device, dtype=gaussians._features_dc.dtype)
        if zero_rest and hasattr(gaussians, "_features_rest"):
            gaussians._features_rest[update_indices.to(device)] = torch.zeros_like(
                gaussians._features_rest[update_indices.to(device)]
            )

    output_ply = Path(output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(output_ply))

    summary = {
        "model_path": str(model_path),
        "checkpoint_path": checkpoint_path,
        "iteration": resolved_iteration,
        "output_ply": str(output_ply),
        "projection_files": [str(p) for p in paths],
        "merge_mode": merge_mode,
        "start_index": start_index,
        "end_index": end_index,
        "updated_count": int(update_indices.numel()),
        "total_points": int(total),
    }
    summary_path = output_ply.with_suffix(".projection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Apply projected SH DC tensors to a 3DGS checkpoint.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_ply", required=True)
    parser.add_argument("--projection", action="append", default=[])
    parser.add_argument("--projection_glob", default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--merge_mode", choices=["average", "last"], default="average")
    parser.add_argument("--keep_rest", action="store_true")
    args = parser.parse_args()

    summary = apply_projection_to_checkpoint(
        model_path=args.model_path,
        output_ply=args.output_ply,
        projection_paths=args.projection,
        projection_glob=args.projection_glob,
        iteration=args.iteration,
        sh_degree=args.sh_degree,
        start_index=args.start_index,
        end_index=args.end_index,
        merge_mode=args.merge_mode,
        zero_rest=not args.keep_rest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
