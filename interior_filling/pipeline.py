import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import torch

from interior_filling.back_projection import (
    _load_tensor,
    point_xy_to_sh_dc,
)
from interior_filling.apply_projection import apply_projection_to_checkpoint
from interior_filling.geometry import run_geometry_filling
from interior_filling.render_slices import render_slices


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pipeline_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resolve_path(path, root=PROJECT_ROOT):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def copy_file(src, dst, overwrite=False):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def image_files(path, pattern="*.png"):
    path = Path(path)
    files = sorted(path.glob(pattern))
    return [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]


def run_shell_commands(commands, cwd=None, dry_run=True, env=None):
    results = []
    for item in commands:
        if isinstance(item, str):
            name = "command"
            command = item
            command_cwd = cwd
        else:
            name = item.get("name", "command")
            command = item["command"]
            command_cwd = item.get("cwd", cwd)
        if command_cwd is not None:
            command_cwd = str(resolve_path(command_cwd))

        record = {
            "name": name,
            "command": command,
            "cwd": command_cwd,
            "dry_run": dry_run,
        }
        if dry_run:
            record["returncode"] = None
            results.append(record)
            continue

        process = subprocess.run(
            command,
            cwd=command_cwd,
            shell=True,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        record["returncode"] = process.returncode
        record["stdout"] = process.stdout
        record["stderr"] = process.stderr
        results.append(record)
        if process.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {process.returncode}")
    return results


def run_geometry_stage(config):
    geometry = config.get("geometry", {})
    if not geometry.get("enabled", True):
        return {"skipped": True}

    metadata = run_geometry_filling(
        model_path=resolve_path(geometry["model_path"]),
        config_path=resolve_path(geometry["config"]),
        output_path=resolve_path(geometry["output_path"]),
        iteration=geometry.get("iteration", -1),
        sh_degree=geometry.get("sh_degree", 3),
        save_debug_ply=geometry.get("save_debug_ply", False),
    )
    return {"skipped": False, "metadata": metadata}


def prepare_mvinpainter_dataset(config, overwrite=False):
    supervision = config.get("supervision", {})
    dataset_root = resolve_path(supervision["dataset_root"])
    frame_pattern = supervision.get("frame_pattern", "*.png")
    views = supervision.get("views", [])
    if not views:
        raise ValueError("supervision.views must contain at least one view")

    manifest = {
        "dataset_root": str(dataset_root),
        "views": [],
        "note": "MVInpainter is external: https://github.com/ewrfcas/MVInpainter",
    }

    for view in views:
        name = view["name"]
        view_dir = dataset_root / name
        images_dir = view_dir / "images"
        masks_dir = view_dir / "masks"
        removal_dir = view_dir / "removal"
        warp_masks_dir = view_dir / "warp_masks"
        obj_bbox_dir = view_dir / "obj_bbox"
        inpainted_dir = view_dir / "inpainted"

        copied = {
            "images": 0,
            "masks": 0,
            "removal": 0,
            "warp_masks": 0,
            "obj_bbox": 0,
            "inpainted": 0,
        }

        source_images = view.get("images")
        if source_images:
            for src in image_files(resolve_path(source_images), frame_pattern):
                copied["images"] += copy_file(src, images_dir / src.name, overwrite)
                copied["removal"] += copy_file(src, removal_dir / src.name, overwrite)

        source_masks = view.get("masks")
        if source_masks:
            for src in image_files(resolve_path(source_masks), frame_pattern):
                copied["masks"] += copy_file(src, masks_dir / src.name, overwrite)
                copied["warp_masks"] += copy_file(src, warp_masks_dir / src.name, overwrite)

        reference_image = view.get("reference_image")
        if reference_image:
            reference_name = view.get("reference_name", "0000.png")
            copied["obj_bbox"] += copy_file(
                resolve_path(reference_image), obj_bbox_dir / reference_name, overwrite
            )
            copied["inpainted"] += copy_file(
                resolve_path(reference_image), inpainted_dir / reference_name, overwrite
            )

        bbox_json = view.get("bbox_json")
        if bbox_json:
            copied["obj_bbox"] += copy_file(
                resolve_path(bbox_json), obj_bbox_dir / Path(bbox_json).name, overwrite
            )

        manifest["views"].append(
            {
                "name": name,
                "view_dir": str(view_dir),
                "copied": copied,
                "source": view,
            }
        )

    write_json(dataset_root / "gaussianfluent_dataset_manifest.json", manifest)
    return manifest


def build_mvinpainter_command(config):
    inpaint = config.get("mvinpainter", {})
    repo_path = Path(inpaint.get("repo_path", "/root/autodl-tmp/MVInpainter"))
    script = inpaint.get("script", "test_nvs.py")
    conda_env = inpaint.get("conda_env")
    cuda_visible_devices = inpaint.get("cuda_visible_devices")

    dataset_root = resolve_path(inpaint["dataset_root"])
    output_path = resolve_path(inpaint["output_path"])
    args = {
        "--load_path": inpaint["load_path"],
        "--dataset_root": str(dataset_root),
        "--output_path": str(output_path),
        "--edited_index": inpaint.get("edited_index", 0),
        "--resume_from_checkpoint": inpaint.get("resume_from_checkpoint", "best"),
        "--val_cfg": inpaint.get("val_cfg", 7.5),
        "--img_height": inpaint.get("img_height", 512),
        "--img_width": inpaint.get("img_width", 512),
        "--sampling_interval": inpaint.get("sampling_interval", 1.0),
        "--nframe": inpaint.get("nframe", 24),
        "--prompt": inpaint.get("prompt", ""),
    }
    if inpaint.get("limit_frame") is not None:
        args["--limit_frame"] = inpaint["limit_frame"]
    if inpaint.get("inference_step") is not None:
        args["--inference_step"] = inpaint["inference_step"]

    command_parts = []
    if cuda_visible_devices is not None:
        command_parts.append(f"CUDA_VISIBLE_DEVICES={shlex.quote(str(cuda_visible_devices))}")
    command_parts.append("python")
    command_parts.append(shlex.quote(str(repo_path / script)))
    for key, value in args.items():
        command_parts.append(f"{key}={shlex.quote(str(value))}")
    if inpaint.get("save_images", True):
        command_parts.append("--save_images")

    command = " ".join(command_parts)
    setup = [f"cd {shlex.quote(str(repo_path))}"]
    if conda_env:
        setup.append("source /root/miniconda3/etc/profile.d/conda.sh")
        setup.append(f"conda activate {shlex.quote(conda_env)}")
    setup.append(command)
    return " && ".join(setup)


def run_mvinpainter_stage(config, dry_run=True):
    inpaint = config.get("mvinpainter", {})
    if not inpaint.get("enabled", True):
        return {"skipped": True}

    command = build_mvinpainter_command(config)
    command_path = resolve_path(inpaint.get("command_path", "output/interior/run_mvinpainter.sh"))
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text("#!/usr/bin/env bash\nset -e\n" + command + "\n", encoding="utf-8")
    command_path.chmod(0o755)

    result = {"skipped": False, "command": command, "command_path": str(command_path)}
    if dry_run:
        result["dry_run"] = True
        return result

    process = subprocess.run(
        ["bash", str(command_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result["returncode"] = process.returncode
    result["stdout"] = process.stdout
    result["stderr"] = process.stderr
    if process.returncode != 0:
        raise RuntimeError(f"MVInpainter failed with exit code {process.returncode}")
    return result


def run_render_stage(config, dry_run=True):
    render_config = config.get("render_slices")
    commands = config.get("render_commands", [])
    result = {}

    if render_config and render_config.get("enabled", True):
        if dry_run and render_config.get("dry_run", False):
            result["render_slices"] = {"skipped": True, "dry_run": True}
        else:
            resolved = dict(render_config)
            if "model_path" in resolved:
                resolved["model_path"] = str(resolve_path(resolved["model_path"]))
            if "output_dir" in resolved:
                resolved["output_dir"] = str(resolve_path(resolved["output_dir"]))
            result["render_slices"] = render_slices(resolved)

    if commands:
        result["commands"] = run_shell_commands(
            commands, cwd=str(PROJECT_ROOT), dry_run=dry_run
        )

    if not result:
        result["skipped"] = True
    return result


def expand_projection_pairs(projection):
    pairs = list(projection.get("pairs", []))
    glob_config = projection.get("glob")
    if not glob_config:
        return pairs

    point_xy_dir = resolve_path(glob_config["point_xy_dir"])
    target_dir = resolve_path(glob_config["target_dir"])
    mask_dir = glob_config.get("mask_dir")
    point_xy_pattern = glob_config.get("point_xy_pattern", "*.pt")
    target_suffix = glob_config.get("target_suffix", ".png")
    output_dir = resolve_path(glob_config["output_dir"])

    for point_xy_path in sorted(point_xy_dir.glob(point_xy_pattern)):
        stem = point_xy_path.stem
        target_path = target_dir / f"{stem}{target_suffix}"
        if not target_path.exists():
            continue
        pair = {
            "point_xy": str(point_xy_path),
            "target_image": str(target_path),
            "output_sh_dc": str(output_dir / f"{stem}_sh_dc.pt"),
        }
        if mask_dir:
            mask_path = resolve_path(mask_dir) / f"{stem}{target_suffix}"
            if mask_path.exists():
                pair["mask_image"] = str(mask_path)
        pairs.append(pair)
    return pairs


def project_saved_pairs(config):
    projection = config.get("projection", {})
    if not projection.get("enabled", True):
        return {"skipped": True}

    mask_mode = projection.get("mask_mode", "black")
    mask_threshold = projection.get("mask_threshold", 0.01)
    results = []
    for pair in expand_projection_pairs(projection):
        point_xy = _load_tensor(pair["point_xy"]).float()
        projected = point_xy_to_sh_dc(
            point_xy,
            pair["target_image"],
            mask_image=pair.get("mask_image"),
            mask_mode=pair.get("mask_mode", mask_mode),
            mask_threshold=pair.get("mask_threshold", mask_threshold),
        )
        output_path = Path(pair["output_sh_dc"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**projected, "source": pair}, output_path)
        results.append(
            {
                "point_xy": pair["point_xy"],
                "target_image": pair["target_image"],
                "mask_image": pair.get("mask_image"),
                "output_sh_dc": str(output_path),
                "valid_count": int(projected["valid"].sum().item()),
            }
        )
    return {"skipped": False, "pairs": results}


def apply_projection_stage(config):
    apply_config = config.get("apply_projection", {})
    if not apply_config.get("enabled", True):
        return {"skipped": True}

    projection_paths = apply_config.get("projection_paths")
    if projection_paths:
        projection_paths = [str(resolve_path(path)) for path in projection_paths]
    projection_glob = apply_config.get("projection_glob")
    if projection_glob:
        projection_glob = str(resolve_path(projection_glob))

    return apply_projection_to_checkpoint(
        model_path=resolve_path(apply_config["model_path"]),
        output_ply=resolve_path(apply_config["output_ply"]),
        projection_paths=projection_paths,
        projection_glob=projection_glob,
        iteration=apply_config.get("iteration", -1),
        sh_degree=apply_config.get("sh_degree", 3),
        start_index=apply_config.get("start_index", 0),
        end_index=apply_config.get("end_index"),
        merge_mode=apply_config.get("merge_mode", "average"),
        zero_rest=apply_config.get("zero_rest", True),
    )


def run_pipeline(config, stage, dry_run=True, overwrite=False):
    workspace = resolve_path(config.get("workspace", "output/interior/pipeline"))
    workspace.mkdir(parents=True, exist_ok=True)
    summary = {"workspace": str(workspace), "stage": stage, "dry_run": dry_run}

    if stage in ("geometry", "all"):
        summary["geometry"] = run_geometry_stage(config)

    if stage in ("render", "all"):
        summary["render"] = run_render_stage(config, dry_run=dry_run)

    if stage in ("prepare", "all"):
        summary["prepare"] = prepare_mvinpainter_dataset(config, overwrite=overwrite)

    if stage in ("inpaint", "all"):
        summary["inpaint"] = run_mvinpainter_stage(config, dry_run=dry_run)

    if stage in ("project", "all"):
        summary["projection"] = project_saved_pairs(config)

    if stage in ("apply", "all"):
        summary["apply_projection"] = apply_projection_stage(config)

    write_json(workspace / "pipeline_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the GaussianFluent interior pipeline.")
    parser.add_argument("--config", required=True, help="Pipeline JSON config.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["geometry", "render", "prepare", "inpaint", "project", "apply", "all"],
    )
    parser.add_argument(
        "--run_external",
        action="store_true",
        help="Actually run render_commands and MVInpainter. Default is dry-run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    summary = run_pipeline(
        config,
        stage=args.stage,
        dry_run=not args.run_external,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
