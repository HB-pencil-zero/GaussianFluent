import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_SPLATTING_PATH = PROJECT_ROOT / "gaussian-splatting"
if GAUSSIAN_SPLATTING_PATH.exists():
    sys.path.append(str(GAUSSIAN_SPLATTING_PATH))

def tensor_to_image(tensor):
    image = tensor.detach().clamp(0, 1).cpu()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("render tensor must have shape 3xHxW")
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def load_render_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_numpy_array(value, default=None):
    if value is None:
        value = default
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def build_camera(model_path, view, defaults):
    from utils.camera_view_utils import get_camera_view

    center = as_numpy_array(
        view.get("center_view_world_space"),
        defaults.get("center_view_world_space"),
    )
    observant = as_numpy_array(
        view.get("observant_coordinates"),
        defaults.get("observant_coordinates"),
    )
    if observant is not None:
        observant = observant.reshape(3, 3)

    return get_camera_view(
        model_path,
        default_camera_index=view.get(
            "default_camera_index",
            defaults.get("default_camera_index", 0),
        ),
        center_view_world_space=center,
        observant_coordinates=observant,
        show_hint=view.get("show_hint", defaults.get("show_hint", False)),
        init_azimuthm=view.get("init_azimuthm", defaults.get("init_azimuthm")),
        init_elevation=view.get("init_elevation", defaults.get("init_elevation")),
        init_radius=view.get("init_radius", defaults.get("init_radius")),
        move_camera=view.get("move_camera", defaults.get("move_camera", False)),
        current_frame=view.get("frame", defaults.get("frame", 0)),
        delta_a=view.get("delta_a", defaults.get("delta_a", 0)),
        delta_e=view.get("delta_e", defaults.get("delta_e", 0)),
        delta_r=view.get("delta_r", defaults.get("delta_r", 0)),
        scales=view.get("scales", defaults.get("scales", 1.1)),
        width=view.get("width", defaults.get("width", 1280)),
        height=view.get("height", defaults.get("height", 720)),
    )


def render_slices(config):
    from gaussian_renderer import render
    from interior_filling.geometry import PipelineParamsNoparse, load_checkpoint

    model_path = config["model_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    point_xy_dir = output_dir / "point_xy"
    image_dir.mkdir(parents=True, exist_ok=True)
    point_xy_dir.mkdir(parents=True, exist_ok=True)

    gaussians, iteration, checkpoint_path = load_checkpoint(
        model_path,
        sh_degree=config.get("sh_degree", 3),
        iteration=config.get("iteration", -1),
    )
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = config.get("compute_cov3D_python", False)

    background = torch.tensor(
        config.get("background", [0.0, 0.0, 0.0]),
        dtype=torch.float32,
        device="cuda",
    )

    defaults = config.get("camera_defaults", {})
    views = config.get("views", [])
    if not views:
        raise ValueError("render config must contain a non-empty views list")

    manifest = {
        "model_path": str(model_path),
        "checkpoint_path": checkpoint_path,
        "iteration": iteration,
        "views": [],
    }

    for idx, view in enumerate(views):
        name = view.get("name", f"{idx:04d}")
        camera = build_camera(model_path, view, defaults)
        render_pkg = render(camera, gaussians, pipeline, background)
        if "point_xy" not in render_pkg:
            raise RuntimeError(
                "gaussian_renderer.render() did not return point_xy. "
                "Use HB-pencil-zero/gaussian-splatting at commit 11b81b5 or newer."
            )

        image_path = image_dir / f"{name}.png"
        point_xy_path = point_xy_dir / f"{name}.pt"
        tensor_to_image(render_pkg["render"]).save(image_path)
        torch.save(render_pkg["point_xy"].detach().cpu(), point_xy_path)

        view_record = dict(view)
        view_record.update(
            {
                "name": name,
                "image": str(image_path),
                "point_xy": str(point_xy_path),
                "visible_count": int(render_pkg.get("visibility_filter", torch.empty(0)).sum().item())
                if "visibility_filter" in render_pkg
                else None,
            }
        )
        manifest["views"].append(view_record)

    manifest_path = output_dir / "render_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Render slice/reference views and save point_xy.")
    parser.add_argument("--config", required=True, help="Render JSON config.")
    args = parser.parse_args()
    manifest = render_slices(load_render_config(args.config))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
