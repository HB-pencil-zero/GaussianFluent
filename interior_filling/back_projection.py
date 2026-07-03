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

C0 = 0.28209479177387814


def rgb_to_sh_dc(rgb):
    """Convert RGB in [0, 1] to the DC SH coefficient used by 3DGS."""
    return (rgb - 0.5) / C0


def load_image_chw(path, device=None):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def load_mask_hw(path, device=None):
    mask = Image.open(path).convert("L")
    array = np.asarray(mask, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _image_to_chw(image, device):
    if isinstance(image, (str, Path)):
        return load_image_chw(image, device=device)

    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)

    image = image.to(device=device, dtype=torch.float32)
    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim != 3:
        raise ValueError("target image must have shape CxHxW or HxWxC")
    if image.shape[0] == 3:
        return image.contiguous()
    if image.shape[-1] == 3:
        return image.permute(2, 0, 1).contiguous()
    raise ValueError("target image must have 3 color channels")


def _mask_to_hw(mask, device):
    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        return load_mask_hw(mask, device=device)

    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    if mask.ndim == 3:
        if mask.shape[0] in (1, 3):
            mask = mask.mean(dim=0)
        elif mask.shape[-1] in (1, 3):
            mask = mask.mean(dim=-1)
        else:
            raise ValueError("mask image must have shape HxW, CxHxW, or HxWxC")
    if mask.ndim != 2:
        raise ValueError("mask image must reduce to shape HxW")
    return mask.contiguous()


def sample_image_bilinear(image_chw, point_xy):
    """Sample CxHxW image at floating point xy coordinates."""
    if image_chw.ndim != 3:
        raise ValueError("image_chw must have shape CxHxW")
    if point_xy.ndim != 2 or point_xy.shape[1] != 2:
        raise ValueError("point_xy must have shape Nx2")

    _, height, width = image_chw.shape
    x = point_xy[:, 0]
    y = point_xy[:, 1]

    valid = (x >= 0) & (y >= 0) & (x < width - 1) & (y < height - 1)
    x = x.clamp(0, width - 1.000001)
    y = y.clamp(0, height - 1.000001)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)

    wx = (x - x0.float()).view(1, -1)
    wy = (y - y0.float()).view(1, -1)

    c00 = image_chw[:, y0, x0]
    c01 = image_chw[:, y1, x0]
    c10 = image_chw[:, y0, x1]
    c11 = image_chw[:, y1, x1]

    c0 = (1.0 - wx) * c00 + wx * c10
    c1 = (1.0 - wx) * c01 + wx * c11
    colors = ((1.0 - wy) * c0 + wy * c1).transpose(0, 1).contiguous()
    return colors, valid


def sample_mask_nearest(mask_hw, point_xy, mode="black", threshold=0.01):
    if mode == "none":
        return torch.ones(point_xy.shape[0], dtype=torch.bool, device=point_xy.device)
    if mask_hw is None:
        return torch.ones(point_xy.shape[0], dtype=torch.bool, device=point_xy.device)

    height, width = mask_hw.shape
    x = torch.round(point_xy[:, 0]).long()
    y = torch.round(point_xy[:, 1]).long()
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height)

    values = torch.ones(point_xy.shape[0], dtype=mask_hw.dtype, device=mask_hw.device)
    values[valid] = mask_hw[y[valid], x[valid]]

    if mode == "black":
        selected = values <= threshold
    elif mode == "white":
        selected = values >= threshold
    elif mode == "nonzero":
        selected = values > threshold
    else:
        raise ValueError("mask mode must be one of: black, white, nonzero, none")

    return selected & valid


def back_project_colors_to_gaussians(
    gaussians,
    point_xy,
    target_image,
    start_index=0,
    end_index=None,
    mask_image=None,
    mask_mode="black",
    mask_threshold=0.01,
    visibility_mask=None,
    zero_rest=True,
):
    """Back-project image colors into Gaussian SH DC coefficients.

    This mutates ``gaussians`` in place. ``point_xy`` must be the screen-space
    coordinates returned by the modified Gaussian rasterizer.
    """
    device = gaussians._features_dc.device
    point_xy = point_xy.to(device=device, dtype=torch.float32)
    target = _image_to_chw(target_image, device)
    mask = _mask_to_hw(mask_image, device)

    total = gaussians._features_dc.shape[0]
    if point_xy.shape[0] != total:
        raise ValueError(
            f"point_xy has {point_xy.shape[0]} points, but gaussians has {total}"
        )
    if end_index is None:
        end_index = total
    if start_index < 0 or end_index > total or start_index > end_index:
        raise ValueError("invalid start_index/end_index range")

    candidate = torch.zeros(total, dtype=torch.bool, device=device)
    candidate[start_index:end_index] = True
    if visibility_mask is not None:
        candidate &= visibility_mask.to(device=device, dtype=torch.bool)

    sampled_rgb, image_valid = sample_image_bilinear(target, point_xy)
    mask_valid = sample_mask_nearest(
        mask,
        point_xy,
        mode=mask_mode,
        threshold=mask_threshold,
    )
    update_mask = candidate & image_valid & mask_valid
    update_indices = torch.nonzero(update_mask, as_tuple=False).flatten()

    with torch.no_grad():
        gaussians._features_dc[update_indices, 0] = rgb_to_sh_dc(
            sampled_rgb[update_indices]
        ).to(gaussians._features_dc.dtype)
        if zero_rest and hasattr(gaussians, "_features_rest"):
            gaussians._features_rest[update_indices] = torch.zeros_like(
                gaussians._features_rest[update_indices]
            )

    return {
        "updated_indices": update_indices,
        "updated_count": int(update_indices.numel()),
        "candidate_count": int(candidate.sum().item()),
    }


def point_xy_to_sh_dc(
    point_xy,
    target_image,
    mask_image=None,
    mask_mode="black",
    mask_threshold=0.01,
):
    """Convert saved screen-space points and an image into SH DC tensors."""
    point_xy = point_xy.float()
    target = _image_to_chw(target_image, point_xy.device)
    mask = _mask_to_hw(mask_image, point_xy.device)
    colors, image_valid = sample_image_bilinear(target, point_xy)
    mask_valid = sample_mask_nearest(
        mask,
        point_xy,
        mode=mask_mode,
        threshold=mask_threshold,
    )
    valid = image_valid & mask_valid
    sh_dc = torch.zeros((point_xy.shape[0], 3), dtype=torch.float32)
    sh_dc[valid.cpu()] = rgb_to_sh_dc(colors[valid].cpu())
    return {"sh_dc": sh_dc, "valid": valid.cpu()}


def render_and_back_project(
    gaussians,
    viewpoint_camera,
    pipeline,
    background,
    target_image,
    start_index=0,
    end_index=None,
    mask_image=None,
    mask_mode="black",
    mask_threshold=0.01,
    zero_rest=True,
):
    from gaussian_renderer import render

    render_pkg = render(viewpoint_camera, gaussians, pipeline, background)
    if "point_xy" not in render_pkg:
        raise RuntimeError(
            "gaussian_renderer.render() did not return point_xy. "
            "Use HB-pencil-zero/gaussian-splatting at commit 11b81b5 or newer."
        )

    result = back_project_colors_to_gaussians(
        gaussians=gaussians,
        point_xy=render_pkg["point_xy"],
        target_image=target_image,
        start_index=start_index,
        end_index=end_index,
        mask_image=mask_image,
        mask_mode=mask_mode,
        mask_threshold=mask_threshold,
        visibility_mask=render_pkg.get("visibility_filter"),
        zero_rest=zero_rest,
    )
    result["render_pkg"] = render_pkg
    return result


def _load_tensor(path):
    path = Path(path)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Back-project image colors to SH DC values using saved point_xy."
    )
    parser.add_argument("--point_xy", required=True, help=".pt or .npy tensor with shape Nx2")
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--output_sh_dc", required=True)
    parser.add_argument("--mask_image", default=None)
    parser.add_argument("--mask_mode", default="black", choices=["black", "white", "nonzero", "none"])
    parser.add_argument("--mask_threshold", type=float, default=0.01)
    args = parser.parse_args()

    point_xy = _load_tensor(args.point_xy).float()
    result = point_xy_to_sh_dc(
        point_xy,
        args.target_image,
        mask_image=args.mask_image,
        mask_mode=args.mask_mode,
        mask_threshold=args.mask_threshold,
    )
    torch.save(result, args.output_sh_dc)
    print(json.dumps({"valid_count": int(result["valid"].sum().item())}, indent=2))


if __name__ == "__main__":
    main()
