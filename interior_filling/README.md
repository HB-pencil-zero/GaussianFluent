# Interior Filling Code

This folder keeps the interior-preparation code separate from scene-specific
simulation demos.

## Scope

The release pipeline is split into three stages:

1. Geometry filling: add internal particles/Gaussians from a trained 3DGS model.
2. Interior texture supervision: render sliced views, inpaint them with an
   external inpainting tool, and collect masks or target images.
3. Texture fitting/back-projection: optimize or directly write SH/color/opacity
   for the filled Gaussians against the generated supervision.

Stage 1 is implemented as a clean command. Stage 3 includes a reusable
back-projection utility for directly writing image colors into Gaussian SH DC
coefficients. Stage 2 is intentionally kept external to this repository.

The research prototype used
[MVInpainter](https://github.com/ewrfcas/MVInpainter) for multi-view inpainting.
MVInpainter is a third-party project, not a GaussianFluent repository, and is
not redistributed with the GaussianFluent code or Hugging Face model assets.
If you need to reproduce that stage, clone MVInpainter separately and follow
its own license, setup, checkpoint, and citation requirements. The local
prototype was tested around MVInpainter commit `323d7f6`.

## One-Command Pipeline

`pipeline.py` ties the stages together without copying third-party inpainting
code into this repository. It can run:

- geometry filling;
- optional scene-specific slice rendering commands;
- built-in slice/reference rendering with saved `point_xy`;
- MVInpainter dataset folder preparation;
- external MVInpainter inference command generation/execution;
- saved `point_xy` to SH DC back-projection.
- application of projected SH DC tensors back into a 3DGS checkpoint.

Start from the example config:

```bash
python -m interior_filling.pipeline \
  --config interior_filling/example_pipeline_config.json \
  --stage all
```

By default this is a dry run for external commands: it prepares local manifests
and writes an executable MVInpainter command script, but it does not launch
MVInpainter or scene-specific render commands. To actually run external
commands:

```bash
python -m interior_filling.pipeline \
  --config interior_filling/example_pipeline_config.json \
  --stage all \
  --run_external
```

Useful partial stages:

```bash
# Only generate filled internal particles.
python -m interior_filling.pipeline --config interior_filling/example_pipeline_config.json --stage geometry

# Only adapt existing rendered images/masks into MVInpainter's folder layout.
python -m interior_filling.pipeline --config interior_filling/example_pipeline_config.json --stage prepare --overwrite

# Only generate/run the external MVInpainter command.
python -m interior_filling.pipeline --config interior_filling/example_pipeline_config.json --stage inpaint

# Only convert saved point_xy and inpainted images into SH DC tensors.
python -m interior_filling.pipeline --config interior_filling/example_pipeline_config.json --stage project

# Only write projected SH DC tensors back to a 3DGS checkpoint and save PLY.
python -m interior_filling.pipeline --config interior_filling/example_pipeline_config.json --stage apply
```

The `render_commands` field is the bridge to the old research scripts that
generated slice views, masks, and optional `point_xy` tensors. Keep those as
explicit commands in the JSON until the camera/slice generator is fully
standardized.

If the view list is already known, `render_slices.py` can render views directly
and save both the RGB render and the modified-rasterizer `point_xy` tensor:

```bash
python -m interior_filling.render_slices \
  --config output/interior/watermelon_pipeline/render_slices.json
```

The final application step can also be called directly:

```bash
python -m interior_filling.apply_projection \
  --model_path model/watermelon \
  --projection_glob "output/interior/watermelon_pipeline/projected_sh_dc/*_sh_dc.pt" \
  --output_ply output/interior/watermelon_pipeline/final/point_cloud.ply \
  --start_index 0
```

## Geometry Filling

Run from the repository root:

```bash
python -m interior_filling.geometry \
  --model_path model/watermelon \
  --config config/watermelon_config.json \
  --output_path output/interior/watermelon \
  --save_debug_ply
```

Outputs:

- `filled_gaussians.pt`: particle positions, covariance, opacity, SH features,
  volume, and transform metadata.
- `metadata.json`: counts and source checkpoint information.
- `filled_particles.ply`: optional debug point cloud when `--save_debug_ply` is
  provided.

The command reads the existing `particle_filling` block from a scene config:

```json
{
  "particle_filling": {
    "n_grid": 256,
    "density_threshold": 5.0,
    "search_threshold": 3.0,
    "max_particles_num": 2000000,
    "max_partciels_per_cell": 1,
    "search_exclude_direction": 5,
    "ray_cast_direction": 4,
    "boundary": null,
    "smooth": false,
    "visualize": false
  }
}
```

If `visualize` is `true`, the new filled particles inherit SH/opacity/covariance
from nearest surface Gaussians. If it is `false`, only simulation particles are
prepared and the original visible Gaussians remain the renderable set.

## Back-Projection

The legacy implementation is `PhysGaussian/utils/filling_utils.py::back_calculate_gaussians`.
The cleaned implementation is in `back_projection.py`.

It relies on the modified Gaussian rasterizer returning one screen-space
coordinate per Gaussian:

```python
render_pkg = render(camera, gaussians, pipeline, background)
point_xy = render_pkg["point_xy"]
```

Then it samples an inpainted/reference image at those coordinates and writes the
sampled RGB into the Gaussian SH DC coefficient:

```python
from interior_filling.back_projection import render_and_back_project

result = render_and_back_project(
    gaussians=gaussians,
    viewpoint_camera=current_camera,
    pipeline=pipeline,
    background=background,
    target_image="output/interior/watermelon/supervision/0010.png",
    mask_image="output/interior/watermelon/masks/0010.png",
    start_index=surface_gaussian_count,
    mask_mode="black",
)
print(result["updated_count"])
```

`mask_mode="black"` matches the old behavior: only projected points whose mask
pixel is near black are updated. Use `mask_mode="none"` to update all projected
points in the selected index range.

For debugging with saved projections, the module can also convert a saved
`point_xy` tensor to SH DC values:

```bash
python -m interior_filling.back_projection \
  --point_xy output/interior/watermelon/point_xy.pt \
  --target_image output/interior/watermelon/supervision/0010.png \
  --mask_image output/interior/watermelon/masks/0010.png \
  --output_sh_dc output/interior/watermelon/sh_dc_from_projection.pt
```

Dependency note: this requires `HB-pencil-zero/gaussian-splatting` commit
`11b81b5` or newer, where `diff-gaussian-rasterization` returns `point_xy`.

## Texture Pipeline Notes

The current research implementation is scattered across:

- `PhysGaussian/utils/filling_utils.py`: experimental slice rendering and
  multi-frame color optimization.
- External [MVInpainter](https://github.com/ewrfcas/MVInpainter): multi-view
  inpainting and mask propagation. Keep this as a separate dependency instead
  of copying its code into GaussianFluent.
- `debug_physgaussian/cdmpmGaussian/normal_vector_proc_nan.py` and
  `phong_model_wm_shs_15.py`: simulation-time normal/Phong helpers, not part of
  geometry filling.

Recommended cleanup direction:

1. Keep MVInpainter as an external dependency with a documented GitHub link,
   commit, checkpoint paths, and upstream license/citation requirements.
2. Move only dataset adapters and slice camera generation into this repository.
3. Replace hard-coded watermelon paths with explicit CLI arguments.
4. Save generated supervision under `output/interior/<scene>/supervision/`.
5. Add a differentiable `fit_texture.py` only after the target image/mask layout
   is stable; direct SH DC back-projection is already available here.
