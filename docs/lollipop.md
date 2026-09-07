# Lollipop Drop and Breaking Simulation

This example contains the lollipop drop test used with GaussianFluent. The simulation script is `gs_simulation/lollipop/gs_simulation_lollipop.py` and the scene parameters are in `config/lollipop_config.json`.

## Run

Put the trained lollipop Gaussian model at `model/lollipop/`, then run:

```bash
python gs_simulation/lollipop/gs_simulation_lollipop.py \\
  --model_path model/lollipop \\
  --output_path output/lollipop \\
  --config config/lollipop_config.json \\
  --render_img --compile_video
```

The rendered frames, simulation data, and `output.mp4` are written to `output/lollipop/`. Use `--output_ply` or `--output_h5` for intermediate formats. `--load_from_saved` renders saved simulation data without rerunning MPM.

## Important parameters

| Parameter | Value | Role |
| --- | ---: | --- |
| `E` | `2e4` | Young's modulus; lowering it makes the material softer and easier to deform. |
| `beta` | `0.5` | Damage/fracture parameter; lowering it generally makes fracture easier to trigger. |
| `xi` | `3.0` | Damage evolution parameter. |
| `density` | `1` | Particle density used by the simulation. |
| `g` | `[0, 0, -15]` | Downward acceleration for the drop test. |
| `surface` | `cut` | Horizontal collider beneath the object. |
| `friction` | `1000` | Keeps contact with the collider from sliding. |
| `frame_num` | `90` | Number of output frames. |
| `frame_dt` | `3e-2` | Time between output frames. |

For a stronger or earlier break, reduce `beta` and/or `E` gradually. Change one parameter at a time. Keep the collider stationary for a stable impact constraint. The result also depends on reconstructed geometry, particle filling, model scale, and impact orientation.

## Notes

The script currently uses the `watermelon` material implementation as its MPM backend; lollipop behavior is controlled by the scene configuration and Gaussian model. A model containing both the candy and its stick is required for a complete drop test.
