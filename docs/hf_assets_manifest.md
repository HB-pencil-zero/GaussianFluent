# Hugging Face Asset Manifest

This file records local GaussianFluent assets that can be cleaned and uploaded
to Hugging Face. It is an upload planning document only; it does not copy or
redistribute third-party code.

## Source Roots

- Main research model root:
  `/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model`
- Main research config root:
  `/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/config`
- Current clean HF staging root:
  `/root/autodl-tmp/hf_clean_upload`

## Upload Boundary

Recommended files per 3DGS scene:

- `model/<scene>/cameras.json`
- `model/<scene>/cfg_args`
- `model/<scene>/input.ply`
- `model/<scene>/point_cloud/iteration_<latest>/point_cloud.ply`
- `config/<scene>_config.json` or the matching simulation config

Do not upload training logs, local caches, conda environments, generated
simulation outputs, or third-party inpainting code/checkpoints.

MVInpainter is an external dependency:
https://github.com/ewrfcas/MVInpainter

The local prototype used MVInpainter around commit `323d7f6`, but MVInpainter is
not part of GaussianFluent and should not be redistributed in the GaussianFluent
HF assets.

## Current HF Staging

`/root/autodl-tmp/hf_clean_upload` currently contains:

| Scene | Included checkpoints | Notes |
| --- | --- | --- |
| `jelly` | `iteration_30000`, `iteration_5000` | Has `cameras.json`, `cfg_args`, `input.ply`. |
| `watermelon` | `iteration_30000` | Has `cameras.json`, `cfg_args`, `input.ply`. |

Use the staging helper from the repository root:

```bash
# Dry run the default minimal package: watermelon + jelly.
python scripts/stage_hf_assets.py --preset minimal --dry_run

# Stage the minimal package under /root/autodl-tmp/hf_clean_upload.
python scripts/stage_hf_assets.py --preset minimal

# Stage additional examples.
python scripts/stage_hf_assets.py --preset single_objects
```

## 3DGS Checkpoint Candidates

These directories have a standard 3DGS layout with `cameras.json`, `cfg_args`,
`input.ply`, and at least one `point_cloud/iteration_*/point_cloud.ply`.

| Scene directory | Size | Latest checkpoint | Likely config |
| --- | ---: | --- | --- |
| `bullet_0_psnr36` | 53M | `iteration_30000` | `bullet_config.json` |
| `bowl` | 56M | `iteration_10000` | `bowl_config.json` |
| `sand_castle` | 66M | `iteration_30000` | `sandcastle_config.json` |
| `lollipop` | 82M | `iteration_10000` | `lollipop_config.json` |
| `kiwi_0.04_psnr42` | 128M | `iteration_30000` | `kiwi_config.json` or `kiwi_physgs_config.json` |
| `milk2` | 138M | `iteration_30000` | `milk2_config.json` |
| `milk_0.03_psnr32` | 148M | `iteration_30000` | `milk_config.json` |
| `jelly` | 158M | `iteration_30000` | `jelly_config_nacc.json` |
| `oreo` | 266M | `iteration_30000` | `oreo_config.json` |
| `cookie` | 459M | `iteration_30000` | `cookie_config.json` |
| `watermelon_fruitninja` | 530M | `iteration_30000` | `watermelon_config_fruitninja.json` |
| `pineple` | 637M | `iteration_30000` | `pineple_config.json` |
| `watermelon` | 686M | `iteration_30000` | `watermelon_config.json` |
| `garden_ours` | 795M | `iteration_150000` | manual check needed |
| `toast` | 812M | `iteration_5000` | `tosta_config.json` |
| `dragonfruit` | 863M | `iteration_30000` | `dragonfruit_config.json` |
| `pumkin` | 997M | `iteration_30000` | `pumkin_config.json` |
| `a752b28d-f` | 1.1G | `iteration_30000` | manual check needed |
| `cake` | 1.1G | `iteration_30000` | `cake_config.json` |
| `kiwi` | 1.3G | `iteration_30000` | `kiwi_config.json` |
| `garden` | 2.4G | `iteration_30000` | manual check needed; has `transform_matrix.txt` |

## FruitNinja Standalone PLY Assets

`/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model/trained_gs_fruitninja`
does not use the standard 3DGS training directory layout. It contains standalone
PLY files plus small JSON metadata/config files:

| Asset | PLY size |
| --- | ---: |
| `apple` | 280M |
| `bread` | 195M |
| `cake` | 166M |
| `orange` | 255M |
| `pomegranate` | 158M |
| `watermelon` | 516M |

Suggested HF layout:

```text
fruitninja/
  apple.ply
  apple.json
  bread.ply
  bread.json
  ...
```

## Release Suggestions

1. Keep the first public HF package small and reproducible:
   `watermelon`, `jelly`, and their release configs.
2. Add a second archive for additional single-object examples:
   `cake`, `cookie`, `dragonfruit`, `kiwi`, `oreo`, `pineple`, `pumkin`,
   `sand_castle`, and `toast`.
3. Publish `trained_gs_fruitninja` as a separate optional asset group because
   its file layout differs from standard 3DGS checkpoints.
4. Keep `garden`, `garden_ours`, and anonymous/hash-named directories out of the
   first release until ownership, license, and scene naming are checked.
