import argparse
import json
import shutil
from pathlib import Path


DEFAULT_MODEL_ROOT = Path("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/model")
DEFAULT_CONFIG_ROOT = Path("/root/autodl-tmp/debug_physgaussian/cdmpmGaussian/config")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/hf_clean_upload")

CONFIG_BY_SCENE = {
    "a752b28d-f": None,
    "bowl": "bowl_config.json",
    "bullet_0_psnr36": "bullet_config.json",
    "cake": "cake_config.json",
    "cookie": "cookie_config.json",
    "dragonfruit": "dragonfruit_config.json",
    "jelly": "jelly_config_nacc.json",
    "kiwi": "kiwi_config.json",
    "kiwi_0.04_psnr42": "kiwi_config.json",
    "lollipop": "lollipop_config.json",
    "milk2": "milk2_config.json",
    "milk_0.03_psnr32": "milk_config.json",
    "oreo": "oreo_config.json",
    "pineple": "pineple_config.json",
    "pumkin": "pumkin_config.json",
    "sand_castle": "sandcastle_config.json",
    "toast": "tosta_config.json",
    "watermelon": "watermelon_config.json",
    "watermelon_fruitninja": "watermelon_config_fruitninja.json",
}

STANDARD_RELEASE_SCENES = [
    "a752b28d-f",
    "bowl",
    "bullet_0_psnr36",
    "cake",
    "cookie",
    "dragonfruit",
    "garden",
    "garden_ours",
    "jelly",
    "kiwi",
    "kiwi_0.04_psnr42",
    "lollipop",
    "milk2",
    "milk_0.03_psnr32",
    "oreo",
    "pineple",
    "pumkin",
    "sand_castle",
    "toast",
    "watermelon",
    "watermelon_fruitninja",
]

FRUITNINJA_ASSET_GROUP = "trained_gs_fruitninja"
RELEASE_ASSETS = STANDARD_RELEASE_SCENES + [FRUITNINJA_ASSET_GROUP]

PRESETS = {
    "minimal": ["watermelon", "jelly"],
    "release": RELEASE_ASSETS,
    "single_objects": [
        "cake",
        "cookie",
        "dragonfruit",
        "kiwi",
        "oreo",
        "pineple",
        "pumkin",
        "sand_castle",
        "toast",
    ],
    "small": ["bowl", "bullet_0_psnr36", "lollipop", "sand_castle"],
}


def latest_checkpoint(scene_dir):
    point_cloud_dir = scene_dir / "point_cloud"
    candidates = []
    for path in point_cloud_dir.glob("iteration_*/point_cloud.ply"):
        try:
            iteration = int(path.parent.name.split("_")[-1])
        except ValueError:
            continue
        candidates.append((iteration, path))
    if not candidates:
        raise FileNotFoundError(f"No point_cloud/iteration_*/point_cloud.ply under {scene_dir}")
    return sorted(candidates)[-1]


def copy_file(src, dst, dry_run=False, overwrite=False):
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        return {"src": str(src), "dst": str(dst), "copied": False, "reason": "exists"}
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return {"src": str(src), "dst": str(dst), "copied": not dry_run, "dry_run": dry_run}


def stage_scene(
    scene,
    model_root,
    config_root,
    output_root,
    dry_run=False,
    overwrite=False,
    stage_config=True,
):
    scene_dir = model_root / scene
    if not scene_dir.exists():
        raise FileNotFoundError(scene_dir)

    iteration, checkpoint = latest_checkpoint(scene_dir)
    staged_model_dir = output_root / "model" / scene
    records = []

    for filename in ("cameras.json", "cfg_args", "input.ply"):
        records.append(
            copy_file(
                scene_dir / filename,
                staged_model_dir / filename,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        )

    records.append(
        copy_file(
            checkpoint,
            staged_model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
            dry_run=dry_run,
            overwrite=overwrite,
        )
    )

    config_name = CONFIG_BY_SCENE.get(scene)
    if config_name and stage_config:
        records.append(
            copy_file(
                config_root / config_name,
                output_root / "config" / config_name,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        )

    return {
        "type": "standard_3dgs",
        "scene": scene,
        "iteration": iteration,
        "source": str(scene_dir),
        "destination": str(staged_model_dir),
        "config": config_name,
        "files": records,
    }


def stage_fruitninja_assets(model_root, output_root, dry_run=False, overwrite=False):
    asset_dir = model_root / FRUITNINJA_ASSET_GROUP
    if not asset_dir.exists():
        raise FileNotFoundError(asset_dir)

    staged_asset_dir = output_root / "model" / FRUITNINJA_ASSET_GROUP
    records = []
    for src in sorted(asset_dir.glob("*")):
        if src.suffix not in {".ply", ".json"}:
            continue
        records.append(
            copy_file(
                src,
                staged_asset_dir / src.name,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        )

    return {
        "type": "standalone_ply_group",
        "scene": FRUITNINJA_ASSET_GROUP,
        "source": str(asset_dir),
        "destination": str(staged_asset_dir),
        "files": records,
    }


def stage_all_configs(config_root, output_root, dry_run=False, overwrite=False):
    records = []
    for src in sorted(config_root.glob("*.json")):
        records.append(
            copy_file(
                src,
                output_root / "config" / src.name,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        )
    return records


def resolve_scenes(args):
    scenes = []
    for preset in args.preset:
        scenes.extend(PRESETS[preset])
    scenes.extend(args.scene)
    seen = set()
    unique = []
    for scene in scenes:
        if scene not in seen:
            seen.add(scene)
            unique.append(scene)
    if not unique:
        unique = PRESETS["minimal"]
    return unique


def main():
    parser = argparse.ArgumentParser(description="Stage GaussianFluent 3DGS assets for Hugging Face upload.")
    parser.add_argument("--model_root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--config_root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--preset", choices=sorted(PRESETS), action="append", default=[])
    parser.add_argument(
        "--configs",
        choices=("all", "matched", "none"),
        default="all",
        help="Which simulation config JSON files to stage.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_root = Path(args.model_root)
    config_root = Path(args.config_root)
    output_root = Path(args.output_root)
    summary = {
        "model_root": str(model_root),
        "config_root": str(config_root),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
        "scenes": [],
        "configs": [],
    }
    for scene in resolve_scenes(args):
        if scene == FRUITNINJA_ASSET_GROUP:
            summary["scenes"].append(
                stage_fruitninja_assets(
                    model_root=model_root,
                    output_root=output_root,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )
            )
        else:
            summary["scenes"].append(
                stage_scene(
                    scene,
                    model_root=model_root,
                    config_root=config_root,
                    output_root=output_root,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    stage_config=args.configs == "matched",
                )
            )

    if args.configs == "all":
        summary["configs"] = stage_all_configs(
            config_root=config_root,
            output_root=output_root,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.configs == "matched":
        summary["configs"] = "matched scene configs only"
    else:
        summary["configs"] = "not staged"

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        with open(output_root / "asset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
