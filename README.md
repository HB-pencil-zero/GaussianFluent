# GaussianFluent: Gaussian Simulation for Dynamic Scenes with Mixed Materials

### [[arXiv](https://arxiv.org/abs/2601.09265)]

Bei Huang<sup>1</sup>, Yixin Chen<sup>1</sup>, Ruijie Lu<sup>1</sup>, Gang Zeng<sup>1</sup>, Hongbin Zha<sup>1</sup>, Yuru Pei<sup>1</sup>, Siyuan Huang<sup>2</sup><br>
<sup>1</sup>Peking University, <sup>2</sup>Beijing Institute for General Artificial Intelligence<br>

![teaser-1.png](static/images/teaser-1.png)

Abstract: *3D Gaussian Splatting (3DGS) has emerged as a prominent 3D representation for high-fidelity and real-time rendering. Prior work has coupled physics simulation with Gaussians, but predominantly targets soft, deformable materials, leaving brittle fracture largely unresolved. This stems from two key obstacles: the lack of volumetric interiors with coherent textures in GS representation, and the absence of fracture-aware simulation methods for Gaussians. To address these challenges, we introduce GaussianFluent, a unified framework for realistic simulation and rendering of dynamic object states. First, it synthesizes photorealistic interiors by densifying internal Gaussians guided by generative models. Second, it integrates an optimized Continuum Damage Material Point Method (CD-MPM) to enable brittle fracture simulation at remarkably high speed. Our approach handles complex scenarios including mixed-material objects and multi-stage fracture propagation, achieving results infeasible with previous methods. Experiments clearly demonstrate GaussianFluent's capability for photo-realistic, real-time rendering with structurally consistent interiors, highlighting its potential for downstream application, such as VR and Robotics.*

## News
- [2026-01-14] Our paper GaussianFluent is on arXiv!
- [2026-04-01] Code Release.

## Cloning the Repository
This repository uses a modified [gaussian-splatting](https://github.com/HB-pencil-zero/gaussian-splatting) as a submodule. Use the following command to clone:

```shell
git clone --recurse-submodules git@github.com:HB-pencil-zero/GaussianFluent.git
```

## Setup

### Python Environment
To prepare the Python environment needed to run GaussianFluent, execute the following commands:
```shell
conda create -n GaussianFluent python=3.9
conda activate GaussianFluent

pip install -r requirements.txt
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```
By default, We use pytorch=2.0.1+cu117.

### Quick Start
We provide pretrained Gaussian Splatting models and their corresponding `.json` config files in the `config` directory.

To simulate a reconstructed 3D Gaussian Splatting scene, run the following command:
```shell
# For watermelon simulation
python gs_simulation/watermelon/gs_simulation_watermelon.py --model_path <path to gs model> --output_path <path to output folder> --config config/watermelon_config.json --render_img --compile_video

# For jelly simulation
python gs_simulation/jelly/gs_simulation_jellynacc.py --model_path <path to gs model> --output_path <path to output folder> --config config/jelly_config_nacc.json --render_img --compile_video
```
The images and video results will be saved to the specified output_path.

## Citation

```
@article{huang2026gaussianfluent,
      title={GaussianFluent: Gaussian Simulation for Dynamic Scenes with Mixed Materials}, 
      author={Huang, Bei and Chen, Yixin and Lu, Ruijie and Zeng, Gang and Zha, Hongbin and Pei, Yuru and Huang, Siyuan},
      journal={arXiv preprint arXiv:2601.09265},
      year={2026},
}
```

## Acknowledgement
This codebase is built upon [PhysGaussian](https://github.com/XPandora/PhysGaussian). We thank the authors for their excellent work.
