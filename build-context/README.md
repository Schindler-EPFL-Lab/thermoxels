# Thermoxels: Learn multimodal 3D models with Thermal Voxels

This paper was accepted as an oral presentation at CISBAT 2025.

[PDF](https://arxiv.org/abs/2504.04448), [cite us](thermoxels.bib)

## Introduction

Thermoxel is a method to build simulation compatible 3D models from multi-modal dataset (with a focuson RGB and thermal).
What to take picture of an object and directly plug it in a finite-element analysis software?
Thermoxels is for you!

![Summary of the method](images/thermoxels_pipeline.png)

<img src="images/animation.gif" alt="Example gif" style="transform: rotate(180deg);">

Thermoxels a voxel-based representation of the scene, where each voxel is associated with a density, a temperature and a color.
We learn a view independent temperature on the foreground object.

## Evaluation

Evaluation is done on the [ThermoScenes](https://zenodo.org/records/15609062) dataset.
See the original [paper](https://www.sciencedirect.com/science/article/abs/pii/S1474034625002381) and [code repo](github.com/Schindler-EPFL-Lab/thermo-nerf) for more details.

## Installation

Install with uv:

```bash
uv sync
```

Thermoxels was tested with PyTorch `1.11.0+cu113` and torchvision `0.12.0+cu113`.
See PyTorch installation instructions [here](https://pytorch.org/get-started/previous-versions/) to find the correct version.

To install Thermoxels on a container environment, you can use our provided Dockerfile in the dockerfile folder.

### FEM

To install the FEM dependencies one need BLAS and Lapack:

```
sudo apt install libblas-dev liblapack-dev petsc-dev
```

Install with uv and extra fem flag:

```bash
export PEP517_BUILD_BACKEND=setuptools.build_meta
uv sync --extra fem
```

It is importnt to export the flag above due to [this issue](https://github.com/astral-sh/uv/issues/10052).

## Train and Evaluate

### Azure

Train on azure with `/scripts/azure/train_thermoxel.py --training-param.scene-name {scene_name}`.

### Local

To train and evaluate Thermoxels, first download Thermoscenes and then use the following
scripts

```bash
python thermoxels/cli/train_thermoxel_model.py --data_dir
{data_dir} --train_dir {train_dir} --n_epoch  {n_epoch} --scene-radius {radius}
```

All the training params are in the `thermoxels/model/training_param.py` and can be modified with the CLI arguments.
Adding `CUDA_LAUNCH_BLOCKING=1` before python launch can sometimes mitigate some cuda issues.

E.g.

```bash
CUDA_LAUNCH_BLOCKING=1 python thermoxels/cli/train_thermoxel_model.py --data_dir dataset/dataset_name --train_dir training/ --n_epoch  10 --scene-radius 10
```
The model will be saved both in Kelvin and Celsius in `param.model_save_path / (str(param.model_save_path.stem) + "_kelvin"` and `param.model_save_path / (str(param.model_save_path.stem) + "_celsius"` respectively.

## Mesh export

You can export a mesh from the trained model using the following script:

```bash
python thermoxels/grid_export/grid_to_stl.py --npz-file ckpt.npz --put-colors --percentile-threshold 90
```

This will save the mesh in obj format in the same folder as the npz file.
Color is derived from the temperature of the voxels using a colormap and the percentile threshold is used to keep only the volxels with a density above threshold.
Filtering of the mesh is natively done, keeping only the largest connected component of the mesh.
If your foreground object vanishes when exporting, turn this option off and perform post filtering of the mesh, for instance with [Meshlab](https://www.meshlab.net/).

## Generate gif of mesh

You can generate gifs of already generated meshes using :

```bash
export XDG_SESSION_TYPE=x11
python thermoxels/grid_export/generate_gif_of_mesh.py --obj-file-path {your_path.obj}
--total-frames {n_frames}
```

The export is needed because libdecor-gtk needs X11 still.

This will generate a gif of the mesh rotating around the x axis.
If needed, you can provide an initial rotation angle prior to the x-axis rotation.

## Simulate

Download your model as a npz file expressed in Kelvin (it should be saved at the end of training in both Celsius and Kelvin).
Then the simulaiton can be run using [jaxfem](https://github.com/deepmodeling/jax-fem) with the command `python thermoxels_fem/cli/run_thermal_simple_regions_simulation_from_ckpt.py --num-steps <X> --input-dir <folder with ckpt file> --output-dir <folder for outputs> --dt 1e-5 --ckpt-npz-filepath <checkpoint file of the model>`

## Contribute

We welcome contributions to Thermoxels!

We format code using ruff and follow PEP8.
The code needs to be type annotated and following our documentation style.
