import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
import tyro

sys.path.append(".")
sys.path.append("./thermoxels")

from plenoxels.opt.util import config_util  # noqa: E402
from plenoxels.opt.util.dataset import datasets  # noqa: E402
from thermoxels.datasets.datasets_utils.colmap_json_to_txt import (
    convert_colmap_json_to_txt,
)
from thermoxels.grid_export.npy_to_mesh import convert_to_hex8_mesh
from thermoxels.model.thermoxel_trainer import ThermoxelTrainer  # noqa: E402
from thermoxels.model.training_param import TrainingParam  # noqa: E402
from thermoxels.renderer_evaluator.model_evaluator import Evaluator
from thermoxels.renderer_evaluator.render_param import RenderParam


def train(param: TrainingParam) -> None:
    start_time = datetime.now()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    factor = 1

    dataset = datasets[param.dataset_type](
        param.data_dir,
        split="train",
        device=device,
        factor=factor,
        n_images=param.n_train,
        rgb_dropout=param.rgb_dropout,
        thermal_dropout=param.thermal_dropout,
        **config_util.build_data_options(param),
    )

    dataset_val = datasets[param.dataset_type](
        param.data_dir,
        split="val",
        rgb_dropout=param.rgb_dropout,
        thermal_dropout=param.thermal_dropout,
        **config_util.build_data_options(param),
    )

    assert param.data_dir is not None

    with open(Path(param.data_dir, "temperature_bounds.json")) as file:
        data = json.load(file)
    t_max = data["absolute_max_temperature"]
    t_min = data["absolute_min_temperature"]

    trainer = ThermoxelTrainer(
        dataset=dataset,
        dataset_val=dataset_val,
        param=param,
        min_temperature=t_min,
        max_temperature=t_max,
    )

    trainer.optimize(factor=factor)
    trainer.eval_step()

    trainer.save_model(
        to_folder=param.model_save_path / (str(param.model_save_path.stem) + "_kelvin")
    )
    trainer.save_model(
        to_folder=param.model_save_path
        / (str(param.model_save_path.stem) + "_celsius"),
        new_scale="Celsius",
    )

    for density in range(0, 100, 10):
        convert_to_hex8_mesh(
            ckpt_path=param.model_save_path
            / (str(param.model_save_path.stem) + "_celsius"),
            model_name_prefix=param.scene_name,
            density_threshold=density,
        )

    # Evaluate the model
    assert param.data_dir is not None
    render_param = RenderParam(
        model_path=param.model_save_path
        / (str(param.model_save_path.stem) + "_kelvin"),
        data_dir=param.data_dir,
        render_dir=Path("outputs") / param.model_save_path.stem,
        nobg=False,
        dataset_type="auto",
        train=True,
        is_thermoxels=param.is_thermoxels,
        metric_path=param.model_save_path,
    )
    dataset_test = datasets[param.dataset_type](
        param.data_dir,
        split="test",
        **{
            "dataset_type": param.dataset_type,
            "seq_id": param.seq_id,
            "epoch_size": param.epoch_size,  # batch_size
            "white_bkgd": param.white_bkgd,
            "hold_every": 8,
            "normalize_by_bbox": param.normalize_by_bbox,
            "data_bbox_scale": param.data_bbox_scale,
            "cam_scale_factor": param.cam_scale_factor,
            "normalize_by_camera": param.normalize_by_camera,
            "permutation": False,
        },
    )

    evaluator = Evaluator(param=render_param, dataset=dataset_test)
    evaluator.save_metric(prefix=param.model_save_path.stem)

    mlflow.log_metric("execution time", (datetime.now() - start_time).total_seconds())


def main() -> None:
    param = tyro.cli(TrainingParam)
    for key, value in asdict(param).items():
        if value is not None:
            try:  # Take in account that some values might already have been logged
                mlflow.log_param(key, value)
            except mlflow.exceptions.RestException:
                pass

    torch.manual_seed(20200823)
    np.random.seed(20200823)

    assert param.data_dir is not None
    data_dir_path = Path(param.data_dir)
    # Transform our nerfstudio dataset into a compatible dataset
    new_dataset = Path("./outputs/dataset")
    convert_colmap_json_to_txt(dataset_path=data_dir_path, save_to=new_dataset)
    param.data_dir = str(new_dataset)

    original_path = param.model_save_path
    param.model_save_path = original_path / "thermoxel"
    train(param)

    if param.t_loss != 0:
        param.t_loss = 0
        param.model_save_path = original_path / "no_t_loss"
        train(param)

    if param.is_thermoxels:
        param.is_thermoxels = False
        param.model_save_path = original_path / "plenoxel_t"
        train(param)

    shutil.rmtree(new_dataset)


if __name__ == "__main__":
    main()
