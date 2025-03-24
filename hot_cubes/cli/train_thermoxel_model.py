import json
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
import tyro

sys.path.append(".")
sys.path.append("./hot_cubes")

from hot_cubes.datasets.datasets_utils.colmap_json_to_txt import (
    convert_colmap_json_to_txt,
)
from hot_cubes.model.thermoxel_trainer import ThermoxelTrainer  # noqa: E402
from hot_cubes.model.training_param import TrainingParam  # noqa: E402
from hot_cubes.renderer_evaluator.model_evaluator import Evaluator
from hot_cubes.renderer_evaluator.render_param import RenderParam
from plenoxels.opt.util import config_util  # noqa: E402
from plenoxels.opt.util.dataset import datasets  # noqa: E402


def main():
    start_time = datetime.now()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(20200823)
    np.random.seed(20200823)
    param = tyro.cli(TrainingParam)
    assert param.data_dir is not None
    factor = 1

    data_dir_path = Path(param.data_dir)
    # Transform our nerfstudio dataset into a compatible dataset
    new_dataset = Path("./outputs/dataset")
    convert_colmap_json_to_txt(dataset_path=data_dir_path, save_to=new_dataset)
    param.data_dir = str(new_dataset)

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

    trainer.save_model(to_folder=param.model_save_path / "kelvin")
    trainer.save_model(
        to_folder=param.model_save_path / "celsius",
        new_scale="Celsius",
    )

    # Evaluate the model
    assert param.data_dir is not None
    render_param = RenderParam(
        model_path=param.model_save_path / "kelvin",
        data_dir=param.data_dir,
        render_dir="./",
        nobg=False,
        dataset_type="auto",
        train=True,
        is_thermoxels=param.is_thermoxels,
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
    evaluator.save_metric(log_only=True)

    mlflow.log_metric("execution time", (datetime.now() - start_time).total_seconds())


if __name__ == "__main__":
    main()
