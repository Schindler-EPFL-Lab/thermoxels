import logging
import sys
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import tyro

sys.path.append(".")
sys.path.append("./hot_cubes")


from hot_cubes.model.thermoxel_trainer import ThermoxelTrainer  # noqa: E402
from hot_cubes.model.training_param import Param  # noqa: E402
from plenoxels.opt.util import config_util  # noqa: E402
from plenoxels.opt.util.dataset import datasets  # noqa: E402


def get_arg() -> Param:

    param = tyro.cli(Param)

    if param.lr_sigma_final >= param.lr_sigma:
        raise RuntimeError("lr_sigma must be >= lr_sigma_final")
    if param.lr_sh_final >= param.lr_sh:
        raise RuntimeError("lr_sh must be >= lr_sh_final")
    if param.lr_temperature_final >= param.lr_temperature:
        raise RuntimeError("lr_temperature must be >= lr_temperature_final")
    if param.freeze_rgb_after > param.n_epoch:
        logging.warning("can only freeze after RGB training")

    for key, value in asdict(param).items():
        if value is not None:
            try:  # Take in account that some values might already have been logged
                mlflow.log_param(key, value)
            except mlflow.exceptions.RestException:
                pass

    Path(param.train_dir).mkdir(parents=True, exist_ok=True)
    return param


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(20200823)
    np.random.seed(20200823)
    param = get_arg()
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

    trainer = ThermoxelTrainer(dataset=dataset, dataset_val=dataset_val, param=param)

    trainer.optimize(factor=factor)


if __name__ == "__main__":
    main()
