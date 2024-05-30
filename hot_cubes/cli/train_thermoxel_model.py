import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hot_cubes.model.thermoxel_trainer import ThermoxelTrainer
from hot_cubes.model.training_param import Param
from plenoxels.opt.util import config_util
from plenoxels.opt.util.dataset import datasets

sys.path.append(".")
sys.path.append("./hot_cubes")


def get_arg() -> Param:
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("general")
    group.add_argument("data_dir", type=str)

    group.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Config yaml file (will override args)",
    )
    group.add_argument(
        "--train_dir",
        "-t",
        type=str,
        default="ckpt",
        help="checkpoint and logging directory",
    )

    args = parser.parse_args()

    param = Param(
        config_file=args.config, train_dir=args.train_dir, data_dir=args.data_dir
    )
    with open(param.config_file, "r") as config_file:
        configs = json.load(config_file)

    param.update_from_dict(configs)
    assert param.lr_sigma_final <= param.lr_sigma, "lr_sigma must be >= lr_sigma_final"
    assert param.lr_sh_final <= param.lr_sh, "lr_sh must be >= lr_sh_final"
    assert param.lr_basis_final <= param.lr_basis, "lr_basis must be >= lr_basis_final"

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
        **config_util.build_data_options(param),
    )

    dataset_val = datasets[param.dataset_type](
        param.data_dir, split="val", **config_util.build_data_options(param)
    )

    trainer = ThermoxelTrainer(dataset=dataset, dataset_val=dataset_val, param=param)

    trainer.optimize(
        param=param, dataset=dataset, factor=factor, dataset_val=dataset_val
    )


if __name__ == "__main__":
    main()
