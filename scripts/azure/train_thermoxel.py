from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure_ml_utils.azure_connect import get_client

from hot_cubes.model.training_param import TrainingParam


def get_data_path_and_thermoscenes_id(
    ml_client, main_scene, scene_name, version
) -> tuple[str, str]:
    """
    Obtain ThermoScenes dataset id and the path to the 'scene_name' dataset under the
    ThermoScenes dataset folder on Azure.

    :returns: A tuple of strings of scene_name' dataset path and ThermoScenes dataset id
    """
    thermoscenes_dataset = ml_client.data.get(name=main_scene, version=version)
    assert thermoscenes_dataset.id is not None
    dataset_path = str(thermoscenes_dataset.path) + str(scene_name) + "/"
    return dataset_path, thermoscenes_dataset.id


@dataclass
class EvalParameters:
    training_param: TrainingParam
    environment_version: str = "latest"  # 78
    experiment_name: str = "train-thermoxel"
    """Experiment name in azure"""

    model_version: str = "1"

    main_scene: str = "ThermoScenes"
    """
    Name of the sub scene/object under the main_scene folder to be assigned to the\
    registered output of the script on Azure
    """
    version: str = "3"

    @property
    def environment(self) -> str:
        return "thermoxel-newtempfield-310"

    def __post_init__(self):
        if self.training_param.scene_name is None:
            raise ValueError("scene_name must be provided")

        if self.training_param.data_dir is not None:
            raise ValueError("data_dir must not be provided")


def main() -> None:
    params = tyro.cli(EvalParameters)
    assert params.training_param.scene_name is not None

    ml_client = get_client()

    dataset_path, thermoscens_dataset_id = get_data_path_and_thermoscenes_id(
        ml_client=ml_client,
        main_scene=params.main_scene,
        scene_name=params.training_param.scene_name,
        version=params.version,
    )
    job_inputs = dict(
        data=Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=dataset_path,
            mode=InputOutputModes.RO_MOUNT,
        ),
        thermoscenes_dataset=Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=thermoscens_dataset_id,
            mode=InputOutputModes.RO_MOUNT,
        ),
    )

    job_name = (
        "train-thermoxel-"
        + params.training_param.scene_name
        + "-"
        + datetime.now().strftime("%d-%m-%Y-%H%M%S")
    )

    cmd = (
        "CUDA_LAUNCH_BLOCKING=1 python3.10 "
        "hot_cubes/cli/train_thermoxel_model.py "
        "--data_dir ${{inputs.data}} "
        "--train_dir ./ "
        "--n_epoch "
        + str(params.training_param.n_epoch)
        + " --scene-radius "
        + str(params.training_param.scene_radius)
        + " --tv_sparsity "
        + str(params.training_param.tv_sparsity)
        + " --tv_sh_sparsity 0.25 "
        "--tv_temp_sparsity 0.1 "
        "--save_every 0 "
        "--eval_every 0 "
        "--log_mse_image "
        "--log_mae_image "
        "--scene-name " + params.training_param.scene_name
    )
    if not params.training_param.is_thermoxels:
        cmd += " --no-is-thermoxels"
        job_name = (
            "train-plenoxel-t-"
            + params.training_param.scene_name
            + "-"
            + datetime.now().strftime("%d-%m-%Y-%H%M%S")
        )
    if params.environment_version.isnumeric():
        env_version_string = ":" + params.environment_version
    else:
        env_version_string = "@" + params.environment_version

    job = command(
        inputs=job_inputs,
        code=".",
        environment=params.environment + env_version_string,
        compute="nerf-a100-2",
        command=cmd,
        experiment_name=params.experiment_name,
        display_name=job_name,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(job)

    assert returned_job.services is not None


if __name__ == "__main__":
    main()
