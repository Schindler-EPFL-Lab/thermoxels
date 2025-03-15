from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure_ml_utils.azure_connect import get_client


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
    environment_version: str = "latest"  # 78
    experiment_name: str = "train-thermoxel"
    """Experiment name in azure"""

    model_version: str = "1"

    main_scene: str = "ThermoScenes"
    """
    Name of the main scene folder that contains different subscenes on Azure. For this\
    project, it is ThermoScenes.
    """
    scene_name: str = "buildingA_spring"
    """
    Name of the sub scene/object under the main_scene folder to be assigned to the\
    registered output of the script on Azure
    """
    version: str = "3"
    """Version of the main_scene dataset"""
    is_thermoxels: bool = True

    @property
    def environment(self) -> str:
        return "thermoxel-newtempfield-310"


def main() -> None:
    params = tyro.cli(EvalParameters)

    ml_client = get_client()

    dataset_path, thermoscens_dataset_id = get_data_path_and_thermoscenes_id(
        ml_client=ml_client,
        main_scene=params.main_scene,
        scene_name=params.scene_name,
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
        + params.scene_name
        + "-"
        + datetime.now().strftime("%d-%m-%Y-%H%M%S")
    )

    cmd = (
        "CUDA_LAUNCH_BLOCKING=1 python3.10 "
        "hot_cubes/cli/train_thermoxel_model.py "
        "--data_dir ${{inputs.data}} "
        "--train_dir ./ "
        "--n_epoch 6 "
        "--scene-radius 3.0 "
        "--tv_sparsity 0.25 "
        "--tv_sh_sparsity 0.25 "
        "--tv_temp_sparsity 0.1 "
        "--save_every 0 "
        "--eval_every 0 "
        "--log_mse_image "
        "--log_mae_image "
        "--scene-name " + params.scene_name
    )
    if not params.is_thermoxels:
        cmd += " --no-is-thermoxels"
        job_name = (
            "train-plenoxel-t-"
            + params.scene_name
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
