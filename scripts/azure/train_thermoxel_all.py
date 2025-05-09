import logging
from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure_ml_utils.azure_connect import get_client

from hot_cubes.model.training_param import TrainingParam

SCENES = {
    "BI-building": TrainingParam(
        scene_name="BI-building",
        scene_radius=1.5,
        tv_sparsity=0.25,
        tv_temp_sparsity=0.25,
        tv_sh_sparsity=0.25,
        t_loss=0.001,
        n_epoch=10,
    ),
    "INR-building": TrainingParam(
        scene_name="INR-building",
        scene_radius=1.5,
        tv_sparsity=0.25,
        tv_temp_sparsity=0.25,
        tv_sh_sparsity=0.25,
        t_loss=0.001,
        n_epoch=10,
    ),
    "MED-building": TrainingParam(
        scene_name="MED-building",
        scene_radius=1.5,
        tv_sparsity=0.25,
        tv_temp_sparsity=0.25,
        tv_sh_sparsity=0.25,
        t_loss=0.001,
        n_epoch=10,
    ),
    "building-sunrise": TrainingParam(
        scene_name="building-sunrise",
        scene_radius=1.5,
        tv_sparsity=0.25,
        tv_temp_sparsity=0.25,
        tv_sh_sparsity=0.25,
        t_loss=0.001,
        n_epoch=10,
    ),
    "buildingA_spring": TrainingParam(
        scene_name="buildingA_spring",
        scene_radius=5,
        t_loss=0,
        n_epoch=10,
    ),
    "buildingA_winter": TrainingParam(
        scene_name="buildingA_winter",
        scene_radius=7,
        lambda_tv_temp=1e-3,
        t_loss=0,
        n_epoch=10,
    ),
    "dorm1": TrainingParam(
        scene_name="dorm1",
        scene_radius=3,
        t_loss=0,
        n_epoch=10,
    ),
    "dorm2": TrainingParam(
        scene_name="dorm2",
        scene_radius=3,
        t_loss=0,
        n_epoch=10,
    ),
    "double_robot": TrainingParam(
        scene_name="double_robot",
        scene_radius=10,
        tv_sparsity=0.25,
        tv_temp_sparsity=0.1,
        tv_sh_sparsity=0.25,
        t_loss=0,
        n_epoch=10,
    ),
    "exhibition_building": TrainingParam(
        scene_name="exhibition_building",
        scene_radius=25,
        t_loss=0,
        n_epoch=10,
    ),
    "freezing_ice_cup": TrainingParam(
        scene_name="freezing_ice_cup",
        scene_radius=7,
        t_loss=0,
        n_epoch=10,
    ),
    "heater_water_cup": TrainingParam(
        scene_name="heater_water_cup",
        scene_radius=3,
        t_loss=0,
        n_epoch=10,
    ),
    "melting_ice_cup": TrainingParam(
        scene_name="melting_ice_cup",
        scene_radius=3,
        t_loss=0,
        n_epoch=10,
    ),
    "raspberrypi": TrainingParam(
        scene_name="raspberrypi",
        scene_radius=10,
        t_loss=0,
        n_epoch=10,
    ),
    "trees": TrainingParam(
        scene_name="trees",
        scene_radius=25,
        t_loss=0,
        n_epoch=10,
    ),
    "heater_water_kettle": TrainingParam(
        scene_name="heater_water_kettle",
        scene_radius=1.5,
        t_loss=0,
        n_epoch=10,
    ),
}


@dataclass
class EvalParameters:
    environment_version: str = "latest"  # 78
    """Experiment name in azure"""

    model_version: str = "1"
    scene: str | None = None

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
        logging.warning("The output folder will be fixed to thermoxel_model.")


def main() -> None:
    params = tyro.cli(EvalParameters)

    ml_client = get_client()
    thermoscenes_dataset = ml_client.data.get(
        name=params.main_scene, version=params.version
    )
    thermoscens_dataset_id = thermoscenes_dataset.id
    assert thermoscenes_dataset.path is not None
    scenes = SCENES
    if params.scene is not None:
        scenes = {params.scene: SCENES[params.scene]}

    for scene in scenes:
        parameters = scenes[scene]
        dataset_path = str(thermoscenes_dataset.path) + str(parameters.scene_name) + "/"
        print(dataset_path)

        job_inputs = dict(
            data=Input(
                type=AssetTypes.URI_FOLDER,  # type: ignore
                path=str(dataset_path),
                mode=InputOutputModes.RO_MOUNT,
            ),
            thermoscenes_dataset=Input(
                type=AssetTypes.URI_FOLDER,  # type: ignore
                path=thermoscens_dataset_id,
                mode=InputOutputModes.RO_MOUNT,
            ),
        )

        job_outputs = {
            "outputs": Output(type=AssetTypes.CUSTOM_MODEL),  # type: ignore
        }

        job_name = (
            "train-thermoxel-"
            + parameters.scene_name
            + "-"
            + datetime.now().strftime("%d-%m-%Y-%H%M%S")
        )

        cmd = (
            "CUDA_LAUNCH_BLOCKING=1 python3.10 "
            "hot_cubes/cli/train_thermoxel_model.py "
            "--data_dir ${{inputs.data}} "
            "--model-save-path ${{outputs.outputs}} "
            "--n-epoch "
            + str(parameters.n_epoch)
            + " --t-loss "
            + str(parameters.t_loss)
            + " --scene-radius "
            + str(parameters.scene_radius)
            + " --tv_sparsity "
            + str(parameters.tv_sparsity)
            + " --tv_sh_sparsity "
            + str(parameters.tv_sh_sparsity)
            + " "
            "--tv_temp_sparsity " + str(parameters.tv_temp_sparsity) + " "
            "--save_every 0 "
            "--eval_every 0 "
            "--log_mse_image "
            "--log_mae_image "
            "--scene-name " + parameters.scene_name
        )
        if not parameters.is_thermoxels:
            cmd += " --no-is-thermoxels"
            job_name = (
                "train-plenoxel-t-"
                + parameters.scene_name
                + "-"
                + datetime.now().strftime("%d-%m-%Y-%H%M%S")
            )
        if params.environment_version.isnumeric():
            env_version_string = ":" + params.environment_version
        else:
            env_version_string = "@" + params.environment_version

        experiment_name = "train-thermoxel-" + parameters.scene_name
        job = command(
            inputs=job_inputs,
            outputs=job_outputs,
            code=".",
            environment=params.environment + env_version_string,
            compute="voxel-a100",
            command=cmd,
            experiment_name=experiment_name,
            display_name=job_name,
            name=job_name,
        )

        returned_job = ml_client.jobs.create_or_update(job)

        assert returned_job.services is not None


if __name__ == "__main__":
    main()
