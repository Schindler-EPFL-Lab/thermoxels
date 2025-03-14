from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure_ml_utils.azure_connect import get_client


@dataclass
class EvalParameters:
    environment_version: str = "latest"  # 78
    experiment_name: str = "train-thermoxel"
    """Experiment name in azure"""

    model_version: str = "1"

    @property
    def environment(self) -> str:
        return "thermoxel-newtempfield-310"


def main() -> None:
    params = tyro.cli(EvalParameters)

    ml_client = get_client()

    data_asset = ml_client.data.get("buildingA_spring-plenoxel", version="3")

    job_name = "train-thermoxel-" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    if params.environment_version.isnumeric():
        env_version_string = ":" + params.environment_version
    else:
        env_version_string = "@" + params.environment_version

    job = command(
        inputs={
            "data": Input(
                path=data_asset.id,
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.MOUNT,
            )
        },
        code=".",
        environment=params.environment + env_version_string,
        compute="nerf-a100-2",
        command=(
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
            "--is_thermoxels "
        ),
        experiment_name=params.experiment_name,
        display_name=job_name,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(job)

    assert returned_job.services is not None


if __name__ == "__main__":
    main()
