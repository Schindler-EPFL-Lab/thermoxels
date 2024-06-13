from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure_ml_utils.azure_connect import get_client


@dataclass
class EvalParameters:
    environment_version: str = "latest"
    experiment_name: str = "plenoxel-train-eval"
    """Experiment name in azure"""

    model_version: str = "1"

    @property
    def environment(self) -> str:
        return "thermoxel-newtempfield-310"


def main() -> None:
    params = tyro.cli(EvalParameters)

    ml_client = get_client()

    data_asset = ml_client.data.get("buildingA_spring-plenoxel", version="1")

    job_name = "train-plenoxel--" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    job = command(
        inputs={
            "data": Input(
                path=data_asset.id,
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.MOUNT,
            )
        },
        code=".",
        environment=params.environment + "@" + params.environment_version,
        compute="voxel-a100",
        command=(
            "python3.10 plenoxels/opt/opt.py ${{inputs.data}} "
            "--train_dir ./ "
            "-c plenoxels/opt/configs/thermoscene.json "
        ),
        experiment_name=params.experiment_name + "plenoxel",
        display_name=job_name,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(job)

    assert returned_job.services is not None


if __name__ == "__main__":
    main()
