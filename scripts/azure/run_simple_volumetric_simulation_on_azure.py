from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import tyro
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client

from thermoxels_fem.cli.run_thermal_simple_regions_simulation_from_ckpt import (
    FEMParameters,
)


@dataclass
class Parameters:
    model: str = "BuildingA"
    env_version: str = "latest"
    exp_name: str = "jaxfem-simulation"

    experiment_parameters: FEMParameters = FEMParameters(
        ckpt_npz_filepath=Path(""), output_dir=Path("outputs")
    )

    @property
    def environmnet(self) -> str:
        return "thermoxel-310"


def main() -> None:
    params = tyro.cli(Parameters)

    ml_client = get_client()
    trained_model = ml_client.data.get(name="thermoxels_trained_models", version="1")
    inputs = {
        "input_model": Input(
            type=AssetTypes.URI_FOLDER,
            path=trained_model.id,
        )
    }

    params.experiment_parameters.ckpt_npz_filepath = Path(
        "${{inputs.input_model}}", params.model, "ckpt2.npz"
    )

    mlflow.set_tracking_uri(mlflow.get_tracking_uri())

    job_name = "run-simple-simulation-" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    if params.env_version.isnumeric():
        env_version_string = ":" + params.env_version
    else:
        env_version_string = "@" + params.env_version

    job = command(
        inputs=inputs,
        code=".",
        environment=params.environmnet + env_version_string,
        compute="nerf-a100-2",
        experiment_name=params.exp_name,
        display_name=job_name,
        name=job_name,
        command=(
            "python3.10 "
            + "thermoxels_fem/cli/run_thermal_simple_regions_simulation_from_ckpt.py "
            + params.experiment_parameters.to_command_string()
        ),
    )

    returned_job = ml_client.jobs.create_or_update(job)

    if returned_job.services is None:
        raise RuntimeError()


if __name__ == "__main__":
    main()
