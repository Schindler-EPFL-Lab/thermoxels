import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import mlflow
import tyro

# Add the parent directories to sys.path
sys.path.append(".")
sys.path.append("./dataclasses-reverse-cli")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from thermoxels_fem.thermoxel_to_volumetric_simple import (
    thermoxel_to_volumetric_density,
)
from thermoxels_fem.volumetric_simple_regions_simulation import (
    VolumetricMeshSimulationSimpleRegions,
)

jax.config.update("jax_enable_x64", True)


@dataclass
class FEMParameters(ReverseCli):
    ckpt_npz_filepath: Path
    output_dir: Path
    num_steps: int = 1
    dt: float = 1e-5
    input_dir: Path = Path("")

    def __post_init__(self) -> None:
        if self.input_dir == Path(""):
            self.input_dir = self.output_dir


def main(fem_parameters: FEMParameters) -> None:
    """
    Initializes and runs a TransientThermalProblem simulation from a Thermoxels
    SparseGrid checkpoint (.npz) located at `ckpt_npz_filepath`.The SparseGrid is
    preprocessed to generate the simulation inputs and saved in `input_dir` as:
    - The hexahedral(hex8)-cell mesh extracted according to the value of
    `density_threshold`, saved in a VTK file named mesh.vtk
    - A precomputed rotation `theta_xyz` in degrees to straighten the mesh saved in a
    JSON file named rotation.json

    The mesh data includes initial temperature information. After the preprocessing,
    the simulation is executed, and the results (including the updated mesh and
    simulation outputs) are saved to `output_dir`.
    """

    mlflow.log_params(
        {
            "input_dir": fem_parameters.input_dir,
            "output_dir": fem_parameters.output_dir,
            "num_steps": fem_parameters.num_steps,
            "dt": fem_parameters.dt,
        }
    )

    start_time = datetime.now()

    print("***", jax.devices())

    density_threshold: float = 22.0
    theta_xyz: list[float] = [70.0, -5.0, 0.0]

    thermoxel_to_volumetric_density(
        ckpt_npz_filepath=fem_parameters.ckpt_npz_filepath,
        output_dir=fem_parameters.input_dir,
        theta_xyz=theta_xyz,
        density_threshold=density_threshold,
    )

    simulation = VolumetricMeshSimulationSimpleRegions(
        input_dir=fem_parameters.input_dir,
        output_dir=fem_parameters.output_dir,
        is_initial_temperature=True,
        dt=fem_parameters.dt,
    )

    simulation.run(num_steps=fem_parameters.num_steps)

    mlflow.log_metric(
        "execution time [s]", (datetime.now() - start_time).total_seconds()
    )


if __name__ == "__main__":
    param = tyro.cli(FEMParameters)
    main(param)
