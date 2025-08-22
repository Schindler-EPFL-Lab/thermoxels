import json
from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from scipy.constants import convert_temperature
from scipy.spatial.transform import Rotation

from thermoxels_fem.thermal_problem import TransientThermalProblem


class VolumetricMeshSimulationSimpleRegions:
    """
    A class for setting up and running a TransientThermalProblem simulation using a
    hexahedral-cell VTK mesh, with customizable initial temperature settings.

    This class:
    - Processes the input VTK mesh and prepares it for simulation.
    - Iteratively solves the transient thermal problem.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        is_initial_temperature: bool,
        dt: float,
    ) -> None:
        """
        The necessary input for the simulation are located in the `input_dir` as such:
        - The hexahedral-cell mesh saved in a VTK file named mesh.vtk
        - A precomputed rotation xyz in degrees to straighten the mesh saved in a JSON
        file named rotation.json

        The results of the transient thermal problem simulation are saved in
        `output_dir`.

        If `is_initial_temperature` is True, the initial temperature at time 0 is taken
        from the VTK mesh; otherwise, a default value is used.
        """
        # Check the input
        self._mesh_filepath = Path(input_dir, "mesh.vtk")
        self._rotation_filepath = Path(input_dir, "rotation.json")
        self._check_input_dir()

        self._is_initial_temperature = is_initial_temperature

        # Create output folders
        self._vtk_dir = Path(output_dir, "vtk")
        self._mesh_dir = Path(output_dir, "msh")
        self._vtk_dir.mkdir(exist_ok=True, parents=True)
        self._mesh_dir.mkdir(exist_ok=True, parents=True)

        # Load the Mesh with meshio, then center and rotate it
        self._meshio_mesh = meshio.read(self._mesh_filepath)
        self._center_and_straighten_mesh()

        resized_ratio = 0.01 / 2000
        self._meshio_mesh.points *= resized_ratio

        min_vals = self._meshio_mesh.points.min(axis=0)
        max_vals = self._meshio_mesh.points.max(axis=0)

        # Instantiate the jax-fem Mesh
        element_type = "HEX8"
        cell_type = get_meshio_cell_type(ele_type=element_type)
        self._mesh = Mesh(
            points=np.array(self._meshio_mesh.points, dtype=jnp.float32),
            cells=self._meshio_mesh.cells_dict[cell_type],
            ele_type=element_type,
        )

        atol_y = 10 * resized_ratio
        atol_x = 8 * resized_ratio

        def is_exterior_wall_simple(point):
            # front facade is the lowest y
            isFrontFacade = jnp.isclose(point[1], min_vals[1], atol=atol_y)
            # Left facade (without cylinder) is the biggest x
            isLeftFacade = jnp.isclose(point[0], max_vals[0], atol=atol_x)
            isRightFacade = jnp.isclose(point[0], min_vals[0], atol=atol_x)
            return isFrontFacade | isLeftFacade | isRightFacade

        def is_interior_wall_simple(point):
            # front facade is the lowest y
            isFrontFacade = jnp.isclose(point[1], min_vals[1], atol=atol_y)
            # Left facade (without cylinder) is the biggest x
            isLeftFacade = jnp.isclose(point[0], max_vals[0], atol=atol_x)
            isRightFacade = jnp.isclose(point[0], min_vals[0], atol=atol_x)
            return ~(isFrontFacade | isLeftFacade | isRightFacade)

        # Simple Regions Definition
        self._location_fns = [is_exterior_wall_simple, is_interior_wall_simple]

        # Instantiate the jax-fem Transient Thermal Problem
        self._transient_thermal_problem = TransientThermalProblem(
            mesh=self._mesh,
            vec=1,
            dim=3,
            ele_type=element_type,
            location_fns=self._location_fns,
            dt=dt,
            gauss_order=1,
            exterior_temperature=convert_temperature(10.0, "Celsius", "Kelvin"),
            interior_temperature=convert_temperature(30.0, "Celsius", "Kelvin"),
        )

        # Set the initial temperature of the simulation at time 0
        self._set_initial_temperature(
            init_points_temperature=(
                self._meshio_mesh.point_data["point_temperature"]
                if self._is_initial_temperature
                else None
            )
        )

    def run(self, num_steps: int) -> None:
        """
        This method first initializes the simulation temperature at time 0. It then runs
        the simulation in a loop. During each iteration, the following steps occur:
        1. The current temperature is saved as the solution for that step.
        2. Internal temperature variables are updated based on the current solution.
        3. The simulation is solved for the next time step.
        4. The temperature is updated accordingly for the next iteration.
        """
        previous_temperature_solution = self._initial_point_temperature

        # Simualtion timeloop iteration
        simulation_times = jnp.arange(
            self._transient_thermal_problem.dt,
            self._transient_thermal_problem.dt * num_steps + 1,
            self._transient_thermal_problem.dt,
        )
        for i in range(len(simulation_times)):
            # Save solutions to local folder.
            vtk_filepath = Path(self._vtk_dir, f"T_{i:05d}.vtu")
            save_sol(
                fe=self._transient_thermal_problem.fes[0],
                sol=previous_temperature_solution,
                sol_file=vtk_filepath,
                point_infos=[("T", previous_temperature_solution)],
            )
            # mlflow.log_artifact(str(vtk_filepath), "./vtk/")
            self._transient_thermal_problem.set_params([previous_temperature_solution])

            # Update T solution
            previous_temperature_solution = solver(self._transient_thermal_problem)[0]
        # mlflow.log_artifacts(str(self._vtk_dir), "./outputs/")

    def _set_initial_temperature(
        self,
        init_points_temperature: None | list[float],
        wall_temperature_celsius: float = 20.0,
        exterior_surface_temperature_celsius: float = 19.0,
        interior_surface_temperature_celsius: float = 21.0,
    ) -> None:
        """
        Set `self.__initial_point_temperature` to the initial temperature of the
        simualtion at time 0.

        If `init_points_temperature` is provided, it is used to set the temperature.

        Otherwise, default values are assigned based on the parameters:
        - `wall_temperature_celsius`: Temperature for points inside the wall.
        - `exterior_surface_temperature_celsius`: Temperature for points on the exterior
        surface.
        - `interior_surface_temperature_celsius`: Temperature for points on the interior
        surface.

        """
        if init_points_temperature is not None:
            self._initial_point_temperature = jnp.array(
                init_points_temperature
            ).reshape(len(init_points_temperature), 1)
            return

        temperatures_kelvin = {
            "wall": convert_temperature(wall_temperature_celsius, "Celsius", "Kelvin"),
            "exterior_wall": convert_temperature(
                exterior_surface_temperature_celsius, "Celsius", "Kelvin"
            ),
            "interior_wall": convert_temperature(
                interior_surface_temperature_celsius, "Celsius", "Kelvin"
            ),
        }

        initial_temperature = np.full(
            (len(self._mesh.points), 1), temperatures_kelvin["wall"]
        )

        masks = {
            "exterior_wall": jnp.array(
                [self._location_fns[0](point) for point in self._mesh.points]
            ),
            "interior_wall": jnp.array(
                [self._location_fns[1](point) for point in self._mesh.points]
            ),
        }

        initial_temperature[masks["exterior_wall"]] = temperatures_kelvin[
            "exterior_wall"
        ]
        initial_temperature[masks["interior_wall"]] = temperatures_kelvin[
            "interior_wall"
        ]
        self._initial_point_temperature = jnp.array(initial_temperature)

    def _center_and_straighten_mesh(self) -> None:
        """
        Process the mesh file located in `self._mesh_filepath` by performing a series of
        operations, and save the fully processed mesh. The processing involve:
        1. Centering the mesh at origin.
        2. Straightening the mesh with the rotation in the JSON file located at
        `self._rotation_filepath`.
        3. Saving the processed mesh at `output_mesh_filepath`.

        :raises RuntimeError: if the mesh file is not found.
        """
        # Center the points of the mesh
        centroid = np.mean(self._meshio_mesh.points, axis=0)
        self._meshio_mesh.points -= centroid

        # Rotate to straighten the mesh
        with open(self._rotation_filepath, "r") as jsonfile:
            rotation_data = json.load(jsonfile)
        theta_xyz: list[float] = [
            rotation_data[name] for name in ["theta_x", "theta_y", "theta_z"]
        ]
        R = Rotation.from_euler(seq="xyz", angles=theta_xyz, degrees=True).as_matrix()
        self._meshio_mesh.points @= R.T

    def _check_input_dir(self) -> None:
        """
        The necessary input for the simulation are:
        - The hexahedral-cell mesh located in `self._mesh_filepath`
        - A precomputed rotation xyz in degrees to straighten the mesh located in
        `self._rotation_filepath`

        :raise: ValueError if any of the input are missing.
        """
        if not (self._mesh_filepath.exists() and self._rotation_filepath.exists()):
            raise ValueError(
                f"Input directory uncomplete or incorect. The directory should contain "
                f"the following files: {self._mesh_filepath.name},"
                f"{self._rotation_filepath.name}, "
            )
