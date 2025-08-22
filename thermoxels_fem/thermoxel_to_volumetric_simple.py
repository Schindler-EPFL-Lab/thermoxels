import json
from pathlib import Path

import meshio
import numpy as np
import pyvista


def thermoxel_to_volumetric_density(
    ckpt_npz_filepath: Path,
    output_dir: Path,
    theta_xyz: list[float],
    density_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a Thermoxels SparseGrid checkpoint (.npz) located at `ckpt_npz_filepath`,
    and save in `output_dir` the necessary input needed for the simulation:
    - The hexahedral(hex8)-cell mesh extracted according to the value of
    `density_threshold`, saved in a VTK file named mesh.vtk
    - A precomputed rotation `theta_xyz` in degrees to straighten the mesh saved in a
    JSON file named rotation.json
    """
    data = np.load(ckpt_npz_filepath)
    # The links of the model describing the grid
    grid = data["links"]
    # The density data at each point of the model
    density_data = data["density_data"]
    # The temperature data [K] for each point of the model
    temperature_data = data["temperature_data"]

    _generate_and_save_volumetric_mesh(
        grid=grid,
        density_data=density_data,
        temperature_data=temperature_data,
        density_threshold=density_threshold,
        output_dir=output_dir,
    )
    _generate_and_save_rotation(theta_xyz=theta_xyz, output_dir=output_dir)
    return grid, density_data


def _generate_and_save_volumetric_mesh(
    grid: np.ndarray,
    density_data: np.ndarray,
    temperature_data: np.ndarray,
    density_threshold: float,
    output_dir: Path,
) -> None:
    """
    Generate and save as mesh.vkt in `output_dir` a hex8 mesh from a Thermoxels
    sparse grid models stored in:
    - `grid`: The links of the model describing the grid
    - `density_data`: The density data at each point of the model
    - `temperature_data`: The temperature data [K] for each point of the model

    The mesh is extracted from the grid according to the value of the
    `density_threshold`.

    :raises RuntimeError: If the mesh file cannot be read by Pyvista.
    """
    density_grid = np.copy(grid)
    temperature_grid = np.copy(grid).astype(float)
    mask = grid >= 0
    indices = grid[mask]

    density = density_data[indices][:, 0].astype(float)
    density_grid[mask] = density

    temperature = temperature_data[indices][:, 0].astype(float)
    temperature_grid[mask] = temperature

    thresholded_grid = density_grid >= density_threshold

    # Lists to define a hexahedral mesh: hex8 cells, points, and temperature data
    cells: list[list[int]] = []
    points: list[tuple[float, float, float]] = []
    point_indices: dict[tuple[float, float, float], int] = {}
    point_temperatures: list[float] = []

    # Generate points and elements for the hexahedral mesh
    nx, ny, nz = thresholded_grid.shape

    for x in range(nx - 1):
        for y in range(ny - 1):
            for z in range(nz - 1):
                if not thresholded_grid[x, y, z]:
                    continue
                # Binary filling
                # Define the 8 corners of the voxel (hexahedron)
                voxel_points = [
                    (x, y, z),
                    (x + 1, y, z),
                    (x + 1, y + 1, z),
                    (x, y + 1, z),
                    (x, y, z + 1),
                    (x + 1, y, z + 1),
                    (x + 1, y + 1, z + 1),
                    (x, y + 1, z + 1),
                ]

                # Add the points if they haven't been added already
                cell: list[int] = []
                for point in voxel_points:
                    if point not in point_indices:
                        point_indices[point] = len(points)
                        points.append(point)
                        point_temperatures.append(
                            temperature_grid[point[0], point[1], point[2]]
                        )
                    cell.append(point_indices[point])
                # Add the hexahedron element (8 indices)
                cells.append(cell)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir, "mesh.vtk")

    # Create the meshio Mesh with temperature data for each point
    meshio_points = np.array(points, dtype=float)
    meshio_cells = [("hexahedron", np.array(cells))]
    meshio_mesh = meshio.Mesh(
        points=meshio_points,
        cells=meshio_cells,
        point_data={"point_temperature": point_temperatures},
    )
    meshio.write(output_path, meshio_mesh)

    # Keep only the biggest connected component of the hex8 mesh
    pv_mesh = pyvista.read(output_path)
    if pv_mesh is None:
        raise RuntimeError("mesh file could not be read by Pyvista")
    pv_mesh.connectivity(extraction_mode="largest", inplace=True)
    pyvista.save_meshio(filename=output_path, mesh=pv_mesh)


def _generate_and_save_rotation(theta_xyz: list[float], output_dir: Path) -> None:
    data = {
        "theta_x": theta_xyz[0],
        "theta_y": theta_xyz[1],
        "theta_z": theta_xyz[2],
    }
    with open(Path(output_dir, "rotation.json"), "w+") as json_file:
        json.dump(data, json_file)
