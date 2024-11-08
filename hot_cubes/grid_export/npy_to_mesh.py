"""
Converts the thermoxels checkpoints to a hexahedron mesh.
"""

import logging
from pathlib import Path

import meshio
import numpy as np
import pyvista
import pyvista.examples
import tyro
from scipy.constants import convert_temperature


def convert_to_hex8_mesh(
    ckpt_path: Path,
    min_temperature_bound: float,
    max_temperature_bound: float,
    density_threshold: float = 22.0,
) -> None:
    """
    Exports a svox2.SparseGrid saved as `npz_file_path` to a hex8 mesh .vtk file.
    If not specified, the `density_threshold` is automatically set to the 90th
    percentile of the density values.
    To extract the temperature data to the mesh, please specify the
    `temperature_bounds_filepath`.
    Output file is saved in the same folder as the input file.
    """

    grid = np.load(Path(ckpt_path, "links.npy"))
    mask = grid >= 0
    idxs = grid[mask]
    density_data = np.load(Path(ckpt_path, "density_data.npy"))
    density = density_data[idxs][:, 0].astype(float)
    grid[mask] = density

    temperature_data = np.load(Path(ckpt_path, "temperature_data.npy"))
    temperature_in_celsius = temperature_data[idxs][:, 0].astype(float)
    temperature_grid = np.load(Path(ckpt_path, "links.npy")).astype(float)
    temperature_grid[mask] = temperature_in_celsius

    # MinMax density values for information
    logging.info(f"min density: {density_data.min()}")
    logging.info(f"max density: {density_data.max()}")

    thresholded_grid = grid >= density_threshold

    logging.info(
        f"Percentage of cells kept: "
        f"{100 * thresholded_grid.sum() / thresholded_grid.size:.4f} %"
    )

    # Create a list of hexahedral elements (8 vertices per voxel)
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

                        raw_temperature = temperature_grid[point[0], point[1], point[2]]
                        temperature_in_celsius = convert_value_from_to_interval(
                            np.clip(raw_temperature, 0, 1),
                            from_start=0.0,
                            from_end=1.0,
                            to_start=min_temperature_bound,
                            to_end=max_temperature_bound,
                        )
                        temperature_in_kelvin = convert_temperature(
                            temperature_in_celsius, "Celsius", "Kelvin"
                        )
                        point_temperatures.append(temperature_in_kelvin)

                    cell.append(point_indices[point])

                # Add the hexahedron element (8 indices)
                cells.append(cell)

    # Export mesh using meshio
    output_path = Path(ckpt_path, "hex8_temperature_mesh.vtk")

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
    pyvista.save_meshio(
        filename=output_path,
        mesh=pv_mesh,
    )

    logging.info(f"Hex8 mesh successfully exported to {output_path}")


def convert_value_from_to_interval(
    value: float, from_start: float, from_end: float, to_start: float, to_end: float
) -> float:
    """
    Takes `value` in the interval `interval_from` and convert it to the corresponding
    value in `interval_to`.

    :returns: the converted value
    :raises: Value Error if intervals do not hold two and only two values, or if
    interval startpoint bigger than endpoint.
    """
    if from_end < from_start or to_end < to_start:
        raise ValueError(
            "Interval startpoint must be less than or equal to the endpoint."
        )
    return to_start + (to_end - to_start) * (value - from_start) / (
        from_end - from_start
    )


if __name__ == "__main__":
    tyro.cli(convert_to_hex8_mesh)
