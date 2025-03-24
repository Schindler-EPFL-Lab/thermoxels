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

from hot_cubes.svox2_temperature import SparseGrid


def convert_to_hex8_mesh(
    ckpt_path: Path,
    model_name_prefix="",
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

    sparsegrid = SparseGrid.load(ckpt_path, normalize=False)

    grid = sparsegrid.links.numpy().astype(float)
    mask = grid >= 0
    idxs = grid[mask]
    density_data = sparsegrid.density_data[idxs][:, 0].detach().numpy().astype(float)
    grid[mask] = density_data

    temperatures = (
        sparsegrid.temperature_data[idxs][:, 0].detach().numpy().astype(float)
    )
    temperature_grid = sparsegrid.links.numpy().astype(float)
    temperature_grid[mask] = temperatures

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

                        temperature = temperature_grid[point[0], point[1], point[2]]
                        point_temperatures.append(temperature)

                    cell.append(point_indices[point])

                # Add the hexahedron element (8 indices)
                cells.append(cell)

    # Export mesh using meshio
    mesh_name = (
        model_name_prefix + ckpt_path.stem + "_" + str(density_threshold) + ".vtu"
    )
    output_path = Path(ckpt_path, mesh_name)

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
