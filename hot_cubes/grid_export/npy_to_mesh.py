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


def convert_to_hex8_mesh(
    ckpt_path: Path,
    density_threshold: float = 22.0,
) -> None:
    """
    Exports a svox2.SparseGrid saved as `npz_file_path` to a hex8 mesh .msh file.
    If not specified, the `density_threshold` is automatically set to the 90th
    percentile of the density values.
    Output file is saved in the same folder as the input file.
    """

    grid = np.load(Path(ckpt_path, "links.npy"))
    mask = grid >= 0
    idxs = grid[mask]
    density_data = np.load(Path(ckpt_path, "density_data.npy"))
    density = density_data[idxs][:, 0].astype(float)
    grid[mask] = density

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
                    cell.append(point_indices[point])

                # Add the hexahedron element (8 indices)
                cells.append(cell)

    # Export mesh using meshio
    output_path = Path(ckpt_path, "hex8_mesh.vtk")
    meshio_points = np.array(points, dtype=float)
    meshio_cells = [("hexahedron", np.array(cells))]
    meshio.write(output_path, meshio.Mesh(points=meshio_points, cells=meshio_cells))

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


if __name__ == "__main__":
    tyro.cli(convert_to_hex8_mesh)
