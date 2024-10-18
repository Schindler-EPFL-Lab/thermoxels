import numpy as np
import meshio
import logging
from pathlib import Path
import argparse

def convert_to_hex8_mesh(
    npz_file_path: Path,
    percentile_threshold: float = 90,
    density_threshold: float | None = None,
    put_colors: bool = False,
    perform_filtering: bool = True,
) -> None:
    """
    Exports a svox2.SparseGrid saved as an npz file to a hex8 mesh .msh file.
    If not specified, the density threshold is automatically set to the 90th percentile
    of the density values.
    Output file is saved in the same folder as the input file.
    """

    grid = np.load("ckpt/links.npy")
    mask = grid >= 0
    idxs = grid[mask]
    density_data = np.load("ckpt/density_data.npy")
    density = density_data[idxs][:, 0].astype(float)
    grid[mask] = density

    # MinMax density values for information
    logging.info(f"min density: {density_data.min()}")
    logging.info(f"max density: {density_data.max()}")

    # Apply the density_threshold to the grid
    density_threshold = 22.0
    if density_threshold is None:
        density_threshold = np.percentile(density_data, percentile_threshold)
    logging.info(f"threshold density value: {density_threshold}")

    thresholded_grid = grid >= density_threshold

    logging.info(
        f"Percentage of cells kept: "
        f"{100 * thresholded_grid.sum() / thresholded_grid.size:.4f} %"
    )

    # Create a list of hexahedral elements (8 vertices per voxel)
    elements = []
    points = []
    point_index = {}

    # Generate points and elements for the hexahedral mesh
    nx, ny, nz = thresholded_grid.shape

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                if thresholded_grid[i, j, k]: # Binary filling
                    # Define the 8 corners of the voxel (hexahedron)
                    voxel_points = [
                        (i, j, k),
                        (i + 1, j, k),
                        (i + 1, j + 1, k),
                        (i, j + 1, k),
                        (i, j, k + 1),
                        (i + 1, j, k + 1),
                        (i + 1, j + 1, k + 1),
                        (i, j + 1, k + 1),
                    ]

                    # Add the points if they haven't been added already
                    element_indices = []
                    for point in voxel_points:
                        if point not in point_index:
                            point_index[point] = len(points)
                            points.append(point)
                        element_indices.append(point_index[point])

                    # Add the hexahedron element (8 indices)
                    elements.append(element_indices)


    points = np.array(points, dtype=float)

    # Export mesh using meshio
    cells = [("hexahedron", np.array(elements))]
    output_file = str(npz_file_path).split(".")[0] + "_output_mesh.msh"
    meshio.write(output_file, meshio.Mesh(points=points, cells=cells))

    logging.info(f"Hex8 mesh successfully exported to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy file to .msh HEX8 format")
    parser.add_argument("--npy-file", type=str, required=True, help="Path to the .npy file")
    parser.add_argument("--msh-file", type=str, required=True, help="Path to the output .msh file")

    args = parser.parse_args()

    convert_to_hex8_mesh(args.npy_file, args.msh_file)
