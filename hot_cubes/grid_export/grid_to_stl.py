import argparse
import logging
from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt


def convert_to_stl(
    npz_file_path: Path,
    percentile_threshold: float = 90,
    density_threshold: float | None = None,
    put_colors: bool = False,
) -> None:
    """
    Exports a svox2.SparseGrid saved as an npz file to a mesh stl file.
    If not specified, the density threshold is automatically set to the 90th percentile
    of the density values.
    Output file is saved in the same folder as the input file.
    """
    data = np.load(npz_file_path)

    # Replace the grid with the density values
    grid = data["links"]
    mask = grid >= 0
    idxs = grid[mask]
    density = data["density_data"][idxs][:, 0]
    grid[mask] = density

    # MinMax density values for information
    logging.info(f"min density: {data['density_data'].min()}")
    logging.info(f"max density: {data['density_data'].max()}")

    # Apply the density_threshold to the grid
    if density_threshold is None:
        density_threshold = np.percentile(density, percentile_threshold)
    logging.info(f"treshold density value: {density_threshold}")

    thresholded_grid = grid >= density_threshold

    logging.info(
        f"Percentage of cells kept: "
        f"{100*thresholded_grid.sum() / thresholded_grid.size:.4f} %"
    )

    # Marching cubes algorithm from trimesh:
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(thresholded_grid)

    # Assign gray colors to the mesh
    mesh.visual.face_colors = [217, 217, 214]

    if put_colors:
        # Use the temperature values to color the mesh
        temp_vector = data["temperature_data"][idxs][:, 0]
        temperature = plt.cm.plasma(temp_vector)
        colors = np.zeros(grid.shape + (3,))
        colors[mask] = temperature[:, :3]

        # Assign colors to each face based on the voxel indices
        face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
        voxel_indices = np.round(face_centroids).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, grid.shape[0] - 1)
        face_colors = colors[
            voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
        ]
        mesh.visual.face_colors = face_colors

    # Export the mesh to a ply file
    output_file = str(npz_file_path).split(".")[0] + "_output_mesh.ply"
    mesh.export(output_file)

    logging.info(f"Mesh successfully exported to {output_file}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_file",
        metavar="path",
        required=True,
        help="path of ckpt.npz",
    )

    parser.add_argument(
        "--percentile_threshold",
        metavar=float,
        default=90,
        required=False,
        help="percentile threshold ",
    )

    parser.add_argument(
        "--colors",
        metavar=bool,
        default=False,
        required=False,
        help="put colors in the mesh",
    )

    args = parser.parse_args()

    convert_to_stl(
        npz_file_path=Path(args.npz_file),
        percentile_threshold=float(args.percentile_threshold),
        put_colors=args.colors,
    )
