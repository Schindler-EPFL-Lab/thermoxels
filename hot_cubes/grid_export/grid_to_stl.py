import argparse
import logging
from pathlib import Path

import numpy as np
import trimesh


def convert_to_stl(
    npz_file_path: Path,
    percentile_threshold: float = 90,
    density_threshold: float | None = None,
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
    output_file = str(npz_file_path).split(".")[0] + "_output_mesh.stl"
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

    args = parser.parse_args()

    convert_to_stl(
        npz_file_path=Path(args.npz_file),
        percentile_threshold=float(args.percentile_threshold),
    )
