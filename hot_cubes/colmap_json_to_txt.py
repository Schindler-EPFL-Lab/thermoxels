import json
import numpy as np
import argparse
import pathlib
import logging


def find_json_structure(dataset_path: pathlib.Path) -> pathlib.Path:
    """
    This function checks the name of the json file used to define dataset's poses
    :returns: path of the json file
    """

    matching_files = list(dataset_path.rglob("transforms*.json"))

    if len(matching_files) == 0:
        raise RuntimeError(
            f"Neither 'transforms.json' nor 'transforms_thermal.json' \
            could be found in {dataset_path}."
        )
    if len(matching_files) > 1:
        raise RuntimeError("Error: More than one 'transforms*.json' file found.")

    logging.info(f"Found unique JSON file: {matching_files[0].name}")
    return matching_files[0]


def convert_colmap_json_to_txt(
    dataset_path: pathlib.Path, save_to: pathlib.Path | None = None
) -> None:
    """
    This function is used to convert a dataset from rebel nerf having a transforms.json
    file to a dataset with a set of poses/image_name.txt used by plenoxel
    Output files are saved in the folder dataset_path/poses/image_name.txt
    """

    if save_to is None:
        save_to = dataset_path

    with open(find_json_structure(dataset_path)) as file:
        data = json.load(file)

    fl_x = data["fl_x"]
    fl_y = data["fl_y"]
    cx = data["cx"]
    cy = data["cy"]

    intrinsic_matrix = np.array(
        [[fl_x, 0, cx, 0], [0, fl_y, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    np.savetxt(dataset_path / "intrinsics.txt", intrinsic_matrix)

    # OpenGL -> OpenCV
    cam_trans = np.diag(np.array([1.0, -1.0, -1.0, 1.0]))

    world_trans = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    (dataset_path / "pose").mkdir(exist_ok=True)

    for frame in data["frames"]:

        text_file_name = pathlib.Path(frame["file_path"]).stem + ".txt"

        c2w = np.array(frame["transform_matrix"])
        c2w = world_trans @ c2w @ cam_trans  # To OpenCV

        full_poses_path = dataset_path / "pose" / text_file_name

        # Save 4x4 OpenCV C2W pose
        np.savetxt(full_poses_path, c2w)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        metavar="path",
        required=True,
        help="the path to the folder of images with transforms.json",
    )

    args = parser.parse_args()
    convert_colmap_json_to_txt(pathlib.Path(args.folder_path))
