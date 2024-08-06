import logging
import os
from pathlib import Path

import imageio
import numpy as np
import open3d as o3d
import tyro


def read_and_prepare_mesh(file_path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()  # Necessary for better visualization aesthetics
    if mesh.has_vertex_colors():
        logging.info("The mesh has vertex colors")
    else:
        logging.info("The mesh has no vertex colors")
    return mesh


def setup_visualizer() -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mesh Animation", width=800, height=600)
    return vis


def rotate_mesh(
    vis: o3d.visualization.VisualizerWithKeyCallback,
    mesh: o3d.geometry.TriangleMesh,
    angle_x: float = 150.0,
    angle_y: float = 0.0,
    angle_z: float = 0.0,
) -> None:
    R = mesh.get_rotation_matrix_from_xyz(
        (angle_x * np.pi / 180, angle_y * np.pi / 180, angle_z * np.pi / 180)
    )  # Rotate around X-axis
    mesh.rotate(R, center=mesh.get_center())
    vis.add_geometry(mesh)


def update_rotation(
    vis: o3d.visualization.VisualizerWithKeyCallback, angle: float = 10.0, axis: int = 0
) -> None:
    ctr = vis.get_view_control()
    ctr.rotate(-10, 0)  # Rotate by 10 degrees about the vertical screen axis


def capture_frames(
    vis: o3d.visualization.VisualizerWithKeyCallback, output_dir, total_frames=360
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i in range(total_frames):
        update_rotation(vis)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{output_dir}/frame_{i:04d}.png")


def create_gif(output_dir: Path, gif_path: Path) -> None:
    image_files = sorted(Path(output_dir).glob("*.png"))
    with imageio.get_writer(gif_path, mode="I", duration=0.1) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    logging.info("GIF created successfully.")


def main(
    obj_file_path: str,
    total_frames: int = 36,
    init_x_angle: float = 0.0,
    init_y_angle: float = 0.0,
    init_z_angle: float = 0.0,
) -> None:
    """
    This functions opens a .obj file, make the object spin around the X-axis,
    captures frames of the spinning object to create a set of frames and a gif saved
    at the same location.

    :param obj_file_path: path of the .obj file as str
    :param total_frames: total number of frames to generate
    :param init_x_angle: initial rotation angle in degrees for x axis
    :param init_y_angle: initial rotation angle in degrees for y axis
    :param init_z_angle: initial rotation angle in degrees for z axis

    """
    obj_file = Path(obj_file_path)
    base_dir = obj_file.parent
    output_frames_dir = base_dir / "frames"
    gif_path = base_dir / "animation.gif"

    # Process mesh
    mesh = read_and_prepare_mesh(obj_file_path)
    vis = setup_visualizer()
    rotate_mesh(vis, mesh, init_x_angle, init_y_angle, init_z_angle)

    # Capture frames and create GIF
    capture_frames(vis, output_frames_dir, total_frames)
    vis.destroy_window()
    create_gif(output_frames_dir, gif_path)


if __name__ == "__main__":
    tyro.cli(main)
