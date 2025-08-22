import logging

import open3d as o3d
import tyro


def load_and_visualize_obj(file_path: str) -> None:
    mesh = o3d.io.read_triangle_mesh(file_path)

    # Check if the mesh is loaded properly
    if not mesh.is_empty():
        logging.info("Mesh loaded successfully!")
        # Compute normals if not present
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Create a visualizer window
        o3d.visualization.draw_geometries([mesh], window_name="3D Object Viewer")
    else:
        logging.info("Failed to load the mesh. Please check the file path.")


def main(
    obj_file_path: str,
) -> None:
    """
    This functions opens a .obj file and visualizes it in a 3D viewer window.

    :param obj_file_path: path of the .obj file as str
    """
    load_and_visualize_obj(obj_file_path)


if __name__ == "__main__":
    tyro.cli(main)
