import unittest
import numpy as np
from pathlib import Path

from hot_cubes.colmap_json_to_txt import convert_colmap_json_to_txt


class TestClomapJsonTotext(unittest.TestCase):
    def test_intrinsic(self) -> None:

        dataset_path = Path("tests/colmap_to_json_test/mock_dataset/")

        assert dataset_path.exists()

        convert_colmap_json_to_txt(dataset_path)

        intrinsic_path = Path("tests/colmap_to_json_test/mock_dataset/intrinsics.txt")

        assert intrinsic_path.exists()

        intrinsic_matrix = np.loadtxt(intrinsic_path)

        ground_truth = np.array(
            [
                [710.7537835874724, 0, 270.68532378655243, 0],
                [0, 717.1744481358219, 291.8659007588129, 0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
            ]
        )

        np.testing.assert_allclose(intrinsic_matrix, ground_truth, rtol=1e-5)

    def test_poses(self) -> None:

        dataset_path = Path("tests/colmap_to_json_test/mock_dataset/")

        assert dataset_path.exists()

        convert_colmap_json_to_txt(dataset_path)

        pose_path = Path(
            "tests/colmap_to_json_test/mock_dataset/pose/frame_eval_00001.txt"
        )

        assert pose_path.exists()

        pose_matrix = np.loadtxt(pose_path)

        ground_truth = np.array(
            [
                [-0.99420619, -0.02986032, 0.10325898, 2.39980927],
                [0.10532561, -0.07882963, 0.99130843, -2.00971301],
                [-0.02146092, 0.99644079, 0.08151796, -1.39536877],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(pose_matrix, ground_truth, rtol=1e-5)
