import sys
import unittest

import numpy as np

from hot_cubes.datasets.thermo_scene_dataset import ThermoSceneDataset

sys.path.append("./plenoxels")  # Avoid having to install plenoxel on GPU-less machines
from plenoxels.opt.util.dataset import datasets  # noqa: E402


class Test_Thermoscene_Dataset(unittest.TestCase):
    def test_thermoscene_dataset(self, data_dir: str = "tests/mock_dataset") -> None:

        dset = datasets["auto"](data_dir, split="train", device="0", epoch_size=1)
        self.assertIsInstance(
            dset, ThermoSceneDataset, "dset is not of type ThermoSceneDataset"
        )

        fx, fy, cx, cy = dset.get_intrinsic_parameters()
        intrinsic_matrix = np.array(
            [
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
            ]
        )
        ground_truth = np.array(
            [
                [358.496168, 0, 239.611546, 0],
                [0, 627.590062, 320.913268, 0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
            ]
        )

        np.testing.assert_allclose(intrinsic_matrix, ground_truth, rtol=1e-5)

    def test_temperature_metadata(self, data_dir: str = "tests/mock_dataset") -> None:

        dset = datasets["ThermoScene"](
            data_dir, split="train", device="0", epoch_size=1
        )

        self.assertEqual(
            dset.t_max, 15.55841227645726, "Wrong absolute max" "temperature"
        )
        self.assertEqual(
            dset.t_min, -15.68152641521192, "Wrong absolute min " "temperature"
        )
