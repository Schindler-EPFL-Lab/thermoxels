import unittest
from plenoxels.opt.util.dataset import datasets
from hot_cubes.utils.thermo_scene_dataset import ThermoSceneDataset
import numpy as np


class Test_Thermoscene_Dataset(unittest.TestCase):
    def test_thermoscene_dataset(
        self, data_dir: str = "tests/test_thermoscene_data_structure/mock_dataset"
    ) -> None:

        dset = datasets["auto"](data_dir, split="train", device="0", epoch_size=1)
        assert isinstance(dset, ThermoSceneDataset), (
            "dset is not of type ThermoSceneDataset"
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
