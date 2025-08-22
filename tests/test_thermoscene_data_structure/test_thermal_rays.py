import unittest
import torch
from thermoxels.datasets.thermo_scene_dataset import ThermoSceneDataset
from thermoxels.datasets.thermal_util import ThermalRays
from pathlib import Path


class TestThermalRays(unittest.TestCase):
    def test_thermal_rays_train(self) -> None:

        dset_path = Path("tests/mock_dataset/")
        dset = ThermoSceneDataset(dset_path, split="train", device="0", epoch_size=1)

        rays = dset.rays
        self.assertIsInstance(rays, ThermalRays)
        self.assertIsInstance(rays.gt_thermal, torch.Tensor)
        self.assertIsInstance(rays.gt, torch.Tensor)
        self.assertEqual(rays.gt_thermal.shape[0], 595 * 480)  # Size of one image from
        # the mock dataset
        self.assertEqual(rays.gt.shape[0], 595 * 480)  # Size of one image from
        # the mock dataset

    def test_thermal_rays_test(self) -> None:

        dset_path = Path("tests/mock_dataset/")
        dset = ThermoSceneDataset(dset_path, split="test", device="0", epoch_size=1)

        dset.gen_rays()
        rays = dset.rays
        self.assertIsInstance(rays, ThermalRays)
        self.assertIsInstance(rays.gt_thermal, torch.Tensor)
        self.assertIsInstance(rays.gt, torch.Tensor)
        self.assertEqual(rays.gt_thermal.shape[0], 1)  # One image in the
        # mock dataset
        self.assertEqual(rays.gt.shape[0], 1)  # One image in the mock dataset
