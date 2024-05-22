import unittest
from pathlib import Path
import imageio.v2 as imageio
import torch
from hot_cubes.renderer_evaluator.thermal_evaluation_metrics import uniform_filter_fn
from hot_cubes.renderer_evaluator.thermal_evaluation_metrics import compute_hssim
import numpy as np


class TestHssim(unittest.TestCase):
    def test_uniform_filter_fn(self) -> None:
        im = torch.ones((8, 8, 1))
        window = torch.ones((4, 4, 1)) / 16

        out = uniform_filter_fn(im, window)

        np.testing.assert_allclose(out.numpy(), torch.ones(1, 1, 2, 2).numpy())

    def test_hssim_values(
        self, img_dir: Path = Path("tests/test_metrics/img.png")
    ) -> None:
        x = torch.eye(3, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.float32
        ).unsqueeze(-1)

        hssim = compute_hssim(x, y, kernel_size=3, stride=1, C2=9e-4)
        self.assertAlmostEqual(hssim, 0.75056, delta=0.0001)

    def test_hssim_equal(
        self, img_dir: Path = Path("tests/test_metrics/img.png")
    ) -> None:

        original_image = imageio.imread(img_dir)
        original_image_tensor = torch.from_numpy(original_image).float().unsqueeze(-1)
        hssim_00 = compute_hssim(original_image_tensor, original_image_tensor)

        self.assertAlmostEqual(hssim_00, 1.0)

    def test_hssim_comparison(
        self, img_dir: Path = Path("tests/test_metrics/img.png")
    ) -> None:

        original_image = imageio.imread(img_dir)

        noise = (
            np.ones_like(original_image)
            * 0.2
            * (original_image.max() - original_image.min())
        )
        rng = np.random.default_rng()
        noise[rng.random(size=noise.shape) > 0.5] *= -1

        distorted_image_1 = np.clip(original_image + noise, 0, 255) / 255
        distorted_image_2 = np.clip(original_image + 5 * noise, 0, 255) / 255

        original_image_tensor = torch.from_numpy(original_image).float().unsqueeze(-1)

        distorted_image_1_tensor = (
            torch.from_numpy(distorted_image_1).float().unsqueeze(-1)
        )
        distorted_image_2_tensor = (
            torch.from_numpy(distorted_image_2).float().unsqueeze(-1)
        )
        hssim_01 = compute_hssim(original_image_tensor, distorted_image_1_tensor)
        hssim_02 = compute_hssim(original_image_tensor, distorted_image_2_tensor)

        self.assertGreater(hssim_01, hssim_02)
