import unittest
from hot_cubes.renderer_evaluator.model_evaluator import Evaluator
from skimage import data
import imageio.v2 as imageio
import torch
import numpy as np


class TestMetricsComparator(unittest.TestCase):

    def test_ssim_comparator(self):
        # Images are taken from skimage library
        original_image = data.astronaut()

        noise = (
            np.ones_like(original_image)
            * 0.2
            * (original_image.max() - original_image.min())
        )
        rng = np.random.default_rng()
        noise[rng.random(size=noise.shape) > 0.5] *= -1

        distorted_image_1 = np.clip(original_image + noise, 0, 255).astype(np.uint8)
        distorted_image_2 = np.clip(original_image + abs(noise), 0, 255).astype(
            np.uint8
        )

        _, _, ssim1 = Evaluator.compute_mse_psnr_ssim(
            torch.tensor(original_image / 255, dtype=torch.double),
            torch.tensor(original_image / 255 + 1e-12, dtype=torch.double),
        )

        self.assertAlmostEqual(ssim1, 1.0)

        _, _, ssim2 = Evaluator.compute_mse_psnr_ssim(
            torch.tensor(original_image / 255), torch.tensor(distorted_image_1 / 255)
        )

        self.assertLess(ssim2, 1.0)

        _, _, ssim3 = Evaluator.compute_mse_psnr_ssim(
            torch.tensor(original_image / 255), torch.tensor(distorted_image_2 / 255)
        )

        self.assertGreater(ssim3, ssim2)

    def test_psnr_comparator(self):
        # Images: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
        original = imageio.imread("tests/test_render/psnr/original_image.png") / 255
        original_tensor = torch.tensor(original, dtype=torch.float32)
        compressed = imageio.imread("tests/test_render/psnr/compressed_image.png") / 255
        compressed_tensor = torch.tensor(compressed, dtype=torch.float32)

        _, psnr1, _ = Evaluator.compute_mse_psnr_ssim(
            original_tensor, compressed_tensor
        )
        self.assertAlmostEqual(psnr1, 43.8629, delta=0.1)

        _, psnr2, _ = Evaluator.compute_mse_psnr_ssim(
            original_tensor, original_tensor + 1e-5
        )
        self.assertGreater(psnr2, psnr1)
