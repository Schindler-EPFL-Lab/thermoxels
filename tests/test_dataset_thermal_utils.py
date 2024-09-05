import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from PIL import Image as PILImage

from hot_cubes.datasets.datasets_utils.rescale_thermal_images import (
    scale,
    scale_image,
    scale_test_to_train,
    unscale_image,
)


class TestDatasetThermalUtils(unittest.TestCase):
    def test_scale_image_down(self) -> None:
        image = np.array([2, 2, 1, -1, 2])
        scaled_img = scale(
            img_array=image, min_origin=0, max_origin=1, min_target=0, max_target=1
        )
        assert_almost_equal(actual=scaled_img, desired=np.array([2, 2, 1, -1, 2]))
        scaled_img = scale(
            img_array=image, min_origin=-1, max_origin=2, min_target=-2, max_target=4
        )
        assert_almost_equal(actual=scaled_img, desired=np.array([4, 4, 2, -2, 4]))

        image = np.array([2, 2, 1, -1, 2])
        scaled_img = scale(
            img_array=image, min_origin=-1, max_origin=2, min_target=0, max_target=1
        )
        assert_almost_equal(
            actual=scaled_img, desired=np.array([1, 1, 0.666, 0, 1]), decimal=3
        )

    def test_unscale_uniform(self) -> None:
        image = np.array([2, 2, 1, -1, 2])
        scaled_img = unscale_image(img_array=image, t_max=2, t_min=-1)
        assert_almost_equal(
            actual=scaled_img, desired=np.array([1, 1, 0.6666, 0, 1]), decimal=3
        )

    def test_scale_uniform(self) -> None:
        image = np.array([1, 1, 0.666, 0, 1])
        scaled_img = scale_image(img_array=image, t_max=2, t_min=-1)
        assert_almost_equal(
            actual=scaled_img, desired=np.array([2, 2, 1, -1, 2]), decimal=1
        )

    def test_conversion(self) -> None:
        image = np.array([2, 2, 1, -1, 2])
        scaled = scale_image(img_array=image, t_min=-1, t_max=2)
        unscaled = unscale_image(img_array=scaled, t_min=-1, t_max=2)
        assert_almost_equal(actual=unscaled, desired=image)

    def test_scale_test_to_train_change_min(self) -> None:
        """test that a temperature lower than the new lower bound is 0"""

        test_array = np.array([[255, 128, 0], [128, 255, 0]])
        test_img = PILImage.fromarray((test_array).astype(np.uint8))
        scaled = scale_test_to_train(
            test_img, t_max=30, t_min=20, t_max_new=30, t_min_new=25
        )

        desired = np.array([[255, 0, 0], [0, 255, 0]], dtype=int)

        assert_almost_equal(
            actual=np.array(scaled),
            desired=desired,
        )

    def test_scale_test_to_train_change_max(self) -> None:
        """test that a temperature higher than the new higher bound is 255"""

        test_array = np.array([[255, 128, 0], [128, 255, 0]])
        test_img = PILImage.fromarray((test_array).astype(np.uint8))
        scaled = scale_test_to_train(
            test_img, t_max=30, t_min=20, t_max_new=25, t_min_new=20
        )

        desired = np.array([[255, 255, 0], [255, 255, 0]], dtype=int)

        assert_almost_equal(
            actual=np.array(scaled),
            desired=desired,
        )

    def test_scale_test_to_train_within_bound(self) -> None:
        """test that a temperature higher than the new higher bound is 255"""

        test_array = np.array([[255, 128, 0], [128, 255, 0]])
        test_img = PILImage.fromarray((test_array).astype(np.uint8))
        scaled = scale_test_to_train(
            test_img, t_max=30, t_min=20, t_max_new=35, t_min_new=15
        )

        desired = np.array([[191, 127, 63], [127, 191, 63]], dtype=int)

        assert_almost_equal(
            actual=np.array(scaled),
            desired=desired,
        )

    def test_scale_test_to_train_no_change(self) -> None:

        test_array = np.array([[255, 128, 0], [128, 255, 0]])
        test_img = PILImage.fromarray((test_array).astype(np.uint8))
        scaled = scale_test_to_train(
            test_img, t_max=30, t_min=20, t_max_new=30, t_min_new=20
        )

        assert_almost_equal(
            actual=np.array(scaled),
            desired=test_array,
        )
