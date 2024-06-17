import json
import logging
import os
from os import path
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from hot_cubes.datasets.thermal_util import ThermalRays
from plenoxels.opt.util.dataset_base import DatasetBase
from plenoxels.opt.util.util import Intrin, similarity_from_cameras


class ThermoSceneDataset(DatasetBase):
    """
    ThermoSceneDataset dataset loader with both rgb and thermal images as features
    """

    focal: float
    c2w_thermal: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    gt_thermal: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: ThermalRays | None
    rays_init: ThermalRays | None

    def __init__(
        self,
        root,
        split,
        epoch_size: int,
        device: str = "cpu",
        scene_scale: float = 1.0,  # Scene scaling
        factor: int = 1,  # Image scaling
        # (on ray gen; use gen_rays(factor) to dynamically change scale)
        scale: float = 1.0,  # Image scaling (on load)
        # TODO : removes when switching to new optimizer script
        permutation: bool = True,
        white_bkgd: bool = True,
        normalize_by_bbox: bool = False,
        data_bbox_scale: float = 1.1,  # Only used if normalize_by_bbox
        cam_scale_factor: float = 0.95,
        normalize_by_camera: bool = True,
        n_images: int | None = None,
        **kwargs,  # TODO : removes when switching to new optimizer script
    ):
        super().__init__()
        if not path.isdir(root):
            raise ValueError(f"'{root}' is not a directory")
        self.root = Path(root)

        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size

        logging.info("Load ThermoScene data %s split %s", self.root, split)

        self.split = split

        self.rgb_img_dir_name = ThermoSceneDataset._look_for_dir(
            base_path=self.root, candidates=["images", "image", "rgb"]
        )
        self.thermal_img_dir_name = ThermoSceneDataset._look_for_dir(
            base_path=self.root, candidates=["thermal"]
        )
        self.pose_dir_name = ThermoSceneDataset._look_for_dir(
            base_path=self.root, candidates=["poses", "pose"]
        )

        self._rgb_img_files = ThermoSceneDataset._split_files(
            img_dir_name=self.rgb_img_dir_name, split=self.split
        )
        self._thermal_img_files = ThermoSceneDataset._split_files(
            img_dir_name=self.thermal_img_dir_name, split=self.split
        )

        all_gt, all_c2w = self.load_images(
            img_dir_name=self.rgb_img_dir_name,
            img_files=self._rgb_img_files,
        )

        all_gt_thermal, all_c2w_thermal = self.load_images(
            img_dir_name=self.thermal_img_dir_name,
            img_files=self._thermal_img_files,
        )
        self.c2w_f64 = torch.stack(all_c2w)
        self.c2w_f64_thermal = torch.stack(all_c2w_thermal)

        self.scene_scale = self.normalize_by_camera(
            cam_scale_factor=cam_scale_factor,
        )

        logging.info("scene_scale {}".format(self.scene_scale))
        self.c2w_f64[:, :3, 3] *= self.scene_scale
        self.c2w = self.c2w_f64.float()

        self.c2w_f64_thermal[:, :3, 3] *= self.scene_scale
        self.c2w_thermal = self.c2w_f64_thermal.float()

        self.gt = torch.stack(all_gt).float() / 255.0
        self.gt_thermal = torch.stack(all_gt_thermal).float() / 255.0

        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]

        self.gt = self.gt.float()
        self.gt_thermal = self.gt_thermal.float()

        if not self.full_size[0] > 0 and self.full_size[1] > 0:
            raise AssertionError("Empty images")

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        if n_images is not None:
            if n_images > self.n_images:
                logging.info(
                    f"using {self.n_images} available training views instead of "
                    f"the requested {n_images}."
                )
                n_images = self.n_images
            self.n_images = n_images
            self.gt = self.gt[0:n_images, ...]
            self.c2w = self.c2w[0:n_images, ...]

        fx, fy, cx, cy = self.get_intrinsic_parameters()

        self.intrins_full = Intrin(fx, fy, cx, cy)

        self.t_max, self.t_min = self.get_temperature_metadata()

        # Rays are not needed for testing
        if self.split == "train":
            self.gen_rays(factor=factor)
            return
        # Rays are not needed for testing
        self.h, self.w = self.h_full, self.w_full
        self.intrins = self.intrins_full

    def get_intrinsic_parameters(self) -> float:

        intrinsic_path = ThermoSceneDataset.find_json_structure(self.root)

        with open(intrinsic_path) as file:
            data = json.load(file)

        fx, fy = data["fl_x"], data["fl_y"]
        cx, cy = data["cx"], data["cy"]

        return fx, fy, cx, cy

    def get_temperature_metadata(self) -> float:

        metadat_path = self.root / Path("temperature_bounds.json")

        with open(metadat_path) as file:
            data = json.load(file)

        t_max = data["absolute_max_temperature"]
        t_min = data["absolute_min_temperature"]

        return t_max, t_min

    @staticmethod
    def find_json_structure(dataset_path: Path) -> Path:
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
        return Path(matching_files[0])

    def normalize_by_camera(self, cam_scale_factor: float) -> float:
        norm_pose_files = sorted(os.listdir(self.pose_dir_name))
        norm_poses = np.stack(
            [
                np.loadtxt((self.pose_dir_name / Path(x))).reshape(-1, 4)
                for x in norm_pose_files
            ],
            axis=0,
        )

        # Select subset of files
        T, camera_scale = similarity_from_cameras(norm_poses)

        self.c2w_f64 = torch.from_numpy(T) @ self.c2w_f64
        self.c2w_f64_thermal = torch.from_numpy(T) @ self.c2w_f64_thermal
        return cam_scale_factor * camera_scale

    def load_images(
        self,
        img_dir_name: Path,
        img_files: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        This function loads thermal and rgb images and return them into lists.
        It also charges the poses matrices, assuming they can be different
        between rgb and thermal.
        :returns: list of rgb and thermal images, list of rgb and thermal pose matrices
        """
        all_gt: list[np.ndarray] = []
        all_c2w: list[np.ndarray] = []

        for img_fname in img_files:
            img_path = img_dir_name / Path(img_fname)
            image = imageio.imread(img_path)
            image = self.load_and_pose(
                img_path=img_path,
                img_fname=img_fname,
                all_c2w=all_c2w,
            )
            all_gt.append(torch.from_numpy(image))

        return all_gt, all_c2w

    def load_and_pose(
        self,
        img_path: Path,
        img_fname: str,
        all_c2w: list,
    ) -> np.ndarray:
        """
        This function loads each image and pose.
        If the image is grayscaled (which is the case of thermal images),
        it converts it to RGB
        :returns: images as np.ndarray
        """
        image = ThermoSceneDataset._load_and_crop_image(img_path, to_crop=True)
        # Check if the image is grayscale (i.e., only one channel)
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        pose_fname = path.splitext(img_fname)[0] + ".txt"
        pose_path = path.join(self.pose_dir_name, pose_fname)
        cam_mtx = np.loadtxt(pose_path).reshape(-1, 4)
        if len(cam_mtx) == 3:
            bottom = np.ndarray([[0.0, 0.0, 0.0, 1.0]])
            cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)

        all_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
        self.full_size = list(image.shape[:2])
        return image

    @staticmethod
    def _split_files(img_dir_name: Path, split: str) -> tuple[list[str], list[str]]:
        """
        This function split the images names according to the train/test/validation
        scenario
        :returns: list of rgb and thermal names for the split case
        """

        orig_img_files = os.listdir(img_dir_name)

        # Access image names
        if split == "train" or split == "test_train" or split == "val":
            train_files, val_files = ThermoSceneDataset._get_train_val_files(
                orig_img_files
            )
            if split == "train":
                img_files = train_files
            if split == "val" or split == "test_train":
                img_files = val_files
        if split == "test":
            img_files = ThermoSceneDataset._get_eval_files(orig_img_files)

        # Check for potential empty lists
        if len(img_files) == 0:
            if split == "train":
                img_files = [x for i, x in enumerate(orig_img_files) if i % 16 != 0]
            else:
                img_files = orig_img_files[::16]

        return img_files

    @staticmethod
    def _get_data_files(orig_img_files: list[str], prefix: str) -> list[str]:
        return [x for x in orig_img_files if x.startswith(prefix)]

    @staticmethod
    def _get_eval_files(
        orig_img_files: list[str],
    ) -> list[str]:
        return ThermoSceneDataset._get_data_files(
            orig_img_files=orig_img_files, prefix="frame_eval_"
        )

    @staticmethod
    def _get_train_val_files(orig_img_files: list[str]) -> tuple[list[str], list[str]]:

        total_files = ThermoSceneDataset._get_data_files(
            orig_img_files=orig_img_files, prefix="frame_train_"
        )

        img_files = [x for i, x in enumerate(total_files) if i % 10 != 0]
        validation_img_files = [x for i, x in enumerate(total_files) if i % 10 == 0]

        return img_files, validation_img_files

    @staticmethod
    def _load_and_crop_image(img_path: Path, to_crop: bool = True) -> np.ndarray:
        """
        This function loads each image offers the option to crop the
        FLIR text on the ThermoSCene dataset.
        :returns: image as np.ndarray
        """
        image = imageio.imread(img_path)
        if not to_crop:
            return image
        height_to_keep = int(image.shape[0] * 0.93)
        cropped_image = image[:height_to_keep]
        return cropped_image

    @staticmethod
    def _look_for_dir(base_path: Path, candidates: list[str]) -> Path:
        for cand in candidates:
            if path.isdir(path.join(base_path, cand)):
                return base_path / Path(cand)
        assert False, "None of " + str(candidates) + " found in data directory"

    def _generate_rays(self, dirs: torch.Tensor, factor: float = 1.0) -> ThermalRays:

        rgb_rays = super()._generate_rays(dirs=dirs, factor=factor)

        if factor != 1:
            gt_thermal = F.interpolate(
                self.gt_thermal.permute([0, 3, 1, 2]),
                size=(self.h, self.w),
                mode="area",
            ).permute([0, 2, 3, 1])
            gt_thermal = gt_thermal.reshape(self.n_images, -1, 1)
        else:
            gt_thermal = self.gt_thermal.reshape(self.n_images, -1, 3)

        if self.split == "train":
            gt_thermal = gt_thermal.reshape(-1, 3)

        return ThermalRays(
            origins=rgb_rays.origins,
            dirs=rgb_rays.dirs,
            gt=rgb_rays.gt,
            gt_thermal=gt_thermal,
        )
