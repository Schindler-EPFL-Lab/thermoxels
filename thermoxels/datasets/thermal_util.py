from plenoxels.opt.util.util import Rays
from dataclasses import dataclass
import torch


@dataclass
class ThermalRays(Rays):
    gt_thermal: torch.Tensor | list[torch.Tensor]

    def to(self, *args, **kwargs) -> "ThermalRays":
        rays = super().to(*args, **kwargs)
        gt_thermal = self.gt_thermal.to(*args, **kwargs)
        return ThermalRays(rays.origins, rays.dirs, rays.gt, gt_thermal)

    def __getitem__(self, key: int) -> "ThermalRays":
        rays = super().__getitem__(key)
        gt_thermal = self.gt_thermal[key]
        return ThermalRays(rays.origins, rays.dirs, rays.gt, gt_thermal)
