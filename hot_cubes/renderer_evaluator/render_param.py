from dataclasses import dataclass


@dataclass
class RenderParam:
    """
    This class is used to store the parameters for rendering the images with
    Plenoxel-based models.
    """

    ckpt: str  # ckpt path, should end with ckpt.npz
    data_dir: str
    render_dir: str = "./"
    config_file: str | None = None
    n_eval: int = 100000
    train: bool = False
    lpips: bool = True
    vidsave: bool = False
    imsave: bool = True
    fps: int = 5
    crop: float = 1.0
    nofg: bool = False
    nobg: bool = False
    blackbg: bool = False
    ray_len: bool = False
    dataset_type: str = "auto"
    scene_scale: float = 1.0
    scale: float = 1.0
    seq_id: int = 1000
    epoch_size: int = 12800
    white_bkgd: bool = True
    normalize_by_bbox: bool = False
    data_bbox_scale: float = 1.2
    cam_scale_factor: float = 0.95
    normalize_by_camera: bool = True
    perm: bool = False
    step_size: float = 0.5
    sigma_thresh: float = 1e-8
    stop_thresh: float = 1e-7
    background_brightness: float = 1.0
    renderer_backend: str = "cuvol"
    random_sigma_std: float = 0.0
    random_sigma_std_background: float = 0.0
    near_clip: float = 0.0
    use_spheric_clip: bool = False
    last_sample_opaque: bool = False
