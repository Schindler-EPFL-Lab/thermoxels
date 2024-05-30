from dataclasses import dataclass


@dataclass
class Param:
    """
    This class contains all the parameters to run a plenoxel optimization.
    All the parameters are set to defaukt values but can be overwritten by a config
    file.
    """

    # Directory settings
    data_dir: str
    render_dir: str = "./"
    config_file: str | None = None
    train_dir: str = "ckpt"

    # General settings
    reso: str = "[[256, 256, 256], [512, 512, 512]]"
    image_type: str = "rgb"
    upsamp_every: int = 3 * 12800
    init_iters: int = 0
    upsample_density_add: float = 0.0
    include_temperature: bool = True
    dataset_type: str = "auto"
    scene_scale: float = 1.0
    scale: float = 1.0
    seq_id: int = 1000
    epoch_size: int = 12800

    # Basis function settings
    basis_type: str = "sh"
    basis_reso: int = 32
    sh_dim: int = 9
    mlp_posenc_size: int = 4
    mlp_width: int = 32

    # Background settings
    background_nlayers: int = 0
    background_reso: int = 512

    # Optimization settings
    n_epoch: int = 5
    batch_size: int = 5000
    sigma_optim: str = "rmsprop"
    sh_optim: str = "rmsprop"
    temp_optim: str = "rmsprop"
    bg_optim: str = "rmsprop"
    basis_optim: str = "rmsprop"

    # Learning rates
    lr_sigma: float = 3e1
    lr_sigma_final: float = 5e-2
    lr_sigma_decay_steps: int = 250000
    lr_sigma_delay_steps: int = 15000
    lr_sigma_delay_mult: float = 1e-2
    lr_sh: float = 1e-2
    lr_sh_final: float = 5e-6
    lr_sh_decay_steps: int = 250000
    lr_sh_delay_steps: int = 0
    lr_sh_delay_mult: float = 1e-2
    lr_fg_begin_step: int = 0
    lr_sigma_bg: float = 3e0
    lr_sigma_bg_final: float = 3e-3
    lr_sigma_bg_decay_steps: int = 250000
    lr_sigma_bg_delay_steps: int = 0
    lr_sigma_bg_delay_mult: float = 1e-2
    lr_color_bg: float = 1e-1
    lr_color_bg_final: float = 5e-6
    lr_color_bg_decay_steps: int = 250000
    lr_color_bg_delay_steps: int = 0
    lr_color_bg_delay_mult: float = 1e-2

    lr_basis: float = 1e-6
    lr_basis_final: float = 1e-6
    lr_basis_decay_steps: int = 250000
    lr_basis_delay_steps: int = 0
    lr_basis_begin_step: int = 0
    lr_basis_delay_mult: float = 1e-2
    rms_beta: float = 0.95

    # Logging and evaluations
    log_per_epoch: int = 2
    save_every: int = 5
    eval_every: int = 1
    init_sigma: float = 0.1
    init_sigma_bg: float = 0.1
    log_mse_image: bool = False
    log_depth_map: bool = True
    log_depth_map_use_thresh: float | None = None

    # Experiments and thresholds
    thresh_type: str = "weight"
    weight_thresh: float = 0.0005 * 512
    density_thresh: float = 5.0
    background_density_thresh: float = 1.0 + 1e-9
    max_grid_elements: int = 44000000
    tune_mode: bool = False
    render_circle: bool = False
    tune_nosave: bool = False

    # Losses and regularization
    lambda_tv: float = 1e-5
    tv_sparsity: float = 0.01
    tv_logalpha: bool = False
    lambda_tv_sh: float = 1e-3
    lambda_tv_temp: float = 1e-3
    lambda_tv_lumisphere: float = 0.0
    tv_lumisphere_sparsity: float = 0.01
    tv_lumisphere_dir_factor: float = 0.0
    tv_decay: float = 1.0
    lambda_l2_sh: float = 0.0
    tv_early_only: int = 1
    tv_contiguous: int = 1
    lambda_sparsity: float = 0.0
    lambda_beta: float = 0.0
    lambda_tv_background_sigma: float = 1e-2
    lambda_tv_background_color: float = 1e-2
    tv_background_sparsity: float = 0.01
    lambda_tv_basis: float = 0.0
    weight_decay_sigma: float = 1.0
    weight_decay_sh: float = 1.0
    lr_decay: bool = True
    n_train: int | None = None
    nosphereinit: bool = False
    tv_sh_sparsity: float = 0.01
    tv_temp_sparsity: float = 0.01
    t_loss: float = 0.01

    # Rendering settings
    white_bkgd: bool = True
    llffhold: int = 8
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
    near_clip: float = 0.00
    use_spheric_clip: bool = False
    enable_random: bool = False
    last_sample_opaque: bool = False

    def update_from_dict(self, updates: dict) -> None:
        """
        This function updats the attributes of the config instance based on a
        dictionary.
        Only existing attributes of the instance which are provided
        in the dictionary will be updated.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
