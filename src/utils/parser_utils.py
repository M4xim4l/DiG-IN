from typing import Dict, List, Optional
from dataclasses import dataclass, field
from .sd_backprop_pipe import OptimParams
from .sdxl_backprop_pipe import OptimParamsXL

@dataclass
class CommonArguments:
    model_path: str = 'CompVis/stable-diffusion-v1-4'
    use_double: bool = False
    resolution: int = 512
    guidance_scale: float = 3.0
    num_ddim_steps: int = 20

    optim_params: OptimParams = OptimParams()

    augmentation_num_cutouts: int = 16
    augmentation_noise_sd: float = 0.005
    augmentation_noise_schedule: str = 'const'

    wandb_project: Optional[str] = None

@dataclass
class CommonArgumentsXL:
    model_path: str = ''
    use_double: bool = False
    resolution: int = 512
    guidance_scale: float = 7.5
    num_ddim_steps: int = 50
    solver: str = 'ddim'

    optim_params: OptimParamsXL = OptimParamsXL()

    augmentation_num_cutouts: int = 16
    augmentation_noise_sd: float = 0.005
    augmentation_noise_schedule: str = 'const'

    wandb_project: Optional[str] = None

@dataclass
class CommonArgumentsXLTurbo(CommonArgumentsXL):
    model_path: str = 'stabilityai/sdxl-turbo'
    resolution: int = 512
    guidance_scale: float = 0.0
    num_ddim_steps: int = 4
    solver: str = 'euler'

@dataclass
class CommonArgumentsXL512(CommonArgumentsXL):
    model_path: str = 'hotshotco/SDXL-512'
    resolution: int = 512
    guidance_scale: float = 3.0
    num_ddim_steps: int = 20
    solver: str = 'ddim'

@dataclass
class CommonArgumentsPlayground512(CommonArgumentsXL):
    model_path: str = 'playgroundai/playground-v2-512px-base'
    resolution: int = 512
    guidance_scale: float = 3.0
    num_ddim_steps: int = 20
    solver: str = 'euler'



def define_common_arguments(parser):
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_double', action='store_true')
    parser.add_argument('--resolution', type=int, default=512)

    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--sgd_steps', type=int, default=20)
    parser.add_argument('--sgd_stepsize', type=float, default=0.05)
    parser.add_argument('--sgd_schedule', type=str, default='cosine')
    parser.add_argument('--normalize_gradient', action='store_true')
    parser.add_argument('--gradient_clipping', type=float, default=0.05)
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--conditioning_lr_factor', type=float, default=1.0)
    parser.add_argument('--uncond_lr_factor', type=float, default=1.0)
    parser.add_argument('--latent_lr_factor', type=float, default=1.0)
    parser.add_argument('--num_ddim_steps', type=int, default=20)
    parser.add_argument('--optimize_latents', action='store_true')
    parser.add_argument('--optimize_conditioning', action='store_true')
    parser.add_argument('--optimize_uncond', action='store_true')
    parser.add_argument('--per_timestep_conditioning', action='store_true')
    parser.add_argument('--per_timestep_uncond', action='store_true')
    parser.add_argument('--conditioning_gradient_masking', action='store_true')
    parser.add_argument('--xa_interpolation_schedule', type=str, default=None)
    parser.add_argument('--augmentation_num_cutouts', type=int, default=32)
    parser.add_argument('--augmentation_noise_sd', type=float, default=0)
    parser.add_argument('--augmentation_noise_schedule', type=str, default='const')
    return parser