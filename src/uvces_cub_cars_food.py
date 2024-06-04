import torch

from diffusers import DDIMScheduler

import os
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from torchvision.datasets import Food101, Flowers102

from utils.utils import make_uvces_and_save
from utils.parser_utils import CommonArguments
from utils.models.load_mae_model import load_cub_mae, load_cub_mae_with_cutout
from utils.models.load_cal_model import load_cal_model, load_cal_model_with_cutout
from utils.models.load_hugginface_model import (load_food101_vit, load_food101_vit_with_cutout,
                                                load_flowers_vit_with_cutout, load_flowers_vit,
                                                load_cub_vit, load_cub_vit_with_cutout)
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad, PromptToPromptParams, MaskBlendParams
from utils.loss_functions import get_loss_function, make_loss_dict
from utils.datasets.cub_classes import IDX2NAME as cub_labels
from utils.datasets.cars_classes import IDX2NAME as cars_labels
from utils.datasets.food101_classes import IDX2NAME as food_labels
from utils.datasets.flowers_classes import IDX2NAME as flowers_labels
from utils.wandb_utils import make_wandb_run
from utils.datasets.car_dataset import CarDataset
from utils.datasets.bird_dataset import BirdDataset
from utils.foreground_segmentation import SegmentationArgs

cub_vces_img_id_start_target = [
    (1271,46,66),
    (1272,46,66),
    (1273,46,66),
    (1274,46,66),
    (1276,46,66),

    (1271,46,67),
    (1272,46,67),
    (1273,46,67),
    (1274,46,67),
    (1276,46,67),

    (1271,46,72),
    (1272,46,72),
    (1273,46,72),
    (1274,46,72),
    (1276,46,72),

    (1271,46,117),
    (1272,46,117),
    (1273,46,117),
    (1274,46,117),
    (1276,46,117),

    (391, 15, 150),
    (393, 15, 150),
    (399, 15, 150),
    (401, 15, 150),
    (401, 15, 150),

    (391, 15, 151),
    (393, 15, 151),
    (399, 15, 151),
    (401, 15, 151),
    (401, 15, 151),

    (391, 15, 152),
    (393, 15, 152),
    (399, 15, 152),
    (401, 15, 152),
    (401, 15, 152),

    (391, 15, 153),
    (393, 15, 153),
    (399, 15, 153),
    (401, 15, 153),
    (401, 15, 153),

    (391, 15, 158),
    (393, 15, 158),
    (399, 15, 158),
    (401, 15, 158),
    (401, 15, 158),

    (391, 15, 160),
    (393, 15, 160),
    (399, 15, 160),
    (401, 15, 160),
    (401, 15, 160),

    (391, 15, 163),
    (393, 15, 163),
    (399, 15, 163),
    (401, 15, 163),
    (401, 15, 163),

    (391, 15, 169),
    (393, 15, 169),
    (399, 15, 169),
    (401, 15, 169),
    (401, 15, 169),
]

car_vces_img_id_start_target = [
    (929,55,143),
    (1741,55,143),

    (929,55,172),
    (1741,55,172),

    (929,55,111),
    (1741,55,111),

    (929,55,33),
    (1741,55,33),

    (929,55,34),
    (1741,55,34),

    (929,55,44),
    (1741,55,44),

    (929,55,14),
    (1741,55,14),
]


@dataclass
class UVCEBlend(MaskBlendParams):
    do_mask_blend: bool = True
    foreground_blend: bool = False
    optimize_mask_blend: bool = False
    initial_mask_blend_schedule: List[str]  = field(default_factory=lambda: [
        'none',
        'threshold_0.3',
    ])
    mask_blend_stepsize: float = 0.05

@dataclass
class UVCEP2P(PromptToPromptParams):
    do_p2p: bool = True
    xa_interpolation_schedule: Optional[List[str]] =  field(default_factory=lambda: [
        'threshold_0.0_0.3',
        'threshold_0.0_0.4',
        'threshold_0.0_0.5',
    ])
    self_interpolation_schedule: Optional[List[str]] = None
    optimize_xa_interpolation_schedule: bool = False
    optimize_self_interpolation_schedule: bool = False
    schedule_stepsize: float = 0.05


@dataclass
class CounterfactualArgs(CommonArguments):
    gpu: int = 0
    seed: int = 42
    num_images: Optional[int] = None
    dataset: Optional[str] = None
    random_images: bool = False

    results_folder: Optional[str] = None
    results_sub_folder: Optional[str] = None
    inversion_folder: Optional[str] = None
    dataset_folder: Optional[str] = None

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])

    loss: str = 'CE'
    loss_weight: float = 1.0

    segmentation_args: SegmentationArgs = SegmentationArgs()
    p2p_params: PromptToPromptParams = UVCEP2P()
    mask_blend_params: MaskBlendParams = UVCEBlend()


def setup() -> CounterfactualArgs:
    default_config: CounterfactualArgs = OmegaConf.structured(CounterfactualArgs)
    cli_args = OmegaConf.from_cli()
    config: CounterfactualArgs = OmegaConf.merge(default_config, cli_args)
    assert config.dataset is not None

    if config.dataset == 'cub':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets/CUB_200_2011'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/cub_counterfactuals'
        config.inversion_folder = config.inversion_folder if config.inversion_folder is not None else 'output_cvpr/cub_inversions'
    elif config.dataset == 'cars':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets/stanford_cars'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/cars_counterfactuals'
        config.inversion_folder = config.inversion_folder if config.inversion_folder is not None else 'output_cvpr/cars_inversions'
    elif config.dataset == 'food101':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/food101_counterfactuals'
        config.inversion_folder = config.inversion_folder if config.inversion_folder is not None else 'output_cvpr/food101_inversions'
    elif config.dataset == 'flowers':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/flowers_counterfactuals'
        config.inversion_folder = config.inversion_folder if config.inversion_folder is not None else 'output_cvpr/flowers_inversions'
    else:
        raise NotImplementedError()

    return config


def main():
    args = setup()

    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        args.gpu = local_rank + args.gpu
        print(f'Rank {local_rank} out of {world_size}')
    else:
        local_rank = 0
        world_size = 1

    #torch.set_deterministic_debug_mode(1)
    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed)

    num_cutouts = args.augmentation_num_cutouts
    dataset = args.dataset
    if dataset == 'cub':
        class_labels = cub_labels
        num_classes = len(class_labels)
        if num_cutouts > 0:
            classifier = load_cub_vit_with_cutout(0.3, num_cutouts, checkpointing=True,
                                                  noise_sd=args.augmentation_noise_sd,
                                                  noise_schedule=args.augmentation_noise_schedule,
                                                  noise_descending_steps=args.optim_params.sgd_steps)
        else:
            classifier = load_cub_vit()

        classifier.to(device)

        img_id_start_target = cub_vces_img_id_start_target
    elif dataset == 'cars':
        class_labels = cars_labels
        num_classes = len(class_labels)
        if num_cutouts > 0:
            classifier = load_cal_model_with_cutout('cars', 0.3, num_cutouts, checkpointing=True,
                                                    noise_sd=args.augmentation_noise_sd,
                                                    noise_schedule=args.augmentation_noise_schedule,
                                                    noise_descending_steps=args.optim_params.sgd_steps)
        else:
            classifier = load_cal_model('cars')

        img_id_start_target = car_vces_img_id_start_target
    elif dataset == 'food101':
        class_labels = food_labels
        num_classes = len(class_labels)
        if num_cutouts > 0:
            classifier = load_food101_vit_with_cutout(0.3, num_cutouts, checkpointing=True,
                                                    noise_sd=args.augmentation_noise_sd,
                                                    noise_schedule=args.augmentation_noise_schedule,
                                                    noise_descending_steps=args.optim_params.sgd_steps)
        else:
            classifier = load_food101_vit()

        img_id_start_target = []
    elif dataset == 'flowers':
        class_labels = flowers_labels
        num_classes = len(class_labels)
        if num_cutouts > 0:
            classifier = load_flowers_vit_with_cutout(0.3, num_cutouts, checkpointing=True,
                                                    noise_sd=args.augmentation_noise_sd,
                                                    noise_schedule=args.augmentation_noise_schedule,
                                                    noise_descending_steps=args.optim_params.sgd_steps)
        else:
            classifier = load_flowers_vit()

        img_id_start_target = []
    else:
        raise NotImplementedError()

    inversion_root_dir = os.path.join(args.inversion_folder)
    if args.random_images:
        assert args.num_images is not None
        if args.dataset == 'cub':
            dataset = BirdDataset(args.dataset_folder, phase='test', transform=None)
        elif args.dataset == 'cars':
            dataset = CarDataset(args.dataset_folder, phase='test', transform=None)
        elif args.dataset == 'food101':
            dataset = Food101(args.dataset_folder, split='test', transform=None)
            dataset.targets = dataset._labels
        elif args.dataset == 'flowers':
            dataset = Flowers102(args.dataset_folder, split='test', transform=None)
            dataset.targets = dataset._labels
        else:
            raise NotImplementedError()

        dataset_targets = dataset.targets
        img_id_start_target = []
        torch.manual_seed(args.gpu)

        while len(img_id_start_target) < args.num_images:
            img_idx = torch.randint(0, len(dataset_targets), (1,), dtype=torch.long).item()
            start_class = dataset_targets[img_idx]
            target_class = torch.randint(0, num_classes, (1,)).item()
            if start_class == target_class:
                continue

            start_label = class_labels[start_class]
            start_class_folder = os.path.join(inversion_root_dir, f'{start_class}_{start_label}')
            inversion_dir = os.path.join(start_class_folder,
                                         f"inversion_{args.num_ddim_steps}_{args.guidance_scale}")
            latent_file = os.path.join(inversion_dir, f'{img_idx}.pt')
            if os.path.isfile(latent_file):
                img_id_start_target.append((img_idx, start_class, target_class))

    if args.use_double:
        classifier.double()
    classifier.to(device)

    pipe: StableDiffusionPipelineWithGrad = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.use_double:
        pipe.convert_to_double()

    pipe.to(device)

    #setup losses
    losses_dicts = []
    loss_function = get_loss_function(args.loss, classifier)
    losses_dicts.append(make_loss_dict(loss_function, args.loss, args.loss_weight))

    assert len(args.regularizers) == len(args.regularizers_ws)
    regularizers_weights = {}
    reg_string = ''
    for reg_name, reg_w in zip(args.regularizers, args.regularizers_ws):
        regularizers_weights[reg_name] = reg_w
        reg_string += f'_{reg_name}_{reg_w}'


    if args.num_images:
        selected_subset = torch.randperm(len(img_id_start_target))[:args.num_images]
        img_id_start_target = [img_id_start_target[i] for i in selected_subset]

    # datetime object containing current date and time
    if args.results_sub_folder is None:
        output_description = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    else:
        output_description = args.results_sub_folder

    result_folder = os.path.join(args.results_folder, output_description)
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, 'config.yaml'), "w") as f:
        OmegaConf.save(args, f)

    make_wandb_run(args.wandb_project, output_description, OmegaConf.to_container(args))

    for linear_idx, (img_idx, start_class, target_class) in enumerate(img_id_start_target):
        if world_size > 1:
            current_split_idx = linear_idx % world_size
            if not current_split_idx == local_rank:
                continue

        make_uvces_and_save(args, classifier, pipe, device, linear_idx, img_idx, class_labels, inversion_root_dir,
                            losses_dicts, result_folder, regularizers_weights, start_class, target_class)


if __name__=="__main__":
    main()