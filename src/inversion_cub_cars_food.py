import os

import numpy as np
import torch
from PIL import Image

#from utils.scheduler import DDIMInverseScheduler
import torchvision.transforms as transforms
from torchvision.datasets import Food101, Flowers102
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler

from omegaconf import OmegaConf
from typing import Optional
from dataclasses import dataclass

from utils.null_text_inversion import NullInversion
from utils.datasets.bird_dataset import BirdDataset as CubDataset
from utils.datasets.car_dataset import CarDataset as CarsDataset
from utils.datasets.food101_classes import IDX2NAME as food_labels
from utils.datasets.cub_classes import IDX2NAME as cub_labels
from utils.datasets.cars_classes import IDX2NAME as cars_labels
from utils.datasets.flowers_classes import IDX2NAME as flowers_labels

@dataclass
class InversionArgs():
    gpu: int = 0
    images_per_class: Optional[int] = None
    load_prompt: bool =True
    dataset: Optional[str] = None
    resolution: int = 512
    null_text_steps: int = 50
    guidance_scale: float = 3.0
    results_folder: Optional[str] = None
    flamingo_captions_folder: Optional[str] = None
    dataset_folder: Optional[str] = None
    num_ddim_steps: int = 20
    model_path: str = 'CompVis/stable-diffusion-v1-4'

def setup() -> InversionArgs:
    default_config: InversionArgs = OmegaConf.structured(InversionArgs)
    cli_args = OmegaConf.from_cli()
    config: InversionArgs = OmegaConf.merge(default_config, cli_args)
    assert config.dataset is not None

    if config.dataset == 'cub':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets/CUB_200_2011'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/cub_inversions'
        config.flamingo_captions_folder = config.flamingo_captions_folder if config.flamingo_captions_folder is not None else 'output_cvpr/cub_captions'
    elif config.dataset == 'cars':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets/stanford_cars'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/cars_inversions'
        config.flamingo_captions_folder = config.flamingo_captions_folder if config.flamingo_captions_folder is not None else 'output_cvpr/cars_captions'
    elif config.dataset == 'food101':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/food101_inversions'
        config.flamingo_captions_folder = config.flamingo_captions_folder if config.flamingo_captions_folder is not None else 'output_cvpr/food101_captions'
    elif config.dataset == 'flowers':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/flowers_inversions'
        config.flamingo_captions_folder = config.flamingo_captions_folder if config.flamingo_captions_folder is not None else 'output_cvpr/flowers_captions'
    else:
        raise NotImplementedError()

    return config


if __name__=="__main__":

    args = setup()

    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        args.gpu = local_rank + args.gpu
        print(f'Rank {local_rank} out of {world_size}')
    else:
        local_rank = 0
        world_size = 1

    if local_rank == 0:
        print(f'Inverting {args.dataset} from {args.dataset_folder}')
        print(f'Loading captions from {args.flamingo_captions_folder}')
        print(f'Writing  inversions to {args.results_folder}')

    pre_crop_size = int(args.resolution * 1.25)
    transform = transforms.Compose([transforms.Resize(pre_crop_size), transforms.CenterCrop(args.resolution)])

    if args.dataset == 'cub':
        in_dataset = CubDataset(args.dataset_folder, phase='test', transform=transform)
        in_labels = cub_labels
    elif args.dataset == 'cars':
        in_dataset = CarsDataset(args.dataset_folder, phase='test', transform=transform)
        in_labels = cars_labels
    elif args.dataset == 'food101':
        in_dataset = Food101(args.dataset_folder, split='test', transform=transform)
        in_dataset.targets = in_dataset._labels
        in_labels = food_labels
    elif args.dataset == 'flowers':
        in_dataset = Flowers102(args.dataset_folder, split='test', transform=transform)
        in_dataset.targets = in_dataset._labels
        in_labels = flowers_labels
    else:
        raise NotImplementedError()

    device = torch.device(f'cuda:{args.gpu}')
    forward_pipe = StableDiffusionPipeline.from_pretrained(args.model_path).to(device)
    forward_pipe.scheduler = DDIMScheduler.from_config(forward_pipe.scheduler.config)
    forward_pipe.safety_checker = lambda images, **kwargs: (images, False)

    null_inversion = NullInversion(forward_pipe, args.num_ddim_steps, args.guidance_scale)

    to_do_list_class_img = []
    dataset_targets = torch.LongTensor(in_dataset.targets)
    for target_class_idx in range(len(torch.unique(dataset_targets))):
        test_class_idcs = torch.nonzero(torch.LongTensor(in_dataset.targets) == target_class_idx,
                                        as_tuple=False).squeeze()
        if args.images_per_class is not None and len(test_class_idcs) > args.images_per_class:
            test_class_idcs = test_class_idcs[:args.images_per_class]
        for in_idx in test_class_idcs:
            to_do_list_class_img.append((target_class_idx, in_idx.item()))

    print(f'Inverting {len(to_do_list_class_img)} images')

    for linear_idx, (target_class, in_idx) in enumerate(to_do_list_class_img):
        if world_size > 1:
            current_split_idx = target_class % world_size
            if not current_split_idx == local_rank:
                continue

        class_label = in_labels[target_class]
        class_folder = os.path.join(args.results_folder, f'{target_class}_{class_label}')

        # make the output folders
        inversion_folder = os.path.join(class_folder, f"inversion_{args.num_ddim_steps}_{args.guidance_scale}")
        os.makedirs(inversion_folder, exist_ok=True)
        flamingo_class_captions_folder = os.path.join(args.flamingo_captions_folder, f'{target_class}_{class_label}')

        # if the input is a folder, collect all the images as a list
        if os.path.isfile(os.path.join(inversion_folder, f"{in_idx}_original.png")):
            continue
        img, _ = in_dataset[in_idx]
        img_np = np.array(img)

        captions_file = os.path.join(flamingo_class_captions_folder, f'{in_idx}_prompt.txt')
        if args.load_prompt:
            if os.path.isfile(captions_file):
                with open(captions_file, 'r') as f:
                    prompt_str = f.read()
            else:
                print(f'Warning: Could not load caption from {captions_file}')
                continue
        else:
            prompt_str = f'an image of a {class_label} macro shot close-up'

        try:
            (_, x_null_reconstructed), x_inv, null_texts = null_inversion.invert(img_np, prompt_str)
            x_null_reconstructed = Image.fromarray(x_null_reconstructed)
            # x_reconstructed = forward_pipe(prompt_str, height=args.resolution, width=args.resolution,
            #                                guidance_scale=1, num_inference_steps=args.num_ddim_steps, latents=x_inv)
            # x_reconstructed = x_reconstructed.images

            # save the inversion
            img.save(os.path.join(inversion_folder, f"{in_idx}_original.png"))
            torch.save(torch.stack(null_texts), os.path.join(inversion_folder, f"{in_idx}_null_texts.pt"))
            torch.save(x_inv[0], os.path.join(inversion_folder, f"{in_idx}.pt"))
            x_null_reconstructed.save(os.path.join(inversion_folder, f"{in_idx}_null_reconstructed.png"))
            # x_reconstructed[0].save(os.path.join(inversion_folder, f"{in_idx}_reconstructed.png"))
            # save the prompt string
            with open(os.path.join(inversion_folder, f"{in_idx}_prompt.txt"), "w") as f:
                f.write(prompt_str)
        except Exception as e:
            print(e)

