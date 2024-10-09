import os, pdb

import argparse
import numpy as np
import torch
import requests
from PIL import Image
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import os
from tqdm.auto import tqdm

#from utils.scheduler import DDIMInverseScheduler
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler

from utils.null_text_inversion import NullInversion
from utils.inet_vce_idcs import VCE_start_class_target


@dataclass
class ImagenetInversionArgs:
    resolution: int = 512
    num_ddim_steps: int = 20

    selected: bool = True
    load_flamingo_prompt: bool = True

    class_idcs: Optional[List] = None
    images_per_class: Optional[int] = None

    gpu: int = 0

    null_text_steps: int = 50
    guidance_scale: float = 1.0

    results_folder: str = 'output_cvpr/imagenet_inversions'
    imagenet_folder: str = '/mnt/datasets/imagenet'
    flamingo_captions_folder: str = 'output_cvpr/imagenet_captions'

    model_path: str = 'CompVis/stable-diffusion-v1-4'

def setup() -> ImagenetInversionArgs:
    default_config: ImagenetInversionArgs = OmegaConf.structured(ImagenetInversionArgs)
    cli_args = OmegaConf.from_cli()
    config: ImagenetInversionArgs = OmegaConf.merge(default_config, cli_args)
    return config


if __name__=="__main__":
    args = setup()
    pre_crop_size = int(args.resolution * 1.25)
    transform = transforms.Compose([transforms.Resize(pre_crop_size), transforms.CenterCrop(args.resolution)])
    in_dataset = ImageNet(args.imagenet_folder, split='val', transform=transform)

    def get_imagenet_labels():
        classes_extended = in_dataset.classes
        labels = []
        for a in classes_extended:
            labels.append(a[0])
        return labels

    in_labels = get_imagenet_labels()

    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        args.gpu = local_rank + args.gpu
        print(f'Rank {local_rank} out of {world_size}')
    else:
        local_rank = 0
        world_size = 1

    device = torch.device(f'cuda:{args.gpu}')
    forward_pipe = StableDiffusionPipeline.from_pretrained(args.model_path).to(device)
    forward_pipe.scheduler = DDIMScheduler.from_config(forward_pipe.scheduler.config)
    forward_pipe.safety_checker = lambda images, **kwargs: (images, False)

    null_inversion = NullInversion(forward_pipe, args.num_ddim_steps, args.guidance_scale)

    to_do_list_class_img = []

    if args.selected:
        for (img_idx, class_idx, _) in VCE_start_class_target:
            to_do_list_class_img.append((class_idx, img_idx))
    else:
        val_imgs_per_class = 50
        images_per_class = args.images_per_class if args.images_per_class is not None else val_imgs_per_class
        if args.class_idcs is None:
            for class_idx in range(1000):
                for i in range(images_per_class):
                    to_do_list_class_img.append((class_idx, class_idx * val_imgs_per_class + i))
        else:
            for class_idx in sorted(set(args.class_idcs)):
                for i in range(images_per_class):
                    to_do_list_class_img.append((class_idx, class_idx * val_imgs_per_class + i))

    if local_rank == 0:
        to_do_list_class_img = tqdm(to_do_list_class_img)

    for linear_idx, (target_class, in_idx) in enumerate(to_do_list_class_img):
        if world_size > 1:
            current_split_idx = linear_idx % world_size
            if not current_split_idx == local_rank:
                continue

        class_label = in_labels[target_class]
        class_folder = os.path.join(args.results_folder, args.model_path, f'{target_class}_{class_label}')

        # make the output folders
        inversion_folder = os.path.join(class_folder, f"inversion_{args.num_ddim_steps}_{args.guidance_scale}")
        os.makedirs(inversion_folder, exist_ok=True)
        flamingo_class_captions_folder = os.path.join(args.flamingo_captions_folder, f'{target_class}_{class_label}')

        # if the input is a folder, collect all the images as a list

        null_text_file = os.path.join(inversion_folder, f"{in_idx}_null_texts.pt")

        if os.path.isfile(null_text_file):
            continue

        if (args.load_flamingo_prompt):
            captions_file = os.path.join(flamingo_class_captions_folder, f'{in_idx}_prompt.txt')
            if os.path.isfile(captions_file):
                with open(captions_file, 'r') as f:
                    prompt_str = f.read()
            else:
                print(f'Warning: Could not load caption from {captions_file}')
                continue
        else:
            prompt_str = f'an image of a {class_label} on background'

        #load image
        img, _ = in_dataset[in_idx]
        img_np = np.array(img)

        (_, x_null_reconstructed), x_inv, null_texts = null_inversion.invert(img_np, prompt_str)
        #x_ddim_reconstructed = Image.fromarray(x_null_reconstructed)
        # x_reconstructed = forward_pipe(prompt_str, height=args.resolution, width=args.resolution,
        #                                guidance_scale=1, num_inference_steps=args.num_ddim_steps, latents=x_inv)
        # x_reconstructed = x_reconstructed.images

        # save the inversion
        torch.save(torch.stack(null_texts), null_text_file)
        torch.save(x_inv[0], os.path.join(inversion_folder, f"{in_idx}.pt"))
        # save the prompt string
        with open(os.path.join(inversion_folder, f"{in_idx}_prompt.txt"), "w") as f:
            f.write(prompt_str)

        img.save(os.path.join(inversion_folder, f"{in_idx}_original.png"))
