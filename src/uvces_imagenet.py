import torch
from diffusers import DDIMScheduler
import os
from omegaconf import OmegaConf
import wandb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from PIL import Image
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field

from utils.models import load_classifier
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad, PromptToPromptParams, MaskBlendParams
from utils.loss_functions import get_loss_function, make_loss_dict
from utils.utils import make_uvces_and_save
from utils.datasets.inet_classes import IDX2NAME as in_labels
from utils.datasets.inet_hierarchy import find_cluster_for_class_idx
from utils.parser_utils import CommonArguments
from utils.foreground_segmentation import SegmentationArgs
from utils.inet_vce_idcs import VCE_start_class_target
from utils.wandb_utils import make_wandb_run

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
    error_visualisation: bool = False

    classifier: str = 'vit_base_patch16_384.augreg_in21k_ft_in1k'

    results_folder: str = 'output_cvpr/imagenet_counterfactuals'
    results_sub_folder: Optional[str] = None
    inversion_folder: str = 'output_cvpr/imagenet_inversions'
    imagenet_folder: str = '/mnt/datasets/imagenet'

    loss: str = 'CE'
    loss_weight: float = 1.0

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])

    segmentation_args: SegmentationArgs = SegmentationArgs()
    p2p_params: PromptToPromptParams = UVCEP2P()
    mask_blend_params: MaskBlendParams = UVCEBlend()

def fill_img_idcs(img_id_start_target, inversion_folder, num_ddim_steps, guidance_scale):
    new_id_start_target = []
    for img_idx, start_class, target_class in img_id_start_target:
        #check if inversion exists
        start_label = in_labels[start_class]
        start_class_folder = os.path.join(inversion_folder, f'{start_class}_{start_label}')
        inversion_dir = os.path.join(start_class_folder,  f"inversion_{num_ddim_steps}_{guidance_scale}")

        if not os.path.isfile(os.path.join(inversion_dir, f'{img_idx}.pt')):
            print(f'Could not find inversion file {img_idx} {start_class}')
            continue

        if target_class is not None:
            new_id_start_target.append((img_idx, start_class, target_class))
        else:
            #find matching in cluster
            in_cluster = find_cluster_for_class_idx(start_class)
            if in_cluster is None:
                print(f'Not cluster found for class {start_class}; Skipping')
            else:
                for t_idx, _ in in_cluster:
                    if t_idx != start_class:
                        new_id_start_target.append((img_idx, start_class, t_idx))

    return new_id_start_target

def find_failure_vce_idcs(classifier, loader, device, inversion_folder, num_ddim_steps, guidance_scale, num_images=None):
    img_idx = 0

    img_id_start_target = []

    for data, target in loader:
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)

            out = classifier(data)
            _, predictions = torch.max(out, dim=1)

        for batch_idx in range(len(data)):
            if target[batch_idx] != predictions[batch_idx]:
                start_class = target[batch_idx].item()
                start_label = in_labels[start_class]
                start_class_folder = os.path.join(inversion_folder, f'{start_class}_{start_label}')
                inversion_dir = os.path.join(start_class_folder, f"inversion_{num_ddim_steps}_{guidance_scale}")

                if not os.path.isfile(os.path.join(inversion_dir, f'{img_idx + batch_idx}.pt')):
                    print(f'Could not find inversion file {img_idx + batch_idx} {start_class}')
                else:
                    img_id_start_target.append((img_idx + batch_idx, start_class, start_class))

        if num_images is not None and len(img_id_start_target) >= num_images:
            break

        img_idx += len(data)

    print(f'Found {len(img_id_start_target)} errors with existing inversion files')
    return img_id_start_target


def setup() -> CounterfactualArgs:
    default_config: CounterfactualArgs = OmegaConf.structured(CounterfactualArgs)
    cli_args = OmegaConf.from_cli()
    config: CounterfactualArgs = OmegaConf.merge(default_config, cli_args)
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

    classifier_name = args.classifier
    classifier = load_classifier(classifier_name, device,
                                 num_cutouts=args.augmentation_num_cutouts, noise_sd=args.augmentation_noise_sd)

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


    inversion_root_dir = os.path.join(args.inversion_folder, args.model_path)

    if args.error_visualisation:
        classifier_resolution = classifier.size
        val_transform = transforms.Compose([
            transforms.Resize(int(1.25 * classifier_resolution)),
            transforms.CenterCrop(classifier_resolution),
            transforms.ToTensor()
        ])
        dataset = ImageNet(args.imagenet_folder, split='val', transform=val_transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
        img_id_start_target = find_failure_vce_idcs(classifier, loader, device, inversion_root_dir,
                                                    args.num_ddim_steps, args.guidance_scale, args.num_images)
    else:
        img_id_start_target = fill_img_idcs(VCE_start_class_target, inversion_root_dir,
                                            args.num_ddim_steps, args.guidance_scale)

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

        make_uvces_and_save(args, classifier, pipe, device, linear_idx, img_idx, in_labels, inversion_root_dir,
                            losses_dicts, result_folder, regularizers_weights, start_class, target_class)



if __name__=="__main__":
    main()