import torch

from diffusers import DDIMScheduler, HeunDiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler

import os
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from typing import List, Optional
from dataclasses import dataclass, field

from src.utils.models import load_classifier
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad
from utils.plotting_utils import plot
from utils.loss_functions import get_loss_function, calculate_confs, make_loss_dict
from utils.datasets.inet_classes import IDX2NAME
from utils.parser_utils import CommonArguments

@dataclass
class GuidedArgs(CommonArguments):
    gpu: int = 0
    seed: int = 42
    num_images: int = 10

    classifier1: str = 'vit_large_patch16_224'
    classifier2: Optional[str] = None
    eval_classifier: Optional[str] = None

    prompt_template: str = 'a photograph of a'

    target_classes: Optional[List[int]] = None
    early_stopping_confidence: Optional[float] = None

    grad_cam: bool = False

    results_folder: str = 'output_cvpr/imagenet_guided'
    results_sub_folder: Optional[str] = None

    solver: str = 'ddim'

    loss: str = 'CE'
    loss_weight: float = 1.0

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])


def setup() -> GuidedArgs:
    default_config: GuidedArgs = OmegaConf.structured(GuidedArgs)
    cli_args = OmegaConf.from_cli()
    config: GuidedArgs = OmegaConf.merge(default_config, cli_args)
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

    device = torch.device(f'cuda:{args.gpu}')

    if args.target_classes:
        class_sort_idcs = torch.LongTensor(args.target_classes)
    else:
        class_sort_idcs = torch.arange(0, 1000, dtype=torch.long)

    loss = args.loss

    pipe = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path)
    if args.solver == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise NotImplementedError()

    if args.use_double:
        pipe.convert_to_double()

    pipe.to(device)

    classifier_name = args.classifier1
    classifier = load_classifier(classifier_name, device, args.augmentation_num_cutouts, args.augmentation_noise_sd)

    if args.classifier2 is not None:
        loss = 'conf'
        classifier2_name = args.classifier2
        classifier2 = load_classifier(classifier2_name, device, args.augmentation_num_cutouts,
                                      args.augmentation_noise_sd)
        result_folder_pre = f'{args.results_folder}_difference_{classifier_name}_{classifier2_name}'
    else:
        classifier2 = None
        result_folder_pre = f'{args.results_folder}'

    eval_classifier_name = args.eval_classifier
    if eval_classifier_name is not None:
        eval_classifier = load_classifier(eval_classifier_name, device, 0, None)
    else:
        eval_classifier = None

    #setup losses
    losses_dicts = []
    loss_function = get_loss_function(loss, classifier, classifier2=classifier2)
    losses_dicts.append(make_loss_dict(loss_function, args.loss, args.loss_weight))

    latent_dim = (pipe.unet.config.in_channels, args.resolution // pipe.vae_scale_factor, args.resolution // pipe.vae_scale_factor)
    torch.manual_seed(0)
    num_classes = 1000
    deterministic_latents = torch.randn((num_classes, args.num_images if args.num_images >= 50 else 50) + latent_dim, dtype=torch.float)

    to_pil = transforms.ToPILImage()

    if args.results_sub_folder is not None:
        result_folder_pre = args.results_sub_folder

    result_folder_pre = os.path.join(args.results_folder, result_folder_pre)
    print(f'Writing results to: {result_folder_pre}')

    os.makedirs(result_folder_pre, exist_ok=True)
    with open(os.path.join(result_folder_pre, 'config.yaml'), "w") as f:
        OmegaConf.save(args, f)

    for class_i, class_idx in enumerate(class_sort_idcs):
        class_idx = class_idx.item()
        for img_idx in range(args.num_images):
            linear_idx = img_idx + class_i * args.num_images
            if world_size > 1:
                current_split_idx = linear_idx % world_size
                if not current_split_idx == local_rank:
                    continue

            target_class = class_idx
            target_label = IDX2NAME[target_class]

            result_folder_postfix = f'{class_idx}_{class_idx}_{target_label}'
            result_folder = os.path.join(result_folder_pre, result_folder_postfix)
            out_file_pth = os.path.join(result_folder, f'{img_idx}.pdf')
            if os.path.isfile(out_file_pth):
                continue
            os.makedirs(result_folder, exist_ok=True)

            latent = deterministic_latents[class_idx, img_idx][None,:]

            prompt_str = f'{args.prompt_template} {target_label}'

            targets_dict = {args.loss: target_class}
            return_values = pipe(targets_dict,
                                 starting_img=None,
                                 losses_dict=losses_dicts,
                                 # optim params
                                 optim_params=args.optim_params,
                                 null_text_embeddings=None,
                                 # SD params
                                 height=args.resolution, width=args.resolution,
                                 num_inference_steps=args.num_ddim_steps,
                                 guidance_scale=args.guidance_scale,
                                 latents=latent,
                                 prompt=prompt_str)

            img_grid = return_values['imgs']
            loss_scores = return_values['loss_scores']
            regularizer_scores = return_values['regularizer_scores']

            confs, preds = calculate_confs(classifier, img_grid, device, target_class=target_class, return_predictions=True)
            max_class_name_length = 8
            if len(target_label) > max_class_name_length:
                title_attribute = target_label[:max_class_name_length] + '.'
            else:
                title_attribute = target_label
            title_attributes = {f'C1 - {title_attribute}': confs}

            if classifier2 is not None:
                confs2, preds2 = calculate_confs(classifier2, img_grid, device, target_class=target_class,
                                               return_predictions=True)
                title_attributes[f'C2 - {title_attribute}'] = confs2

            if eval_classifier is not None:
                eval_confs, eval_preds = calculate_confs(eval_classifier, img_grid, device, target_class=target_class, return_predictions=True)
                title_attributes[f'Eval - {title_attribute}'] = eval_confs

            min_loss_idx = torch.argmin(torch.FloatTensor(loss_scores['total'])).item()

            min_loss_img = img_grid[min_loss_idx]
            pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
            pil_sd.save(os.path.join(result_folder, f'{img_idx}_sd.png'))
            pil_ours = to_pil(torch.clamp(min_loss_img, 0, 1))
            pil_ours.save(os.path.join(result_folder, f'{img_idx}_ours.png'))

            plot(None, None, img_grid, title_attributes,
                 os.path.join(result_folder, f'{img_idx}.pdf'), loss_scores=loss_scores,
                 regularizer_scores=regularizer_scores)

if __name__=="__main__":
    main()