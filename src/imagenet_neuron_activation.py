from datetime import datetime

import math
import torch

from diffusers import DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler

import os
import argparse
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from utils.models.load_robust_model import load_madry_l2_with_cutout, load_madry_l2
from utils.models.load_timm_model import load_timm_model, load_timm_model_with_cutout
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad, MaskBlendParams
from utils.plotting_utils import plot, plot_attention
from utils.loss_functions import get_feature_loss_function, calculate_neuron_activations, make_loss_dict, calculate_confs
from utils.datasets.inet_classes import IDX2NAME
from utils.parser_utils import CommonArguments
from utils.foreground_segmentation import SegmentationArgs
from utils.attribution_with_gradient import ActivationCaption

@dataclass
class NeuronBlend(MaskBlendParams):
    do_mask_blend: bool = True
    foreground_blend: bool = True
    optimize_mask_blend: bool = False
    initial_mask_blend_schedule: List[str]  = field(default_factory=lambda: [
        'threshold_0.3',
    ])
    mask_blend_stepsize: float = 0.05

DEFAULT_CLASS_NEURON_PAIRS = [(83, 565), (120, 870), (537, 0), (146, 1697), (2, 1697), (576, 1772), (14, 660),
                              (50, 341), (309, 595), (309, 1797), (537, 579), (537, 1855), (695, 1567), (695, 1635)]

@dataclass
class NeuronArgs(CommonArguments):
    gpu: int = 0
    seed: int = 42
    num_images: int = 10

    classifier: str = 'madry_l2'
    prompt_template: str = 'a photograph of a'
    target_class_neurons: List[List[int]] = field(default_factory=lambda: DEFAULT_CLASS_NEURON_PAIRS)
    spurious_neurons: bool = False

    results_folder: str = 'output_cvpr/imagenet_neurons'
    results_sub_folder: Optional[str] = None
    mask_blend_params: MaskBlendParams = NeuronBlend()

    solver: str = 'ddim'

    loss: str = 'neuron_activatione'
    loss_weight: float = 1.0

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])
    segmentation_args: SegmentationArgs = SegmentationArgs()

def setup() -> NeuronArgs:
    default_config: NeuronArgs = OmegaConf.structured(NeuronArgs)
    cli_args = OmegaConf.from_cli()
    config: NeuronArgs = OmegaConf.merge(default_config, cli_args)
    return config

foreground_replacements = {
    'fiddler crab': 'crab',
    'prairie chicken': 'chicken',
    'howler monkey': 'monkey',
    'great white shark': 'shark',
    'American alligator': 'alligator',
}

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

    device = torch.device(f'cuda:{args.gpu}')
    
    if args.spurious_neurons:
        target_classes_neurons = torch.load('src/utils/spurious_class_neuron_pairs.pth')
    else:
        target_classes_neurons = args.target_class_neurons

    for target_i, (target_class, target_neuron) in enumerate(target_classes_neurons):
        if IDX2NAME[target_class] in foreground_replacements:
            print(f'{target_i}: {IDX2NAME[target_class]} - {foreground_replacements[IDX2NAME[target_class]]}')
        else:
            print(f'{target_i}: {IDX2NAME[target_class]}')

    loss = args.loss

    pipe = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path)
    if args.solver == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        solver_order = 1
    elif args.solver == 'heun':
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        solver_order = 2
    elif args.solver == 'dpm':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        solver_order = 2
    elif args.solver == 'pndm':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        solver_order = 2
    else:
        raise NotImplementedError()

    if args.use_double:
        pipe.convert_to_double()

    pipe.to(device)

    if args.classifier == 'madry_l2':
        if args.augmentation_num_cutouts > 0:
            classifier = load_madry_l2_with_cutout(cut_power=0.3, num_cutouts=args.augmentation_num_cutouts, noise_sd=args.augmentation_noise_sd,
                                                        noise_schedule=args.augmentation_noise_schedule,
                                                        noise_descending_steps=args.optim_params.sgd_steps)
        else:
            classifier = load_madry_l2()
        last_layer = classifier.model.fc
    elif 'convnext' in args.classifier.lower():
        if args.augmentation_num_cutouts > 0:
            classifier = load_timm_model_with_cutout(args.classifier, cut_power=0.3, num_cutouts=args.augmentation_num_cutouts,
                                                    noise_sd=args.augmentation_noise_sd,
                                                    noise_schedule=args.augmentation_noise_schedule,
                                                    noise_descending_steps=args.optim_params.sgd_steps)
        else:
            classifier = load_timm_model(args.classifier)
        last_layer = classifier.model.head.fc
    else:
        raise NotImplementedError()

    classifier.to(device)

    layer_activations = ActivationCaption(classifier, [last_layer])
    classifier_cam = None
    classifier_cam_target_layers = None

    result_folder_pre = f'{args.results_folder}'

    assert len(args.regularizers) == len(args.regularizers_ws)
    regularizers_weights = {}
    reg_string = ''
    for reg_name, reg_w in zip(args.regularizers, args.regularizers_ws):
        regularizers_weights[reg_name] = reg_w
        reg_string += f'_{reg_name}_{reg_w}'


    #setup losses
    losses_dicts = []
    loss_function = get_feature_loss_function(args.loss, classifier, layer_activations)
    losses_dicts.append(make_loss_dict(loss_function, args.loss, args.loss_weight))

    to_tensor = transforms.ToTensor()

    latent_dim = (pipe.unet.config.in_channels, args.resolution // pipe.vae_scale_factor, args.resolution // pipe.vae_scale_factor)
    torch.manual_seed(args.seed)
    num_classes = 1000
    deterministic_latents = torch.randn((num_classes, 50 if args.num_images < 50 else args.num_images) + latent_dim, dtype=torch.float)

    to_pil = transforms.ToPILImage()

    if args.results_sub_folder is None:
        output_description = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    else:
        output_description = args.results_sub_folder

    result_folder_pre = os.path.join(result_folder_pre, output_description)
    os.makedirs(result_folder_pre, exist_ok=True)
    print(f'Writing results to: {result_folder_pre}')

    with open(os.path.join(result_folder_pre, 'config.yaml'), "w") as f:
        OmegaConf.save(args, f)

    for target_i, (target_class, target_neuron) in enumerate(target_classes_neurons):
        target_label = IDX2NAME[target_class]

        result_folder_postfix = f'{target_class}_{target_label}_neuron_{target_neuron}'
        result_folder = os.path.join(result_folder_pre, result_folder_postfix)
        os.makedirs(result_folder, exist_ok=True)

        if target_label in foreground_replacements:
            prompt_foreground_key_words = [word for word in foreground_replacements[target_label].split()]
        else:
            prompt_foreground_key_words = [word for word in target_label.split()]

        for img_idx in range(args.num_images):
            linear_idx = target_i * args.num_images + img_idx
            if world_size > 1:
                current_split_idx = linear_idx % world_size
                if not current_split_idx == local_rank:
                    continue

            out_file_pth = os.path.join(result_folder, f'{img_idx}.pdf')
            
            if os.path.isfile(out_file_pth):
                continue
            
            target_preds = torch.zeros(args.num_images, dtype=torch.long)
            
            latent = deterministic_latents[target_class, img_idx][None,:]

            prompt_str = f'{args.prompt_template} {target_label}'

            targets_dict = {args.loss: [target_class, target_neuron]}
            return_values = pipe(targets_dict,
                                 starting_img=None,
                                 # loss and regs
                                 losses_dict=losses_dicts,
                                 regularizers_weights=regularizers_weights,
                                 # optim params
                                 optim_params=args.optim_params,
                                 null_text_embeddings=None,
                                 #segmentation based regularizers
                                 prompt_foreground_key_words = prompt_foreground_key_words,
                                 segmentation_args=args.segmentation_args,
                                 #
                                 mask_blend_params=args.mask_blend_params,
                                 # SD params
                                 height=args.resolution, width=args.resolution,
                                 num_inference_steps=args.num_ddim_steps,
                                 guidance_scale=args.guidance_scale,
                                 latents=latent,
                                 prompt=prompt_str)

            with torch.no_grad():
                img_grid = return_values['imgs']
                loss_scores = return_values['loss_scores']
                regularizer_scores = return_values['regularizer_scores']
                px_foreground_segmentation = return_values['px_foreground_segmentation']
                words_attention_masks = return_values['words_attention_masks']
                initial_img = return_values['initial_img']

                _, activations = calculate_neuron_activations(classifier, layer_activations, img_grid, device, target_neuron, loss)
                target_confs = calculate_confs(classifier, img_grid, device, target_class=target_class)

                min_loss_idx = torch.argmin(torch.FloatTensor(loss_scores['total'])).item()
                min_loss_img = img_grid[min_loss_idx]

                pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
                pil_sd = pil_sd.save(os.path.join(result_folder, f'{img_idx}_sd.png'))
                pil_ours = to_pil(torch.clamp(min_loss_img, 0, 1))
                pil_ours = pil_ours.save(os.path.join(result_folder, f'{img_idx}_ours.png'))

                title_attributes = {
                    f'Neuron {target_neuron}': activations,
                    target_label: target_confs
                }
                plot(initial_img, None, img_grid, title_attributes,
                     os.path.join(result_folder, f'{img_idx}.pdf'), loss_scores=loss_scores,  regularizer_scores=regularizer_scores)

                if px_foreground_segmentation is not None:
                    plot_attention(initial_img, px_foreground_segmentation, words_attention_masks,
                                   os.path.join(result_folder, f'{img_idx}_attention.pdf'))

