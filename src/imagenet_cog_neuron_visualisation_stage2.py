import math
import torch

from diffusers import DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler

import os
import argparse
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from utils.models import load_classifier
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad, MaskBlendParams
from utils.plotting_utils import plot, plot_attention
from utils.loss_functions import get_feature_loss_function, calculate_neuron_activations, make_loss_dict, calculate_confs
from utils.datasets.inet_classes import IDX2NAME
from utils.parser_utils import CommonArguments
from utils.foreground_segmentation import SegmentationArgs
from utils.attribution_with_gradient import ActivationCaption

from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

DEFAULT_TARGET_NEURONS = [310, 312, 318, 319, 322, 334, 338, 373, 386, 402, 410, 412, 424, 434, 473, 474, 476, 478, 483,
                          446, 494, 495, 498, 505, 507, 530, 534, 553, 571, 583, 589, 593, 595, 599, 608, 618, 619, 623,
                          633, 682, 694, 707, 720, 725, 754, 767, 770, 777, 784, 798, 799, 816, 870, 899, 908, 910, 914,
                          919, 924, 932, 53, 1700, 462, 10, 13, 56, 58, 60, 68, 71, 73, 90, 138, 150, 168, 291, 292]

@dataclass
class NeuronArgs(CommonArguments):
    gpu: int = 0
    seed: int = 42
    num_images: int = 10

    classifier: str = 'seresnet152d.ra2_in1k'
    prompt_template: str = 'a photograph of a'
    min_word_count: int = 1
    sd_batch_size: int = 8
    target_neurons: List[int] = field(default_factory=lambda: DEFAULT_TARGET_NEURONS)

    results_folder: str = 'output_cvpr/imagenet_cogvlm_neurons'
    coq_responses_folder: str = 'output_cvpr/imagenet_coqvlm_neurons/seresnet152d.ra2_in1k'
    results_sub_folder: Optional[str] = None

    mask_blend_params: Optional[MaskBlendParams] = None
    segmentation_args: Optional[SegmentationArgs] = None

    solver: str = 'ddim'
    loss: str = 'neuron_activation'
    loss_weight: float = 1.0

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])

def setup() -> NeuronArgs:
    default_config: NeuronArgs = OmegaConf.structured(NeuronArgs)
    cli_args = OmegaConf.from_cli()
    config: NeuronArgs = OmegaConf.merge(default_config, cli_args)
    return config

def plot_sd_random_search(word_sd_images, word_sd_activations, words, neuron_idx, out_file_path):
    num_rows = len(word_sd_images)
    num_cols = len(next(iter(word_sd_activations.values())))

    scale_factor = 2.0
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
    axs = axs.reshape((num_rows, num_cols))
    for i, word in enumerate(words):
        word_images = word_sd_images[word]
        word_activations = word_sd_activations[word]

        for j in range(num_cols):
            ax = axs[i, j]
            ax.axis('off')
            img = word_images[j].permute(1, 2, 0).cpu().detach().float()
            ax.imshow(img, interpolation='lanczos')
            title = f'{word}\nNeuron{neuron_idx}: {word_activations[j]:.4f}'
            ax.set_title(title)

    plt.tight_layout()
    fig.savefig(out_file_path)
    plt.close(fig)

@torch.no_grad()
def calculate_top_word(args, layer_activations, device, local_rank, world_size):
    pipe = DiffusionPipeline.from_pretrained(args.model_path, use_safetensors=True, torch_dtype=torch.float16)
    if args.solver == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise NotImplementedError()

    pipe.to(device)
    pipe.safety_checker = None
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.results_sub_folder is None:
        result_sub_folder = args.classifier
    else:
        result_sub_folder = args.results_sub_folder
    result_folder = os.path.join(args.results_folder, result_sub_folder)
    os.makedirs(result_folder, exist_ok=True)

    top_words_per_neuron = {}

    for linear_idx, neuron_idx in enumerate(args.target_neurons):
        if world_size > 1:
            current_split_idx = linear_idx % world_size
            if not current_split_idx == local_rank:
                continue

        coq_words = {}
        coq_dicts = torch.load(os.path.join(args.coq_responses_folder, f'neuron_{neuron_idx}_top_images_coq_responses.pt'))
        for coq_dict in coq_dicts:
            coq_response = coq_dict['response']
            if not '[' in coq_response and ']' in coq_response:
                print(f'Warning: [] not in response: {coq_response}')
                continue

            response_words = coq_response.split('[')[1].split(']')[0].split(',')
            for i in range(len(response_words)):
                response_words[i] = response_words[i].lower()
            for word in response_words:
                if word in coq_words:
                    coq_words[word] = coq_words[word] + 1
                else:
                    coq_words[word] = 1

        word_mean_activation = {}
        word_sd_images = {}
        word_sd_activations = {}
        for word, word_count in coq_words.items():
            if word_count < args.min_word_count:
                continue

            prompt = f'{args.prompt_template} {word}'
            images = pipe(prompt, num_images_per_prompt=args.sd_batch_size, num_inference_steps=args.num_ddim_steps,
                          output_type='pt').images
            _ = layer_activations(images, augment=False)
            act = layer_activations.activations[0][0]

            neuron_acts = act[:, neuron_idx]
            mean_act = torch.mean(neuron_acts)
            word_mean_activation[word] = mean_act.item()
            word_sd_images[word] = images.cpu().detach()
            word_sd_activations[word] = neuron_acts.detach().cpu()

        if len(word_mean_activation) == 0:
            print('Did not find a word with min_word_count - skipping')
            continue

        words = []
        activations = []

        for word, word_mean_activation in word_mean_activation.items():
            words.append(word)
            activations.append(word_mean_activation)

        activations = torch.tensor(activations)
        sort_ids = torch.argsort(activations, descending=True)
        words_sorted = []
        for sort_idx in sort_ids:
            words_sorted.append(words[sort_idx])

        top_words_per_neuron[neuron_idx] = words_sorted

        print(f'Maximally activating word {words[sort_ids[0]]}: {activations[sort_ids[0]]:.4f}')
        out_file_path = os.path.join(result_folder, f'neuron_{neuron_idx}_word_sd_search.png')
        plot_sd_random_search(word_sd_images, word_sd_activations, words_sorted, neuron_idx, out_file_path)

    return top_words_per_neuron


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

    device = torch.device(args.gpu)

    classifier = load_classifier(args.classifier, device,
                                 num_cutouts=args.augmentation_num_cutouts, noise_sd=args.augmentation_noise_sd)
    if args.classifier == 'madry_l2' or 'resnet' in args.classifier:
        last_layer = classifier.model.fc
    elif 'convnext' in args.classifier.lower():
        last_layer = classifier.model.head.fc
    elif 'vit' in args.classifier.lower():
        last_layer = classifier.model.head
    elif 'resnet' in args.classifier.lower():
        last_layer = classifier.model.head
    else:
        raise NotImplementedError()

    layer_activations = ActivationCaption(classifier, [last_layer])

    top_words_per_neuron = calculate_top_word(args, layer_activations, device, local_rank, world_size)

    if args.results_sub_folder is None:
        result_sub_folder = args.classifier
    else:
        result_sub_folder = args.results_sub_folder
    result_folder = os.path.join(args.results_folder, result_sub_folder)
    os.makedirs(result_folder, exist_ok=True)

    pipe = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path, device_map=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # setup losses
    losses_dicts = []
    loss_function = get_feature_loss_function(args.loss, classifier, layer_activations)
    losses_dicts.append(make_loss_dict(loss_function, args.loss, args.loss_weight))

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    latent_dim = (pipe.unet.config.in_channels, args.resolution // pipe.vae_scale_factor, args.resolution // pipe.vae_scale_factor)
    torch.manual_seed(args.seed)
    deterministic_latents = torch.randn((50 if len(args.target_neurons) < 50 else len(args.target_neurons), 50 if args.num_images < 50 else args.num_images) + latent_dim, dtype=torch.float)

    for linear_idx, neuron_idx in enumerate(args.target_neurons):
        if world_size > 1:
            current_split_idx = linear_idx % world_size
            if not current_split_idx == local_rank:
                continue

        neuron_result_folder = os.path.join(result_folder, f'neuron_{neuron_idx}')
        os.makedirs(neuron_result_folder, exist_ok=True)
        neuron_top_word = top_words_per_neuron[neuron_idx][0]

        print(f'Generating for neuron {neuron_idx} - {neuron_top_word}')
        for img_idx in range(args.num_images):
            out_file_path = os.path.join(neuron_result_folder, f"neuron_{neuron_idx}_{neuron_top_word.replace(' ', '')}_{img_idx}.pdf")

            latent = deterministic_latents[linear_idx, img_idx][None, :]
            prompt_str = f'{args.prompt_template} {neuron_top_word}'

            targets_dict = {args.loss: [None, neuron_idx]}
            return_values = pipe(targets_dict,
                                 starting_img=None,
                                 # loss and regs
                                 losses_dict=losses_dicts,
                                 regularizers_weights=None,
                                 # optim params
                                 optim_params=args.optim_params,
                                 null_text_embeddings=None,
                                 # segmentation based regularizers
                                 prompt_foreground_key_words=None,
                                 segmentation_args=args.segmentation_args,
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
                initial_img = return_values['initial_img']

                _, activations = calculate_neuron_activations(classifier, layer_activations, img_grid, device,
                                                              neuron_idx, args.loss)
                min_loss_idx = torch.argmin(torch.FloatTensor(loss_scores['total'])).item()
                min_loss_img = img_grid[min_loss_idx]

                pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
                pil_sd.save(os.path.join(neuron_result_folder, f'{img_idx}_sd.png'))
                pil_ours = to_pil(torch.clamp(min_loss_img, 0, 1))
                pil_ours.save(os.path.join(neuron_result_folder, f'{img_idx}_ours.png'))

                title_attributes = {
                    f'Neuron {neuron_idx}': activations,
                }
                plot(initial_img, None, img_grid, title_attributes,
                     out_file_path, loss_scores=loss_scores,
                     regularizer_scores=regularizer_scores)


if __name__=="__main__":
    main()