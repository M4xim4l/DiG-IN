import torch
from diffusers import DDIMScheduler
import os
from omegaconf import OmegaConf
import wandb
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from typing import List, Optional, Union, Iterable
from dataclasses import dataclass, field

from utils.models import load_classifier
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad, PromptToPromptParams, MaskBlendParams
from utils.plotting_utils import plot, plot_attention, plot_p2p_xa_schedule_search
from utils.loss_functions import get_loss_function, calculate_confs, make_loss_dict
from utils.loss_functions import get_feature_loss_function, calculate_neuron_activations
from utils.datasets.inet_classes import IDX2NAME as in_labels
from utils.datasets.inet_hierarchy import find_cluster_for_class_idx
from utils.parser_utils import CommonArguments
from utils.foreground_segmentation import SegmentationArgs
from utils.wandb_utils import make_wandb_run
from utils.attribution_with_gradient import ActivationCaption

@dataclass
class UVCEBlend(MaskBlendParams):
    do_mask_blend: bool = True
    foreground_blend: bool = True
    optimize_mask_blend: bool = False
    initial_mask_blend_schedule: List[str]  = field(default_factory=lambda: [
        'threshold_0.3',
    ])
    mask_blend_stepsize: float = 0.05

@dataclass
class UVCEP2P(PromptToPromptParams):
    do_p2p: bool = True
    xa_interpolation_schedule: Optional[List[str]] =  field(default_factory=lambda: [
        'threshold_0.0_0.3',
    ])
    self_interpolation_schedule: Optional[List[str]] = None
    optimize_xa_interpolation_schedule: bool = False
    optimize_self_interpolation_schedule: bool = False
    schedule_stepsize: float = 0.05

DEFAULT_CLASS_NEURON_PAIRS = [(83, 565), (120, 870), (537, 0), (146, 1697), (2, 1697), (576, 1772), (14, 660),
                              (50, 341), (309, 595), (309, 1797), (537, 579), (537, 1855), (695, 1567), (695, 1635)]

@dataclass
class CounterfactualArgs(CommonArguments):
    gpu: int = 0
    seed: int = 42
    num_images_per_neuron: Optional[int] = None

    classifier: str = 'madry_l2'

    target_class_neurons: List[List[int]] = field(default_factory=lambda: DEFAULT_CLASS_NEURON_PAIRS)
    additional_evaluation_classes: List[List[int]] = field(default_factory=lambda: [])

    results_folder: str = 'output_cvpr/imagenet_neuron_counterfactuals'
    results_sub_folder: Optional[str] = None
    inversion_folder: str = 'output_cvpr/imagenet_inversions'
    imagenet_folder: str = '/mnt/datasets/imagenet'

    loss: str = 'neuron_activation'
    loss_weight: float = 1.0

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])

    segmentation_args: SegmentationArgs = SegmentationArgs()
    p2p_params: Optional[PromptToPromptParams] = UVCEP2P()
    mask_blend_params: MaskBlendParams = UVCEBlend()

foreground_replacements = {
    'fiddler crab': 'crab',
    'prairie chicken': 'chicken',
    'howler monkey': 'monkey',
    'great white shark': 'shark',
    'American alligator': 'alligator',
    'bullfrog': 'frog',
    'house finch': 'finch',
}

def fill_img_idcs(target_class_neurons, inversion_folder, num_ddim_steps, guidance_scale, additional_evaluation_classes, num_images_per_neuron=None):
    class_img_idcs = {}
    target_classes = [a for (a,b) in target_class_neurons]
    for class_idx in sorted(set(target_classes)):
        start_label = in_labels[class_idx]
        start_class_folder = os.path.join(inversion_folder, f'{class_idx}_{start_label}')
        inversion_dir = os.path.join(start_class_folder, f"inversion_{num_ddim_steps}_{guidance_scale}")

        if not os.path.isdir(inversion_dir):
            print(f'Warning: No inversion dir {inversion_dir}')
            continue
        for file in os.listdir(inversion_dir):
            # null text file format: {IMG_IDX}_null_texts.pt
            if 'null_texts.pt' in file:
                img_idx = int(file.split('_')[0])

                if not class_idx in class_img_idcs:
                    class_img_idcs[class_idx] = []
                class_img_idcs[class_idx].append(img_idx)

    class_neuron_img_idcs = []
    eval_classes_img_idcs = []
    for linear_idx, (class_idx, neuron_idx) in enumerate(target_class_neurons):
        class_img_files = class_img_idcs.get(class_idx, [])
        if num_images_per_neuron is None:
            class_num_images = len(class_img_files)
        elif num_images_per_neuron > len(class_img_files):
            class_num_images = len(class_img_files)
            print(f'Warning:  {class_idx} ({in_labels[class_idx]})Found {class_num_images} inverted images but requested {num_images_per_neuron}')
        else:
            class_num_images = num_images_per_neuron

        random_idcs = torch.randperm(len(class_img_files))[:class_num_images]
        for random_idx in random_idcs:
            img_idx = class_img_files[random_idx]
            class_neuron_img_idcs.append((class_idx, neuron_idx, img_idx))
            if additional_evaluation_classes is not None:
                try:
                    eval_classes_img_idcs.append(additional_evaluation_classes[linear_idx])
                except:
                    pass

    return class_neuron_img_idcs, eval_classes_img_idcs

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

    classifier = load_classifier(args.classifier, device,
                                 num_cutouts=0, noise_sd=0.0)
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

    pipe: StableDiffusionPipelineWithGrad = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.use_double:
        pipe.convert_to_double()

    pipe.to(device)

    #setup losses
    losses_dicts = []
    loss_function = get_feature_loss_function(args.loss, classifier, layer_activations)
    losses_dicts.append(make_loss_dict(loss_function, args.loss, args.loss_weight))

    to_tensor = transforms.ToTensor()

    assert len(args.regularizers) == len(args.regularizers_ws)
    regularizers_weights = {}
    reg_string = ''
    for reg_name, reg_w in zip(args.regularizers, args.regularizers_ws):
        regularizers_weights[reg_name] = reg_w
        reg_string += f'_{reg_name}_{reg_w}'

    to_pil = transforms.ToPILImage()

    inversion_root_dir = os.path.join(args.inversion_folder, args.model_path)
    class_neuron_img_idcs, additional_evaluation_classes = fill_img_idcs(args.target_class_neurons, inversion_root_dir,
                                                                         args.num_ddim_steps, args.guidance_scale,
                                                                         args.additional_evaluation_classes,
                                                                         args.num_images_per_neuron)

    # datetime object containing current date and time
    if args.results_sub_folder is None:
        output_description = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    else:
        output_description = args.results_sub_folder

    result_folder_pre = os.path.join(args.results_folder, output_description)
    os.makedirs(result_folder_pre, exist_ok=True)
    with open(os.path.join(result_folder_pre, 'config.yaml'), "w") as f:
        OmegaConf.save(args, f)

    make_wandb_run(args.wandb_project, output_description, OmegaConf.to_container(args))

    for linear_idx, (target_class, target_neuron, img_idx) in enumerate(class_neuron_img_idcs):
        if world_size > 1:
            current_split_idx = linear_idx % world_size
            if not current_split_idx == local_rank:
                continue

        target_label = in_labels[target_class]

        if args.p2p_params is not None and args.p2p_params.do_p2p:
            prompt_to_prompt_replacements = (target_label, target_label)
        else:
            prompt_to_prompt_replacements = None

        start_class_folder = os.path.join(inversion_root_dir, f'{target_class}_{target_label}')
        inversion_dir = os.path.join(start_class_folder,
                                     f"inversion_{args.num_ddim_steps}_{args.guidance_scale}")

        result_folder_postfix = f'{target_class}_{target_label}_neuron_{target_neuron}'
        result_folder = os.path.join(result_folder_pre, result_folder_postfix)
        os.makedirs(result_folder, exist_ok=True)

        out_pdf = os.path.join(result_folder, f'{img_idx}.pdf')
        if os.path.isfile(out_pdf):
            continue

        latent = torch.load(os.path.join(inversion_dir, f'{img_idx}.pt'), map_location='cpu')[None, :]
        null_texts_embeddings = torch.load(os.path.join(inversion_dir, f'{img_idx}_null_texts.pt'), map_location='cpu')
        assert len(null_texts_embeddings) == args.num_ddim_steps

        if args.use_double:
            latent = latent.double()
            null_texts_embeddings = null_texts_embeddings.double()

        captions_file = os.path.join(inversion_dir,  f'{img_idx}_prompt.txt')
        if os.path.isfile(captions_file):
            with open(captions_file, 'r') as f:
                prompt_str = f.read()
        else:
            print(f'Warning: Could not load caption from {captions_file}')
            continue

        if target_label in foreground_replacements:
            prompt_foreground_key_words = [word for word in foreground_replacements[target_label].split()]
        else:
            prompt_foreground_key_words = [word for word in target_label.split()]

        original_img = Image.open(os.path.join(inversion_dir, f'{img_idx}_original.png'))
        original_tensor = to_tensor(original_img).to(device)
        if args.use_double:
            original_tensor = original_tensor.double()

        try:
            targets_dict = {args.loss: [target_class, target_neuron]}
            return_values = pipe(targets_dict,
                                 starting_img=original_tensor,
                                 # loss and regs
                                 losses_dict=losses_dicts,
                                 regularizers_weights=regularizers_weights,
                                 #optim params
                                 optim_params = args.optim_params,
                                 null_text_embeddings=null_texts_embeddings,
                                 #segmentation based regularizers
                                 prompt_foreground_key_words = prompt_foreground_key_words,
                                 segmentation_args=args.segmentation_args,
                                 #Prompt-To-Prompt params
                                 p2p_params = args.p2p_params,
                                 p2p_replacements=prompt_to_prompt_replacements,
                                 # Mask-blend params
                                 mask_blend_params=args.mask_blend_params,
                                 #SD params
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

                min_loss_idx = torch.argmin(torch.FloatTensor(loss_scores['total'])).item()

                min_loss_img = img_grid[min_loss_idx]

                _, activations = calculate_neuron_activations(classifier, layer_activations, img_grid, device,
                                                                       target_neuron, args.loss)
                target_confs = calculate_confs(classifier, img_grid, device, target_class=target_class)

                title_attributes = {
                    f'Neuron {target_neuron}': activations,
                    target_label: target_confs
                }

                try:
                    for add_eval_class_idx in additional_evaluation_classes[linear_idx]:
                        add_eval_label = in_labels[add_eval_class_idx]
                        add_eval_confs = calculate_confs(classifier, img_grid, device, target_class=add_eval_class_idx)
                        title_attributes[add_eval_label] = add_eval_confs
                except:
                    pass

                pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
                pil_sd.save(os.path.join(result_folder, f'{img_idx}_sd.png'))
                pil_ours = to_pil(torch.clamp(min_loss_img, 0, 1))
                pil_ours.save(os.path.join(result_folder, f'{img_idx}_ours.png'))

                plot(original_tensor, initial_img, img_grid, title_attributes, out_pdf, loss_scores=loss_scores, regularizer_scores=regularizer_scores)
                if px_foreground_segmentation is not None:
                    plot_attention(initial_img, px_foreground_segmentation, words_attention_masks,
                                   os.path.join(result_folder, f'{img_idx}_attention.pdf'))

        except KeyError as e:
            print(e)
            pass

if __name__=="__main__":
    main()