import math
import torch

from utils.models import load_classifier
from utils.attribution_with_gradient import ActivationCaption

import os
import argparse
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from torchvision.io import read_image
import matplotlib.pyplot as plt
from tqdm import tqdm

@dataclass
class NeuronArgs:
    gpu: int = 0
    seed: int = 42
    num_images: int = 10

    num_images_to_label: int = 5

    classifier: str = 'seresnet152d.ra2_in1k'
    target_neurons: List[List[int]] = field(default_factory=lambda: [[319,373,494,798], [507,571,784], [53, 530, 633, 899], [292, 424, 583, 618, 694, 770]])
    neuron_imgs_root: str = 'output_cvpr/imagenet_coqvlm_neurons'

    results_folder: str = 'output_cvpr/imagenet_coqvlm_neurons/similar_neurons'
    results_sub_folder: Optional[str] = None

def setup() -> NeuronArgs:
    default_config: NeuronArgs = OmegaConf.structured(NeuronArgs)
    cli_args = OmegaConf.from_cli()
    config: NeuronArgs = OmegaConf.merge(default_config, cli_args)
    return config

def load_image(path):
    img = read_image(path).float() / 255.
    return img

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

    if args.results_sub_folder is None:
        result_sub_folder = args.classifier
    else:
        result_sub_folder = args.results_sub_folder
    result_folder = os.path.join(args.results_folder, result_sub_folder)
    os.makedirs(result_folder, exist_ok=True)

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

    with torch.no_grad():
        for neuron_pair in tqdm(args.target_neurons):
            paired_imgs = []
            paired_imgs_activations = []
            for neuron_idx in neuron_pair:
                neuron_imgs_folder = os.path.join(args.neuron_imgs_root, args.classifier, f'neuron_{neuron_idx}')
                neuron_imgs = []
                for file in os.listdir(neuron_imgs_folder):
                    if '_ours' in file:
                        neuron_imgs.append(load_image(os.path.join(neuron_imgs_folder, file)))

                neuron_imgs = torch.stack(neuron_imgs, dim=0)
                paired_imgs.append(neuron_imgs)

                _ = layer_activations(neuron_imgs.to(device), augment=False)
                act = layer_activations.activations[0][0]

                paired_imgs_activations.append(act)

            scale_factor = 4.0
            num_cols = max([len(a) for a in paired_imgs])
            num_rows = len(paired_imgs)
            fig, axs = plt.subplots(num_rows, num_cols,
                                    figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
            out_file_path = os.path.join(result_folder, f"{'_'.join(map(str, neuron_pair))}.pdf")
            for i in range(num_rows):
                for j in range(num_cols):
                    ax = axs[i, j]
                    ax.axis('off')
                    img = paired_imgs[i][j].permute(1, 2, 0).cpu().detach().float()
                    ax.imshow(img, interpolation='lanczos')
                    title = f''
                    for lin_idx, neuron_idx in enumerate(neuron_pair):
                        if lin_idx > 0:
                            title += '\n'
                        title += f'Neuron {neuron_idx}: {paired_imgs_activations[i][j, neuron_idx]:.4f}'
                    ax.set_title(title)

            plt.tight_layout()
            fig.savefig(out_file_path)
            plt.close(fig)


if __name__ == "__main__":
    main()
