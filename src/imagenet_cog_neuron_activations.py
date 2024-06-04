import torch

import os
import argparse
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from utils.models import load_classifier
from utils.attribution_with_gradient import ActivationCaption
from utils.datasets.inet_classes import IDX2NAME as in_labels
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet
from tqdm import tqdm

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

@dataclass
class NeuronArgs:
    gpu: int = 0
    batch_size: int = 128
    num_workers: int = 32

    augmentation_num_cutouts: int = 0
    augmentation_noise_sd: float = 0.0
    augmentation_noise_schedule: str = 'const'

    imgs_per_class: Optional[int] = None
    classifier: str = 'seresnet152d.ra2_in1k'

    imagenet_folder: str = '/mnt/datasets/imagenet'
    results_folder: str = 'output_cvpr/imagenet_neuron_statistics'
    results_sub_folder: Optional[str] = None

def compute_cam_map(img, classifier, cam_layer, layer_activations, neuron_idx, device, augment=True):
    class ActivationClassifier(torch.nn.Module):
        def __init__(self, classifier, activation_classifier):
            super().__init__()
            self.classifier = classifier
            self.activation_classifier = activation_classifier

        def forward(self, x):
            bs = len(x)
            y = self.activation_classifier(x, augment=augment)
            activations = layer_activations.activations[0][0]
            if augment and bs > 1:
                raise NotImplementedError()
            elif augment:
                activations = torch.mean(activations, dim=0, keepdim=True)
            return activations

    activation_classifier = ActivationClassifier(classifier, layer_activations)
    cam_target_layer = [cam_layer]
    classifier_cam = HiResCAM(model=activation_classifier, target_layers=cam_target_layer)

    # compute cam
    for layer in cam_target_layer:
        for param in layer.parameters():
            param.requires_grad_(True)

    targets = [ClassifierOutputTarget(neuron_idx)]
    with torch.enable_grad():
        cam_map = classifier_cam(input_tensor=img[None, :].to(device).requires_grad_(True), targets=targets, aug_smooth=True)
    classifier_imgs_cams_maps = torch.from_numpy(cam_map)
    if len(classifier_imgs_cams_maps) > 1:
        classifier_imgs_cams_maps = torch.mean(classifier_imgs_cams_maps, dim=0, keepdim=True)
    # C H W -> H W C
    cam_np = classifier_imgs_cams_maps.detach().expand(3, -1, -1).cpu().permute(1, 2, 0).numpy()

    colormap = cv2.COLORMAP_JET
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    for layer in cam_target_layer:
        for param in layer.parameters():
            param.grad = None
            param.requires_grad_(False)

    return heatmap

class ImageNetSubset(ImageNet):
    def __init__(self, root, imgs_per_class=None, split='train', transform=None):
        super().__init__(root, split=split, transform=transform)

        if imgs_per_class is not None:
            targets = torch.tensor(self.targets)
            self.subset_idcs = []
            for class_idx in range(1000):
                matching_idcs = torch.nonzero(targets == class_idx, as_tuple=False).squeeze()
                self.subset_idcs.append(matching_idcs[:min(imgs_per_class, len(matching_idcs))])

            self.subset_idcs = torch.cat(self.subset_idcs)
        else:
            self.subset_idcs = torch.arange(super().__len__(), dtype=torch.long)

    def __len__(self):
        return len(self.subset_idcs)

    def __getitem__(self, idx):
        subset_idx = self.subset_idcs[idx]
        return super().__getitem__(subset_idx.item())

    def get_filename_from_idx(self, idx):
        subset_idx = self.subset_idcs[idx]
        filename = self.imgs[subset_idx][0]
        return filename


def setup() -> NeuronArgs:
    default_config: NeuronArgs = OmegaConf.structured(NeuronArgs)
    cli_args = OmegaConf.from_cli()
    config: NeuronArgs = OmegaConf.merge(default_config, cli_args)
    return config

def main():
    args = setup()
    device = torch.device(f'cuda:{args.gpu}')

    classifier = load_classifier(args.classifier, device, args.augmentation_num_cutouts, args.augmentation_noise_sd)

    if args.classifier == 'madry_l2' or 'resnet' in args.classifier:
        last_layer = classifier.model.fc
        cam_layer = classifier.model.layer4[-1]
    elif 'convnext' in args.classifier.lower():
        last_layer = classifier.model.head.fc
        cam_layer = classifier.model.norm_pre
    elif 'vit' in args.classifier.lower():
        last_layer = classifier.model.head
        cam_layer = classifier.model.blocks[-1].norm1
    else:
        raise NotImplementedError()

    classifier.to(device)

    layer_activations = ActivationCaption(classifier, [last_layer])

    classifier_resolution = classifier.size
    val_transform = transforms.Compose([
        transforms.Resize(int(1.25 * classifier_resolution)),
        transforms.CenterCrop(classifier_resolution),
        transforms.ToTensor()
    ])

    dataset = ImageNetSubset(args.imagenet_folder, args.imgs_per_class, split='train', transform=val_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    if args.results_sub_folder is None:
        result_sub_folder = args.classifier
    else:
        result_sub_folder = args.results_sub_folder
    result_folder = os.path.join(args.results_folder, result_sub_folder)
    os.makedirs(result_folder, exist_ok=True)

    activations_file = os.path.join(result_folder, f'classifier_activations_{args.imgs_per_class}.pt')

    if not os.path.isfile(activations_file):
        class_activations_img_idcs = {}
        img_idx = 0
        with torch.no_grad():
            for data, target in tqdm(loader):
                data = data.to(device)
                _ = layer_activations(data, augment=False)
                act = layer_activations.activations[0][0]

                act = act.cpu()
                for i, t in enumerate(target):
                    t = t.item()
                    if t not in class_activations_img_idcs:
                        class_activations_img_idcs[t] = []

                    class_activations_img_idcs[t].append((act[i].cpu(), img_idx))
                    img_idx += 1

        torch.save(class_activations_img_idcs, activations_file)
    else:
        class_activations_img_idcs = torch.load(activations_file)

    #this code plots the top neurons for each class - not required for CogVLM visualisations
    if False:
        num_neurons_per_class = 5
        num_images_per_neuron_per_class = 10
        with torch.no_grad():
            for class_idx in tqdm(class_activations_img_idcs.keys()):
                class_imgs_stacked = [img_idx   for (_, img_idx) in class_activations_img_idcs[class_idx]]
                class_activations_stacked = torch.stack([acts   for (acts, _) in class_activations_img_idcs[class_idx]])
                class_fc = last_layer.weight[class_idx, :].cpu()
                class_activations_stacked_weighted = class_activations_stacked * class_fc[None, :]
                mean_activation = torch.mean(class_activations_stacked, dim=0)
                mean_activation_weighted = torch.mean(class_activations_stacked_weighted, dim=0)

                max_neurons = torch.argsort(mean_activation_weighted, descending=True)[:num_neurons_per_class]

                caption_file = os.path.join(result_folder, f'{class_idx}_{in_labels[class_idx]}_top_neurons.txt')
                with open(caption_file, 'w') as f:
                    for i in range(num_neurons_per_class):
                        neuron_idx = max_neurons[i]
                        max_activating_imgs = torch.argsort(class_activations_stacked[:, neuron_idx], descending=True)
                        f.write(f'{neuron_idx.item()} - Mean activation: {mean_activation[neuron_idx]:.5f} - Weight {class_fc[class_idx]:.5f} - Mean weighted: {mean_activation_weighted[neuron_idx]:.5f}\n')
                        for j in range(num_images_per_neuron_per_class):
                            img_idx = class_imgs_stacked[max_activating_imgs[j].item()]
                            img_filename = dataset.get_filename_from_idx(img_idx)
                            f.write(f'{img_filename}\t{class_activations_stacked[max_activating_imgs[j], neuron_idx]:.8f}\n')

                #plot top scoring images
                neuron_plot_file = os.path.join(result_folder, f'{class_idx}_{in_labels[class_idx]}_top_neurons.png')

                num_cols = num_images_per_neuron_per_class
                num_rows = 2 * num_neurons_per_class
                scale_factor = 2.0
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))

                for i in range(num_neurons_per_class):
                    neuron_idx = max_neurons[i]
                    mean_neuron_i = mean_activation[neuron_idx]
                    max_activating_imgs = torch.argsort(class_activations_stacked[:, neuron_idx], descending=True)
                    for j in range(num_cols):
                        img_idx = class_imgs_stacked[max_activating_imgs[j]]
                        img, _ = dataset[img_idx]
                        img_np = img.cpu().permute(1, 2, 0).numpy()
                        heatmap = compute_cam_map(img, classifier, cam_layer, layer_activations, neuron_idx, device)
                        ax = axs[2*i,j]
                        ax.axis('off')
                        img = img.permute(1, 2, 0).cpu().detach()
                        ax.imshow(img, interpolation='lanczos')
                        if j == 0:
                            title = f'Weight {neuron_idx}: {class_fc[neuron_idx]:.4f}\n'
                        else:
                            title = ''

                        title += f'Weighted: {class_activations_stacked_weighted[max_activating_imgs[j], neuron_idx]:.4f}\n'
                        title += f'Neuron {neuron_idx}: {class_activations_stacked[max_activating_imgs[j], neuron_idx]:.4f}'

                        ax.set_title(title)

                        ax = axs[2*i+1,j]
                        ax.axis('off')
                        ax.imshow(0.5 * heatmap + 0.5 * img_np, interpolation='lanczos')

                plt.tight_layout()
                fig.savefig(neuron_plot_file)
                plt.close(fig)

    activations_stacked = []
    img_idcs_stacked = []

    for class_idx, class_act_idcs in class_activations_img_idcs.items():
        for act, img_idx in class_act_idcs:
            activations_stacked.append(act)
            img_idcs_stacked.append(img_idx)

    activations_stacked = torch.stack(activations_stacked, dim=0)

    num_classifier_neurons = activations_stacked.shape[1]
    num_imgs_per_neuron = 20
    with torch.no_grad():
        for neuron_idx in tqdm(range(num_classifier_neurons)):
            mean_neuron_i = torch.mean(activations_stacked[:, neuron_idx])
            max_activating_imgs = torch.argsort(activations_stacked[:, neuron_idx], descending=True)[:num_imgs_per_neuron]

            caption_file = os.path.join(result_folder, f'neuron_{neuron_idx}_top_images.txt')

            with open(caption_file, 'w') as f:
                for j in range(num_imgs_per_neuron):
                    img_idx = img_idcs_stacked[max_activating_imgs[j].item()]
                    img_filename = dataset.get_filename_from_idx(img_idx)
                    f.write(f'{img_filename}\t{activations_stacked[max_activating_imgs[j], neuron_idx]:.8f}\n')

            neuron_plot_file = os.path.join(result_folder, f'neuron_{neuron_idx}_top_images.png')
            #plot top scoring images
            num_cols = num_imgs_per_neuron
            num_rows = 2
            scale_factor = 2.0
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))


            for j in range(num_cols):
                img_idx = img_idcs_stacked[max_activating_imgs[j].item()]
                img, class_idx = dataset[img_idx]
                class_label = in_labels[class_idx]
                img_np = img.cpu().permute(1, 2, 0).numpy()
                heatmap = compute_cam_map(img, classifier, cam_layer, layer_activations, neuron_idx, device)
                ax = axs[0,j]
                ax.axis('off')
                img = img.permute(1, 2, 0).cpu().detach()
                ax.imshow(img, interpolation='lanczos')
                ax.set_title(
                    f'Neuron {neuron_idx}: {activations_stacked[max_activating_imgs[j], neuron_idx]:.4f}\nClass: {class_label}')

                ax = axs[1,j]
                ax.axis('off')
                ax.imshow(0.5 * heatmap + 0.5 * img_np, interpolation='lanczos')

            plt.tight_layout()
            fig.savefig(neuron_plot_file)
            plt.close(fig)

if __name__=="__main__":
    main()

