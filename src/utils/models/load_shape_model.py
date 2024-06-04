"""
Read PyTorch model from .pth.tar checkpoint.
"""
import os
import sys
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
from torch.utils import model_zoo
from .model_wrappers import NormalizationAndResizeWrapper, NormalizationAndCutoutWrapper, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def _load_texture_vs_shape_model(model_name):
    model_urls = {
        'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
        'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
        'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    }

    if "resnet50" in model_name:
        print("Using the ResNet50 architecture.")
        model = torchvision.models.resnet50(pretrained=False)
        checkpoint = model_zoo.load_url(model_urls[model_name])
        image_size = 224
    else:
        raise ValueError("unknown model architecture.")

    #remove module created by DataParallel
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = val

    model.load_state_dict(new_state_dict)
    return model, image_size

def load_sin_model():
    model, size = _load_texture_vs_shape_model('resnet50_trained_on_SIN')
    model = NormalizationAndResizeWrapper(model, size=size, mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                               std=torch.tensor(IMAGENET_DEFAULT_STD))
    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model

def load_sin_model_with_cutout(cut_power, num_cutouts, noise_sd=0, noise_schedule='constant',
                                noise_descending_steps=None,
                               batches=1, checkpointing=False):

    model, size = _load_texture_vs_shape_model('resnet50_trained_on_SIN')
    model = NormalizationAndCutoutWrapper(model, size=size, mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                               std=torch.tensor(IMAGENET_DEFAULT_STD), cut_power=cut_power,
                                          noise_sd=noise_sd,noise_schedule=noise_schedule, noise_descending_steps=noise_descending_steps,
                                          num_cutouts=num_cutouts, checkpointing=checkpointing, batches=batches)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
