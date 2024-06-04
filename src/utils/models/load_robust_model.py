import torch
import torchvision.models as torch_models
import os
from .model_wrappers import NormalizationAndResizeWrapper, NormalizationAndCutoutWrapper,\
    IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

def load_madry_l2():
    import torch

    model = torch_models.resnet50()
    state_dict_file = f'madry_model/madry_l2.pt'
    state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = NormalizationAndResizeWrapper(model, 224,
                                          mean=torch.FloatTensor(IMAGENET_DEFAULT_MEAN),
                                          std=torch.FloatTensor(IMAGENET_DEFAULT_STD))
    model.eval()
    return model
def load_madry_l2_with_cutout(cut_power, num_cutouts, noise_sd=0, noise_schedule='constant',
                                noise_descending_steps=None,
                                batches=1, checkpointing=False):

    model = torch_models.resnet50()
    state_dict_file = f'madry_model/madry_l2.pt'
    state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = NormalizationAndCutoutWrapper(model, size=224,mean=torch.FloatTensor(IMAGENET_DEFAULT_MEAN),
                                          std=torch.FloatTensor(IMAGENET_DEFAULT_STD), cut_power=cut_power,
                                          noise_sd=noise_sd, noise_schedule=noise_schedule,
                                          noise_descending_steps=noise_descending_steps,
                                          num_cutouts=num_cutouts, checkpointing=checkpointing, batches=batches)
    model.eval()
    return model
