from .model_wrappers import NormalizationAndCutoutWrapper, NormalizationAndResizeWrapper, \
    NormalizationAndRandomNoiseWrapper, IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

import torch
from .mae_models_vit import vit_large_patch16

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    if arch == 'mae_vit_large_patch16':
        model = vit_large_patch16(drop_path_rate=0.2,
        global_pool=True)
    else:
        raise NotImplementedError()

    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def load_cub_mae_with_cutout(cut_power, num_cutouts, checkpointing=False, auto_resize=True,
                             noise_sd=0, noise_schedule='constant', noise_descending_steps=None):
    chkpt_dir = 'mae_models/cub_mae_large/checkpoint-49.pth'
    model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')

    if auto_resize:
        img_size = 224
    else:
        img_size = None

    model = NormalizationAndCutoutWrapper(model, size=img_size, mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                          std=torch.tensor(IMAGENET_DEFAULT_STD),
                                          cut_power=cut_power, num_cutouts=num_cutouts, checkpointing=checkpointing,
                                          noise_sd=noise_sd, noise_schedule=noise_schedule, noise_descending_steps=noise_descending_steps)

    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_cub_mae(auto_resize=True):
    chkpt_dir = 'mae_models/cub_mae_large/checkpoint-49.pth'
    model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')

    if auto_resize:
        img_size = 224
    else:
        img_size = None

    model = NormalizationAndResizeWrapper(model, size=img_size, mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                          std=torch.tensor(IMAGENET_DEFAULT_STD))

    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


