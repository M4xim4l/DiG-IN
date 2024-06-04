import torch
import timm

from .model_wrappers import NormalizationAndCutoutWrapper, NormalizationAndResizeWrapper, \
    NormalizationAndRandomNoiseWrapper


def load_timm_model_with_cutout(model_name, cut_power, num_cutouts, noise_sd=0, noise_schedule='constant',
                                noise_descending_steps=None,
                                batches=1, checkpointing=False):
    model = timm.create_model(model_name, pretrained=True)
    cfg = model.default_cfg
    assert cfg["input_size"][1] == cfg["input_size"][2]
    size = cfg["input_size"][1]
    if checkpointing:
        try:
            model.set_grad_checkpointing(True)
            checkpointing = False
        except:
            print('Model does not support Timm checkpointing')

    model = NormalizationAndCutoutWrapper(model, size=size, mean=torch.tensor(cfg["mean"]),
                                          std=torch.tensor(cfg["std"]), cut_power=cut_power,
                                          noise_sd=noise_sd, noise_schedule=noise_schedule, noise_descending_steps=noise_descending_steps,
                                          num_cutouts=num_cutouts, checkpointing=checkpointing, batches=batches)
    model = model.eval()
    return model

def load_timm_model_with_random_noise(model_name, noise_sd, num_samples, batches=1, checkpointing=False):
    model = timm.create_model(model_name, pretrained=True)
    cfg = model.default_cfg
    assert cfg["input_size"][1] == cfg["input_size"][2]
    size = cfg["input_size"][1]
    model = NormalizationAndRandomNoiseWrapper(model, size=size, mean=torch.tensor(cfg["mean"]),
                                               std=torch.tensor(cfg["std"]), noise_sd=noise_sd, num_samples=num_samples,
                                               batches=batches, checkpointing=checkpointing)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model

def load_timm_model(model_name, auto_resize=True):
    model = timm.create_model(model_name, pretrained=True)
    cfg = model.default_cfg
    assert cfg["input_size"][1] == cfg["input_size"][2]
    if auto_resize:
        size = cfg["input_size"][1]
    else:
        size = None
    model = NormalizationAndResizeWrapper(model, size=size, mean=torch.tensor(cfg["mean"]),
                                          std=torch.tensor(cfg["std"]))

    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
