from transformers import ViTForImageClassification, ViTImageProcessor, SwinForImageClassification
import torch
from .model_wrappers import NormalizationAndResizeWrapper, NormalizationAndCutoutWrapper
import torch.nn as nn


class ViTReturnWrapper(nn.Module):
    def __init__(self, model, id2label=None):
        super().__init__()
        self.model = model
        if id2label is not None:
            self.register_buffer('id2label', id2label)
        else:
            self.id2label = None

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        logits = outputs['logits']
        if self.id2label is not None:
            for i in range(logits.shape[0]):
                logits[i, :] = logits[i, self.id2label]
        return logits

def load_food101_vit():
    model_name = 'nateraw/food'
    processor = ViTImageProcessor.from_pretrained(model_name)
    net = ViTForImageClassification.from_pretrained(model_name)
    net = ViTReturnWrapper(net)
    assert processor.size['height'] == processor.size['width']
    net = NormalizationAndResizeWrapper(net, size=processor.size['height'], mean=torch.tensor(processor.image_mean),
                                          std=torch.tensor(processor.image_std))

    net = net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def load_food101_vit_with_cutout(cut_power, num_cutouts, checkpointing=False, auto_resize=True,
                    noise_sd=0, noise_schedule=None, noise_descending_steps=None ):
    model_name = 'nateraw/food'
    processor = ViTImageProcessor.from_pretrained(model_name)
    net = ViTForImageClassification.from_pretrained(model_name)
    net = ViTReturnWrapper(net)
    assert processor.size['height'] == processor.size['width']
    net = NormalizationAndCutoutWrapper(net, size=processor.size['height'], mean=torch.tensor(processor.image_mean),
                                          std=torch.tensor(processor.image_std),
                                          cut_power=cut_power, num_cutouts=num_cutouts,
                                        checkpointing=checkpointing,
                                        noise_sd=noise_sd, noise_schedule=noise_schedule,
                                        noise_descending_steps=noise_descending_steps)
    net = net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def load_cub_vit_with_cutout(cut_power, num_cutouts, checkpointing=False, auto_resize=True,
                    noise_sd=0, noise_schedule=None, noise_descending_steps=None ):
    model_name = 'PwNzDust/vit_cub'
    processor = ViTImageProcessor.from_pretrained(model_name)
    net = ViTForImageClassification.from_pretrained(model_name)
    net = ViTReturnWrapper(net)
    assert processor.size['height'] == processor.size['width']
    net = NormalizationAndCutoutWrapper(net, size=processor.size['height'], mean=torch.tensor(processor.image_mean),
                                          std=torch.tensor(processor.image_std),
                                          cut_power=cut_power, num_cutouts=num_cutouts,
                                        checkpointing=checkpointing,
                                        noise_sd=noise_sd, noise_schedule=noise_schedule,
                                        noise_descending_steps=noise_descending_steps)
    net = net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net

def load_cub_vit():
    model_name = 'PwNzDust/vit_cub'
    processor = ViTImageProcessor.from_pretrained(model_name)
    net = ViTForImageClassification.from_pretrained(model_name)
    net = ViTReturnWrapper(net)
    assert processor.size['height'] == processor.size['width']
    net = NormalizationAndResizeWrapper(net, size=processor.size['height'], mean=torch.tensor(processor.image_mean),
                                          std=torch.tensor(processor.image_std))

    net = net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def load_flowers_vit():
    model_name = 'andriydovgal/mvp_flowers'
    processor = ViTImageProcessor.from_pretrained(model_name)
    net = ViTForImageClassification.from_pretrained(model_name)
    net = ViTReturnWrapper(net)
    assert processor.size['height'] == processor.size['width']
    net = NormalizationAndResizeWrapper(net, size=processor.size['height'], mean=torch.tensor(processor.image_mean),
                                          std=torch.tensor(processor.image_std))

    net = net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net

def load_flowers_vit_with_cutout(cut_power, num_cutouts, checkpointing=False, auto_resize=True,
                    noise_sd=0, noise_schedule=None, noise_descending_steps=None ):
    model_name = 'andriydovgal/mvp_flowers'
    processor = ViTImageProcessor.from_pretrained(model_name)
    net = ViTForImageClassification.from_pretrained(model_name)
    id2label = torch.zeros(102, dtype=torch.long)
    for k, v in net.config.id2label.items():
        id2label[int(v) - 1] = k
    net = ViTReturnWrapper(net, id2label)
    assert processor.size['height'] == processor.size['width']
    net = NormalizationAndCutoutWrapper(net, size=processor.size['height'], mean=torch.tensor(processor.image_mean),
                                          std=torch.tensor(processor.image_std),
                                          cut_power=cut_power, num_cutouts=num_cutouts,
                                        checkpointing=checkpointing,
                                        noise_sd=noise_sd, noise_schedule=noise_schedule,
                                        noise_descending_steps=noise_descending_steps)
    net = net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net

