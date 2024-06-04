import os
import torch
import open_clip
from .model_wrappers import NormalizationAndResizeWrapper, NormalizationAndCutoutWrapper, IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN


class ClipImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, img, normalize=True):
        img_emb = self.model.encode_image(img)
        if normalize:
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        return img_emb.float()


class ClipTextEncoder(torch.nn.Module):
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


    def forward(self, text, normalize=True):
        text_tokens = self.tokenizer(text).to(self.device)
        text_encoding = self.model.encode_text(text_tokens)
        if normalize:
            text_encoding = text_encoding / text_encoding.norm(dim=-1, keepdim=True)

        return text_encoding


def load_clip(device, model_name='EVA01-g-14-plus', pretrained=None):
    model, a, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    img_encoder = ClipImageEncoder(model)
    img_encoder = NormalizationAndResizeWrapper(img_encoder, model.visual.image_size[0],
                                      mean=torch.FloatTensor(model.visual.image_mean),
                                      std=torch.FloatTensor(model.visual.image_std))

    text_encoder = ClipTextEncoder(model, tokenizer)
    img_encoder.eval()
    text_encoder.eval()
    return img_encoder, text_encoder

def load_clip_with_cutout(device, cut_power, num_cutouts, model_name='EVA02-L-14-336', pretrained='merged2b_s6b_b61k',
                          noise_sd=0, noise_schedule='constant', noise_descending_steps=None, batches=1, checkpointing=False):
    model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    img_encoder = ClipImageEncoder(model)
    model.visual.set_grad_checkpointing(True)
    assert model.visual.image_size[0] == model.visual.image_size[1]
    img_encoder = NormalizationAndCutoutWrapper(img_encoder, size=model.visual.image_size[0], mean=torch.FloatTensor(model.visual.image_mean),
                                          std=torch.FloatTensor(model.visual.image_std), cut_power=cut_power,
                                          noise_sd=noise_sd, noise_schedule=noise_schedule,
                                          noise_descending_steps=noise_descending_steps,
                                          num_cutouts=num_cutouts, checkpointing=False, batches=batches)

    text_encoder = ClipTextEncoder(model, tokenizer, device)
    img_encoder.eval()
    text_encoder.eval()
    return img_encoder, text_encoder


if __name__=="__main__":
    open_clip.list_pretrained()