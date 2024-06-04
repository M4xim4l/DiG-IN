import os
import torch
import open_clip
from open_clip.zero_shot_classifier import build_zero_shot_classifier
from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES
from open_clip.transform import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

import pathlib
from .model_wrappers import NormalizationAndResizeWrapper, NormalizationAndCutoutWrapper

class ZeroShotCLIP(torch.nn.Module):

    def __init__(self, device, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', templates='openai',
                 dataset='imagenet', weights_path='./zeroshot_model'):
        super().__init__()

        self.model = open_clip.create_model(model_name=model_name, pretrained=pretrained, device=device)

        # Compute zero shot weights
        weights_path = os.path.join(weights_path, f'{dataset}_model_{model_name}_templates_{templates}.pth')
        if os.path.isfile(weights_path):
            self.weights = torch.load(weights_path).to(device)
        else:
            weights_dir = os.path.dirname(weights_path)
            pathlib.Path(weights_dir).mkdir(exist_ok=True, parents=True)
            tokenizer = open_clip.get_tokenizer(model_name)

            if templates == 'openai':
                prompt_templates = OPENAI_IMAGENET_TEMPLATES
            weights = build_zero_shot_classifier(self.model, tokenizer, IMAGENET_CLASSNAMES, prompt_templates,
                                                 num_classes_per_batch=64, device=device, use_tqdm=True)

            torch.save(weights, weights_path)
            self.weights = weights

    def forward(self, img, normalize=True):

        img_emb = self.model.encode_image(img)
        if normalize:
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        return 100.0 * img_emb.float() @ self.weights


def load_clip_zero_shot(device):
    model = ZeroShotCLIP(device)
    model = NormalizationAndResizeWrapper(model, 224,
                                          mean=torch.FloatTensor(OPENAI_DATASET_MEAN),
                                          std=torch.FloatTensor(OPENAI_DATASET_STD))
    model.eval()

    return model


def load_clip_zero_shot_with_cutout(device, cut_power, num_cutouts, noise_sd=0, noise_schedule='constant',
                                    noise_descending_steps=None,
                                    batches=1, checkpointing=False):
    model = ZeroShotCLIP(device)
    model = NormalizationAndCutoutWrapper(model, size=224, mean=torch.FloatTensor(OPENAI_DATASET_MEAN),
                                          std=torch.FloatTensor(OPENAI_DATASET_STD), cut_power=cut_power,
                                          noise_sd=noise_sd, noise_schedule=noise_schedule,
                                          noise_descending_steps=noise_descending_steps,
                                          num_cutouts=num_cutouts, checkpointing=checkpointing, batches=batches)
    model.eval()
    return model