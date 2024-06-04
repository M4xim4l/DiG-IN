import os

import torch
from torchvision.datasets import ImageFolder, ImageNet
import torchvision.transforms as transforms

from omegaconf import OmegaConf
from typing import Optional
from dataclasses import dataclass
import argparse
from utils.datasets.inet_classes import IDX2NAME

from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel


from utils.cogvlm.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from sat.resources.urls import MODEL_URLS
from sat.mpu import get_model_parallel_world_size

#need to import to set up registry hooks
import utils.cogvlm.models.cogagent_model
import utils.cogvlm.models.cogvlm_model
from tqdm import tqdm

class SpuriousImagenet(ImageFolder):
    def __init__(self, root: str, **kwargs) -> None:
        root = self.root = os.path.expanduser(root)
        super().__init__(root, **kwargs)
        self.root = root

        self.class_to_in_class = {}
        for v,i in self.class_to_idx.items():
            in_class = int(v.split('_')[2])
            self.class_to_in_class[i] = in_class

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        target = self.class_to_in_class[target]
        return img, target
@dataclass
class CogImageNetArgs:
    gpu: int = 0
    spurious_imagenet_folder: str = '/mnt/amaximilian19/spurious_imagenet/images/'

    #CoqVLM parameters
    max_length: int = 2048
    top_p: float = 0.4
    top_k: int = 1
    temperature: float = 0.8
    quant: Optional[int] = None

    from_pretrained: str = "cogagent-vqa"
    local_tokenizer: str = "lmsys/vicuna-7b-v1.5"
    fp16: bool = False
    bf16: bool = True
    stream_chat: bool = False

def setup() -> CogImageNetArgs:
    default_config: CogImageNetArgs = OmegaConf.structured(CogImageNetArgs)
    cli_args = OmegaConf.from_cli()
    config: CogImageNetArgs = OmegaConf.merge(default_config, cli_args)
    return config


if __name__=="__main__":
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

    #load demo images
    in_labels = IDX2NAME
    spurious_dataset = SpuriousImagenet(args.spurious_imagenet_folder)

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=0,
        rank=0,
        world_size=1,
        model_parallel_size=1,
        mode='inference',
        skip_init=True,
        build_only=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else device,
        bf16=args.bf16,
        fp16=args.fp16,
        **vars(args)
    ), overwrite_args={})
    model = model.eval()
    #assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
    print("[Language processor version]:", language_processor_version)
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None

    if args.quant:
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.to(device)

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    class_imgs_decisions = {}
    for idx in tqdm(range(len(spurious_dataset))):
        data, target_class = spurious_dataset[idx]
        class_label = in_labels[target_class]

        with torch.no_grad():
            history = None
            cache_image = None

            query = f'Does this image contain a {class_label}? Please only answer with yes or no.'

            response, history, cache_image = chat(
                None,
                model,
                text_processor_infer,
                image_processor,
                query,
                history=history,
                cross_img_processor=cross_image_processor,
                image=data,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
                top_k=args.top_k,
                invalid_slices=text_processor_infer.invalid_slices,
                args=args
                )

            decision = 1 if 'yes' in response.lower() else 0

            if not target_class in class_imgs_decisions:
                class_imgs_decisions[target_class] = []

            class_imgs_decisions[target_class].append(decision)

    total_yes = 0
    total_imgs = 0
    for class_idx, class_decision in class_imgs_decisions.items():
        class_yes = sum(class_decision)
        spurious_fraction = class_yes / len(class_decision)
        print(f'{in_labels[class_idx]}: {class_yes} out of {len(class_decision)} ({spurious_fraction:.3f})')

        total_yes += class_yes
        total_imgs += len(class_decision)

    print(f'Total: {total_yes} out of {total_imgs} ({total_yes/total_imgs:.3f})')