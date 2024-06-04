import os

import torch
from torchvision.datasets import ImageNet
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



@dataclass
class CogImageNetArgs:
    gpu: int = 0
    num_images: int = 50
    class_idx: Optional[int] = None

    resolution: int = 512

    results_folder: str = 'output_cvpr/imagenet_captions_cog'
    imagenet_folder: str = '/mnt/datasets/imagenet'

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

    #load demo images
    pre_crop_size = int(args.resolution * 1.25)
    transform = transforms.Compose([transforms.Resize(pre_crop_size), transforms.CenterCrop(args.resolution)])

    in_labels = IDX2NAME
    in_val_dataset = ImageNet(args.imagenet_folder, split='val', transform=transform)

    if args.class_idx is None:
        target_classes = torch.arange(0, 1000, dtype=torch.long)
    else:
        target_classes = torch.LongTensor([args.class_idx])

    os.makedirs(args.results_folder, exist_ok=True)
    with open(os.path.join(args.results_folder, 'config.yaml'), "w") as f:
        OmegaConf.save(args, f)

    for target_class in target_classes:
        if world_size > 1:
            current_split_idx = target_class % world_size
            if not current_split_idx == local_rank:
                continue

        target_class = target_class.item()
        class_label = in_labels[target_class]
        val_class_idcs = torch.nonzero(torch.LongTensor(in_val_dataset.targets) == target_class, as_tuple=False).squeeze()

        # make the output folders
        class_folder = os.path.join(args.results_folder, f'{target_class}_{class_label}')
        os.makedirs(class_folder, exist_ok=True)

        with torch.no_grad():
            for img_idx in range(args.num_images):
                in_idx = val_class_idcs[img_idx]

                caption_file = os.path.join(class_folder, f"{in_idx}_prompt.txt")
                if os.path.isfile(caption_file):
                    continue

                imagenet_img, _ = in_val_dataset[in_idx]
                history = None
                cache_image = None

                query = f'Caption the following image containing a {class_label} in a short single sentence. The answer has to contain the object {class_label}. Keep the answer a short as possible.'

                response, history, cache_image = chat(
                    None,
                    model,
                    text_processor_infer,
                    image_processor,
                    query,
                    history=history,
                    cross_img_processor=cross_image_processor,
                    image=imagenet_img,
                    max_length=args.max_length,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    invalid_slices=text_processor_infer.invalid_slices,
                    args=args
                    )

                image_caption = response

                print(f"{image_caption}")
                imagenet_img.save(os.path.join(class_folder, f"{in_idx}_original.png"))

                with open(caption_file, 'w') as f:
                    f.write(image_caption)

