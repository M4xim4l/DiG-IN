# -*- encoding: utf-8 -*-
import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#need to import to set up registry hooks

import math
import torch
import os
import argparse
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel

from sat.resources.urls import MODEL_URLS
from sat.mpu import get_model_parallel_world_size

import utils.cogvlm.models.cogagent_model
import utils.cogvlm.models.cogvlm_model


from utils.cogvlm.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor


DEFAULT_TARGET_NEURONS = [310, 312, 318, 319, 322, 334, 338, 373, 386, 402, 410, 412, 424, 434, 473, 474, 476, 478, 483,
                          446, 494, 495, 498, 505, 507, 530, 534, 553, 571, 583, 589, 593, 595, 599, 608, 618, 619, 623,
                          633, 682, 694, 707, 720, 725, 754, 767, 770, 777, 784, 798, 799, 816, 870, 899, 908, 910, 914,
                          919, 924, 932, 53, 1700, 462, 10, 13, 56, 58, 60, 68, 71, 73, 90, 138, 150, 168, 291, 292]


@dataclass
class NeuronArgs:
    gpu: int = 0
    seed: int = 42
    num_images: int = 10

    num_images_to_label: int = 5
    coq_prompt: str = 'list the most important objects in the image in a list format starting with [ and ending with ] without a full sentence'

    classifier: str = 'seresnet152d.ra2_in1k'
    target_neurons: List[int] = field(default_factory=lambda: DEFAULT_TARGET_NEURONS)

    results_folder: str = 'output_cvpr/imagenet_cogvlm_neurons'
    neuron_statistics_folder: str = 'output_cvpr/imagenet_neuron_statistics'
    results_sub_folder: Optional[str] = None

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


def setup() -> NeuronArgs:
    default_config: NeuronArgs = OmegaConf.structured(NeuronArgs)
    cli_args = OmegaConf.from_cli()
    config: NeuronArgs = OmegaConf.merge(default_config, cli_args)
    return config


def main():
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
        local_rank=local_rank,
        rank=local_rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else device,
        bf16=args.bf16,
        fp16=args.fp16,
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

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

    if args.results_sub_folder is None:
        result_sub_folder = args.classifier
    else:
        result_sub_folder = args.results_sub_folder
    result_folder = os.path.join(args.results_folder, result_sub_folder)
    os.makedirs(result_folder, exist_ok=True)


    with torch.no_grad():
        for neuron_idx in args.target_neurons:
            top_neurons_file = os.path.join(args.neuron_statistics_folder, args.classifier, f'neuron_{neuron_idx}_top_images.txt')
            top_neuron_imgs = []
            top_neuron_imgs_activatiosn = []

            print(f'\nNeuron {neuron_idx}')
            neuron_coq_response_file = os.path.join(result_folder, f'neuron_{neuron_idx}_top_images_coq_responses.pt')
            if os.path.isfile(neuron_coq_response_file):
                continue

            with open(top_neurons_file, 'r') as f:
                for line in f:
                    image_path = line.split('\t')[0]
                    activation = float(line.rstrip().split('\t')[1])
                    top_neuron_imgs.append(image_path)
                    top_neuron_imgs_activatiosn.append(activation)
                    if len(top_neuron_imgs) >= args.num_images_to_label:
                        break

            top_neuron_dicts = []
            for image_path, activation in zip(top_neuron_imgs, top_neuron_imgs_activatiosn):
                history = None
                cache_image = None
                assert image_path is not None

                query = args.coq_prompt
                assert query is not None

                response, history, cache_image = chat(
                    image_path,
                    model,
                    text_processor_infer,
                    image_processor,
                    query,
                    history=history,
                    cross_img_processor=cross_image_processor,
                    image=cache_image,
                    max_length=args.max_length,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    invalid_slices=text_processor_infer.invalid_slices,
                    args=args
                    )

                print(f"{image_path}: {response}")
                top_neuron_dicts.append({
                    'image_path': image_path,
                    'activation': activation,
                    'response': response
                })

            torch.save(top_neuron_dicts, neuron_coq_response_file)

if __name__ == "__main__":
    main()
