import os

import numpy as np
import torch
from utils.datasets.bird_dataset import BirdDataset as CubDataset
from utils.datasets.car_dataset import CarDataset as CarsDataset
from utils.datasets.cub_classes import IDX2NAME as cub_labels
from utils.datasets.cars_classes import IDX2NAME as cars_labels
from utils.datasets.food101_classes import IDX2NAME as food_labels
from utils.datasets.flowers_classes import IDX2NAME as flowers_labels
import torchvision.transforms as transforms
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from torchvision.datasets import Food101, Flowers102
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from typing import Optional
from dataclasses import dataclass
import time



cub_imgs_idcs_captions = [
    (3253, 'an image of a American Redstart sitting on a hand'),
    (2490, 'an image of a Red legged Kittiwake flying over the ocean'),
    (868, 'an image of a American Crow standing on concrete'),
    (5589, 'an image of a American Three toed Woodpecker in a tree trunk'),
    (475, 'an image of a Painted Bunting eating from a bird feeder'),
    (959, 'an image of a Mangrove Cuckoo sitting in a tree with flowers'),
    (1856, 'an image of a Herring Gull flying in the sky'),
    (5945, 'an image of a Winter Wren standing in the snow'),
    (2603, 'an image of a Mallard standing in grass'),
    (2300, 'an image of a Tropical Kingbird sitting on a cable'),
    (316, 'an image of a Rusty Blackbird standing in a marsh'),
    (25, 'an image of a Black footed Albatross flying over water'),
    (2070, 'an image of a Green Violetear next to a leaf and a pink flower'),
    (3938, 'an image of a White crowned Sparrow sitting on white flowers'),
    (5655, 'an image of a Red bellied Woodpecker eating from a seed feeder'),
    (2029, 'an image of a Ruby throated Hummingbird macro shot black background'),
    (1659, 'an image of a Pine Grosbeak sitting on red berries and blossom'),
    (1619, 'an image of a Blue Grosbeak next to a bird house in front of bushes'),
    (4276, 'an image of a Caspian Tern at the beach in front of the sea'),
    (2318, 'an image of a Gray Kingbird sitting on a metal pipe brown background'),
    (4867, 'an image of a Cape May Warbler drinking from a red bird feeder green background'),
    (502, 'an image of a Cardinal hanging from a blue bird cage'),
    (5343, 'an image of a Tennessee Warbler between branches and leafs'),
]

cars_imgs_idcs_captions = [
    (7180, 'an image of a Jaguar XK XKR 2012 driving on a road with trees in the back'),
    (1740, 'an image of a BMW X3 SUV 2012 standing on the beach in front of the sea'),
    (5331, 'an image of a Porsche Panamera Sedan 2012 driving on a mountain road'),
    (5378, 'an image of a Lamborghini Diablo Coupe 2001 standing in front of an old building'),
    (7445, 'an image of a Ferrari 458 Italia Coupe 2012 parking in front of a modern building'),
    (629, 'an image of a Ferrari 458 Italia Convertible 2012 parking on grass'),
    (5310, 'an image of a BMW 3 Series Sedan 2012 photographed inside a showroom'),
    (3318, 'an image of a Hyundai Sonata Hybrid Sedan 2012 in a showroom'),
    (6967, 'an image of a Jaguar XK XKR 2012 driving on a race track'),
    (4934, 'an image of a Chevrolet Corvette ZR1 2012 parking in a parking lot'),
    (7787, 'an image of a Ford F-450 Super Duty Crew Cab 2012 on white background'),
    (6582, 'an image of a Bentley Arnage Sedan 2009 driving through the city'),
    (8137, 'an image of a Suzuki Aerio Sedan 2007 on the street in front of fall trees'),
    (6428, 'an image of a Jeep Wrangler SUV 2012 in front of a countryside valley'),
    (354, 'an image of a Nissan Juke Hatchback 2012 parking in front of glass windows'),
    (4320, 'an image of a Chevrolet Sonic Sedan 2012 macro shot white background'),
    (6542, 'an image of a Mercedes-Benz C-Class Sedan 2012 parking on gravel road with blue sky'),
    (1827, 'an image of a Buick Verano Sedan 2012 standing on road in front of a rock wall '),
    (2167, 'an image of a Ferrari FF Coupe 2012 in front of a blurry forest'),
    (5217, 'an image of a Chevrolet HHR SS 2010 parking on raod cargo containers '),
    (550, 'an image of a Bentley Continental GT Coupe 2012 in the dessert sunny sky'),
    (5337, 'an image of a Chevrolet Impala Sedan 2007 in the lot of a car shop'),
    (560, 'an image of a Audi S5 Convertible 2012 driving road countryside background sky'),
    (5841, 'an image of a Chevrolet Silverado 1500 Extended Cab 2012 standing on road in front of mountains'),
    (7113, 'an image of a Jaguar XK XKR 2012 driving on a race track'),
    (5225, 'an image of a Audi TT RS Coupe 2012 driving in front of snowey buildings trees'),
]

food_imgs_idcs_captions = [
    (25821, 'an image of chocolate cake on a plate with powder sugar'),
    (13657, 'an image of a crab cakes on green sauce'),
    (46629, 'an image of a donuts next to each other'),
    (64392, 'an image of a macarons next to each other in a box'),
    (16658, 'an image of a baby back ribs on a wodden board with a glass in the background'),
    (53221, 'an image of a bruschetta on a plate on a table with a water bottle and striped tablecloth'),
    (65560, 'an image of a seaweed salad on a plate'),
    (25960, 'an image of a chocolate cake with chocolate sauce ice cream strawberries and whipped cream'),
    (49736, 'an image of a spaghetti carbonara on a plate'),
    (88, 'an image of a churros on a plate next to sauces in glass bowls on a wooden table'),
    (34075, 'an image of a onion rings served with ketchup on a plate'),
    (37075, 'an image of a deviled eggs served on a black board'),
    (71649, 'an image of a ice cream in a cup with whipped cream'),
    (11035, 'an image of a pulled pork sandwich on paper with fries'),
    (18025, 'an image of a frozen yogurt in a paper bowl'),
    (26817, 'an image of a tiramisu in a bowl on a plate with a spoon'),
    (35267, 'an image of a grilled salmon on a plate with baby potatoes'),
    (4905, 'an image of a panna cotta in a bowl on a plate with a spoon'),
    (71229, 'an image of a cheesecake in a freezer'),
    (27123, 'an image of a spaghetti bolognese on a platewith fork and knife on a table'),
]

flowers_imgs_idcs_captions = [
    (25821, 'an image of chocolate cake on a plate with powder sugar'),
    (13657, 'an image of a crab cakes on green sauce'),
    (46629, 'an image of a donuts next to each other'),
    (64392, 'an image of a macarons next to each other in a box'),
    (16658, 'an image of a baby back ribs on a wodden board with a glass in the background'),
    (53221, 'an image of a bruschetta on a plate on a table with a water bottle and striped tablecloth'),
    (65560, 'an image of a seaweed salad on a plate'),
    (25960, 'an image of a chocolate cake with chocolate sauce ice cream strawberries and whipped cream'),
    (49736, 'an image of a spaghetti carbonara on a plate'),
    (88, 'an image of a churros on a plate next to sauces in glass bowls on a wooden table'),
    (34075, 'an image of a onion rings served with ketchup on a plate'),
    (37075, 'an image of a deviled eggs served on a black board'),
    (71649, 'an image of a ice cream in a cup with whipped cream'),
    (11035, 'an image of a pulled pork sandwich on paper with fries'),
    (18025, 'an image of a frozen yogurt in a paper bowl'),
    (26817, 'an image of a tiramisu in a bowl on a plate with a spoon'),
    (35267, 'an image of a grilled salmon on a plate with baby potatoes'),
    (4905, 'an image of a panna cotta in a bowl on a plate with a spoon'),
    (71229, 'an image of a cheesecake in a freezer'),
    (27123, 'an image of a spaghetti bolognese on a platewith fork and knife on a table'),
]


@dataclass
class OpenFlamingoArgs:
    gpu: int = 0
    num_images: int = 50
    class_idx: Optional[int] = None

    resolution: int = 224

    dataset: Optional[str] = None
    results_folder: Optional[str] = None
    dataset_folder: Optional[str] = None

    plot_random: bool = False

    lang_encoder: str = 'anas-awadalla/mpt-7b'
    flamingo_checkpoint: str = 'openflamingo/OpenFlamingo-9B-vitl-mpt7b'
    cross_attn_every_n_layers: int = 4

def setup() -> OpenFlamingoArgs:
    default_config: OpenFlamingoArgs = OmegaConf.structured(OpenFlamingoArgs)
    cli_args = OmegaConf.from_cli()
    config: OpenFlamingoArgs = OmegaConf.merge(default_config, cli_args)
    if config.dataset == 'cub':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets/CUB_200_2011'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/cub_captions'
    elif config.dataset == 'cars':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets/stanford_cars'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/cars_captions'
    elif config.dataset == 'food101':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/food101_captions'
    elif config.dataset == 'flowers':
        config.dataset_folder = config.dataset_folder if config.dataset_folder is not None else '/mnt/datasets'
        config.results_folder = config.results_folder if config.results_folder is not None else 'output_cvpr/flowers102_captions'
    else:
        raise NotImplementedError()

    return config

#torchrun --nproc-per-node=8 src/open_flamingo_cub_cars.py dataset cars


if __name__=="__main__":
    args = setup()
    #load demo images
    pre_crop_size = int(args.resolution * 1.25)
    transform = transforms.Compose([transforms.Resize(pre_crop_size), transforms.CenterCrop(args.resolution)])

    if args.dataset == 'cub':
        train_dataset = CubDataset(args.dataset_folder, phase='train', transform=transform)
        test_dataset = CubDataset(args.dataset_folder, phase='test', transform=transform)
        labels = cub_labels
        demo_imgs_idcs_captions = cub_imgs_idcs_captions
    elif args.dataset == 'cars':
        train_dataset = CarsDataset(args.dataset_folder, phase='train', transform=transform)
        test_dataset = CarsDataset(args.dataset_folder, phase='test', transform=transform)
        labels = cars_labels
        demo_imgs_idcs_captions = cars_imgs_idcs_captions
    elif args.dataset == 'food101':
        train_dataset = Food101(args.dataset_folder, split='train', transform=transform)
        test_dataset = Food101(args.dataset_folder, split='test', transform=transform)
        test_dataset.targets = test_dataset._labels
        labels = food_labels
        demo_imgs_idcs_captions = food_imgs_idcs_captions
    elif args.dataset == 'flowers':
        train_dataset = Flowers102(args.dataset_folder, split='train', transform=transform)
        test_dataset = Flowers102(args.dataset_folder, split='test', transform=transform)
        labels = flowers_labels
        demo_imgs_idcs_captions = flowers_imgs_idcs_captions
    else:
        raise NotImplementedError()

    if args.plot_random:
        num_rows = 20
        num_cols = 20
        num_imgs = num_cols * num_rows
        scale_factor = 4.0

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
        random_idcs = torch.randperm(len(train_dataset))[:num_imgs]

        for i in range(num_rows):
            for j in range(num_cols):
                dataset_idx = random_idcs[i * num_cols + j]
                img, class_idx = train_dataset[dataset_idx]
                class_name = labels[class_idx]

                img = np.array(img)

                ax = axs[i, j]
                ax.axis('off')

                title = f'{dataset_idx}\n{class_name}'
                ax.set_title(title)
                ax.imshow(img, interpolation='lanczos')
        plt.tight_layout()
        fig.savefig(f'{args.dataset}_train_imgs.pdf')
        plt.close(fig)
    else:
        num_demo_imgs = len(demo_imgs_idcs_captions)
        demo_images = torch.zeros((1, num_demo_imgs, 1, 3, args.resolution, args.resolution))

        if "WORLD_SIZE" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            args.gpu = local_rank + args.gpu
            print(f'Rank {local_rank} out of {world_size}')
        else:
            local_rank = 0
            world_size = 1


        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=args.lang_encoder,
            tokenizer_path=args.lang_encoder,
            cross_attn_every_n_layers=args.cross_attn_every_n_layers
        )

        checkpoint_path = hf_hub_download(args.flamingo_checkpoint, "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)

        loaded = False
        c = 0
        gpu_i = local_rank
        gpu_offset = args.gpu - local_rank
        while not loaded:
            try:
                device = torch.device(f'cuda:{args.gpu}')
                model = model.to(device)
                loaded = True
            except:
                gpu_i += 1
                if gpu_i == world_size:
                    gpu_i = 0
                    time.sleep(10)
                    c = c - 1
                    if c <= 0:
                        print(f'Rank: {local_rank} waiting - Offset {gpu_offset}')
                        c = 18
                else:
                    args.gpu = gpu_i + gpu_offset
                    time.sleep(0.5)

        tokenizer = tokenizer

        tokenizer_demo_input = ''

        for i, (imagenet_idx, caption) in enumerate(demo_imgs_idcs_captions):
            train_img, _ = train_dataset[imagenet_idx]
            train_img = image_processor(train_img)
            demo_images[0, i, 0] = train_img
            tokenizer_demo_input += f'<image>{caption}.<|endofchunk|>'

        num_classes = len(labels)
        if args.class_idx is None or args.class_idx > num_classes:
            target_classes = torch.arange(0, num_classes, dtype=torch.long)
        else:
            target_classes = torch.LongTensor([args.class_idx])

        os.makedirs(args.results_folder, exist_ok=True)
        with open(os.path.join(args.results_folder, 'config.yaml'), "w") as f:
            OmegaConf.save(args, f)

        for target_class_idx in target_classes:
            if world_size > 1:
                current_split_idx = target_class_idx % world_size
                if not current_split_idx == local_rank:
                    continue

            class_label = labels[target_class_idx.item()]
            test_class_idcs = torch.nonzero(torch.LongTensor(test_dataset.targets) == target_class_idx, as_tuple=False).squeeze()

            # make the output folders
            class_folder = os.path.join(args.results_folder, f'{target_class_idx}_{class_label}')
            os.makedirs(class_folder, exist_ok=True)

            with torch.no_grad():
                for img_idx in range(min(args.num_images, len(test_class_idcs))):
                    in_idx = test_class_idcs[img_idx]

                    caption_file = os.path.join(class_folder, f"{in_idx}_prompt.txt")
                    if os.path.isfile(caption_file):
                        continue

                    test_img, _ = test_dataset[in_idx]
                    test_img_processed = image_processor(test_img)[None, None, None]

                    vision_x = torch.cat([demo_images, test_img_processed], dim=1).to(device)
                    generic_caption = f'an image of a {class_label} '
                    tokenizer_input = tokenizer_demo_input + '<image>' + generic_caption
                    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
                    lang_x = tokenizer(
                        [tokenizer_input],
                        return_tensors="pt",
                    )
                    lang_x = lang_x.to(device)

                    generated_text = model.generate(
                        vision_x=vision_x,
                        lang_x=lang_x["input_ids"],
                        attention_mask=lang_x["attention_mask"],
                        max_new_tokens=20,
                        num_beams=3,
                    )

                    generated_text = generated_text.cpu()

                    text_decoded = tokenizer.decode(generated_text[0])
                    image_caption = text_decoded.split('<image>')[-1]

                    if '.' in image_caption:
                        image_caption = image_caption.split('.')[0]

                    if len(image_caption) < len(generic_caption):
                        print(f'Generated caption: {image_caption} did not contain generic, replacing {generic_caption}')
                        image_caption = generic_caption

                    test_img.save(os.path.join(class_folder, f"{in_idx}_original.png"))

                    with open(caption_file, 'w') as f:
                        f.write(image_caption)

