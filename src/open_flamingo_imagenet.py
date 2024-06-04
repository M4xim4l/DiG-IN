import os

import torch
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

from omegaconf import OmegaConf
from typing import Optional
from dataclasses import dataclass

from utils.datasets.inet_classes import IDX2NAME



demo_imgs_idcs_captions = [
    (411428, 'an image of an admiral sitting on a purple flower in front of green background'),
    (1278006, 'an image of bolete in the grass'),
    (1178950, 'an image of a traffic light in front of a tree and a white church'),
    (136098, 'an image of a koala hanging on a tree'),
    (1184094, 'an image of guacamole in a bowl with flatbread in the background'),
    (45134,'an image of a leatherback turtle on the beach at the sea'),
    (176902, 'an image of an European gallinule standing in the grass'),
    (392182, 'an image of a rhinoceros beetle sitting on a tree stamp with grass in the background'),
    (368716, 'an image of a leopard lying on sand in the savanna'),
    (362277, 'an image of a persian cat lying on top of a book with glasses and a chair in the background'),
    (1193941, 'an image of pretzels lying on top of a kitchen cloth'),
    (701468, 'an image of an electric guitar standing in front of wooden planks'),
    (1194988, 'an image of a cheese burger with fries next to it and wrapping paper in the back'),
    (257954, 'an image of a soft-coated wheaten terrier standing on a brick walkway with bushes'),
    (78721, 'an image of a night snake lying on a wooden tree trunk'),
    (373870, 'an image of a tiger jumping off wooden logs behind a fence'),
    (139796, 'an image of a sea anemone underwater on rocks'),
    (266074, 'an image of a labrador retriever being walked on leash by his owner and bushes'),
    (920017, 'an image of a pickup being driven in the fog'),
    (883691, 'an image of a overskirt being worn by a woman'),
    (585198, 'an image of a bottlecap on wooden table planks'),
    (931679, 'an image of a planetarium with people standing in front of it'),
    (999545, 'an image of a schooner sailing on the ocean in the sunset'),
    (999545, 'an image of a tiger beetle sitting on a leaf macro shot'),
]

@dataclass
class OpenFlamingoImageNetArgs:
    gpu: int = 0
    num_images: int = 50
    class_idx: Optional[int] = None

    resolution: int = 224

    results_folder: str = 'output_cvpr/imagenet_captions'
    imagenet_folder: str = '/mnt/datasets/imagenet'

    lang_encoder: str = 'anas-awadalla/mpt-7b'
    flamingo_checkpoint: str = 'openflamingo/OpenFlamingo-9B-vitl-mpt7b'
    cross_attn_every_n_layers: int = 4

def setup() -> OpenFlamingoImageNetArgs:
    default_config: OpenFlamingoImageNetArgs = OmegaConf.structured(OpenFlamingoImageNetArgs)
    cli_args = OmegaConf.from_cli()
    config: OpenFlamingoImageNetArgs = OmegaConf.merge(default_config, cli_args)
    return config

def plot_random_images():
    import matplotlib.pyplot as plt
    pre_crop_size = int(args.resolution * 1.25)
    transform = transforms.Compose(
        [transforms.Resize(pre_crop_size), transforms.CenterCrop(args.resolution), transforms.ToTensor()])
    in_train_dataset = ImageNet(args.imagenet_folder, split='train', transform=transform)

    num_rows = 10
    num_cols = 10

    scale_factor = 4.0
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))

    for i in range(num_rows):
        for j in range(num_cols):
            idx = torch.randint(0, len(in_train_dataset), (1,)).item()
            img, target = in_train_dataset[idx]
            ax = axs[i, j]
            ax.axis('off')
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f'{idx}\n{IDX2NAME[target]}')

    plt.show()


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

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=args.lang_encoder,
        tokenizer_path=args.lang_encoder,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers
    )

    checkpoint_path = hf_hub_download(args.flamingo_checkpoint, "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    model = model.to(device)
    tokenizer = tokenizer

    #load demo images
    pre_crop_size = int(args.resolution * 1.25)
    transform = transforms.Compose([transforms.Resize(pre_crop_size), transforms.CenterCrop(args.resolution)])

    in_train_dataset = ImageNet(args.imagenet_folder, split='train', transform=transform)


    in_labels = IDX2NAME

    num_demo_imgs = len(demo_imgs_idcs_captions)
    demo_images = torch.zeros((1, num_demo_imgs, 1, 3, args.resolution, args.resolution))

    tokenizer_demo_input = ''

    for i, (imagenet_idx, caption) in enumerate(demo_imgs_idcs_captions):
        imagenet_img, _ = in_train_dataset[imagenet_idx]
        imagenet_img = image_processor(imagenet_img)
        demo_images[0, i, 0] = imagenet_img
        tokenizer_demo_input += f'<image>{caption}.<|endofchunk|>'

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
                imagenet_img_processed = image_processor(imagenet_img)[None, None, None]

                vision_x = torch.cat([demo_images, imagenet_img_processed], dim=1).to(device)
                tokenizer_input = tokenizer_demo_input + f'<image>an image of a {class_label} '
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

                imagenet_img.save(os.path.join(class_folder, f"{in_idx}_original.png"))

                with open(caption_file, 'w') as f:
                    f.write(image_caption)

