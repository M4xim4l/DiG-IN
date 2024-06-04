import os

import torch
import wandb
from PIL import Image
from torchvision import transforms as transforms

from .loss_functions import calculate_confs, calculate_lp_distances, calculate_lpips_distances
from .plotting_utils import plot, plot_attention



def make_uvces_and_save(args, classifier, pipe, device, linear_idx, img_idx, class_labels, inversion_root_dir,
                        losses_dicts, result_root_folder, regularizers_weights, start_class, target_class):
    start_label = class_labels[start_class]
    target_label = class_labels[target_class]
    if args.p2p_params is not None and args.p2p_params.do_p2p:
        prompt_to_prompt_replacements = (start_label, target_label)
    else:
        prompt_to_prompt_replacements = None
    start_class_folder = os.path.join(inversion_root_dir, f'{start_class}_{start_label}')
    inversion_dir = os.path.join(start_class_folder,
                                 f"inversion_{args.num_ddim_steps}_{args.guidance_scale}")
    result_folder = os.path.join(result_root_folder,
                                 f'{start_class}_{start_label}/')
    os.makedirs(result_folder, exist_ok=True)

    target_label_cleaned = target_label.replace('\\', '').replace('/', '')
    out_pdf = os.path.join(result_folder, f'{img_idx}_{target_class}_{target_label_cleaned}.pdf')

    if os.path.isfile(out_pdf):
        return

    latent_file = os.path.join(inversion_dir, f'{img_idx}.pt')
    if not os.path.isfile(latent_file):
        print(f'Couldnt find {latent_file} - Skipping')
        return

    latent = torch.load(latent_file, map_location='cpu')[None, :]
    null_texts_embeddings = torch.load(os.path.join(inversion_dir, f'{img_idx}_null_texts.pt'), map_location='cpu')
    assert len(null_texts_embeddings) == args.num_ddim_steps
    if args.use_double:
        latent = latent.double()
        null_texts_embeddings = null_texts_embeddings.double()
    captions_file = os.path.join(inversion_dir, f'{img_idx}_prompt.txt')
    if os.path.isfile(captions_file):
        with open(captions_file, 'r') as f:
            prompt_str = f.read()
    else:
        print(f'Warning: Could not load caption from {captions_file}')
        # continue
    os.makedirs(result_folder, exist_ok=True)
    prompt_foreground_key_words = [word for word in start_label.split()]

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    original_img = Image.open(os.path.join(inversion_dir, f'{img_idx}_original.png'))
    original_tensor = to_tensor(original_img).to(device)
    if args.use_double:
        original_tensor = original_tensor.double()

    targets_dict = {args.loss: target_class}
    try:
        return_values = pipe(targets_dict,
                             starting_img=original_tensor,
                             losses_dict=losses_dicts,
                             # loss and regs
                             regularizers_weights=regularizers_weights,
                             # optim params
                             optim_params=args.optim_params,
                             null_text_embeddings=null_texts_embeddings,
                             # segmentation based regularizers
                             prompt_foreground_key_words=prompt_foreground_key_words,
                             segmentation_args=args.segmentation_args,
                             # Prompt-To-Prompt params
                             p2p_params=args.p2p_params,
                             p2p_replacements=prompt_to_prompt_replacements,
                             # Mask-blend params
                             mask_blend_params=args.mask_blend_params,
                             # SD params
                             height=args.resolution, width=args.resolution,
                             num_inference_steps=args.num_ddim_steps,
                             guidance_scale=args.guidance_scale,
                             latents=latent,
                             prompt=prompt_str)
    except KeyError as e:
        print(e)
        return

    with torch.no_grad():
        img_grid = return_values['imgs']
        loss_scores = return_values['loss_scores']
        regularizer_scores = return_values['regularizer_scores']
        px_foreground_segmentation = return_values['px_foreground_segmentation']
        words_attention_masks = return_values['words_attention_masks']
        pre_p2p_image = return_values['initial_img']

        start_confs = calculate_confs(classifier, img_grid, device, target_class=start_class)
        target_confs = calculate_confs(classifier, img_grid, device, target_class=target_class)

        total_losses = torch.FloatTensor(loss_scores['total'])
        min_loss_idx = torch.argmin(total_losses).item()

        min_loss_img = img_grid[min_loss_idx]
        # torch.save(min_loss_img, os.path.join(result_folder, f'{img_idx}.pth'))

        original_img.save(os.path.join(result_folder, f'{img_idx}.png'))
        pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
        pil_sd.save(os.path.join(result_folder, f'{img_idx}_{target_class}_{target_label_cleaned}_sd.png'))
        pil_ours = to_pil(torch.clamp(min_loss_img, 0, 1))
        pil_ours.save(os.path.join(result_folder, f"{img_idx}_{target_class}_{target_label_cleaned}_ours.png"))


        title_attributes = {}
        max_class_name_length = 8
        if len(start_label) > max_class_name_length:
            title_attribute = start_label[:max_class_name_length] + '.'
        else:
            title_attribute = start_label
        title_attributes[title_attribute] = start_confs

        if len(target_label) > max_class_name_length:
            title_attribute = target_label[:max_class_name_length] + '.'
        else:
            title_attribute = target_label
        title_attributes[title_attribute] = target_confs

        plot(original_tensor, pre_p2p_image, img_grid, title_attributes, out_pdf,
             loss_scores=loss_scores,
             regularizer_scores=regularizer_scores,
             wandb_name=f'history' if args.wandb_project is not None else None,
             wandb_step=linear_idx)

        plot_attention(original_tensor, px_foreground_segmentation, words_attention_masks,
                       os.path.join(result_folder, f'{img_idx}_attention.pdf'),
                       wandb_name=f'attention' if args.wandb_project is not None else None,
                       wandb_step=linear_idx)

        if args.wandb_project is not None:
            lp_distances = calculate_lp_distances([img_grid[0], min_loss_img], [original_tensor, original_tensor],
                                                  ps=(1., 2.))
            lpips_distances = calculate_lpips_distances([img_grid[0], min_loss_img], [original_tensor, original_tensor])
            wandb_log_dict = {
                'loss': loss_scores[min_loss_idx],
                'l2_best': lp_distances[2.][1],
                'l2_p2p': lp_distances[2.][0],
                'l1_best': lp_distances[1.][1],
                'l1_p2p': lp_distances[1.][0],
                'lpips_best': lpips_distances[1],
                'lpips_p2p': lpips_distances[0],
                'start_class_conf': start_confs[min_loss_idx],
                'target_class_conf': target_confs[min_loss_idx],
                'start_p2p_ours': [wandb.Image(img, caption=cap) for img, cap in zip([original_img, pil_sd, pil_ours],
                                                                                     [prompt_str, 'P2P', 'UVCE'])]
            }
            if regularizer_scores:
                for reg_n, reg_score in regularizer_scores.items():
                    wandb_log_dict[reg_n] = reg_score
            wandb.log(wandb_log_dict,
                      step=linear_idx)
