import math
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import wandb


def plot_p2p_xa_schedule_search(original_image, p2p_xa_schedule_search_results, filename):
    scale_factor = 4.0
    num_cols = max(2, math.ceil(math.sqrt(len(p2p_xa_schedule_search_results))))
    num_rows = 1 + math.ceil(len(p2p_xa_schedule_search_results) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))


    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            ax = axs[row_idx, col_idx]
            ax.axis('off')

            if row_idx == 0:
                if  col_idx == 0 and original_image is not None:
                    img = original_image.permute(1, 2, 0).cpu().detach()
                    ax.set_title('Original')
                    ax.imshow(img, interpolation='lanczos')
            else:
                img_idx = (row_idx - 1)* num_cols + col_idx
                if img_idx >= len(p2p_xa_schedule_search_results):
                    continue
                else:
                    img = p2p_xa_schedule_search_results[img_idx]['img']
                    loss = p2p_xa_schedule_search_results[img_idx]['loss']
                    xa_schedule = p2p_xa_schedule_search_results[img_idx]['xa_schedule_name']
                    self_schedule = p2p_xa_schedule_search_results[img_idx]['self_schedule_name']
                    blend_schedule = p2p_xa_schedule_search_results[img_idx]['blend_schedule_name']

                    img = torch.clamp(img.permute(1, 2, 0).cpu().detach(), min=0.0, max=1.0)

                    ax.axis('off')
                    ax.imshow(img, interpolation='lanczos')
                    ax.set_title(f'{xa_schedule}\n{self_schedule}\n{blend_schedule}\n{loss:.3f}')

    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def plot(original_image, original_reconstructed, trajectory_imgs, title_attributes,
         filename, loss_scores=None, regularizer_scores=None, model_grad_cams=None, wandb_name=None, wandb_step=None):
    scale_factor = 4.0
    num_cols = max(2, math.ceil(math.sqrt(len(trajectory_imgs))))
    num_rows = math.ceil(len(trajectory_imgs) / num_cols)

    if model_grad_cams is not None:
        num_sub_rows = 1 + model_grad_cams.shape[0]
    else:
        num_sub_rows = 1

    total_rows = 1 + num_sub_rows * num_rows

    fig, axs = plt.subplots(total_rows, num_cols, figsize=(scale_factor * num_cols, total_rows * 1.3 * scale_factor))

    # plot original:
    axs[0, 0].axis('off')
    if original_image is not None:
        img = original_image.permute(1, 2, 0).cpu().detach()
        axs[0, 0].set_title('Original')
        axs[0, 0].imshow(img, interpolation='lanczos')

    axs[0, 1].axis('off')
    if original_reconstructed is not None:
        img = original_reconstructed.permute(1, 2, 0).cpu().detach()
        axs[0, 1].set_title('Original Null-Reconstructed')
        axs[0, 1].imshow(img, interpolation='lanczos')

    for j in range(2, num_cols):
        axs[0, j].axis('off')

    # plot counterfactuals
    for outer_row_idx in range(0, num_rows):
        row_idx = 1 + outer_row_idx * num_sub_rows
        for sub_row_idx in range(num_sub_rows):
            for col_idx in range(num_cols):
                img_idx = outer_row_idx * num_cols + col_idx
                ax = axs[row_idx + sub_row_idx, col_idx]
                if img_idx >= len(trajectory_imgs):
                    ax.axis('off')
                    continue
                if sub_row_idx == 0:
                    img = trajectory_imgs[img_idx]
                    img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)

                    ax.axis('off')
                    ax.imshow(img, interpolation='lanczos')

                    title = ''
                    if title_attributes is not None:
                        #should be a dict with tensor inside
                        for attribute_idx, (title_attribute, attribute_values) in enumerate(title_attributes.items()):
                            if attribute_idx == 0:
                                title += f'{img_idx} - {title_attribute}: {attribute_values[img_idx]:.5f}'
                            else:
                                title += f'\n{title_attribute}: {attribute_values[img_idx]:.5f}'

                    if loss_scores is not None:
                        for loss_name, loss_values in loss_scores.items():
                            title += f'\n{loss_name}: {loss_values[img_idx]:.5f}'

                    if regularizer_scores is not None:
                        for reg_name, reg_s in regularizer_scores.items():
                            title += f'\n{reg_name}: {reg_s[img_idx]:.5f}'

                    ax.set_title(title)
                else:
                    #heatmap
                    img = trajectory_imgs[img_idx]
                    img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)
                    cam = model_grad_cams[sub_row_idx - 1, img_idx]
                    #chw to hwc
                    img_np = img.numpy()
                    cam_np = cam.permute(1, 2, 0).numpy()

                    colormap = cv2.COLORMAP_JET
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), colormap)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = np.float32(heatmap) / 255

                    image_weight = 0.5
                    cam = (1 - image_weight) * heatmap + image_weight * img_np
                    cam = cam / np.max(cam)

                    ax.axis('off')
                    ax.set_title(f'CAM classifier {sub_row_idx - 1}')

                    ax.imshow(cam, interpolation='lanczos')

    plt.tight_layout()
    fig.savefig(filename)

    if wandb_name is not None:
        wandb.log({wandb_name: fig}, step=wandb_step)

    plt.close(fig)

def plot_attention(original_image, segmentation, words_attention_masks, filename, wandb_name=None, wandb_step=None):
    scale_factor = 4.0
    num_cols = 2 + len(words_attention_masks)
    num_rows = 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))

    for i in range(0, num_cols):
        if i == 0:
            img = original_image.cpu().detach()
            title = 'original'
        elif i == 1:
            img = segmentation.cpu().detach()
            title = 'segmentation'
        else:
            word = list(words_attention_masks.keys())[i-2]
            img = words_attention_masks[word].cpu().detach()
            title = word

        img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)

        ax = axs[i]
        ax.axis('off')
        ax.set_title(title)
        if i < 2:
            ax.imshow(img, interpolation='lanczos')
        else:
            ax.imshow(img, cmap='viridis', interpolation='lanczos')

    plt.tight_layout()
    fig.savefig(filename)

    if wandb_name is not None:
        wandb.log({wandb_name: fig}, step=wandb_step)


    plt.close(fig)
