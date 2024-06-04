import numpy as np
import torch
from kornia.morphology import erosion, dilation
from torchvision.transforms import functional as TF
from typing import List, Optional
from dataclasses import dataclass, field
from torch.distributions.categorical import Categorical

from .cross_attention import XA_STORE_INITIAL_CONDITIONAL_ONLY

try:
    from segment_anything_hq import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError as e:
    print('Could not import SAM HQ, pls install from: https://github.com/SysCV/sam-hq via: pip install segment-anything-hq')
    print(e)
    SAM_AVAILABLE = False

@dataclass
class SegmentationArgs:
    xa_map_resolution: int = 16
    xa_start_t: float  = 0.5
    xa_end_t: float = 1.0
    xa_segmentation_squash: Optional[str] = 'sqrt_0.3'

    sam_segmentation: bool = True
    sam_prompt: str = 'point'
    sam_num_points: int = 5
    sam_foreground_threshold: Optional[float] = 0.35
    sam_bb_quantile: Optional[float] = 0.97
    postprocessing: List[str] = field(default_factory=lambda: ['erosion_3', 'dilation_20', 'blur_45_30'])

def get_old_segmentation_args() -> SegmentationArgs:
    args = SegmentationArgs()
    args.sam_segmentation = False
    args.sam_foreground_threshold = None
    args.sam_bb_quantile = None
    args.postprocessing = []
    return args

def calculate_segmentation(starting_img, foreground_token_mask, latent_dimension, px_dimension, xa_maps,
                           word_to_token_embeddings, segmentation_args: SegmentationArgs):
    with torch.no_grad():

        latents_mask, px_mask, words_attention_masks = \
            calculate_xa_foreground_segmentation(foreground_token_mask, latent_dimension, px_dimension, xa_maps,
                                                 word_to_token_embeddings,
                                                 xa_map_resolution=segmentation_args.xa_map_resolution,
                                                 start_t=segmentation_args.xa_start_t,
                                                 end_t=segmentation_args.xa_end_t,
                                                 squash_function=segmentation_args.xa_segmentation_squash)
        if segmentation_args.sam_segmentation:
            px_mask = sam_segmentation(starting_img, px_mask, segmentation_args)

        latents_mask, px_mask = segmentation_post_processing(latents_mask, px_mask,
                                                         postprocessing= segmentation_args.postprocessing)
    return latents_mask, px_mask, words_attention_masks


def segmentation_post_processing(latents_mask, px_mask, postprocessing: List[str]):
    mask = px_mask
    for post in postprocessing:
        if 'erosion_' in post:
            kernel_size = int(post.split('_')[1])
            erosion_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float, device=px_mask.device)
            mask = erosion(mask[None, :, :, :], erosion_kernel).squeeze(dim=0)
        elif 'dilation_' in post:
            kernel_size = int(post.split('_')[1])
            dilation_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float, device=px_mask.device)
            mask = dilation(mask[None, :, :, :], dilation_kernel).squeeze(dim=0)
        elif 'blur_' in post:
            kernel_size = int(post.split('_')[1])
            kernel_sigma = float(post.split('_')[2])
            mask = TF.gaussian_blur(mask, kernel_size, kernel_sigma)
        else:
            raise ValueError('Please use postprocessing format: erosion_SIZE, dilation_SIZE, blur_SIZE_SIGMA')

    latent_mask = TF.resize(mask,(latents_mask.shape[1], latents_mask.shape[2]))

    return latent_mask, mask



#uses SAM HQ to calculate the segmentation on the original high-res image using a bounding box calculated from the
#XA segmentation
def sam_segmentation(img, px_xa_foreground_mask, segmentation_args: SegmentationArgs):
    assert SAM_AVAILABLE
    with torch.no_grad():
        model_type = "vit_h"  # "vit_l/vit_b/vit_h/vit_tiny"
        sam_checkpoint = "sam_hq/sam_hq_vit_h.pth"
        sam = sam_model_registry[model_type](checkpoint=None)
        with open(sam_checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        info = sam.load_state_dict(state_dict, strict=False)
        print(info)

        sam = sam.to(img.device)

        foreground_px = torch.nonzero(px_xa_foreground_mask.squeeze(dim=0) > segmentation_args.sam_foreground_threshold, as_tuple=True)
        if segmentation_args.sam_prompt == 'bb':
            try:
                y_sorted = torch.sort(foreground_px[0])[0]
                x_sorted = torch.sort(foreground_px[1])[0]

                bb_top = y_sorted[int((1 - segmentation_args.sam_bb_quantile) * len(y_sorted))].item()
                bb_bot = y_sorted[int(segmentation_args.sam_bb_quantile * len(y_sorted))].item()

                bb_l = x_sorted[int((1 - segmentation_args.sam_bb_quantile) * len(y_sorted))].item()
                bb_r = x_sorted[int(segmentation_args.sam_bb_quantile * len(y_sorted))].item()

                predictor = SamPredictor(sam)
                img_np = ((255 * img).permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8)
                predictor.set_image(img_np)
            except:
                return px_xa_foreground_mask

                #XYXY format
                input_box = np.array([bb_l, bb_top, bb_bot, bb_r])
                input_point, input_label = None, None
                hq_token_only = True
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    multimask_output=False,
                    hq_token_only=hq_token_only,
                )

                masks = torch.from_numpy(masks).to(img.device).float()
                sam_px_segmentation = masks
                return sam_px_segmentation
        elif segmentation_args.sam_prompt == 'point':

            try:
                foreground_xy_vals = [(x.item(), y.item(), px_xa_foreground_mask[0, y, x]) for (y, x) in
                                      zip(foreground_px[0], foreground_px[1])]
                vals = torch.tensor([abc[2] for abc in foreground_xy_vals])
                vals_normalized = vals / torch.sum(vals)

                input_points = []
                for idx in range(segmentation_args.sam_num_points):
                    if len(input_points) == 0:
                        prob = vals_normalized
                    else:
                        xys = torch.stack([torch.tensor([abc[0], abc[1]]) for abc in foreground_xy_vals])
                        dists = torch.cdist(torch.stack(input_points).float(), xys.float())
                        min_dists, _ = torch.min(dists, dim=0)
                        eps = 0.00001
                        dist_prob_normalized = (min_dists + eps)/ torch.sum(min_dists + eps)
                        prob = 0.5 * prob + 0.5 * dist_prob_normalized

                    m = Categorical(vals_normalized)

                    sample_idx = m.sample((1,))

                    abc = foreground_xy_vals[sample_idx]
                    input_points.append(torch.tensor([abc[0], abc[1]]))
            except:
                return px_xa_foreground_mask


            input_points = torch.stack(input_points, dim=0).numpy()
            input_labels = torch.ones(len(input_points)).numpy()
            input_box = None
            hq_token_only = True

            predictor = SamPredictor(sam)
            img_np = ((255 * img).permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8)
            predictor.set_image(img_np)

            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )

            masks = torch.from_numpy(masks).to(img.device).float()
            sam_px_segmentation = masks
            return sam_px_segmentation
        else:
            raise ValueError


#Calculates the segmentation of the foreground object given the XA maps and applies a post processing squash function
def calculate_xa_foreground_segmentation(foreground_token_mask, latent_dimension, px_dimension, xa_maps,
                                         word_to_token_embeddings, xa_map_resolution=16, start_t=0.75, end_t=1.0,
                                         squash_function=None):
    matching_dim_xa_maps = []
    xa_map_num_pixels = xa_map_resolution ** 2
    for t_idx, ca_maps_t in enumerate(xa_maps.values()):
        t_current = 1 - t_idx / (len(xa_maps.keys()) - 1)
        if t_current < start_t or t_current > end_t:
            continue

        for ca_map_t in ca_maps_t.values():
            h = ca_map_t.shape[0]
            if ca_map_t.shape[1] == xa_map_num_pixels:
                #only use conditional XA part
                if XA_STORE_INITIAL_CONDITIONAL_ONLY:
                    matching_dim_xa_maps.append(ca_map_t)
                else:
                    matching_dim_xa_maps.append(ca_map_t[h // 2:])
    stacked_xa_maps = torch.cat(matching_dim_xa_maps, dim=0)

    latent_foreground_segmentation = None
    px_foreground_segmentation = None
    words_attention_masks = {}
    for i in range(len(word_to_token_embeddings) + 1):
        if i == 0:
            stacked_xa_maps_foreground = stacked_xa_maps[:, :, foreground_token_mask.to(stacked_xa_maps.device)]
        else:
            word = list(word_to_token_embeddings.keys())[i - 1]
            token_embeddings_word = word_to_token_embeddings[word]
            word_token_mask = torch.zeros_like(foreground_token_mask)
            word_token_mask[torch.LongTensor(token_embeddings_word)] = 1
            stacked_xa_maps_foreground = stacked_xa_maps[:, :, word_token_mask.to(stacked_xa_maps.device)]

        xa_map_segmentation = stacked_xa_maps_foreground.mean(dim=2).mean(dim=0)
        xa_map_segmentation /= (xa_map_segmentation.max() + 1e-8)
        xa_map_segmentation_spatial = xa_map_segmentation.view(1, xa_map_resolution, xa_map_resolution)

        latent_xa_map_segmentation_spatial = torch.clamp(
            TF.resize(xa_map_segmentation_spatial, size=[latent_dimension, latent_dimension],
                      interpolation=TF.InterpolationMode.BICUBIC), 0.0, 1.0)
        px_xa_map_segmentation_spatial = torch.clamp(
            TF.resize(xa_map_segmentation_spatial, size=[px_dimension, px_dimension],
                      interpolation=TF.InterpolationMode.BICUBIC), 0.0, 1.0)
        if i == 0:
            latent_foreground_segmentation = apply_xa_segmentation_squash(squash_function, latent_xa_map_segmentation_spatial)
            px_foreground_segmentation = apply_xa_segmentation_squash(squash_function, px_xa_map_segmentation_spatial)
        else:
            words_attention_masks[word] = px_xa_map_segmentation_spatial.detach().cpu()
    return latent_foreground_segmentation, px_foreground_segmentation, words_attention_masks


def apply_xa_segmentation_squash(squash_function, xa_map_segmentation_spatial):
    if squash_function == 'none' or squash_function is None:
        pass
    elif 'sqr' in squash_function or 'linear' in squash_function or 'sqrt' in squash_function:
        cut_off_begin = float(squash_function.split('_')[1])
        xa_map_segmentation_spatial = (1/(1 - cut_off_begin)) * torch.clamp(xa_map_segmentation_spatial - cut_off_begin, 0.0, 1.0)
        if 'sqrt' in squash_function:
            xa_map_segmentation_spatial = torch.pow(xa_map_segmentation_spatial, 0.5)
        elif 'sqr' in squash_function:
            xa_map_segmentation_spatial = torch.pow(xa_map_segmentation_spatial, 2.0)
    elif 'threshold' in squash_function:
        threshold_value = float(squash_function.split('_')[1])
        xa_map_segmentation_spatial = xa_map_segmentation_spatial > threshold_value
    else:
        raise NotImplementedError()

    return xa_map_segmentation_spatial

