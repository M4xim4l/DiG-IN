import os
import torch
import open_clip
from .models.model_wrappers import NormalizationAndResizeWrapper
from torchvision.transforms.functional import resize

class ClipImageEncoder(torch.nn.Module):
    def __init__(self, model, output_tokens=True):
        super().__init__()
        self.model = model
        self.output_tokens = output_tokens

    def forward(self, img, normalize=True):
        self.model.visual.output_tokens = self.output_tokens
        if self.output_tokens:
            _, tokens = self.model.encode_image(img)
            return tokens
        else:
            img_emb = self.model.encode_image(img)
            if normalize:
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            return img_emb.float()

def get_embeddings_from_tokens(tokens, visual_model):
    assert visual_model.attn_pool is None
    assert visual_model.pool_type == 'avg'
    if visual_model.final_ln_after_pool:
        pooled = tokens.mean(dim=1)
        pooled = visual_model.ln_post(pooled)
    else:
        pooled = tokens.mean(dim=1)

    embed = pooled @ visual_model.proj
    return embed

class MaskedLatentDist(torch.nn.Module):
    def __init__(self, device, model_name: str = 'ViT-L-14-CLIPA-336', pretrained: str = 'datacomp1b'):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)

        img_encoder = ClipImageEncoder(model)
        model.visual.set_grad_checkpointing(True)
        assert model.visual.image_size[0] == model.visual.image_size[1]
        self.clip_input_size = model.visual.image_size[0]
        self.img_encoder = NormalizationAndResizeWrapper(img_encoder, size=self.clip_input_size,
                                                    mean=torch.FloatTensor(model.visual.image_mean),
                                                    std=torch.FloatTensor(model.visual.image_std))
        self.img_encoder.eval()
        self.img_encoder = self.img_encoder.to(device)

        #use patchify to calculate average of mask over patch regions
        transformer_patchify_conv = model.visual.conv1
        self.mask_averaging_conv = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                              kernel_size=transformer_patchify_conv.kernel_size,
                                              stride=transformer_patchify_conv.kernel_size, bias=False)
        if len(transformer_patchify_conv.kernel_size) == 2:
            one_over_kernel_numel = 1. / (transformer_patchify_conv.kernel_size[0] * transformer_patchify_conv.kernel_size[1])
        else:
            raise NotImplementedError()

        self.mask_averaging_conv.weight.data.fill_(one_over_kernel_numel)
        self.mask_averaging_conv.to(device)
    def forward(self, in0, in1, masks):
        if len(in0.shape) == 3:
            in0 = in0[None, :, : ,:]
        if len(in1.shape) == 3:
            in1 = in1[None, :, :, :]
        if len(masks.shape) == 3:
            masks = masks[None, :, :, :]

        assert len(in0) == len(in1)
        assert len(in0) == len(masks)

        #forward through transformer and get tokens pre-projection
        y0_tokens = self.img_encoder(in0)
        y1_tokens = self.img_encoder(in1)

        #rescale masks
        with torch.no_grad():
            masks_reshaped = []
            for i in range(len(masks)):
                mask_i_rescaled = resize(masks[i], self.clip_input_size)
                masks_reshaped.append(mask_i_rescaled[None, :, :, :])
            masks_reshaped = torch.cat(masks_reshaped, dim=0)

            #patchify masks
            token_mask_weights = self.mask_averaging_conv(masks_reshaped)  # shape = [*, width, grid, grid]
            token_mask_weights = token_mask_weights.reshape(token_mask_weights.shape[0], token_mask_weights.shape[1], -1)  # shape = [*, width, grid ** 2]
            token_mask_weights = token_mask_weights.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        #mask tokens
        y0_tokens_masked = token_mask_weights * y0_tokens
        y1_tokens_masked = token_mask_weights * y1_tokens

        #project masked tokens
        visual_model = self.img_encoder.model.model.visual
        y0_embed = get_embeddings_from_tokens(y0_tokens_masked, visual_model)
        y1_embed = get_embeddings_from_tokens(y1_tokens_masked, visual_model)

        cosine_similarity = torch.nn.functional.cosine_similarity(y0_embed, y1_embed)
        distance = 1.0 - cosine_similarity
        return distance