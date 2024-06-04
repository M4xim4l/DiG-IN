from collections import OrderedDict

import math
from diffusers import StableDiffusionXLPipeline
from typing import Callable, List, Optional, Union, Tuple, Dict, Any

import torch
import torch.nn.functional as F

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from .cross_attention import prepare_unet, free_unet, get_and_release_last_attention, \
    get_initial_cross_attention_all_timesteps, p2p_reshape_initial_cross_attention, get_interpolation_factor
from diffusers.schedulers import KarrasDiffusionSchedulers
from .sd_backprop_pipe import OptimParams, PromptToPromptParams, create_scheduler, setup_regularizers
from .foreground_segmentation import calculate_segmentation, SegmentationArgs
from dataclasses import dataclass, field

@dataclass
class OptimParamsXL(OptimParams):
    optimize_add: bool = True
    per_timestep_add: bool = True
    add_lr_factor: float = 1.0

class DenoisingLoop(torch.nn.Module):
    def __init__(self, unet, scheduler, progress_bar, num_inference_steps, timesteps, do_classifier_free_guidance,
                 guidance_scale, guidance_rescale, optim_params: OptimParamsXL, pass_timestep_to_xa=False,
                 p2p_enabled=False, p2p_params: Optional[PromptToPromptParams] = None, per_timestep_null_text=False,
                 do_checkpointing=True, generator_seed=None):
        super().__init__()
        self.scheduler = scheduler
        self.progress_bar = progress_bar
        self.per_timestep_conditioning_delta = optim_params.per_timestep_conditioning_delta
        self.per_timestep_uncond_delta = optim_params.per_timestep_uncond_delta
        self.per_timestep_null_text = per_timestep_null_text
        self.per_timestep_add_text = optim_params.per_timestep_add
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.do_checkpointing = do_checkpointing
        self.solver_order = optim_params.solver_order
        self.unet = unet
        self.xa_interpolation_schedule = p2p_params.xa_interpolation_schedule if p2p_params is not None else None
        self.self_interpolation_schedule = p2p_params.self_interpolation_schedule if p2p_params is not None else None
        self.pass_timestep_to_xa = pass_timestep_to_xa
        self.p2p_enabled = p2p_enabled
        self.p2p_mask_blend = p2p_params.p2p_mask_blend if p2p_params is not None else None
        self.loss_steps = optim_params.loss_steps
        self.generator_seed = generator_seed

    def get_timestep_variable(self, variable, delta, per_timestep, i):
        if delta is not None:
            if per_timestep:
                delta_i = delta[i]
            else:
                delta_i = delta

            return variable + delta_i.to(variable.dtype)
        else:
            return variable

    def get_step_function(self, force_no_mask_blend=False):
        def step_function(latents, uncond_embeds, uncond_embeds_delta, conditional_embeds, conditional_embeds_delta,
                          add_text_embeds, add_text_delta, add_time_ids, i, t, blend_latent_templates,
                          latents_xa_foreground_mask, generator, extra_step_kwargs):
            per_timestep_idx = i // self.solver_order + i % self.solver_order

            #conditioning
            cond_i = self.get_timestep_variable(conditional_embeds, conditional_embeds_delta,
                                                self.per_timestep_conditioning_delta, per_timestep_idx)

            if self.do_classifier_free_guidance:
                # null text
                if uncond_embeds is not None and self.per_timestep_null_text:
                    uncond_i = uncond_embeds[per_timestep_idx]
                else:
                    uncond_i = uncond_embeds

                uncond_i = self.get_timestep_variable(uncond_i, uncond_embeds_delta,
                                                      self.per_timestep_uncond_delta, per_timestep_idx)

                prompt_embeds = torch.cat([uncond_i, cond_i], dim=0)
            else:
                prompt_embeds = cond_i

            text_i = self.get_timestep_variable(add_text_embeds, add_text_delta,
                                       self.per_timestep_add_text, per_timestep_idx)

            #if self.p2p_enabled and not force_no_mask_blend and self.p2p_mask_blend is not None:
                # assert blend_latent_templates is not None
                # assert latents_xa_foreground_mask is not None
                #
                # mask_blend_factor = 1. - get_interpolation_factor(i, self.timesteps, self.p2p_mask_blend)
                # if mask_blend_factor > 0.0:
                #     blend_i = blend_latent_templates[i]
                #
                #     blend_strength = 0.5
                #     mask_multi = blend_strength * mask_blend_factor * (1. - latents_xa_foreground_mask[:, None, :, :])
                #     latents = mask_multi * blend_i + (1. - mask_multi) * latents

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if self.p2p_enabled:
                xa_attention_initial_interpolation_factor = \
                    get_interpolation_factor(i, len(self.timesteps), self.xa_interpolation_schedule)
                self_attention_initial_interpolation_factor = \
                    get_interpolation_factor(i, len(self.timesteps), self.self_interpolation_schedule)
                xa_args = {"timestep": t,
                           "xa_attention_initial_interpolation_factor": xa_attention_initial_interpolation_factor,
                           "self_attention_initial_interpolation_factor": self_attention_initial_interpolation_factor}
            elif self.pass_timestep_to_xa:
                xa_args = {"timestep": t}
            else:
                xa_args = None

            added_cond_kwargs = {"text_embeds": text_i, "time_ids": add_time_ids}
            unet_out = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=xa_args,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=True,
            )
            noise_pred = unet_out.sample

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            extra_step_kwargs['generator'] = generator
            scheduler_out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs,
                                                return_dict=True)
            latents = scheduler_out.prev_sample
            pred_original_sample = scheduler_out.pred_original_sample

            return latents, pred_original_sample

        return step_function

    def __call__(self, uncond_embeds, conditional_embeds, uncond_embeds_delta, conditional_embeds_delta, latents,
                 add_text_embeds, add_text_delta, add_time_ids, extra_step_kwargs, blend_latent_templates=None,
                 latents_xa_foreground_mask=None,
                 return_all_steps=False, force_no_mask_blend=False):
        self.scheduler.set_timesteps(self.num_inference_steps, device=latents.device)
        pred_originals = OrderedDict()

        if self.generator_seed is not None:
            generator = torch.Generator(conditional_embeds.device)
            generator.manual_seed(self.generator_seed)
        else:
            generator = None

        for i, t in enumerate(self.timesteps):
            step_function = self.get_step_function(force_no_mask_blend=force_no_mask_blend)
            step_return = step_function(latents, uncond_embeds, uncond_embeds_delta,
                                        conditional_embeds, conditional_embeds_delta,
                                        add_text_embeds, add_text_delta, add_time_ids,
                                        i, t, blend_latent_templates,
                                        latents_xa_foreground_mask, generator,
                                        extra_step_kwargs)
            latents, pred_original_sample = step_return

            if i >= (len(self.timesteps) - self.loss_steps) or return_all_steps:
                pred_originals[i] = pred_original_sample
            else:
                pred_originals[i] = None

        return pred_originals


def enable_gradient_checkpointing(model: torch.nn.Module):
    model.enable_gradient_checkpointing()
    model.train()
    module_different_train_behaviour = (torch.nn.modules.batchnorm._BatchNorm,
                                        torch.nn.modules.instancenorm._InstanceNorm,
                                        torch.nn.modules.dropout._DropoutNd,
                                        )
    for module_name, module in model.named_modules():
        if isinstance(module, module_different_train_behaviour):
            module.eval()
            #print(module_name)

def create_optimizer(initial_latents, uncond_embeds_delta, conditional_embeds_delta, add_delta, params: OptimParamsXL):
    optim_variables = []
    assert params.optimize_conditioning or params.optimize_latents or params.optimize_uncond
    if params.optimize_latents:
        initial_latents.requires_grad_(True)
        optim_variables.append({'params': [initial_latents], "lr": params.sgd_stepsize * params.latent_lr_factor})
    if add_delta is not None and params.optimize_add:
        add_delta.requires_grad_(True)
        optim_variables.append({'params': [add_delta], "lr": params.sgd_stepsize * params.add_lr_factor})
    if params.optimize_conditioning:
        conditional_embeds_delta.requires_grad_(True)
        optim_variables.append({'params': [conditional_embeds_delta], "lr": params.sgd_stepsize * params.conditioning_lr_factor})
    if uncond_embeds_delta is not None and params.optimize_uncond:
        uncond_embeds_delta.requires_grad_(True)
        optim_variables.append({'params': [uncond_embeds_delta], "lr": params.sgd_stepsize * params.uncond_lr_factor})
    if params.sgd_optim == 'adam':
        optim = torch.optim.Adam(optim_variables, lr=params.sgd_stepsize)
    elif params.sgd_optim == 'adamw':
        optim = torch.optim.AdamW(optim_variables, lr=params.sgd_stepsize)
    elif params.sgd_optim == 'sgd':
        optim = torch.optim.SGD(optim_variables, lr=params.sgd_stepsize, momentum=0.9)
    else:
        raise NotImplementedError()

    optim_variables_list = []
    for ov in optim_variables:
        optim_variables_list.extend(ov['params'])
    return optim, optim_variables_list

class StableDiffusionXLPipelineWithGrad(StableDiffusionXLPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`StableDiffusionXLPipeline.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.StableDiffusionXLPipeline.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, tokenizer=tokenizer,
                         tokenizer_2=tokenizer_2, unet=unet, image_encoder=image_encoder, feature_extractor=feature_extractor,
                         scheduler=scheduler, force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.watermark = None
        enable_gradient_checkpointing(self.unet)
        enable_gradient_checkpointing(self.vae)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing

    def encode_image_to_latent(self, img_in):
        img_rescaled = self.image_processor.preprocess(img_in)
        latent = self.vae.encode(img_rescaled).latent_dist.mean
        latent = latent * self.vae.config.scaling_factor
        return latent

    def decode_latent_to_img(self, latent_in):
        latents = 1 / self.vae.config.scaling_factor * latent_in
        image = self.vae.decode(latents).sample
        image = self.image_processor.postprocess(image, output_type='pt', do_denormalize=[True])
        return image

    @torch.no_grad()
    def __call__(
        self,
        targets_dict=None,
        losses_dict=None,
        starting_img=None,
        # loss and regs
        regularizers_weights={},
        # optim params
        optim_params: OptimParamsXL = OptimParams(),
        null_text_embeddings=None,
        # segmentation based regularizers
        prompt_foreground_key_words: List[str] = None,
        segmentation_args: SegmentationArgs = None,
        # Prompt-To-Prompt params
        p2p_params: Optional[PromptToPromptParams] = None,
        p2p_replacements: Optional[Tuple[str, str]] = None,
        #
        offload_clip_to_cpu=False,
        #
        generator_seed=None,
        #SD default params
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        conditional_embeds: Optional[torch.FloatTensor] = None,
        uncond_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            conditional_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            uncond_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            conditional_embeds,
            uncond_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = conditional_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        with torch.no_grad():
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            (
                conditional_embeds,
                uncond_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=conditional_embeds,
                negative_prompt_embeds=uncond_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )
            word_to_token_embeddings = self.get_word_to_token_embeddings(prompt)

        #calculate foreground token mask for segmentation calculation
        foreground_token_mask = None
        if prompt_foreground_key_words is not None:
            foreground_token_mask = torch.zeros(conditional_embeds.shape[1], device=conditional_embeds.device, dtype=torch.bool)
            for word in prompt_foreground_key_words:
                if word in word_to_token_embeddings:
                    for word_position in word_to_token_embeddings[word]:
                        foreground_token_mask[word_position] = 1

        if offload_clip_to_cpu:
            text_encoders = (
                [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
            )
            for te in text_encoders:
                te.to(torch.device('cpu'))

        #prompt to prompt initialization
        if p2p_replacements is not None:
            assert p2p_params is not None
            p2p_enabled = True
            p2p_prompt = prompt
            p2p_self_interpolation_schedule = p2p_params.self_interpolation_schedule
            p2p_xa_interpolation_schedule = p2p_params.xa_interpolation_schedule
            #replace words and note the positions in the prompt
            source_word, target_word = p2p_replacements
            assert source_word in prompt
            p2p_prompt = p2p_prompt.replace(source_word, target_word)

            with torch.no_grad():
                p2p_conditional_embeds, p2p_uncond_embeds = self.encode_prompt(
                    p2p_prompt,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                )
                p2p_word_to_token_embeddings = self.get_word_to_token_embeddings(p2p_prompt)
        else:
            assert p2p_params is None
            p2p_conditional_embeds = None
            p2p_enabled = False
            p2p_self_interpolation_schedule = None
            p2p_xa_interpolation_schedule = None

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        initial_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            conditional_embeds.dtype,
            device,
            generator,
            latents,
        )

        initial_latents = initial_latents.to(conditional_embeds.dtype)
        initial_latents = initial_latents.detach()
        initial_latents_norm = torch.norm(initial_latents.view(-1), p=2)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=conditional_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=conditional_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        #setup optimizer
        if optim_params.optimize_add:
            if optim_params.per_timestep_add:
                add_shape = (num_inference_steps,) + add_text_embeds.shape
            else:
                add_shape = add_text_embeds.shape

            add_delta = torch.zeros(add_shape, device=device, dtype=add_text_embeds.dtype)
        else:
            add_delta = None

        if optim_params.optimize_conditioning:
            if optim_params.per_timestep_conditioning_delta:
                conditioning_delta_shape = (num_inference_steps,) + conditional_embeds.shape
            else:
                conditioning_delta_shape = conditional_embeds.shape

            conditional_embeds_delta = torch.zeros(conditioning_delta_shape,
                                                   device=device, dtype=conditional_embeds.dtype)
        else:
            conditional_embeds_delta = None

        if uncond_embeds is not None and optim_params.optimize_uncond:
            if optim_params.per_timestep_uncond_delta:
                uncond_delta_shape = (num_inference_steps,) + uncond_embeds.shape
            else:
                uncond_delta_shape = uncond_embeds.shape

            uncond_embeds_delta = torch.zeros(uncond_delta_shape,
                                                   device=device, dtype=uncond_embeds.dtype)
        else:
            uncond_embeds_delta = None

        if uncond_embeds is not None and null_text_embeddings is not None:
            print('Using pre-defined Null-Text prompts')
            uncond_embeds = null_text_embeddings.to(device)
            per_timestep_null_text = True
        else:
            per_timestep_null_text = False


        optim, optim_variables = create_optimizer(initial_latents, uncond_embeds_delta, conditional_embeds_delta,
                                 add_delta, optim_params)
        optim_scheduler = create_scheduler(optim, optim_params)

        #freeze non optim parameters
        for module in [self.unet, self.vae, self.text_encoder]:
            for param in module.parameters():
                param.requires_grad_(False)

        if starting_img is not None:
            starting_latent = self.encode_image_to_latent(starting_img)
        else:
            starting_latent = None


        imgs_cpu = torch.zeros((optim_params.sgd_steps + 1, 3, height, width))
        loss_scores = {}
        regularizer_scores = {}

        regularizers_fs_ws_names, requires_xa_foreground_mask\
            = setup_regularizers(regularizers_weights, device)

        #Attention Storage
        xa_store_initial_attention = requires_xa_foreground_mask or p2p_xa_interpolation_schedule is not None
        xa_store_last_attention = False
        self_store_initial_attention = p2p_self_interpolation_schedule is not None
        self.unet = prepare_unet(self.unet, xa_store_initial_attention_map=xa_store_initial_attention,
                                 xa_store_last_attention_map=xa_store_last_attention,
                                 self_store_initial_attention_map=self_store_initial_attention,
                                 self_store_last_attention_map=False, store_in_ram=True, store_dtype=None)

        #DDIM Loop with gradient checkpointing
        pass_timestep_to_xa = xa_store_initial_attention or self_store_initial_attention
        ddim_loop = DenoisingLoop(self.unet, self.scheduler, self.progress_bar, num_inference_steps, timesteps,
                                  do_classifier_free_guidance, guidance_scale, guidance_rescale, optim_params,
                                  pass_timestep_to_xa=pass_timestep_to_xa,
                                  p2p_enabled=p2p_enabled, p2p_params=p2p_params,
                                  per_timestep_null_text=per_timestep_null_text,
                                  generator_seed=generator_seed)


        xa_store_initial_attention = requires_xa_foreground_mask or p2p_xa_interpolation_schedule is not None
        xa_store_last_attention = False
        self_store_initial_attention = p2p_self_interpolation_schedule is not None
        self.unet = prepare_unet(self.unet, xa_store_initial_attention_map=xa_store_initial_attention,
                                 xa_store_last_attention_map=xa_store_last_attention,
                                 self_store_initial_attention_map=self_store_initial_attention,
                                 self_store_last_attention_map=False, store_in_ram=True, store_dtype=None)

        with (torch.no_grad()):
            latents_xa_foreground_mask, px_xa_foreground_mask, reference_xa_maps, \
            words_attention_masks, initial_image, initial_intermediate_latents \
                = self.initial_denoising_loop(ddim_loop, starting_img, initial_latents, conditional_embeds,
                                              conditional_embeds_delta, uncond_embeds, uncond_embeds_delta,
                                              add_text_embeds, add_delta, add_time_ids, foreground_token_mask,
                                              requires_xa_foreground_mask, timesteps, width, device,
                                              word_to_token_embeddings, segmentation_args, extra_step_kwargs,
                                              p2p_enabled)

            #if we do not use p2p interpolation, we can go back to more efficient Attention implementation
            if not p2p_enabled:
                ddim_loop.pass_timestep_to_xa = False
                free_unet(self.unet)

            #if no starting image for regularisation is passed, use the first generated image
            if starting_img is None and initial_intermediate_latents is not None:
                starting_img = initial_image
                starting_latent = next(reversed(initial_intermediate_latents.values()))

            #remove latents if no blending is enabled
            if not p2p_enabled or p2p_params.p2p_mask_blend is None:
                initial_intermediate_latents = None

            #reshape cross attention if p2p is enabled and replace prompt embedding
            if p2p_enabled:
                raise NotImplementedError()
                # p2p_reshape_initial_cross_attention(self.unet, timesteps, word_to_token_embeddings,
                #                                     p2p_word_to_token_embeddings, conditional_embeds,
                #                                     p2p_conditional_embeds, p2p_replacements, device)
                # conditional_embeds = p2p_conditional_embeds

            # SGD loop
            for outer_iteration in range(optim_params.sgd_steps + 1):
                #calculate gradient of loss wrt to last latent x0
                with torch.enable_grad():
                    intermediate_preds = ddim_loop(uncond_embeds, conditional_embeds,
                                                 uncond_embeds_delta, conditional_embeds_delta,
                                                 initial_latents,
                                                 add_text_embeds, add_delta, add_time_ids,
                                                 extra_step_kwargs,
                                                 blend_latent_templates=initial_intermediate_latents,
                                                 latents_xa_foreground_mask=latents_xa_foreground_mask)


                    with torch.no_grad():
                        non_augmented_loss, image = self.calculate_loss(intermediate_preds, conditional_embeds_delta,
                                                                        uncond_embeds_delta, losses_dict, targets_dict,
                                                                        reference_xa_maps,
                                                                        latents_xa_foreground_mask,
                                                                        px_xa_foreground_mask, regularizers_fs_ws_names,
                                                                        optim_params.loss_steps,
                                                                        optim_params.loss_steps_schedule, starting_img,
                                                                        starting_latent, loss_scores=loss_scores,
                                                                        regularizer_scores=regularizer_scores,
                                                                        augment=False)
                    imgs_cpu[outer_iteration] = image.detach().cpu()
                    if outer_iteration == optim_params.sgd_steps:
                        break

                    loss, _ = self.calculate_loss(intermediate_preds, conditional_embeds_delta, uncond_embeds_delta,
                                                  losses_dict, targets_dict, reference_xa_maps,
                                                  latents_xa_foreground_mask, px_xa_foreground_mask,
                                                  regularizers_fs_ws_names, optim_params.loss_steps,
                                                  optim_params.loss_steps_schedule, starting_img, starting_latent,
                                                  augment=True)

                print_string = f'{outer_iteration} - Loss w. reg: {loss.item():.5f}' \
                               f' - Loss w. reg non augmented: {non_augmented_loss.item():.5f}'
                for loss_name, loss_s in loss_scores.items():
                    if loss_name != 'total':
                        print_string += f' - {loss_name}: {loss_s[-1]:.5f}'
                for reg_name, reg_s in regularizer_scores.items():
                    print_string += f' - {reg_name}: {reg_s[-1]:.5f}'

                if optim_params.early_stopping_loss is not None and loss < optim_params.early_stopping_loss:
                    imgs_cpu = imgs_cpu[:(1 + outer_iteration)]
                    print('Early stopping criterion reached')
                    break

                print(print_string)
                loss.backward(inputs=optim_variables)

                if optim_params.gradient_clipping is not None:
                    for optim_variable_dict in optim.param_groups:
                        for var in optim_variable_dict['params']:
                            torch.nn.utils.clip_grad_norm_(var, optim_params.gradient_clipping, 'inf')
                if optim_params.normalize_gradients:
                    for optim_variable_dict in optim.param_groups:
                        for var in optim_variable_dict['params']:
                            grad_non_normalized = var.grad
                            grad_norm = torch.norm(grad_non_normalized.view(-1), p=2).item()
                            var.grad /= grad_norm

                #update parameters
                optim.step()
                if optim_params.optimize_latents:
                    scale_factor = initial_latents_norm / torch.norm(initial_latents.view(-1), p=2)
                    initial_latents.mul_(scale_factor)
                if optim_scheduler is not None:
                    optim_scheduler.step()
                optim.zero_grad()

        #free_unet(self.unet)
        # Offload all models
        self.maybe_free_model_hooks()

        return_values = {
            'imgs': imgs_cpu,
            'loss_scores': loss_scores,
            'regularizer_scores': regularizer_scores,
            'px_foreground_segmentation': px_xa_foreground_mask.cpu() if px_xa_foreground_mask is not None else None,
            'latents_foreground_segmentation': latents_xa_foreground_mask.cpu() if latents_xa_foreground_mask is not None else None,
            'words_attention_masks': words_attention_masks,
            'initial_img': initial_image.cpu() if initial_image is not None else None,
        }

        return return_values

    def initial_denoising_loop(self, ddim_loop: DenoisingLoop, starting_img, initial_latents, conditional_embeds,
                               conditional_embeds_delta, uncond_embeds, uncond_embeds_delta, add_text_embeds, add_delta,
                               add_time_ids, foreground_token_mask, requires_xa_foreground_mask, timesteps, width,
                               device, word_to_token_embeddings, segmentation_args, extra_step_kwargs, p2p_enabled):
        latents_xa_foreground_mask = None
        px_xa_foreground_mask = None
        reference_xa_maps = None
        words_attention_masks = None
        initial_img = None
        intermediate_latents = None
        with torch.no_grad():
            if p2p_enabled or requires_xa_foreground_mask:
                intermediate_latents = ddim_loop(uncond_embeds, conditional_embeds,
                                                 uncond_embeds_delta, conditional_embeds_delta,
                                                 initial_latents, add_text_embeds, add_delta,
                                                 add_time_ids, extra_step_kwargs,
                                                 extra_step_kwargs,
                                                 return_all_steps=True, force_no_mask_blend=True)

                latents = next(reversed(intermediate_latents.values()))
                initial_img = self.decode_latent_to_img(latents).detach().squeeze(dim=0)
                if starting_img is None:
                    starting_img = initial_img

                # save first cross attention map, used for segmentation calculation or prompt-2-prompt style xa editing
                if requires_xa_foreground_mask:
                    reference_xa_maps = get_initial_cross_attention_all_timesteps(self.unet, timesteps)
                    # calculate latent "segmentation" based on foreground word
                    assert segmentation_args is not None
                    latents_xa_foreground_mask, px_xa_foreground_mask, words_attention_masks = \
                        calculate_segmentation(starting_img, foreground_token_mask, width // self.vae_scale_factor,
                                                             width, reference_xa_maps, word_to_token_embeddings,
                                                             segmentation_args=segmentation_args)
                    latents_xa_foreground_mask = latents_xa_foreground_mask.to(device)
                    px_xa_foreground_mask = px_xa_foreground_mask.to(device)


        return (latents_xa_foreground_mask, px_xa_foreground_mask, reference_xa_maps,
                words_attention_masks, initial_img, intermediate_latents)



    def calculate_loss(self, intermediate_preds, conditional_embeds_delta, uncond_embeds_delta, losses_dict,
                       targets_dict, reference_xa_maps, latents_xa_foreground_mask, px_xa_foreground_mask,
                       regularizers_fs_ws_names, loss_steps, loss_steps_schedule, starting_img, starting_latent,
                       loss_scores=None, regularizer_scores=None, augment=True):

        loss = 0

        for i, latents in intermediate_preds.items():
            if not augment:
                #without augment we only evaluate the loss value of the final image
                if i < (len(intermediate_preds) - 1):
                    continue
                else:
                    loss_w_i = 1.0
            else:
                if i < (len(intermediate_preds) - loss_steps):
                    continue
                else:
                    if loss_steps_schedule == 'uniform':
                        loss_w_i = 1 / loss_steps
                    elif loss_steps_schedule == 'linear':
                        loss_w_i = (1 + loss_steps - len(intermediate_preds) + i) / sum(range(1 + loss_steps))
                    else:
                        raise ValueError()

            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.decode_latent_to_img(latents)
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            if losses_dict is not None:
                if not isinstance(losses_dict, (tuple, list)):
                    losses_dict = [losses_dict]
                for loss_dict in losses_dict:
                    loss_function = loss_dict['loss']
                    loss_name = loss_dict['name']
                    loss_w = loss_dict['weight']

                    loss_target = targets_dict[loss_name]
                    loss_value = loss_function(image, loss_target, augment=augment)
                    loss += loss_w * loss_w_i * loss_value

                    #log sub lossees
                    if loss_scores is not None and (i == len(intermediate_preds) - 1):
                        if not loss_name in loss_scores:
                            loss_scores[loss_name] = []
                        loss_scores[loss_name].append(loss_value.detach().item())

            regularizer_kwargs = {
                'image': image,
                'latents': latents,
                'starting_img': starting_img,
                'starting_latent': starting_latent,
                'conditional_embeds_delta': conditional_embeds_delta,
                'uncond_embeds_delta': uncond_embeds_delta,
                'reference_xa': reference_xa_maps,
                'latents_xa_foreground_mask': latents_xa_foreground_mask,
                'px_xa_foreground_mask': px_xa_foreground_mask,
            }

            for (regularizer_f, regularizer_w, regularizer_name) in regularizers_fs_ws_names:
                reg_score = regularizer_f(**regularizer_kwargs)
                loss = loss + loss_w_i * regularizer_w * reg_score
                #log regularizers
                if regularizer_scores is not None and (i == len(intermediate_preds) - 1):
                    if not regularizer_name in regularizer_scores:
                        regularizer_scores[regularizer_name] = []
                    regularizer_scores[regularizer_name].append(reg_score.detach().item())

            #log total loss
            if loss_scores is not None and (i == len(intermediate_preds) - 1):
                if not 'total' in loss_scores:
                    loss_scores['total'] = []
                loss_scores['total'].append(loss.detach().item())

        return loss, image

    def get_word_to_token_embeddings(self, prompt):
        # positions
        single_word_enc = [(x, self.tokenizer(x,
                                 padding=False,
                                 max_length=self.tokenizer.model_max_length,
                                 truncation=True,
                                 return_tensors="pt",
                                 )['input_ids']) for x in prompt.split()]

        #chars that will be ignored
        replacements = [(',', ''), ('.', '')]

        # clip encoding always has a 49406/49407 at the start/end of sentence which does not correspond to an input word
        token_map_idx = 1

        word_to_token_embeddings = {}
        for word, token in single_word_enc:
            for char, replacement in replacements:
                word = word.replace(char, replacement)
            for ids in token.view(-1):
                # ignore start/end of sentence tokens
                if ids.item() != 49406 and ids.item() != 49407:
                    if word in word_to_token_embeddings:
                        word_to_token_embeddings[word].append(token_map_idx)
                    else:
                        word_to_token_embeddings[word] = [token_map_idx]
                    token_map_idx += 1

        return word_to_token_embeddings


