import itertools
from collections import OrderedDict
from typing import Callable, List, Optional, Union, Tuple
from typing import List, Optional
from dataclasses import dataclass, field

import kornia.filters
import torch
import torch.nn.functional as F

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.utils import (
    logging,
)
from lpips import LPIPS
from functools import partial

from .masked_lpips import MaskedLPIPS
from .masked_latent_dist import MaskedLatentDist
from .cross_attention import (prepare_unet, free_unet, get_and_release_last_attention,
                              get_initial_cross_attention_all_timesteps, p2p_reshape_initial_cross_attention,
                              get_interpolation_factor)
from .foreground_segmentation import calculate_segmentation, SegmentationArgs

logger = logging.get_logger(__name__)


@dataclass
class OptimParams:
    sgd_steps: int = 20
    sgd_stepsize: float = 0.01
    sgd_optim: str = 'adam'
    sgd_scheduler: Optional[str] = None
    latent_lr_factor: float = 0.1
    conditioning_lr_factor: float = 1.0
    uncond_lr_factor: float = 1.0
    early_stopping_loss: Optional[float] = None
    loss_steps: int = 3
    loss_steps_schedule: str = 'linear'
    solver_order: int = 1
    optimize_latents: bool = True
    optimize_conditioning: bool = True
    per_timestep_conditioning_delta: bool = True
    optimize_uncond: bool = True
    per_timestep_uncond_delta: bool = True
    normalize_gradients: bool = False
    gradient_clipping: Optional[float] = None

@dataclass
class PromptToPromptParams:
    do_p2p: bool = True
    xa_interpolation_schedule: Optional[List[str]] =  field(default_factory=lambda: [
        #'threshold_0.0_0.1',
        #'threshold_0.0_0.2',
        'threshold_0.0_0.3',
        'threshold_0.0_0.4',
        'threshold_0.0_0.5',
    ])
    self_interpolation_schedule: Optional[List[str]] = None
    optimize_xa_interpolation_schedule: bool = False
    optimize_self_interpolation_schedule: bool = False
    schedule_stepsize: float = 0.05

@dataclass
class MaskBlendParams:
    do_mask_blend: bool = True
    foreground_blend: bool = True
    optimize_mask_blend: bool = False
    initial_mask_blend_schedule: List[str]  = field(default_factory=lambda: [
        #'none',
        #'inv-threshold_0.8',
        #'inv-threshold_0.5',
        'threshold_0.3',
        # 'threshold_0.8',
        # 'threshold_0.5'
    ])
    mask_blend_stepsize: float = 0.05

class DenoisingLoop(torch.nn.Module):
    def __init__(self, unet, scheduler, progress_bar, num_inference_steps, timesteps, do_classifier_free_guidance,
                 guidance_scale, optim_params: OptimParams, pass_timestep_to_xa=False, per_timestep_null_text=False,
                 do_checkpointing=True):
        super().__init__()
        self.scheduler = scheduler
        self.progress_bar = progress_bar
        self.per_timestep_conditioning_delta = optim_params.per_timestep_conditioning_delta
        self.per_timestep_uncond_delta = optim_params.per_timestep_uncond_delta
        self.per_timestep_null_text = per_timestep_null_text
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.guidance_scale = guidance_scale
        self.do_checkpointing = do_checkpointing
        self.solver_order = optim_params.solver_order
        self.unet = unet
        self.pass_timestep_to_xa = pass_timestep_to_xa
        self.loss_steps = optim_params.loss_steps

    def get_step_function(self):
        def step_function(latents, uncond_embeds, uncond_embeds_delta, conditional_embeds, conditional_embeds_delta, i,
                          t, blend_latent_templates, blend_mask, blend_mask_schedule, xa_interpolation_schedule,
                          self_interpolation_schedule, extra_step_kwargs):
            per_timestep_idx = i // self.solver_order + i % self.solver_order
            if conditional_embeds_delta is not None:
                if self.per_timestep_conditioning_delta:
                    conditional_embeds_delta_i = conditional_embeds_delta[per_timestep_idx]
                else:
                    conditional_embeds_delta_i = conditional_embeds_delta
                cond_i = conditional_embeds + conditional_embeds_delta_i
            else:
                cond_i = conditional_embeds

            if self.per_timestep_null_text:
                uncond_i = uncond_embeds[per_timestep_idx]
            else:
                uncond_i = uncond_embeds

            if uncond_embeds_delta is not None:
                if self.per_timestep_uncond_delta:
                    uncond_embeds_delta_i = uncond_embeds_delta[per_timestep_idx]
                else:
                    uncond_embeds_delta_i = uncond_embeds_delta
                uncond_i = uncond_i + uncond_embeds_delta_i

            prompt_embeds = torch.cat([uncond_i, cond_i])

            #
            if blend_mask is not None and blend_mask_schedule is not None:
                assert blend_latent_templates is not None
                mask_blend_factor_i = blend_mask_schedule[i]
                blend_template_i = blend_latent_templates[i]
                blend_mask_i = mask_blend_factor_i * blend_mask[:, None]
                latents = blend_mask_i * blend_template_i + (1. - blend_mask_i) * latents

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if xa_interpolation_schedule is not None or self_interpolation_schedule is not None:
                xa_int_i = xa_interpolation_schedule[i] if xa_interpolation_schedule is not None else 0
                self_int_i = self_interpolation_schedule[i] if self_interpolation_schedule is not None else 0
                xa_args = {"timestep": t,
                           "xa_attention_initial_interpolation_factor": xa_int_i,
                           "self_attention_initial_interpolation_factor": self_int_i}
            elif self.pass_timestep_to_xa:
                xa_args = {"timestep": t}
            else:
                xa_args = None
            unet_out = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,
                                 cross_attention_kwargs=xa_args)
            noise_pred = unet_out.sample

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
            latents = scheduler_out.prev_sample
            pred_original_sample = scheduler_out.pred_original_sample

            return latents, pred_original_sample

        return step_function

    def __call__(self, uncond_embeds, conditional_embeds, uncond_embeds_delta, conditional_embeds_delta, latents,
                 extra_step_kwargs, xa_interpolation_schedule=None, self_interpolation_schedule=None,
                 blend_latent_templates=None, blend_mask=None, blend_mask_schedule=None, return_all_steps=False,
                 return_intermediate_latents=False):
        self.scheduler.set_timesteps(self.num_inference_steps, device=latents.device)
        pred_originals = OrderedDict()
        intermediate_latents = OrderedDict()
        for i, t in enumerate(self.timesteps):
            if return_intermediate_latents:
                intermediate_latents[i] = latents.detach().clone()

            step_function = self.get_step_function()
            step_return = step_function(latents, uncond_embeds, uncond_embeds_delta,
                                        conditional_embeds, conditional_embeds_delta,
                                        i, t,
                                        blend_latent_templates,
                                        blend_mask,
                                        blend_mask_schedule,
                                        xa_interpolation_schedule,
                                        self_interpolation_schedule,
                                        extra_step_kwargs)
            latents, pred_original_sample = step_return

            if i >= (len(self.timesteps) - self.loss_steps) or return_all_steps:
                pred_originals[i] = pred_original_sample
            else:
                pred_originals[i] = None


        if return_intermediate_latents:
            return pred_originals, intermediate_latents
        else:
            return pred_originals


def create_optimizer(initial_latents, uncond_embeds_delta, conditional_embeds_delta, params: OptimParams,
                     blend_mask=None, blend_mask_lr=None, xa_schedule=None, self_schedule=None, schedule_lr=None):
    optim_variables = []
    assert params.optimize_conditioning or params.optimize_latents or params.optimize_uncond
    if params.optimize_latents:
        initial_latents.requires_grad_(True)
        optim_variables.append({'params': [initial_latents], "lr": params.sgd_stepsize * params.latent_lr_factor})
    if params.optimize_conditioning:
        conditional_embeds_delta.requires_grad_(True)
        optim_variables.append({'params': [conditional_embeds_delta], "lr": params.sgd_stepsize * params.conditioning_lr_factor})
    if uncond_embeds_delta is not None and params.optimize_uncond:
        uncond_embeds_delta.requires_grad_(True)
        optim_variables.append({'params': [uncond_embeds_delta], "lr": params.sgd_stepsize * params.uncond_lr_factor})

    if blend_mask is not None:
        assert blend_mask_lr is not None
        blend_mask.requires_grad_(True)
        optim_variables.append({'params': [blend_mask], "lr": blend_mask_lr})

    if xa_schedule is not None:
        assert schedule_lr is not None
        xa_schedule.requires_grad_(True)
        optim_variables.append({'params': [xa_schedule], "lr": schedule_lr})

    if self_schedule is not None:
        assert schedule_lr is not None
        self_schedule.requires_grad_(True)
        optim_variables.append({'params': [self_schedule], "lr": schedule_lr})

    if params.sgd_optim == 'adam':
        optim = torch.optim.Adam(optim_variables, lr=params.sgd_stepsize)
    elif params.sgd_optim == 'adamw':
        optim = torch.optim.AdamW(optim_variables, lr=params.sgd_stepsize)
    elif params.sgd_optim == 'sgd':
        optim = torch.optim.SGD(optim_variables, lr=params.sgd_stepsize, momentum=0.9)
    else:
        raise NotImplementedError()

    return optim

def create_scheduler(optim, optim_params):
    if optim_params.sgd_scheduler is None or optim_params.sgd_scheduler == 'none':
        optim_scheduler = None
    elif optim_params.sgd_scheduler == 'cosine':
        optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, optim_params.sgd_steps,
                                                                     eta_min=optim_params.sgd_stepsize / 10)
    else:
        raise NotImplementedError()
    return optim_scheduler

def setup_regularizers(regularizers_weights, device):
    regularizers_fs_ws_names = []
    requires_xa_foreground_mask = False
    for regularizer_name, regularizer_w in regularizers_weights.items():
        if regularizer_name == 'px_l1':
            def reg(**kwargs):
                starting_img = kwargs['starting_img']
                image = kwargs['image'].squeeze()
                px_dist = F.l1_loss(image, starting_img, reduction='mean')
                return px_dist
        elif regularizer_name == 'px_l2':
            def reg(**kwargs):
                starting_img = kwargs['starting_img']
                image = kwargs['image'].squeeze()
                px_dist = F.mse_loss(image, starting_img, reduction='mean')
                return px_dist
        elif regularizer_name == 'px_lpips':
            loss_fn_alex = LPIPS(net='alex').to(device)
            def reg(**kwargs):
                starting_img = kwargs['starting_img']
                image = kwargs['image']
                px_dist = loss_fn_alex(image, starting_img[None, :, :, :], normalize=True).mean()
                return px_dist
        elif regularizer_name == 'latent_l1':
            def reg(**kwargs):
                starting_latent = kwargs['starting_latent']
                latents = kwargs['latents']

                latent_dist = F.l1_loss(latents.view(-1), starting_latent.view(-1), reduction='mean')
                return latent_dist
        elif regularizer_name == 'latent_l2':
            def reg(**kwargs):
                starting_latent = kwargs['starting_latent']
                latents = kwargs['latents']

                latent_dist = F.mse_loss(latents.view(-1), starting_latent.view(-1), reduction='mean')
                return latent_dist
        elif regularizer_name in ['px_foreground_l2', 'px_background_l2', 'px_foreground_l1', 'px_background_l1']:
            requires_xa_foreground_mask = True

            def reg_fn(regularizer_name, **kwargs):
                xa_foreground_mask = kwargs['px_xa_foreground_mask']
                if 'background' in regularizer_name:
                    mask = (1. - xa_foreground_mask)
                elif 'foreground' in regularizer_name:
                    mask = xa_foreground_mask
                else:
                    raise NotImplementedError()

                starting_img = kwargs['starting_img']
                image = kwargs['image'].squeeze()
                if 'l2' in regularizer_name:
                    px_dist = torch.mean(mask * (image - starting_img)**2)
                elif 'l1' in regularizer_name:
                    px_dist = torch.mean(mask * (image - starting_img).abs())
                else:
                    raise NotImplementedError()

                return px_dist

            reg = partial(reg_fn, regularizer_name)
        elif regularizer_name in ['px_foreground_lpips', 'px_background_lpips']:
            requires_xa_foreground_mask = True
            loss_fn_alex_masked = MaskedLPIPS(net='alex').to(device)
            def reg_fn(regularizer_name, **kwargs):
                xa_foreground_mask = kwargs['px_xa_foreground_mask']
                if 'background' in regularizer_name:
                    mask = (1. - xa_foreground_mask)
                elif 'foreground' in regularizer_name:
                    mask = xa_foreground_mask
                else:
                    raise NotImplementedError()

                starting_img = kwargs['starting_img']
                image = kwargs['image']
                px_dist = loss_fn_alex_masked(image, starting_img[None, :, :, :],
                                              mask[None, :, :, :], normalize=True).mean()
                return px_dist

            reg = partial(reg_fn, regularizer_name)
        elif regularizer_name in ['px_foreground_clip', 'px_background_clip']:
            requires_xa_foreground_mask = True
            loss_fn_clip_masked = MaskedLatentDist(device)
            def reg_fn(regularizer_name, **kwargs):
                xa_foreground_mask = kwargs['px_xa_foreground_mask']
                if 'background' in regularizer_name:
                    mask = (1. - xa_foreground_mask)
                elif 'foreground' in regularizer_name:
                    mask = xa_foreground_mask
                else:
                    raise NotImplementedError()

                starting_img = kwargs['starting_img']
                image = kwargs['image']
                px_dist = loss_fn_clip_masked(image, starting_img[None, :, :, :],
                                              mask[None, :, :, :]).mean()
                return px_dist

            reg = partial(reg_fn, regularizer_name)
        elif regularizer_name in ['latent_foreground_l2', 'latent_background_l2', 'latent_foreground_l1', 'latent_background_l1']:
            requires_xa_foreground_mask = True
            def reg_fn(regularizer_name, **kwargs):
                starting_latent = kwargs['starting_latent']
                latents = kwargs['latents']
                xa_foreground_mask = kwargs['latents_xa_foreground_mask']
                if 'background' in regularizer_name:
                    mask = (1. - xa_foreground_mask[None, :])
                elif 'foreground' in regularizer_name:
                    mask = xa_foreground_mask[None, :]
                else:
                    raise NotImplementedError()

                if 'l2' in regularizer_name:
                    latent_dist = torch.mean(mask * (latents - starting_latent)**2)
                elif 'l1' in regularizer_name:
                    latent_dist = torch.mean(mask * (latents - starting_latent).abs())
                else:
                    raise NotImplementedError()

                return latent_dist

            reg = partial(reg_fn, regularizer_name)
        else:
            raise NotImplementedError(regularizer_name)

        regularizers_fs_ws_names.append((reg, regularizer_w, regularizer_name))

    return regularizers_fs_ws_names, requires_xa_foreground_mask

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

class StableDiffusionPipelineWithGrad(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 image_encoder: CLIPVisionModelWithProjection = None,
                 requires_safety_checker: bool = True,
                 ):
        super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                         safety_checker=safety_checker, feature_extractor=feature_extractor,
                         requires_safety_checker=requires_safety_checker)
        enable_gradient_checkpointing(self.unet)
        enable_gradient_checkpointing(self.vae)

    def decode_latent_to_img(self, latent_in):
        latents = 1 / self.vae.config.scaling_factor * latent_in
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5)
        image = torch.clamp(image, 0.0, 1.0)
        return image

    def encode_image_to_latent(self, img_in):
        img_rescaled = 2 * img_in[None,:,:,:] - 1
        latent = self.vae.encode(img_rescaled).latent_dist.mean
        latent = latent * self.vae.config.scaling_factor
        return latent

    def convert_to_double(self):
        self.unet.double()
        self.text_encoder.double()
        #stuff in image space is typically fine in single precision
        #self.vae.double()

    @torch.no_grad()
    def __call__(self,
                 targets_dict=None,
                 losses_dict=None,
                 starting_img=None,
                 #loss and regs
                 regularizers_weights=None,
                 #optim params
                 optim_params: OptimParams = OptimParams(),
                 null_text_embeddings=None,
                 #segmentation based regularizers
                 prompt_foreground_key_words: List[str] = None,
                 segmentation_args: SegmentationArgs = None,
                 #Prompt-To-Prompt params
                 p2p_params: Optional[PromptToPromptParams] = None,
                 p2p_replacements: Optional[Tuple[str, str]] = None,
                 #Mask-blend params
                 mask_blend_params: Optional[MaskBlendParams] = None,
                 #SD params
                 height: Optional[int] = None, width: Optional[int] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 latents: Optional[torch.FloatTensor] = None,
                 prompt: Union[str, List[str]] = None,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 eta: float = 0.0,
                 prompt_embeds: Optional[torch.FloatTensor] = None,
                 negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 ):
        num_images_per_prompt = 1


        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, 1, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            print('We only support optimization with guidance scale > 1.0')

        # 3. Encode input prompt
        assert prompt_embeds is None and negative_prompt_embeds is None

        with torch.no_grad():
            conditional_embeds, uncond_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
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

        #prompt to prompt initialization
        if p2p_params is not None and p2p_params.do_p2p:
            assert p2p_replacements is not None
            p2p_enabled = True
            p2p_prompt = prompt
            #replace words and note the positions in the prompt
            source_word, target_word = p2p_replacements
            assert source_word in prompt
            p2p_prompt = p2p_prompt.replace(source_word, target_word)

            with (torch.no_grad()):
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
            p2p_conditional_embeds = None
            p2p_enabled = False

        #Setup
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        assert generator is None
        initial_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            uncond_embeds.dtype,
            device,
            generator,
            latents,
        )
        #initial_latents = initial_latents.to(self.dtype)
        initial_latents = initial_latents.detach()
        initial_latents_norm = torch.norm(initial_latents.view(-1), p=2)

        if optim_params.optimize_conditioning:
            if optim_params.per_timestep_conditioning_delta:
                conditioning_delta_shape = (num_inference_steps,) + conditional_embeds.shape
            else:
                conditioning_delta_shape = conditional_embeds.shape

            conditional_embeds_delta = torch.zeros(conditioning_delta_shape,
                                                   device=device, dtype=conditional_embeds.dtype)
        else:
            conditional_embeds_delta = None

        if optim_params.optimize_uncond:
            if optim_params.per_timestep_uncond_delta:
                uncond_delta_shape = (num_inference_steps,) + uncond_embeds.shape
            else:
                uncond_delta_shape = uncond_embeds.shape

            uncond_embeds_delta = torch.zeros(uncond_delta_shape,
                                                   device=device, dtype=uncond_embeds.dtype)
        else:
            uncond_embeds_delta = None

        if null_text_embeddings is not None:
            print('Using pre-defined Null-Text prompts')
            uncond_embeds = null_text_embeddings.to(device)
            per_timestep_null_text = True
        else:
            per_timestep_null_text = False

        #freeze non optim parameters
        for module in [self.unet, self.vae, self.text_encoder]:
            for param in module.parameters():
                param.requires_grad_(False)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if starting_img is not None:
            starting_latent = self.encode_image_to_latent(starting_img)
        else:
            starting_latent = None

        #prepare regularizers
        regularizers_weights = regularizers_weights if regularizers_weights is not None else {}
        regularizers_fs_ws_names, requires_xa_foreground_mask\
            = setup_regularizers(regularizers_weights, device)

        imgs_cpu = torch.zeros((optim_params.sgd_steps + 1, 3, height, width))
        loss_scores = {}
        regularizer_scores = {}

        #Attention Storage
        xa_store_initial_attention = requires_xa_foreground_mask or (p2p_params is not None and p2p_params.xa_interpolation_schedule is not None)
        xa_store_last_attention = False
        self_store_initial_attention = p2p_params is not None and p2p_params.self_interpolation_schedule is not None
        self.unet = prepare_unet(self.unet, xa_store_initial_attention_map=xa_store_initial_attention,
                                 xa_store_last_attention_map=xa_store_last_attention,
                                 self_store_initial_attention_map=self_store_initial_attention,
                                 self_store_last_attention_map=False, store_in_ram=True, store_dtype=None)

        #DDIM Loop with gradient checkpointing
        pass_timestep_to_xa = xa_store_initial_attention or self_store_initial_attention
        ddim_loop = DenoisingLoop(self.unet, self.scheduler, self.progress_bar, num_inference_steps, timesteps,
                                  do_classifier_free_guidance, guidance_scale, optim_params,
                                  pass_timestep_to_xa=pass_timestep_to_xa,
                                  per_timestep_null_text=per_timestep_null_text)

        # 7. SGD loop
        with (torch.no_grad()):
            (latents_xa_foreground_mask, px_xa_foreground_mask, reference_xa_maps,
             words_attention_masks, initial_final_img, initial_final_latent, initial_intermediate_latents) \
                = self.initial_denoising_loop(ddim_loop, starting_img, initial_latents, conditional_embeds,
                                              conditional_embeds_delta, uncond_embeds, uncond_embeds_delta,
                                              foreground_token_mask, requires_xa_foreground_mask, timesteps, width,
                                              device, word_to_token_embeddings, segmentation_args, extra_step_kwargs,
                                              p2p_enabled)

            #if we do not use p2p interpolation, we can go back to more efficient Attention implementation
            if not p2p_enabled:
                ddim_loop.pass_timestep_to_xa = False
                free_unet(self.unet)

            #if no starting image for regularisation is passed, use the first generated image
            if starting_img is None and initial_final_img is not None and initial_final_latent is not None:
                starting_img = initial_final_img
                starting_latent = initial_final_latent

            #mask blend
            if mask_blend_params is not None and mask_blend_params.do_mask_blend:
                assert starting_latent is not None
                assert latents_xa_foreground_mask is not None
                if mask_blend_params.foreground_blend:
                    blend_mask = latents_xa_foreground_mask
                else:
                    blend_mask = 1.0 - latents_xa_foreground_mask
            else:
                #remove latents if no blending is enabled
                initial_intermediate_latents = None
                blend_mask = None
                blend_mask_schedule = None

            #reshape cross attention if p2p is enabled and replace prompt embedding
            if p2p_enabled:
                p2p_reshape_initial_cross_attention(self.unet, timesteps, word_to_token_embeddings,
                                                    p2p_word_to_token_embeddings, conditional_embeds,
                                                    p2p_conditional_embeds, p2p_replacements, device)
                conditional_embeds = p2p_conditional_embeds
            else:
                p2p_xa_interpolation_schedule = None
                p2p_self_interpolation_schedule = None

            if p2p_enabled or blend_mask is not None:
                p2p_xa_interpolation_schedule, p2p_self_interpolation_schedule, blend_mask_schedule, schedule_search_results = self.find_best_p2p_blend_schedule(
                    ddim_loop,
                    uncond_embeds, conditional_embeds,
                    uncond_embeds_delta, conditional_embeds_delta,
                    initial_latents,
                    initial_intermediate_latents,
                    losses_dict,
                    targets_dict,
                    latents_xa_foreground_mask,
                    px_xa_foreground_mask,
                    regularizers_fs_ws_names,
                    starting_img, starting_latent,
                    extra_step_kwargs,
                    device,
                    num_inference_steps,
                    p2p_params,
                    blend_params=mask_blend_params,
                    blend_mask=blend_mask)
            else:
                schedule_search_results = None

            # setup optimizer
            create_optimizer_kwargs = {}
            if blend_mask is not None and mask_blend_params.optimize_mask_blend:
                create_optimizer_kwargs['blend_mask'] = blend_mask_schedule
                create_optimizer_kwargs['blend_mask_lr'] = mask_blend_params.mask_blend_stepsize

            if p2p_xa_interpolation_schedule is not None and p2p_params.optimize_xa_interpolation_schedule:
                create_optimizer_kwargs['xa_schedule'] = p2p_xa_interpolation_schedule
                create_optimizer_kwargs['schedule_lr'] = p2p_params.schedule_stepsize

            if p2p_self_interpolation_schedule is not None and p2p_params.optimize_self_interpolation_schedule:
                create_optimizer_kwargs['self_schedule'] = p2p_self_interpolation_schedule
                create_optimizer_kwargs['schedule_lr'] = p2p_params.schedule_stepsize

            optim = create_optimizer(initial_latents, uncond_embeds_delta, conditional_embeds_delta, optim_params,
                                     **create_optimizer_kwargs)

            optim_scheduler = create_scheduler(optim, optim_params)

            # SGD loop
            for outer_iteration in range(optim_params.sgd_steps + 1):
                #calculate gradient of loss wrt to last latent x0
                with torch.enable_grad():
                    intermediate_preds = ddim_loop(uncond_embeds, conditional_embeds,
                                                   uncond_embeds_delta, conditional_embeds_delta,
                                                   initial_latents, extra_step_kwargs,
                                                   xa_interpolation_schedule=p2p_xa_interpolation_schedule,
                                                   self_interpolation_schedule=p2p_self_interpolation_schedule,
                                                   blend_latent_templates=initial_intermediate_latents,
                                                   blend_mask_schedule=blend_mask_schedule,
                                                   blend_mask=blend_mask)

                    with torch.no_grad():
                        non_augmented_loss, image = self.calculate_loss(intermediate_preds, conditional_embeds_delta,
                                                                        uncond_embeds_delta, losses_dict, targets_dict,
                                                                        latents_xa_foreground_mask,
                                                                        px_xa_foreground_mask, regularizers_fs_ws_names,
                                                                        1, 'uniform', starting_img,
                                                                        starting_latent, loss_scores=loss_scores,
                                                                        regularizer_scores=regularizer_scores,
                                                                        augment=False)
                    imgs_cpu[outer_iteration] = image.detach().cpu()
                    if outer_iteration == optim_params.sgd_steps:
                        break

                    loss, _ = self.calculate_loss(intermediate_preds, conditional_embeds_delta, uncond_embeds_delta,
                                                  losses_dict, targets_dict, latents_xa_foreground_mask,
                                                  px_xa_foreground_mask, regularizers_fs_ws_names,
                                                  optim_params.loss_steps, optim_params.loss_steps_schedule,
                                                  starting_img, starting_latent, augment=True)


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
                loss.backward()

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
                if blend_mask_schedule is not None and mask_blend_params.optimize_mask_blend:
                    blend_mask_schedule.clip_(0.0, 1.0)
                if p2p_xa_interpolation_schedule is not None and p2p_params.optimize_xa_interpolation_schedule:
                    p2p_xa_interpolation_schedule.clip_(0.0, 1.0)
                if p2p_self_interpolation_schedule is not None and p2p_params.optimize_self_interpolation_schedule:
                    p2p_self_interpolation_schedule.clip_(0.0, 1.0)

                # if blend_mask_schedule is not None:
                #     print([f'{i}: {v:.3f}' for (i,v) in enumerate(blend_mask_schedule)])
                # if p2p_xa_interpolation_schedule is not None:
                #     print([f'{i}: {v:.3f}' for (i,v) in enumerate(p2p_xa_interpolation_schedule)])

                optim.zero_grad()

        free_unet(self.unet)

        return_values = {
            'imgs': imgs_cpu,
            'loss_scores': loss_scores,
            'regularizer_scores': regularizer_scores,
            'px_foreground_segmentation': px_xa_foreground_mask.cpu() if px_xa_foreground_mask is not None else None,
            'latents_foreground_segmentation': latents_xa_foreground_mask.cpu() if latents_xa_foreground_mask is not None else None,
            'words_attention_masks': words_attention_masks,
            'initial_img': initial_final_img.cpu() if initial_final_img is not None else None,
            'schedule_search_results': schedule_search_results
        }

        return return_values


    def initial_denoising_loop(self, ddim_loop: DenoisingLoop, starting_img, initial_latents, conditional_embeds,
                               conditional_embeds_delta, uncond_embeds, uncond_embeds_delta, foreground_token_mask,
                               requires_xa_foreground_mask, timesteps, width, device, word_to_token_embeddings,
                               segmentation_args, extra_step_kwargs, p2p_enabled):
        latents_xa_foreground_mask = None
        px_xa_foreground_mask = None
        reference_xa_maps = None
        words_attention_masks = None
        initial_final_latent = None
        initial_final_img = None
        intermediate_latents = None
        with torch.no_grad():
            if p2p_enabled or requires_xa_foreground_mask:
                predicted_originals, intermediate_latents = ddim_loop(uncond_embeds, conditional_embeds,
                                                 uncond_embeds_delta, conditional_embeds_delta,
                                                 initial_latents, extra_step_kwargs,
                                                 xa_interpolation_schedule=None,
                                                 self_interpolation_schedule=None,
                                                 return_all_steps=True, return_intermediate_latents=True)

                initial_final_latent = next(reversed(predicted_originals.values()))
                initial_final_img = self.decode_latent_to_img(initial_final_latent).detach().squeeze(dim=0)
                if starting_img is None:
                    starting_img = initial_final_img

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
                words_attention_masks, initial_final_img, initial_final_latent, intermediate_latents)




    def calculate_loss(self, intermediate_preds, conditional_embeds_delta, uncond_embeds_delta,  losses_dict,
                       targets_dict, latents_xa_foreground_mask, px_xa_foreground_mask, regularizers_fs_ws_names, loss_steps,
                       loss_steps_schedule, starting_img, starting_latent, loss_scores=None, regularizer_scores=None,
                       augment=True):

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

            latents = latents.to(torch.float32)
            image = self.decode_latent_to_img(latents)

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

    #modified encoding that returns the positions of words in encoding that each word in input corresponds to
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

    def find_best_p2p_blend_schedule(self,
                                     ddim_loop,
                                     uncond_embeds, conditional_embeds,
                                     uncond_embeds_delta, conditional_embeds_delta,
                                     initial_latents,
                                     initial_intermediate_latents,
                                     losses_dict,
                                     targets_dict,
                                     latents_xa_foreground_mask,
                                     px_xa_foreground_mask,
                                     regularizers_fs_ws_names,
                                     starting_img, starting_latent,
                                     extra_step_kwargs,
                                     device,
                                     num_inference_steps,
                                     p2p_params: PromptToPromptParams,
                                     blend_params: MaskBlendParams,
                                     blend_mask=None):

        schedule_dicts = []

        dtype = conditional_embeds.dtype
        def get_interpolation_schedule_all_timesteps(schedule):
            if schedule is None or schedule == 'none':
                return None

            schedule_values = [get_interpolation_factor(i, num_inference_steps, schedule) for i in
                                             range(num_inference_steps)]
            schedule_values = torch.tensor(schedule_values, dtype=dtype, device=device)
            return schedule_values

        if p2p_params is not None and p2p_params.do_p2p:
            search_xa_schedules = p2p_params.xa_interpolation_schedule
            if isinstance(search_xa_schedules, str):
                search_xa_schedules = [search_xa_schedules]
            if search_xa_schedules is None:
                search_xa_schedules = [None]
            search_self_schedules = p2p_params.self_interpolation_schedule
            if isinstance(search_self_schedules, str):
                search_self_schedules = [search_self_schedules]
            if search_self_schedules is None:
                search_self_schedules = [None]
        else:
            #if p2p is disabled we cannot search since the images will be constant
            if blend_params is None or blend_params.do_mask_blend is None:
                return None, None, None, None
            elif len(blend_params.initial_mask_blend_schedule) == 1:
                mask_blend_schedule = get_interpolation_schedule_all_timesteps(blend_params.initial_mask_blend_schedule[0])
                return None, None, mask_blend_schedule, None
            else:
                raise ValueError('Searching for more than one blend schedule not enabled without P2P')

        if blend_params is not None and blend_params.do_mask_blend:
            assert blend_mask is not None
            search_blend_schedules = blend_params.initial_mask_blend_schedule
            if search_blend_schedules is None:
                search_blend_schedules = [None]
        else:
            search_blend_schedules = [None]

        #don't search
        if len(search_xa_schedules) == 1 and len(search_self_schedules) == 1 and len(search_blend_schedules) == 1:
            p2p_xa_interpolation_schedule = get_interpolation_schedule_all_timesteps(search_xa_schedules[0])
            p2p_self_interpolation_schedule = get_interpolation_schedule_all_timesteps(search_self_schedules[0])
            mask_blend_schedule = get_interpolation_schedule_all_timesteps(search_blend_schedules[0])
            schedule_dicts = None
        else:
            for xa_sched, self_sched, blend_sched in itertools.product(search_xa_schedules, search_self_schedules, search_blend_schedules):
                # make schedules
                p2p_xa_interpolation_schedule = get_interpolation_schedule_all_timesteps(xa_sched)
                p2p_self_interpolation_schedule = get_interpolation_schedule_all_timesteps(self_sched)
                mask_blend_schedule = get_interpolation_schedule_all_timesteps(blend_sched)
                with torch.no_grad():
                    intermediate_preds = ddim_loop(uncond_embeds, conditional_embeds,
                                                   uncond_embeds_delta, conditional_embeds_delta,
                                                   initial_latents, extra_step_kwargs,
                                                   xa_interpolation_schedule=p2p_xa_interpolation_schedule,
                                                   self_interpolation_schedule=p2p_self_interpolation_schedule,
                                                   blend_latent_templates=initial_intermediate_latents,
                                                   blend_mask_schedule=mask_blend_schedule,
                                                   blend_mask=blend_mask)

                    non_augmented_loss, image = self.calculate_loss(intermediate_preds, conditional_embeds_delta,
                                                                    uncond_embeds_delta, losses_dict, targets_dict,
                                                                    latents_xa_foreground_mask, px_xa_foreground_mask,
                                                                    regularizers_fs_ws_names, 1,
                                                                    'uniform', starting_img,
                                                                    starting_latent, augment=False)

                    schedule_dicts.append({
                        'xa_schedule_name': xa_sched,
                        'self_schedule_name': self_sched,
                        'blend_schedule_name': blend_sched,
                        'xa_schedule': p2p_xa_interpolation_schedule.cpu() if p2p_xa_interpolation_schedule is not None else None,
                        'self_schedule': p2p_self_interpolation_schedule.cpu() if p2p_self_interpolation_schedule is not None else None,
                        'mask_blend_schedule': mask_blend_schedule.cpu() if mask_blend_schedule is not None else None,
                        'img': image.detach().cpu().squeeze(dim=0),
                        'loss': non_augmented_loss.cpu(),
                    })

            losses_tensor = torch.tensor([a['loss'] for a in schedule_dicts])
            min_loss_idx = torch.argmin(losses_tensor)
            p2p_xa_interpolation_schedule = schedule_dicts[min_loss_idx]['xa_schedule']
            p2p_self_interpolation_schedule = schedule_dicts[min_loss_idx]['self_schedule']
            mask_blend_schedule = schedule_dicts[min_loss_idx]['mask_blend_schedule']
            if p2p_xa_interpolation_schedule is not None:
                p2p_xa_interpolation_schedule = p2p_xa_interpolation_schedule.to(device)
            if p2p_self_interpolation_schedule is not None:
                p2p_self_interpolation_schedule = p2p_self_interpolation_schedule.to(device)
            if mask_blend_schedule is not None:
                mask_blend_schedule = mask_blend_schedule.to(device)
        return p2p_xa_interpolation_schedule, p2p_self_interpolation_schedule, mask_blend_schedule, schedule_dicts