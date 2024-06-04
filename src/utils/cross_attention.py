import itertools

import torch
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0
import math


#Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py
def prepare_unet(unet: UNet2DConditionModel, xa_store_initial_attention_map=False, xa_store_last_attention_map=False,
                 self_store_initial_attention_map=False, self_store_last_attention_map=False, store_in_ram=False,
                 softmax_dtype=torch.float, store_dtype=None):

    required = xa_store_initial_attention_map or xa_store_last_attention_map or self_store_initial_attention_map or self_store_last_attention_map
    if not required:
        return unet

    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn1" in name:
            sa_store_in_ram = True if store_in_ram is None else store_in_ram
            pix2pix_zero_attn_procs[name] = TrackAttentionProcessor(store_last_attention_map=self_store_last_attention_map,
                                                                    store_initial_attention_map=self_store_initial_attention_map,
                                                                    is_xa=False, store_in_ram=sa_store_in_ram,
                                                                    softmax_dtype=softmax_dtype, store_dtype=store_dtype)
            #module.requires_grad_(True)
        if "attn2" in name:
            xa_store_in_ram = False if store_in_ram is None else store_in_ram
            pix2pix_zero_attn_procs[name] = TrackAttentionProcessor(store_last_attention_map=xa_store_last_attention_map,
                                                                    store_initial_attention_map=xa_store_initial_attention_map,
                                                                    is_xa=True, store_in_ram=xa_store_in_ram,
                                                                    softmax_dtype=softmax_dtype, store_dtype=store_dtype)
            #module.requires_grad_(True)

    unet.set_attn_processor(pix2pix_zero_attn_procs)
    return unet


def get_interpolation_factor(i, num_timesteps, interpolation_schedule):
    if interpolation_schedule is None or interpolation_schedule == 'none':
       return 0
    elif 'cosine' in interpolation_schedule:
        part = interpolation_schedule.split('_')
        if len(part) == 2:
            tilt = float(part[1])
        else:
            tilt = 1.

        return 0.5 * (1 + math.cos(math.pi * (i / num_timesteps)**tilt))
    elif 'threshold' in interpolation_schedule:
        part = interpolation_schedule.split('_')
        if len(part) == 3:
            threshold_min = float(part[1])
            threshold_max = float(part[2])
        elif len(part) == 2:
            threshold_min = 0.0
            threshold_max = float(part[1])
        else:
            raise ValueError()

        in_range = (i / num_timesteps) < threshold_max and (i/num_timesteps) >= threshold_min
        if 'inv' in interpolation_schedule:
            return 0. if in_range else 1.
        else:
            return 1. if in_range else 0.
    else:
        raise NotImplementedError(f'Attention schedule: {interpolation_schedule} not supported')

def free_unet(unet):
    unet.set_attn_processor(AttnProcessor2_0())

XA_STORE_INITIAL_CONDITIONAL_ONLY = True

class TrackAttentionProcessor:
    """An attention processor class to store the attention weights.
    We support two modes:
    Track the initial attention for each timestep, useful for prompt-to-prompt style interpolation to cpu (little memory cost)
    OR
    Store the last attention map for each timestep which requires A LOT of memory but required for regularization on XA maps
    in combination with checkpointing
    """

    def __init__(self, store_initial_attention_map=False, store_last_attention_map=False, is_xa=True,
                 store_in_ram=False, xa_store_initial_conditional_only=XA_STORE_INITIAL_CONDITIONAL_ONLY,
                 softmax_dtype=torch.float, store_dtype=None):
        self.store_last_attention_map = store_last_attention_map
        self.store_initial_attention_map = store_initial_attention_map
        self.last_attention_map = {}
        self.initial_attention_map = {}
        self.replacement_mapper = None
        self.store_in_ram = store_in_ram
        self.is_xa = is_xa
        self.xa_store_initial_conditional_only = xa_store_initial_conditional_only
        self.store_dtype=store_dtype
        self.softmax_dtype = softmax_dtype

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        xa_attention_initial_interpolation_factor=0,
        self_attention_initial_interpolation_factor=0,
        timestep=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        #attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs = get_attention_scores(query, key, self.softmax_dtype, attn.scale, attention_mask)

        #replace attention prompt 2 prompt style
        if self.is_xa and xa_attention_initial_interpolation_factor > 0:
            if self.store_initial_attention_map and timestep.item() in self.initial_attention_map.keys():
                #only replace conditional part
                h = attention_probs.shape[0]
                target_uncond, target_cond = attention_probs.chunk(2)
                source_initial_t = self.initial_attention_map[timestep.item()].detach().to(attention_probs.device)
                if self.xa_store_initial_conditional_only:
                    source_cond = source_initial_t.to(attention_probs.dtype)
                else:
                    source_cond = source_initial_t[h // 2:, :, :].to(attention_probs.dtype)
                target_cond = attention_probs[h // 2:, :, :]

                #ws = (1. - xa_attention_initial_interpolation_factor)
                #print('Warning: please double check this')
                ws = xa_attention_initial_interpolation_factor
                wt = 1. - ws
                replaced_attention_cond = wt * target_cond + ws * source_cond
                attention_probs = torch.cat([target_uncond, replaced_attention_cond], dim=0)
        if not self.is_xa and self_attention_initial_interpolation_factor > 0:
            if self.store_initial_attention_map and timestep.item() in self.initial_attention_map.keys():
                source_initial_t = self.initial_attention_map[timestep.item()].to(attention_probs.device)
                source_cond = source_initial_t
                target_cond = attention_probs

                #ws = (1. - self_attention_initial_interpolation_factor)
                ws = self_attention_initial_interpolation_factor
                wt = 1. - ws
                attention_probs = wt * target_cond + ws * source_cond

        if self.store_last_attention_map:
            #we use them for explicit regularization so they need to be differentiable
            self.last_attention_map[timestep.item()] = attention_probs
        with torch.no_grad():
            if self.store_initial_attention_map and timestep.item() not in self.initial_attention_map.keys():
                attention_probs_ = attention_probs.detach()
                attention_probs_.requires_grad_(False)
                if self.is_xa and self.xa_store_initial_conditional_only:
                    #storing only the second half saves memory
                    attention_probs_ = attention_probs_[attention_probs_.shape[0] // 2:, :, :]
                if self.store_in_ram:
                    attention_probs_ = attention_probs_.cpu()
                if self.store_dtype is not None:
                    attention_probs_ = attention_probs_.to(self.store_dtype)

                self.initial_attention_map[timestep.item()] = attention_probs_

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def get_attention_scores(query, key, softmax_dtype, scale, attention_mask=None):
    dtype = query.dtype

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=scale,
    )

    attention_scores = attention_scores.to(softmax_dtype)
    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(dtype)

    return attention_probs


def get_and_release_last_attention(unet, timestep):
    # add the cross attention map to the dictionary
    xa_maps_t = []
    xa_map_names = []
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            attn_mask = module.processor.last_attention_map[timestep.item()]
            assert attn_mask is not None
            xa_maps_t.append(attn_mask)
            xa_map_names.append(module_name)
            module.processor.last_attention_map[timestep.item()] = None

    return xa_maps_t, xa_map_names


def get_initial_cross_attention_all_timesteps(unet, timesteps):
    xa_maps = {}
    for t in timesteps:
        xa_maps_t = {}
        xa_map_names = []
        for name in unet.attn_processors.keys():
            module_name = name.replace(".processor", "")
            module = unet.get_submodule(module_name)
            if "attn2" in name:
                attn_mask = module.processor.initial_attention_map[t.item()]
                assert attn_mask is not None
                xa_maps_t[module_name] = attn_mask

        xa_maps[t.item()] = xa_maps_t

    return xa_maps


def p2p_reshape_initial_cross_attention(unet, timesteps, word_to_token_embeddings, p2p_word_to_token_embeddings,
                                        conditional_embeds, p2p_conditional_embeds, prompt_to_prompt_replacements, device):
    ori_seq, p2p_seq = prompt_to_prompt_replacements

    #using p2p without changing prompt can be used as regularization during optimization to enforce similar shapes
    if ori_seq == p2p_seq:
        return

    ori_words = ori_seq.split(' ')
    p2p_words = p2p_seq.split(' ')

    ori_words_mean_clip = torch.zeros((len(ori_words), conditional_embeds.shape[2]), device=device)
    p2p_words_mean_clip = torch.zeros((len(p2p_words), conditional_embeds.shape[2]), device=device)
    for w_idx, ori_word in enumerate(ori_words):
        word_token_idcs = word_to_token_embeddings[ori_word]
        ori_words_mean_clip[w_idx] = torch.mean(conditional_embeds[0, torch.LongTensor(word_token_idcs).to(device), : ], dim=0)

    for w_idx, p2p_word in enumerate(p2p_words):
        word_token_idcs = p2p_word_to_token_embeddings[p2p_word]
        p2p_words_mean_clip[w_idx] = torch.mean(p2p_conditional_embeds[0, torch.LongTensor(word_token_idcs).to(device), : ], dim=0)

    d_cos = torch.zeros((len(p2p_words), len(ori_words)), device=device)
    for i, i_clip in enumerate(p2p_words_mean_clip):
        for j, j_clip in enumerate(ori_words_mean_clip):
            #turn into distance in range [0,2]
            d_cos[i,j] = 1. - torch.cosine_similarity(i_clip, j_clip, dim=0)

    print(f'P2P words: {p2p_words}')
    print(f'Original words: {ori_words}')
    print(f'P2p-Ori distance {d_cos}')

    #associate words with each other based on clip distance
    target_source_map = torch.zeros((len(p2p_words), len(ori_words)), device=device)
    if len(ori_words) == len(p2p_words):
        #1 to 1 mappings:
        all_possible_assignments = list(itertools.permutations([i for i in range(len(ori_words))]))
        assignment_costs = torch.zeros(len(all_possible_assignments), device=device)
        for i, assignment in enumerate(all_possible_assignments):
            assignment = torch.LongTensor(list(assignment)).to(device)
            assignment_costs[i] = d_cos[torch.arange(len(ori_words), device=device), assignment].sum()

        min_assignment_idx = torch.argmin(assignment_costs)
        min_assignment = all_possible_assignments[min_assignment_idx.item()]
        for i in range(len(ori_words)):
            target_source_map[i, min_assignment[i]] = 1.0
    else:
        #interpolate
        softmax_interp = True
        softmax_temperature = 1.
        for i in range(len(p2p_words)):
            if softmax_interp:
                target_source_map[i, :] = torch.softmax(softmax_temperature * d_cos[i,:], dim=0)
            else:
                target_source_map[i, :] = d_cos[i,:] / torch.sum(d_cos[i,:])

    #now do the attention reshaping
    for t in timesteps:
        for name in unet.attn_processors.keys():
            module_name = name.replace(".processor", "")
            module = unet.get_submodule(module_name)
            if "attn2" in name:
                attn_mask = module.processor.initial_attention_map[t.item()].to(device)
                assert attn_mask is not None

                p2p_token_idcs = [p2p_word_to_token_embeddings[p2p_rep] for p2p_rep in p2p_seq.split(' ')]
                ori_token_idcs = [word_to_token_embeddings[ori_rep] for ori_rep in ori_seq.split(' ')]

                p2p_token_idcs_joined = sum(p2p_token_idcs, [])
                ori_token_idcs_joined = sum(ori_token_idcs, [])
                if (max(p2p_token_idcs_joined) - min(p2p_token_idcs_joined)) != (len(p2p_token_idcs_joined) - 1):
                    print('Warning P2P: Non-Contiguous P2P token sequence')
                if (max(ori_token_idcs_joined) - min(ori_token_idcs_joined)) != (len(ori_token_idcs_joined) - 1):
                    print('Warning P2P: Non-Contiguous original token sequence')

                #copy original attention, everything left of the first replaced token will be fine
                new_attn = attn_mask.clone().detach()
                h = attn_mask.shape[0]
                #copy everything right of the last replaced token
                p2p_idx = max(p2p_token_idcs_joined) + 1
                ori_idx = max(ori_token_idcs_joined) + 1

                conditional_start_idx = h // 2 if XA_STORE_INITIAL_CONDITIONAL_ONLY else 0

                while (p2p_idx < attn_mask.shape[2]):
                    #only copy attention part belongig to conditional
                    new_attn[conditional_start_idx:, :, p2p_idx] = attn_mask[conditional_start_idx:, :, ori_idx]
                    p2p_idx += 1
                    ori_idx = min(ori_idx + 1, attn_mask.shape[2] - 1)

                #now do actual attention manipulation
                for p2p_word_idx, p2p_word in enumerate(p2p_words):
                    word_map = target_source_map[p2p_word_idx]
                    p2p_word_token_idcs = p2p_token_idcs[p2p_word_idx]
                    #first zero out
                    for p2p_idx in p2p_word_token_idcs:
                        new_attn[conditional_start_idx:, :, p2p_idx] = 0
                    #now fill
                    for ori_word_token_idcs, ori_w in zip(ori_token_idcs, word_map):
                        ori_attns = attn_mask[conditional_start_idx:, :, torch.LongTensor(ori_word_token_idcs).to(device)]
                        ori_attns_mean = torch.mean(ori_attns, dim=2)
                        new_attn[conditional_start_idx:, :, p2p_idx] += ori_w.item() * ori_attns_mean

                initial_device = module.processor.initial_attention_map[t.item()].device
                module.processor.initial_attention_map[t.item()] = new_attn.to(initial_device)

