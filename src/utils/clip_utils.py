from typing import Optional
import torch

#from transfomers/models/clip/modeling_clip.py
#split token encoding and the clip forward pass
def clip_encode_tokens(input_ids, clip_model, position_ids=None):
    if input_ids is None:
        raise ValueError("You have to specify input_ids")

    text_model = clip_model.text_model
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    hidden_states = text_model.embeddings(input_ids=input_ids, position_ids=position_ids)
    return hidden_states

from transformers.models.clip.modeling_clip import _expand_mask
def clip_model_forward(hidden_states,
                       clip_model,
                       attention_mask: Optional[torch.Tensor] = None,
                       position_ids: Optional[torch.Tensor] = None,
                       output_attentions: Optional[bool] = None,
                       output_hidden_states: Optional[bool] = None,
                       return_dict: Optional[bool] = None):

    text_model = clip_model.text_model
    output_attentions = output_attentions if output_attentions is not None else text_model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else text_model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else text_model.config.use_return_dict

    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    bsz, seq_len = hidden_states.size()[:2]
    causal_attention_mask = text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_model.final_layer_norm(last_hidden_state)
    return last_hidden_state
