import pdb

import torch
import torch.nn.functional as F


def generate_one(model, input_ids):
    # 准备模型的输入
    model_inputs = {
        "input_ids": input_ids
    }
    # 调用模型获取logits
    outputs = model(**model_inputs)
    logits = outputs.logits[:, -1, :]
    return logits

def forward_first_k_layers(model, input_ids, k, past_seen_token_num, kv_cache=None):
    seq_len = len(input_ids)
    input_ids = input_ids[past_seen_token_num:].unsqueeze(dim=0).to(model.device)
    attention_mask = torch.ones(1, seq_len).to(model.device)
    position_ids = torch.arange(past_seen_token_num, seq_len).unsqueeze(dim=0).to(model.device)
    
    # 准备模型的输入
    model_inputs = {
        "k": k,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids, 
        "past_key_values": kv_cache,
    }
    # 调用模型获取logits
    # pdb.set_trace()
    last_hidden_states, kv_cache = model.forward_first_k_layers(**model_inputs)
    assert len(last_hidden_states.shape) == 3 and last_hidden_states.size(0) == 1
    assert last_hidden_states.size(1) == seq_len - past_seen_token_num
    last_hidden_states = last_hidden_states.squeeze(dim=0)
    return last_hidden_states, kv_cache

def forward_from_k_layer(model, input_ids, input_hidden_states, k, past_seen_token_num, kv_cache=None):
    seq_len = len(input_ids)
    assert input_hidden_states.size(1) == seq_len - past_seen_token_num
    attention_mask = torch.ones(1, seq_len).to(model.device)
    position_ids = torch.arange(past_seen_token_num, seq_len).unsqueeze(dim=0).to(model.device)

    # 准备模型的输入

    model_inputs = {
        "k": k,
        "inputs_embeds": input_hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids, 
        "past_key_values": kv_cache,
        # "cache_position": torch.arange(seq_len).unsqueeze(dim=0).to(model.device),
    }
    # pdb.set_trace()
    # 调用模型获取logits
    outputs = model.forward_from_k_layer(**model_inputs)
    logits = outputs.logits[:, -1, :]
    kv_cache = outputs.past_key_values
    return logits, kv_cache

def generate(model, tokenizer, input_text, max_new_length, device):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    LLM_input_ids_length = input_ids.shape[1]
    # 生成过程
    for _ in range(max_new_length - 1):
        # 准备模型的输入
        model_inputs = {
            "input_ids": input_ids
        }
        # 调用模型获取logits
        outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token_idx = torch.argmax(probs, dim=-1)
        next_token_idx = next_token_idx.view(-1, 1)
        if next_token_idx.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat((input_ids, next_token_idx), dim=1)

    output_text = tokenizer.decode(input_ids[:, LLM_input_ids_length:].tolist()[0],
                                       skip_special_tokens=False)  # 跳过开始标记
    return output_text
