import pdb

import torch

from src.Model_generator import generate_one, forward_first_k_layers, forward_from_k_layer
from transformers.cache_utils import DynamicCache

class DeepEnsemblerGenerator():
    """
    DeepEnsemblerGenerator controls the text generation process of the ensemble of multiple models.
    """
    def __init__(self, model_list, tokenizer_list, device_list, ensembler, layer_alignment):
        self.model_list = model_list
        self.tokenizer_list = tokenizer_list
        self.device_list = device_list
        self.ensembler = ensembler
        self.layer_alignment = layer_alignment

    def ensemble_generate(self, llm_input_text_list, max_new_tokens=400):
        """
        Given the input, we fuse the hidden states.
        """
        main_model_id = 0
        model_num = len(self.model_list)
        current_input_string = llm_input_text_list[0]
        llm_inputs_list = []
        kv_cache_list = [None for i in range(model_num)]
        past_seen_tokens_num_list = [0 for i in range(model_num)]

        for index, llm_input_text in enumerate(llm_input_text_list):
            llm_input = self.tokenizer_list[index].encode(llm_input_text, return_tensors="pt").to(self.device_list[main_model_id])
            llm_inputs_list.append(llm_input[0])
        # try: 
        #     assert len(set(["".join(self.tokenizer_list[i[0]].convert_ids_to_tokens(i[1])) for i in enumerate(llm_inputs_list)])) == 1, "We assume that all model receives the same input string, but this is not the case now."
        # except:
        #     print(["".join(self.tokenizer_list[i[0]].convert_ids_to_tokens(i[1])) for i in enumerate(llm_inputs_list)])
        #     pdb.set_trace()

        for gen_step in range(max_new_tokens):
            hidden_state_list = []
            # 1. 所有模型前向计算
            for index in range(len(llm_inputs_list)):
                # # test code:
                # kv_cache_list[index] = None
                # past_seen_tokens_num_list[index] = 0
                hidden_states, kv_cache = forward_first_k_layers(self.model_list[index], 
                                                                 llm_inputs_list[index], 
                                                                 self.layer_alignment[index], 
                                                                 past_seen_tokens_num_list[index],
                                                                 kv_cache=kv_cache_list[index])
                
                hidden_states = hidden_states.to(self.device_list[0])
                hidden_state_list.append(hidden_states)
                kv_cache_list[index] = kv_cache

            
            # 2. token对齐
            for index in range(1, len(llm_inputs_list)):
                alignment_matrix = align_tokens(llm_inputs_list[index][past_seen_tokens_num_list[index]:].tolist(), 
                                                llm_inputs_list[0][past_seen_tokens_num_list[0]:].tolist(),
                                                self.tokenizer_list[index], 
                                                self.tokenizer_list[0])
                alignment_matrix = alignment_matrix.to(self.device_list[0]).type_as(hidden_state_list[index])
                hidden_state_list[index] = torch.mm(alignment_matrix.T, hidden_state_list[index])
            # 3. 表示融合
            # 0. 加了一步操作，只更新最后面的10个token
            pdb.set_trace()
            changed_token_num = 10
            unchanged_main_token_state = hidden_state_list[0][:-changed_token_num]
            hidden_state_list = [i[-changed_token_num:].clone().detach() for i in hidden_state_list]
            
            aggregated_hidden_state = self.ensembler.fuse(hidden_state_list)

            if len(unchanged_main_token_state) > 0:
                aggregated_hidden_state = torch.cat([unchanged_main_token_state, aggregated_hidden_state], dim=0)

            aggregated_hidden_state = aggregated_hidden_state.unsqueeze(dim=0)  # (T, d) => (1, T, d)
            # 4. 主模型继续前向计算
            
            ensemble_output, kv_cache = forward_from_k_layer(self.model_list[0], 
                                                   llm_inputs_list[0], 
                                                   aggregated_hidden_state, 
                                                   self.layer_alignment[0],
                                                   past_seen_tokens_num_list[0],
                                                   kv_cache=kv_cache_list[0])

            kv_cache_list[0] = kv_cache

            # 5. 决定next token
            
            # print(ensemble_output[0,:5])
            # new_outputs = self.model_list[0](**self.tokenizer_list[0](current_input_string, return_tensors="pt").to(self.device_list[main_model_id]))
            # new_logits = new_outputs.logits
            # print(new_logits[0][-1][:5])
            

            next_token_idx = torch.argmax(ensemble_output, dim=-1)
            print(self.tokenizer_list[0].decode(next_token_idx))
            llm_input = torch.cat([llm_inputs_list[main_model_id], next_token_idx.to(self.model_list[main_model_id].device)])
            current_input_string = self.tokenizer_list[main_model_id].decode(llm_input.tolist(), skip_special_tokens=True)
            # print(main_model_id)
            if next_token_idx.item() == self.tokenizer_list[main_model_id].eos_token_id:
                break
            if next_token_idx.item() == self.tokenizer_list[main_model_id].unk_token_id:
                current_input_string += "<unk>"

            # 6. 更新输入
            for index in range(len(llm_inputs_list)):
                # self.tokenizer_list[0].convert_ids_to_tokens(llm_inputs_list[0]["input_ids"][0])
                # llm_inputs_list[0]["input_ids"].shape
                # self.tokenizer_list[1].convert_ids_to_tokens(llm_inputs_list[1]["input_ids"][0])
                # llm_inputs_list[1]["input_ids"].shape

                new_input_ids = self.tokenizer_list[index].encode(current_input_string, return_tensors="pt")[0].to(self.device_list[main_model_id])
                # test code:
                # self.tokenizer_list[index].convert_ids_to_tokens(llm_inputs_list[index])
                # llm_inputs_list[index].shape
                # self.tokenizer_list[index].convert_ids_to_tokens(new_input_ids)
                # new_input_ids.shape
                # pdb.set_trace()
                common_pref_tokens_num = get_common_prefix_tokens_num(llm_inputs_list[index], new_input_ids)
                past_seen_tokens_num_list[index] = common_pref_tokens_num
                kv_cache_list[index] = update_kv_cache(kv_cache_list[index], common_pref_tokens_num)
                llm_inputs_list[index] = new_input_ids

        return current_input_string


def align_tokens(src_input_ids, tgt_input_ids, src_tokenizer, tgt_tokenizer):
        """
        """
        src_input_tokens = src_tokenizer.convert_ids_to_tokens(src_input_ids)
        tgt_input_tokens = tgt_tokenizer.convert_ids_to_tokens(tgt_input_ids)
        # assert len("".join(src_input_tokens)) == len("".join(tgt_input_tokens)), "received two different sequences"
        alignment_matrix = torch.zeros(len(src_input_ids), len(tgt_input_ids))

        def get_cum_ids(token_list):
            cum_ids = []
            cum_len = 0
            for token in token_list:
                cum_ids.append(set(range(cum_len, cum_len + len(token))))
                cum_len += len(token)
            return cum_ids

        src_input_cum_ids = get_cum_ids(src_input_tokens)
        tgt_input_cum_ids = get_cum_ids(tgt_input_tokens)
        for i in range(len(src_input_tokens)):
            for j in range(len(tgt_input_tokens)):
                src_span = src_input_cum_ids[i]
                tgt_span = tgt_input_cum_ids[j]
                alignment_matrix[i][j] = len(src_span.intersection(tgt_span)) / len(src_span)
        
        return alignment_matrix

def get_common_prefix_tokens_num(l1, l2):
    i = 0
    for i in range(min(len(l1), len(l2))):
        if l1[i] == l2[i]:
            i += 1
        else:
            break
    return i

def update_kv_cache(kv_cache, new_past_seen_token_num):
    """
    kv_cache: a list of each layer's (key_cache, value_cache), where key_cache is a tensor with shape (batch_num, head_num, seq_len, head_dim)
    """
    if isinstance(kv_cache, DynamicCache):
        # print("before update, the shape of key cache is:", kv_cache.key_cache[0].shape)
        for layer_id, layer_cache in enumerate(kv_cache.key_cache):
            kv_cache.key_cache[layer_id] = kv_cache.key_cache[layer_id][:,:,:new_past_seen_token_num,:]
        for layer_id, layer_cache in enumerate(kv_cache.value_cache):
            kv_cache.value_cache[layer_id] = kv_cache.value_cache[layer_id][:,:,:new_past_seen_token_num,:]
        # print("after update, the shape of key cache is:", kv_cache.key_cache[0].shape)
    elif isinstance(kv_cache, tuple):
        # print("before update, the shape of key cache is:", kv_cache[0][0].shape)
        kv_cache = list(kv_cache)
        for layer_id, layer_cache in enumerate(kv_cache):
            key_cache, value_cache = layer_cache
            kv_cache[layer_id] = (key_cache[:,:,:new_past_seen_token_num,:],
                                  value_cache[:,:,:new_past_seen_token_num,:])
        kv_cache = tuple(kv_cache)
        # print("after update, the shape of key cache is:", kv_cache[0][0].shape)
    else:
        raise Exception("Unknow KV cache type")
    return kv_cache

