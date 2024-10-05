import json
import random
import os
import pdb
import torch
import sys
sys.path.append("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang")


from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import k_index, read_multiline_json
from utils.anchor_words import get_common_vocab_list_by_tokenizer
from model_config import model_paths
from datasets import load_from_disk

device="auto"


def extract_anchor_embeddings(model_name_list, data, output_dir, anchor_num=4000, seed=1):
    random.seed(seed)

    model_num = len(model_name_list)
    print(f"Models: {model_name_list}")
    anchor_embeddings_list = []
    for i in range(model_num):
        anchor_embeddings_list.append([])

    # 1. Load data
    random.shuffle(data)

    # 2. Load model and tokenizer
    model_list = []
    tokenizer_list = []
    print("Vocabulary size:")
    for model_name in model_name_list:
        model_list.append(AutoModelForCausalLM.from_pretrained(model_paths[model_name],
                                                     device_map=device,
                                                     torch_dtype="auto",
                                                     trust_remote_code=True))
        tokenizer = AutoTokenizer.from_pretrained(model_paths[model_name], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer_list.append(tokenizer)
        print(f"{model_name}: {len(tokenizer.get_vocab())}")
    
    common_token_index_list = get_common_vocab_list_by_tokenizer(tokenizer_list) # {"token": xxx, "index": [index_in_model1, index_in_model2, ...]}
    common_token_set = [w["token"] for w in common_token_index_list]
    print(f"Common tokens: {len(common_token_set)}")

    # 3. tokenize sentence and find the aligned token pair and their indices in the sentence.
    count = 0
    batch_count = 0
    batch_size = 10000
    sample_per_sent = max(10, int(anchor_num / len(data)))
    anchor_token_set = Counter(['llllllll'])
    for tmp_j in tqdm(range(anchor_num)):
        sent = data[tmp_j % len(data)]
        token_ids_list = [tokenizer.encode(sent) for tokenizer in tokenizer_list]

        position_pair_list = []
        
        cand_token_set = set()
        for token_id in token_ids_list[0]:
            sampled_token = tokenizer_list[0].convert_ids_to_tokens(token_id)
            if sampled_token in common_token_set \
                and anchor_token_set[sampled_token] < anchor_num / len(common_token_set) * 5 \
                and token_id > 3 \
                and len(set([token_ids_list[i].count(tokenizer_list[i].convert_tokens_to_ids(sampled_token)) for i in range(model_num)])) == 1:
                cand_token_set.add(sampled_token)
        
        cand_token_list = random.sample(cand_token_set, min(len(cand_token_set), sample_per_sent))

        for sampled_token in cand_token_list:
            k = random.choice(list(range(token_ids_list[0].count(tokenizer_list[0].convert_tokens_to_ids(sampled_token)))))
            try:
                position_pair_list.append([k_index(token_ids_list[i], tokenizer_list[i].convert_tokens_to_ids(sampled_token), k+1)
                    for i in range(model_num)])
            except Exception as e:
                pdb.set_trace()

        # 5. model forward to obtain the representation
        with torch.no_grad():
            for i in range(model_num):
                tokenizer = tokenizer_list[i]
                model = model_list[i]
                selected_positions = [p[i] for p in position_pair_list]
                
                model_input = {"input_ids": tokenizer.encode(sent, return_tensors="pt").to(model.device), 
                                "return_dict": True, 
                                "output_hidden_states": True}
                
                # print(torch.cuda.memory_allocated("cuda:0")/(1024**2), "MB")
                model_output = model(**model_input)['hidden_states'] # a tuple of (L+1) tensors, each one is (B, T, d)
                # print(torch.cuda.memory_allocated("cuda:0")/(1024**2), "MB")
                
                selected_hidden_states = [layer_output[0][selected_positions].clone() for layer_output in model_output] # a tuple of (L+1) tensors, each one is (d) 
                del model_output
                # print(torch.cuda.memory_allocated("cuda:0")/(1024**2), "MB")
                anchor_embeddings_list[i].append(selected_hidden_states)
                torch.cuda.empty_cache()

        count += len(selected_positions)
        batch_count += len(selected_positions)
        anchor_token_set.update(cand_token_list)

        if batch_count >= batch_size:
            res_anchor_embeddings = {} # to be a dict {"model": (layer_num, anchor_num, d)}
    
            for i in range(model_num):  #  for each model
                anchor_embeddings = anchor_embeddings_list[i]  # (anchor_num, layer_num, d)
                reshaped_anchor_embeddings = []
                
                for l in range(len(anchor_embeddings[0])):  # for each layer
                    reshaped_anchor_embeddings.append(torch.concatenate([hidden_states[l] for hidden_states in anchor_embeddings]))
                res_anchor_embeddings[model_name_list[i]] = reshaped_anchor_embeddings
            output_path = os.path.join(output_dir, '_'.join(model_name_list) + f"_{anchor_num}anchors_seed{seed}_{int(count/batch_size)}.pt")
            print(f"anchor embeddings saved in: {output_path}")
            torch.save(res_anchor_embeddings, output_path)

            for i in range(model_num):
                anchor_embeddings_list[i] = []
            batch_count = 0
            torch.cuda.empty_cache()

        if count >= anchor_num:
            if len(anchor_embeddings_list[0]) > 0:
                res_anchor_embeddings = {} # to be a dict {"model": (layer_num, anchor_num, d)}
        
                for i in range(model_num):
                    anchor_embeddings = anchor_embeddings_list[i]  # (anchor_num, layer_num, d)
                    reshaped_anchor_embeddings = []
                    
                    for l in range(len(anchor_embeddings[0])):
                        reshaped_anchor_embeddings.append(torch.concatenate([hidden_states[l] for hidden_states in anchor_embeddings]))
                    res_anchor_embeddings[model_name_list[i]] = reshaped_anchor_embeddings
                output_path = os.path.join(output_dir, '_'.join(model_name_list) + f"_{anchor_num}anchors_seed{seed}_{int(count/batch_size) + 1}.pt")
                print(f"anchor embeddings saved in: {output_path}")
                torch.save(res_anchor_embeddings, output_path)

            break

    
    # anchor_embeddings_list: (model_num, anchor_num, layer_num, d)
    res_anchor_embeddings = {} # to be a dict {"model": (layer_num, anchor_num, d)}
    
    for i in range(model_num):
        anchor_embeddings = anchor_embeddings_list[i]  # (anchor_num, layer_num, d)
        reshaped_anchor_embeddings = []
        
        for l in range(len(anchor_embeddings[0])):
            reshaped_anchor_embeddings.append(torch.stack([hidden_states[l] for hidden_states in anchor_embeddings]))
        res_anchor_embeddings[model_name_list[i]] = reshaped_anchor_embeddings
    output_path = os.path.join(output_dir, '_'.join(model_name_list) + f"_{anchor_num}anchors_seed{seed}.pt")
    print(f"anchor embeddings saved in: {output_path}")
    torch.save(res_anchor_embeddings, output_path)


if __name__ == "__main__":
    # data_path = "/data/home/cpfu/ychuang/reimplement_deepen/datasets/TriviaQA/wikipedia-dev-1900.jsonl"
    # data = read_multiline_json(data_path)
    # data = [i["question"] for i in data]

    data_path = "/data/home/cpfu/ychuang/Downloads/minipile/"
    dataset = [" ".join(i['text'].split()[:100]) for i in load_from_disk(data_path)["train"]]
    output_dir = "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/anchor_embeddings/"
    extract_anchor_embeddings(["llama2-13b", "mistral-7b"], dataset, output_dir, anchor_num=200000)