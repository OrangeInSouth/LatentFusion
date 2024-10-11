import json
import random
import os
import pdb
import argparse
import torch
import sys
proj_path = "/share/home/fengxiaocheng/ychuang/LatentFusion"

sys.path.append(proj_path)

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
    batch_index = 0
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
            # (1) obtain all models' hidden states and predicted tokens
            hidden_states_per_model = []
            pred_tokens_per_model = []

            for i in range(model_num):
                tokenizer = tokenizer_list[i]
                model = model_list[i]
                selected_positions = [p[i] for p in position_pair_list]
                
                model_input = {"input_ids": tokenizer.encode(sent, return_tensors="pt").to(model.device), 
                                "return_dict": True, 
                                "output_hidden_states": True}
                
                # print(torch.cuda.memory_allocated("cuda:0")/(1024**2), "MB")
                pdb.set_trace()
                model_output = model(**model_input) # a tuple of (L+1) tensors, each one is (B, T, d)
                interal_hidden_states = model_output['hidden_states']
                logits = model_output['logits'][0]
                pred_tokens = tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1))
                # selected_pred_tokens = pred_tokens[selected_positions]
                # selected_hidden_states = [layer_output[0][selected_positions].clone() for layer_output in interal_hidden_states] # a tuple of (L+1) tensors, each one is (d) 

                pred_tokens_per_model.append(pred_tokens)
                hidden_states_per_model.append(interal_hidden_states)

            # (2) filter sampled token position again according to whether predicting the same token
            position_pair_list = [pp for pp in position_pair_list if output_same_token(pred_tokens_per_model[0][pp[0]], pred_tokens_per_model[1][pp[1]])]

            # (3) withdraw the embedding of the sample token
            for i in range(model_num):
                selected_positions = [p[i] for p in position_pair_list]
                anchor_embeddings_list[i].append([layer_output[selected_positions].squeeze(dim=0).clone() for layer_output in hidden_states_per_model[i]])
            
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
            output_path = os.path.join(output_dir, '_'.join(model_name_list) + f"_{anchor_num}anchors_seed{seed}_{batch_index}.pt")
            print(f"anchor embeddings saved in: {output_path}")
            torch.save(res_anchor_embeddings, output_path)

            for i in range(model_num):
                anchor_embeddings_list[i] = []
            batch_count = 0
            batch_index += 1
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
    # res_anchor_embeddings = {} # to be a dict {"model": (layer_num, anchor_num, d)}
    
    # for i in range(model_num):
    #     anchor_embeddings = anchor_embeddings_list[i]  # (anchor_num, layer_num, d)
    #     reshaped_anchor_embeddings = []
        
    #     for l in range(len(anchor_embeddings[0])):
    #         reshaped_anchor_embeddings.append(torch.stack([hidden_states[l] for hidden_states in anchor_embeddings]))
    #     res_anchor_embeddings[model_name_list[i]] = reshaped_anchor_embeddings
    # output_path = os.path.join(output_dir, '_'.join(model_name_list) + f"_{anchor_num}anchors_seed{seed}.pt")
    # print(f"anchor embeddings saved in: {output_path}")
    # torch.save(res_anchor_embeddings, output_path)

def output_same_token(t1, t2):
    return t1 in t2 or t2 in t1


def parse_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")

    # Add arguments with default values
    parser.add_argument('--anchor-num', type=int, default=1000000, help='Number of Anchors used to estimation the embedding projection.')
    # parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # data_path = "/data/home/cpfu/ychuang/reimplement_deepen/datasets/TriviaQA/wikipedia-dev-1900.jsonl"
    # data = read_multiline_json(data_path)
    # data = [i["question"] for i in data]

    # Load data and perform truncate
    args = parse_args()
    anchor_num = args.anchor_num
    data_path = "/share/home/fengxiaocheng/ychuang/Downloads/minipile/"
    dataset = [" ".join(i['text'].split()[:100]) for i in load_from_disk(data_path)["train"]]

    output_dir = f"{proj_path}/experiments/anchor_embeddings_v3/"

    extract_anchor_embeddings(["llama2-13b", "mistral-7b"], dataset, output_dir, anchor_num=anchor_num)
