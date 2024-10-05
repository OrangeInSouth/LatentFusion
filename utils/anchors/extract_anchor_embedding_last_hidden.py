import json
import random
import os
import pdb
import torch
import sys
sys.path.append("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang")

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import k_index
from utils.anchor_words import get_common_vocab_list_by_tokenizer

device="auto"

seed = 1
random.seed(seed)
anchor_num = 4000
data_path = "/data/home/cpfu/ychuang/reimplement_deepen/datasets/TriviaQA/wikipedia-dev-1900.jsonl"
output_dir = "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/anchor_embeddings/"
model_path = "/data3/cpfu/baohangli/ModelsHub/"
models = ["mistralai/Mistral-7B-v0.1",
        "internlm-20b",
        "Skywork/Skywork-13B-base",
        "Llama-2-13b-hf",
        "01-ai/Yi-6B-hf",
        "TigerResearch/tigerbot-13b-base-v2/",
        "Nanbeige/Nanbeige-16B-Base"]

if __name__ ==  "__main__":

    main_anchor_embeddings = []
    assist_anchor_embeddings = []
    # 1. Load data
    f = open(data_path)
    data = [json.loads(line) for line in f.readlines()]
    f.close()
    data = [i["question"] for i in data]
    random.shuffle(data)

    # 2. Load model and tokenizer
    main_model_path = f"{model_path}/{models[3]}"
    assist_model_path = f"{model_path}/{models[0]}"

    main_model = AutoModelForCausalLM.from_pretrained(main_model_path,
                                                     device_map=device,
                                                     torch_dtype="auto",
                                                     trust_remote_code=True)
    assist_model = AutoModelForCausalLM.from_pretrained(assist_model_path,
                                                     device_map=device,
                                                     torch_dtype="auto",
                                                     trust_remote_code=True)
    
    main_tokenizer = AutoTokenizer.from_pretrained(main_model_path, trust_remote_code=True)
    assist_tokenizer = AutoTokenizer.from_pretrained(assist_model_path, trust_remote_code=True)

    main_tokenizer.pad_token = main_tokenizer.eos_token
    assist_tokenizer.pad_token = assist_tokenizer.eos_token

    common_token_index_list = get_common_vocab_list_by_tokenizer(main_tokenizer, assist_tokenizer)
    # {"common_vocab_token": xxx, "main_model_vocab_index": xxx, "assist_model_vocab_index": xxx}
    common_token_set = [w["common_vocab_token"] for w in common_token_index_list]

    # 3. tokenize sentence and find the aligned token pair and their indices in the sentence.
    count = 0
    anchor_token_set = set()
    for tmp_j in tqdm(range(anchor_num)):
        sent = data[tmp_j % len(data)]
        main_token_id_list = main_tokenizer.encode(sent)
        assist_token_id_list = assist_tokenizer.encode(sent)

        MAX_trial_times = 10
        
        tmp_i = 0
        while tmp_i < MAX_trial_times:
            tmp_i += 1
            # 4. sample a common token from the selected sentence
            main_sampled_token_id = random.choice(main_token_id_list)
            sampled_token = main_tokenizer.convert_ids_to_tokens(main_sampled_token_id)
            
            if sampled_token in common_token_set and main_sampled_token_id > 3 and sampled_token not in anchor_token_set: # if sampled token is a common token
                assist_sampled_token_id = assist_tokenizer.convert_tokens_to_ids(sampled_token)
                assert assist_sampled_token_id > 0
                try:
                    assert main_token_id_list.count(main_sampled_token_id) == assist_token_id_list.count(assist_sampled_token_id)
                except Exception as e:
                    continue
                
                k = random.choice(range(main_token_id_list.count(main_sampled_token_id)))
                main_anchor_index = k_index(main_token_id_list, main_sampled_token_id, k+1)
                assist_anchor_index = k_index(assist_token_id_list, assist_sampled_token_id, k+1)

                # 5. model forward to obtain the representation
                main_model_inputs = {"input_ids": main_tokenizer.encode(sent, return_tensors="pt").to(main_model.device), 
                                     "return_dict": True, 
                                     "output_hidden_states": True}
                with torch.no_grad():
                    main_outputs = main_model(**main_model_inputs)
                    pdb.set_trace()
                    main_last_hidden_states = main_outputs['hidden_states'][-1][0] # L*d
                    main_anchor_embedding = main_last_hidden_states[main_anchor_index]

                    assist_model_inputs = {"input_ids": assist_tokenizer.encode(sent, return_tensors="pt").to(assist_model.device), 
                                        "return_dict": True, 
                                        "output_hidden_states": True}
                    assist_outputs = assist_model(**assist_model_inputs)
                    assist_last_hidden_states = assist_outputs['hidden_states'][-1][0]
                    assist_anchor_embedding = assist_last_hidden_states[assist_anchor_index]

                    main_anchor_embeddings.append((main_anchor_embedding))
                    assist_anchor_embeddings.append((assist_anchor_embedding))
                count += 1
                break

        
        if count == anchor_num:
            break

    main_anchor_embeddings = torch.stack(main_anchor_embeddings)
    assist_anchor_embeddings = torch.stack(assist_anchor_embeddings)
    output_path = os.path.join(output_dir, f"llama2_7b-mistral_7b.pt")
    torch.save({"llama2_7b": main_anchor_embeddings,
                "mistral_7b": assist_anchor_embeddings}, output_path)