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
seed += 1000  # set a different value of seed
random.seed(seed)
sample_num = 2000

data_path = "/data/home/cpfu/ychuang/reimplement_deepen/datasets/TriviaQA/wikipedia-dev-1900.jsonl"
anchor_embedding_file = "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/anchor_embeddings/llama2_7b-mistral_7b.pt"
model_path = "/data3/cpfu/baohangli/ModelsHub/"
models = ["mistralai/Mistral-7B-v0.1",
        "internlm-20b",
        "Skywork/Skywork-13B-base",
        "Llama-2-13b-hf",
        "01-ai/Yi-6B-hf",
        "TigerResearch/tigerbot-13b-base-v2/",
        "Nanbeige/Nanbeige-16B-Base"]

temperature = 10
def transform_to_relative_embedding(embedding, anchor_embeddings):
    rel_embedding = torch.cosine_similarity(embedding, anchor_embeddings)
    pdb.set_trace()
    rel_embedding = (rel_embedding * temperature).softmax(dim=-1)
    return rel_embedding


def measure_relative_embedding_consistency(embedding1, embedding2, method = "l2"):
    if method == "l2":
        return torch.nn.functional.mse_loss(embedding1, embedding2).item()
    if method == "cos":
        return torch.cosine_similarity(embedding1, embedding2, dim=0).item()


def avg(l):
    return sum(l) / len(l)


if __name__ ==  "__main__":

    # 1. Load data
    f = open(data_path)
    data = [json.loads(line) for line in f.readlines()]
    f.close()
    data = [i["question"] for i in data]
    random.shuffle(data)

    # 2. Load anchor embeddings
    anchor_embeddings = torch.load(anchor_embedding_file)
    main_anchor_embeddings = anchor_embeddings["llama2_7b"]
    assist_anchor_embeddings = anchor_embeddings["mistral_7b"]

    # 3. Load model and tokenizer
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

    common_token_index_list = get_common_vocab_list_by_tokenizer([main_tokenizer, assist_tokenizer])
    # {"common_vocab_token": xxx, "main_model_vocab_index": xxx, "assist_model_vocab_index": xxx}
    common_token_set = [w["token"] for w in common_token_index_list]

    # 4. tokenize sentence and find the aligned token pair and their indices in the sentence.
    count = 0
    count_same_output = 0
    count_different_output = 0
    similarity_same_output = []
    similarity_different_output = []
    similarity_main_intra_sentence = []
    similarity_main_inter_sentence = []
    similarity_assist_intra_sentence = []
    similarity_assist_inter_sentence = []

    for tmp_j in tqdm(range(sample_num)):
        sent = data[tmp_j % len(data)]
        main_token_id_list = main_tokenizer.encode(sent)
        assist_token_id_list = assist_tokenizer.encode(sent)

        MAX_trial_times = 10
        
        tmp_i = 0
        while tmp_i < MAX_trial_times:
            tmp_i += 1
            # 5. sample a common token from the selected sentence
            main_sampled_token_id = random.choice(main_token_id_list)
            sampled_token = main_tokenizer.convert_ids_to_tokens(main_sampled_token_id)
            
            if sampled_token in common_token_set and main_sampled_token_id > 3: # if sampled token is a common token
                assist_sampled_token_id = assist_tokenizer.convert_tokens_to_ids(sampled_token)
                assert assist_sampled_token_id > 0
                try:
                    assert main_token_id_list.count(main_sampled_token_id) == assist_token_id_list.count(assist_sampled_token_id)
                except Exception as e:
                    continue
                
                k = random.choice(range(main_token_id_list.count(main_sampled_token_id)))
                main_sampled_index = k_index(main_token_id_list, main_sampled_token_id, k+1)
                assist_sampled_index = k_index(assist_token_id_list, assist_sampled_token_id, k+1)

                rel_embedding_similarity_consistent = []
                rel_embedding_similarity_inconsistent = []
                # 6. model forward to obtain the representation
                with torch.no_grad():
                    main_model_inputs = {"input_ids": main_tokenizer.encode(sent, return_tensors="pt").to(main_model.device), 
                                     "return_dict": True, 
                                     "output_hidden_states": True}
                    main_outputs = main_model(**main_model_inputs)
                    
                    main_last_hidden_states = main_outputs['hidden_states'][-1][0] # L*d
                    main_sampled_embedding = main_last_hidden_states[main_sampled_index]
                    main_sentence_embedding = main_last_hidden_states.mean(dim=0)
                    main_sampled_logits = main_outputs['logits'][0][main_sampled_index] # d
                    main_sampled_output = main_tokenizer.convert_ids_to_tokens(main_sampled_logits.topk(5)[1]) # top-5 tokens in the output distribution
                    # print("main output:", main_sampled_output)

                    assist_model_inputs = {"input_ids": assist_tokenizer.encode(sent, return_tensors="pt").to(assist_model.device), 
                                        "return_dict": True, 
                                        "output_hidden_states": True}
                    assist_outputs = assist_model(**assist_model_inputs)
                    assist_last_hidden_states = assist_outputs['hidden_states'][-1][0]
                    assist_sampled_embedding = assist_last_hidden_states[assist_sampled_index]
                    assist_sentence_embedding = assist_last_hidden_states.mean(dim=0)
                    assist_sampled_logits = assist_outputs['logits'][0][assist_sampled_index] # d
                    assist_sampled_output = assist_tokenizer.convert_ids_to_tokens(assist_sampled_logits.topk(5)[1])
                    # print("assist output:", assist_sampled_output)

                    main_relative_embedding = transform_to_relative_embedding(main_sampled_embedding, main_anchor_embeddings)
                    assist_relative_embedding = transform_to_relative_embedding(assist_sampled_embedding, assist_anchor_embeddings)
                    relative_embedding_similarity = measure_relative_embedding_consistency(main_relative_embedding, assist_relative_embedding)
                    # print("Relative Embedding Similarity:", relative_embedding_similarity)
                    
                    def output_same_token(t1, t2):
                        return t1 in t2 or t2 in t1
                    
                    if output_same_token(main_sampled_output[0], assist_sampled_output[0]):
                        count_same_output += 1
                        similarity_same_output.append(relative_embedding_similarity)
                    else:
                        count_different_output += 1
                        similarity_different_output.append(relative_embedding_similarity)

                    # print("Baseline:")
                    main_sentence_relative_embedding = transform_to_relative_embedding(main_sentence_embedding, main_anchor_embeddings)
                    assist_sentence_relative_embedding = transform_to_relative_embedding(assist_sentence_embedding, assist_anchor_embeddings)
                    
                    sim_main_intra = measure_relative_embedding_consistency(main_relative_embedding, main_sentence_relative_embedding)
                    # print("Main Intra-Sentence:", sim_main_intra)
                    similarity_main_intra_sentence.append(sim_main_intra)

                    sim_main_inter = measure_relative_embedding_consistency(main_relative_embedding, assist_sentence_relative_embedding)
                    # print("Main Inter-Sentence:", sim_main_inter)
                    similarity_main_inter_sentence.append(sim_main_inter)

                    sim_assist_intra = measure_relative_embedding_consistency(assist_relative_embedding, assist_sentence_relative_embedding)
                    # print("Assist Intra-Sentence:", sim_assist_intra)
                    similarity_assist_intra_sentence.append(sim_assist_intra)

                    sim_assist_inter = measure_relative_embedding_consistency(assist_relative_embedding, main_sentence_relative_embedding)
                    # print("Assist Inter-Sentence:", sim_assist_inter)
                    similarity_assist_inter_sentence.append(sim_assist_inter)
                    
                    # neighbour_anchors = list(set(main_relative_embedding.topk(10)[1].tolist() + assist_relative_embedding.topk(10)[1].tolist()))
                    # main_new_relative_embedding = main_relative_embedding[neighbour_anchors]
                    # assist_new_relative_embedding = assist_relative_embedding[neighbour_anchors]
                    # print("Modified Relative Embedding Similarity:", torch.cosine_similarity(main_new_relative_embedding, assist_new_relative_embedding, dim=0))
                    
                    # pdb.set_trace()
                break


    print(f"count_same_output: {count_same_output}")
    print(f"count_different_output: {count_different_output}")
    print(f"similarity_same_output: {avg(similarity_same_output)}")
    print(f"similarity_different_output: {avg(similarity_different_output)}")
    print(f"similarity_main_intra_sentence: {avg(similarity_main_intra_sentence)}")
    print(f"similarity_main_inter_sentence: {avg(similarity_main_inter_sentence)}")
    print(f"similarity_assist_intra_sentence: {avg(similarity_assist_intra_sentence)}")
    print(f"similarity_assist_inter_sentence: {avg(similarity_assist_inter_sentence)}")