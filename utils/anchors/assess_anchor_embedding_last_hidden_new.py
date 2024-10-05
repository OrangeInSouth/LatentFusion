import json
import random
import os
import pdb
import torch
import sys
sys.path.append("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang")

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import k_index, read_multiline_json, avg, standalize
from utils.anchor_words import get_common_vocab_list_by_tokenizer
from model_config import model_paths
device="auto"


models = ["llama2-13b", "mistral-7b"]
anchor_num = sys.argv[1]
knn = sys.argv[2]

anchor_num = int(anchor_num)
knn = int(knn)

# For Cosine Similarity
temperatures = [10, 10]
# For Eucidean Distance
# temperatures = [5, 5]

models = "_".join(models)
data_path = "/data/home/cpfu/ychuang/reimplement_deepen/datasets/TriviaQA/wikipedia-dev-1900.jsonl"
anchor_embedding_file = f"/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/anchor_embeddings/{models}_{anchor_num}anchors_seed1.pt"
res_file = f"/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/anchor_embeddings/{models}_{anchor_num}anchors_{knn}_seed1_consistency.txt"

sample_num = 100  # S
seed = 1
seed += 1000  # set a different value of seed
random.seed(seed)


def transform_to_relative_embedding(embedding, anchor_embeddings, temperature, method="cos"): # cos Euclidean Distance

    if method == "cos":
        return (torch.cosine_similarity(embedding, anchor_embeddings, dim=-1) * temperature).softmax(dim=-1)
    elif method == "kernel":
        # pdb.set_trace()
        return (embedding * anchor_embeddings).sum(dim=-1)
        # return standalize((embedding * anchor_embeddings).sum(dim=-1))
    elif method == "Euclidean Distance":
        d = (embedding - anchor_embeddings).norm(dim=-1)
        s = standalize(d * -1)
        r = (s * temperature).softmax(dim=-1)
        return r




def measure_cross_model_relative_embedding_consistency(embedding1, embedding2, method = "l2"):
    """
    embedding1: (N1, A)
    embedding2: (N2, A)
    return (N1, N2)
    """
    if method == "l2":
        return ((embedding1.unsqueeze(dim=1) - embedding2.unsqueeze(dim=0)) ** 2).mean(dim=-1)
    elif method == "cos":
        return torch.cosine_similarity(embedding1.unsqueeze(dim=1), embedding2.unsqueeze(dim=0), dim=-1)
    elif method == "MuKNN":
        pdb.set_trace()
        s1_knn = embedding1.topk(knn, dim=1)[1].tolist()
        s2_knn = embedding2.topk(knn, dim=1)[1].tolist()

        res = torch.zeros(embedding1.size(0), embedding2.size(0), dtype=torch.float)
        for i in range(embedding1.size(0)):
            for j in range(embedding2.size(0)):
                res[i][j] = len(set(s1_knn[i]) & set(s2_knn[j]))/ knn
        return res


        # s1 = torch.zeros_like(embedding1, dtype=torch.bool)
        # s1.scatter_(1, embedding1.topk(k, dim=1)[1], 1)

        # s2 = torch.zeros_like(embedding2, dtype=torch.bool)
        # s2.scatter_(1, embedding2.topk(k, dim=1)[1], 1)

        # pdb.set_trace()

        # return (s1.unsqueeze(dim=1) & s2.unsqueeze(dim=0)).float().mean(dim=-1)




def measure_relative_embedding_consistency(embedding1, embedding2, method = "l2"):
    if method == "l2":
        return torch.nn.functional.mse_loss(embedding1, embedding2).item()
    if method == "cos":
        return torch.cosine_similarity(embedding1, embedding2, dim=0).item()
    

if __name__ ==  "__main__":
    model_name_list = ["llama2-13b", "mistral-7b"]
    model_num = len(model_name_list)

    # 1. Load data
    data = read_multiline_json(data_path)
    data = [i["question"] for i in data]
    random.shuffle(data)

    # 2. Load anchor embeddings
    anchor_embeddings_dict = torch.load(anchor_embedding_file)
    anchor_embedding_list = [torch.stack(anchor_embeddings_dict[name][1:]) for name in model_name_list]
    
    # 3. Load model and tokenizer
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

    # 4. Tokenize sentence and find the aligned token pair and their indices in the sentence.   
    layer_align_matrix_list = []  # (S, N1 * N2)
    
    rand_sent_rel_embed_list = []  # (2, N, A)
    intra_sent_rel_emb_sim_list = [] # (2, S), indicate the similarity of relative representations between the sampled token embedding and the sentence embedding.
    inter_sent_rel_emb_sim_list = [] # (2, S), indicate the similarity of relative representations between the sampled token embedding and the irrelevant sentence embedding.
    for i in range(model_num):
        rand_sent_rel_embed_list.append(None)
        intra_sent_rel_emb_sim_list.append([])
        inter_sent_rel_emb_sim_list.append([])

    for tmp_j in tqdm(range(sample_num)):
        sent = data[tmp_j % len(data)]
        token_ids_list = [tokenizer.encode(sent) for tokenizer in tokenizer_list]

        MAX_trial_times = 10
        
        for tmp_i in range(MAX_trial_times):
            # 5. Sample a common token from the selected sentence
            sampled_token_index = random.choice(token_ids_list[0])
            sampled_token = tokenizer_list[0].convert_ids_to_tokens(sampled_token_index)
            
            if sampled_token in common_token_set and sampled_token_index > 3: # if sampled token is a common token
                sampled_token_index_list = [tokenizer.convert_tokens_to_ids(sampled_token) for tokenizer in tokenizer_list]
                assert all([index > 0 for index in sampled_token_index_list])
                
                try:
                    assert len(set([token_ids_list[i].count(sampled_token_index_list[i]) for i in range(model_num)])) == 1
                except Exception:
                    continue
                
                # Deduce the position of the k-th ct
                k = random.choice(range(token_ids_list[0].count(sampled_token_index)))
                position_list = [k_index(token_ids_list[i], sampled_token_index_list[i], k+1) for i in range(model_num)]

                # 6. model forward to obtain the representation
                with torch.no_grad():
                    selected_position_rel_embed_list = []

                    for i in range(model_num):
                        tokenizer = tokenizer_list[i]
                        model = model_list[i]
                        selected_position = position_list[i]
                        # pdb.set_trace()
                        anchor_embeddings = anchor_embedding_list[i]  # (N, A, d)
                        temperature = temperatures[i]
                        
                        model_input = {"input_ids": tokenizer.encode(sent, return_tensors="pt").to(model.device), 
                                     "return_dict": True, 
                                     "output_hidden_states": True}

                        model_output = model(**model_input) # a tuple of (L+1) tensors, each one is (B, L, d)
                        # (1) get the hidden states of all positions in all layers: (N, L*d) 
                        all_hidden_states = [layer_output[0] for layer_output in model_output['hidden_states'][1:]]
                        # (2) get the sentence embedding in all layers: (N, d)
                        sent_embed = [layer_states.mean(dim=0) for layer_states in all_hidden_states]
                        sent_embed = torch.stack(sent_embed)
                        # (3) get the selected token's embedding in all layers: (N, d)
                        sampled_token_embed = [layer_states[selected_position] for layer_states in all_hidden_states]
                        sampled_token_embed = torch.stack(sampled_token_embed)
                        # (4) Convert absolute embeddings to relative embeddings
                        sentence_rel_embed = transform_to_relative_embedding(sent_embed.unsqueeze(dim=1), anchor_embeddings, temperature)  # (N, A)
                        sampled_token_rel_embed = transform_to_relative_embedding(sampled_token_embed.unsqueeze(dim=1), anchor_embeddings, temperature)  # (N, A)
                        selected_position_rel_embed_list.append(sampled_token_rel_embed)
                        
                        intra_sent_rel_emb_sim_list[i].append(measure_relative_embedding_consistency(sampled_token_rel_embed, sentence_rel_embed))
                        if rand_sent_rel_embed_list[i] is not None:
                            # pdb.set_trace()
                            inter_sent_rel_emb_sim_list[i].append(measure_relative_embedding_consistency(sampled_token_rel_embed, rand_sent_rel_embed_list[i]))
                        rand_sent_rel_embed_list[i] = sentence_rel_embed

                    # Calculate the Cross-Model Relative Embedding Consistency 
                    xmrec = measure_cross_model_relative_embedding_consistency(selected_position_rel_embed_list[0], selected_position_rel_embed_list[1], method="MuKNN")
                    layer_align_matrix_list.append(xmrec)


                break


    print("Cross-Model Aligment Score in Terms of Relative Representations:")
    layer_align_matrix_list = torch.stack(layer_align_matrix_list)
    layer_align_matrix = layer_align_matrix_list.mean(dim=0)
    print(f"Alignment: mean ({layer_align_matrix.mean()}), var({layer_align_matrix.var()}) max({layer_align_matrix.max()}), min({layer_align_matrix.min()})")
    layer_align_matrix = layer_align_matrix.tolist()
    layer_align_matrix.reverse()

    f = open(res_file, 'w+')
    for line in layer_align_matrix:
        line_str = "\t".join([str(round(i, 2)) for i in line]) + '\n'
        print(line_str)
        f.write(line_str)
        # print("\t".join([str(round(i, 2)) for i in line]))
    f.close()

    layer_align_matrix = torch.tensor(layer_align_matrix)
    print(f"40-32 alignment: {layer_align_matrix[0][31]}")
    print(f"mean alignment: {avg([avg(line) for line in layer_align_matrix])}")
    print(f"max alignment: {max([max(line) for line in layer_align_matrix])}")
