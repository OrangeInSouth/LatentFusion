import torch
import os
from tqdm import tqdm
import pdb
import re

anchor_dir = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/anchor_embeddings/"
anchor_prefix = "llama2-13b_mistral-7b_1000000anchors_seed1"

models = ["llama2-13b", "mistral-7b"]
layer_pair = [40, 32]

output_path = f"{anchor_dir}/{anchor_prefix}_layer{'-'.join([str(i) for i in layer_pair])}.pt"

files = [f for f in os.listdir(anchor_dir) if re.match(fr"^{re.escape(anchor_prefix)}_\d+\.pt$", f)]
files = sorted(files)

states = []
for file in tqdm(files):
    state = torch.load(f"{anchor_dir}/{file}", map_location=torch.device('cpu'))
    single_layer_state = {}
    for index, model in enumerate(models):
        try:
            assert len(state[model][layer_pair[index]].shape) == 2
            single_layer_state[model] = state[model][layer_pair[index]]
        except Exception:
            pdb.set_trace()
    states.append(single_layer_state)

res_state = {}
for model in models:
    res_state[model] = torch.concatenate([state[model] for state in states])


torch.save(res_state, output_path)

print(f"Merged {len(res_state[models[0]])} anchors.")
print(f"Saved into: {output_path}")
